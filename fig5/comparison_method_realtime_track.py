import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import networkx as nx
import multiprocessing as mp
import warnings
import torch

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_FIG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_FIG_DIR, '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'train'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'fig3'))

from comparison_method import (
    td_method_adaptive,
    katz_method_adaptive,
    extended_tia_method_adaptive,
    RLAgentWrapper,
    GNN_AVAILABLE,
    IncrementalPyGBuilder,
    generate_graph,
    generate_random_targets,
    generate_localized_targets,
    compute_params_from_lambda,
)

print("✓ Successfully imported from comparison_method")

class Config:
    N_NODES          = 1024
    TARGET_RATIO     = 0.05
    LAMBDA           = 20
    SIMULATION_TIMES = 100
    NUM_BINS         = 100
    NUM_WORKERS      = 70
    MODEL_PATH       = os.path.join(_REPO_ROOT, 'train',
                                    'gnn_ppo_dual_stream_n50-100__seed42__1769653775',
                                    'model.pt')
    SAVE_DIR         = os.path.join(_FIG_DIR, 'data')
    GRAPH_PARAMS     = {
        'ba_m_range':    (2, 4),
        'ws_k_range':    (4, 8),
        'ws_beta_range': (0.1, 0.3),
    }

_worker_rl_agent = None

METRICS = ['distances', 'local_betweenness', 'cost_benefit', 'isolated_ratio']


def compute_distance_to_targets(G: nx.Graph, node: int,
                                 target_nodes: set) -> float:
    if node not in G or node in target_nodes:
        return 0.0
    min_dist = float('inf')
    for t in target_nodes:
        if t not in G:
            continue
        try:
            d = nx.shortest_path_length(G, node, t)
            if d < min_dist:
                min_dist = d
            if min_dist == 1:
                return 1.0
        except nx.NetworkXNoPath:
            pass
    return min_dist if min_dist != float('inf') else float('inf')


def compute_local_betweenness(G: nx.Graph, node: int,
                               target_nodes: list) -> float:
    if node not in G or node in target_nodes:
        return 0.0
    try:
        bc = nx.betweenness_centrality_subset(
            G, sources=list(target_nodes),
            targets=list(G.nodes()), normalized=True
        )
        return bc.get(node, 0.0)
    except Exception:
        return 0.0


def map_to_bins(data: List[float], num_bins: int) -> np.ndarray:
    if not data:
        return np.zeros(num_bins)
    n = len(data)
    if n == 1:
        return np.full(num_bins, data[0])
    return np.interp(
        np.linspace(0, n - 1, num_bins),
        np.arange(n),
        data
    )


class RealtimeTrackingSimulator:
    def __init__(self, G: nx.Graph, targets: List[int],
                 move_prob: float, attack_per_step: int,
                 seed: int = None):
        self.original_graph      = G.copy()
        self.current_graph       = G.copy()
        self.target_nodes        = set(targets)
        self.n_initial_targets   = len(targets)   # constant denominator for isolated_ratio
        self.move_prob           = move_prob
        self.attack_per_step     = attack_per_step
        self.rng                 = np.random.RandomState(seed)

        self.n_nodes             = G.number_of_nodes()
        self.n_non_targets       = self.n_nodes - len(targets)
        self.removed_nodes       = set()
        self.attacks_this_round  = 0

        self.initial_lcc_size    = self._lcc_size()
        self.cumulative_anc      = 0.0
        self.anc_count           = 0

        self.tracked_data: Dict[str, List[float]] = {m: [] for m in METRICS}


    def _lcc_size(self) -> int:
        comps = list(nx.connected_components(self.current_graph))
        return len(max(comps, key=len)) if comps else 0

    def _targets_in_lcc(self) -> int:
        comps = list(nx.connected_components(self.current_graph))
        if not comps:
            return 0
        lcc = max(comps, key=len)
        return sum(1 for t in self.target_nodes if t in lcc)

    def is_terminal(self) -> bool:
        return self._targets_in_lcc() == 0

    def _move_targets(self):
        if self.move_prob <= 0:
            return
        movements = []
        claimed   = set()
        for t in list(self.target_nodes):
            if t not in self.current_graph:
                continue
            if self.rng.random() > self.move_prob:
                continue
            nbrs = [n for n in self.current_graph.neighbors(t)
                    if n not in self.removed_nodes
                    and n not in self.target_nodes
                    and n not in claimed]
            if nbrs:
                dest = self.rng.choice(nbrs)
                movements.append((t, dest))
                claimed.add(dest)
        for old, new in movements:
            self.target_nodes.discard(old)
            self.target_nodes.add(new)


    def step(self, action: int) -> bool:
        if (action in self.target_nodes or action in self.removed_nodes
                or action not in self.current_graph):
            return self.is_terminal()

        dist = compute_distance_to_targets(
            self.current_graph, action, self.target_nodes
        )
        self.tracked_data['distances'].append(dist)

        lb = compute_local_betweenness(
            self.current_graph, action, list(self.target_nodes)
        )
        self.tracked_data['local_betweenness'].append(lb)

        prev_lcc = self._lcc_size()
        self.removed_nodes.add(action)
        self.current_graph.remove_node(action)
        self.attacks_this_round += 1

        curr_lcc = self._lcc_size()
        self.cumulative_anc += curr_lcc / max(1, self.initial_lcc_size)
        self.anc_count      += 1

        delta_pc  = 1.0 / max(1, self.n_non_targets)
        delta_lcc = (prev_lcc - curr_lcc) / max(1, self.initial_lcc_size)
        self.tracked_data['cost_benefit'].append(delta_lcc / delta_pc)

        targets_in_lcc  = self._targets_in_lcc()
        isolated        = self.n_initial_targets - targets_in_lcc
        self.tracked_data['isolated_ratio'].append(
            isolated / max(1, self.n_initial_targets)
        )

        if self.is_terminal():
            return True

        if self.attacks_this_round >= self.attack_per_step:
            self._move_targets()
            self.attacks_this_round = 0

        return False

    def get_tracked_data(self) -> Dict[str, List[float]]:
        return self.tracked_data



def run_heuristic_realtime(
        G: nx.Graph, targets: List[int], method_func,
        move_prob: float, attack_per_step: int,
        seed: int, batch_mode: bool = False,
) -> Dict[str, List[float]]:
    sim       = RealtimeTrackingSimulator(G, targets, move_prob, attack_per_step, seed)
    max_steps = G.number_of_nodes()

    if batch_mode:
        while not sim.is_terminal():
            quota   = sim.attack_per_step - sim.attacks_this_round
            if quota <= 0:
                if not sim.is_terminal():
                    sim._move_targets()
                sim.attacks_this_round = 0
                quota = sim.attack_per_step
            ranking = method_func(sim.current_graph, list(sim.target_nodes))
            batch   = [n for n in ranking
                       if n not in sim.removed_nodes
                       and n not in sim.target_nodes
                       and n in sim.current_graph][:quota]
            if not batch:
                break
            for node in batch:
                done = sim.step(node)
                if done:
                    break
            else:
                continue
            break
    else:
        for _ in range(max_steps):
            if sim.is_terminal():
                break
            try:
                order = method_func(sim.current_graph, list(sim.target_nodes))
                order = [n for n in order
                         if n not in sim.removed_nodes
                         and n not in sim.target_nodes
                         and n in sim.current_graph]
                if not order:
                    break
                if sim.step(order[0]):
                    break
            except Exception:
                break

    return sim.get_tracked_data()


def run_rl_realtime(
        G: nx.Graph, targets: List[int], rl_agent: RLAgentWrapper,
        move_prob: float, attack_per_step: int, seed: int,
) -> Dict[str, List[float]]:
    sim       = RealtimeTrackingSimulator(G, targets, move_prob, attack_per_step, seed)
    max_steps = G.number_of_nodes()
    builder   = IncrementalPyGBuilder(sim.original_graph, sim.target_nodes)

    for _ in range(max_steps):
        if sim.is_terminal():
            break
        try:
            data, nodes = builder.build_data(
                sim.current_graph,
                sim.target_nodes,
                sim.removed_nodes,
                move_prob,
                sim.attacks_this_round,
                attack_per_step,
            )
            action_idx = rl_agent.get_action_deterministic(data)
            action     = nodes[action_idx]
            if action in sim.target_nodes or action not in sim.current_graph:
                break
            if sim.step(action):
                break
        except Exception:
            break

    return sim.get_tracked_data()


def _init_worker(model_path: str):
    global _worker_rl_agent
    torch.set_num_threads(1)
    if model_path and os.path.exists(model_path) and GNN_AVAILABLE:
        try:
            _worker_rl_agent = RLAgentWrapper(model_path, device='cpu')
        except Exception as e:
            _worker_rl_agent = None
            print(f"  RL worker init error: {e}")
    else:
        _worker_rl_agent = None


def _worker_task(params):
    (sim_idx, gtype, tdist, graph_params,
     n_nodes, target_ratio, lambda_val, num_bins) = params

    seed = 42 + sim_idx * 1000

    G = generate_graph(gtype, n_nodes, graph_params, seed)
    if tdist == 'random':
        targets = generate_random_targets(G, target_ratio, seed)
    else:
        targets = generate_localized_targets(G, target_ratio, seed)

    move_prob, attack_ratio = compute_params_from_lambda(lambda_val)
    n_targets       = len(targets)
    attack_per_step = max(1, int(attack_ratio * n_targets))

    run_results = {}

    method_configs = [
        ('Adaptive TD',   td_method_adaptive,           True),
        ('Adaptive Katz', katz_method_adaptive,          False),
        ('Adaptive TIA',  extended_tia_method_adaptive,  True),
    ]
    for m_name, m_func, batch in method_configs:
        try:
            raw = run_heuristic_realtime(
                G.copy(), targets, m_func,
                move_prob, attack_per_step, seed, batch
            )
            run_results[m_name] = {
                metric: map_to_bins(raw[metric], num_bins)
                for metric in METRICS
            }
        except Exception:
            run_results[m_name] = {
                metric: np.zeros(num_bins) for metric in METRICS
            }

    if _worker_rl_agent is not None:
        try:
            raw = run_rl_realtime(
                G.copy(), targets, _worker_rl_agent,
                move_prob, attack_per_step, seed
            )
            run_results['GNN-RL'] = {
                metric: map_to_bins(raw[metric], num_bins)
                for metric in METRICS
            }
        except Exception:
            run_results['GNN-RL'] = {
                metric: np.zeros(num_bins) for metric in METRICS
            }

    return run_results


def save_to_csv(stats_results: Dict, num_bins: int, save_dir: str,
                n_nodes: int, lambda_val: float) -> str:
    time_points = np.linspace(0, 1, num_bins)
    rows = []
    for config, methods_dict in stats_results.items():
        for method, metric_dict in methods_dict.items():
            for b, t in enumerate(time_points):
                row = {'config': config, 'method': method, 'time_bin': round(t, 6)}
                for metric in METRICS:
                    row[f'{metric}_mean'] = float(metric_dict.get(f'{metric}_mean',
                                                                   np.zeros(num_bins))[b])
                    row[f'{metric}_std']  = float(metric_dict.get(f'{metric}_std',
                                                                   np.zeros(num_bins))[b])
                    n = metric_dict.get('n_runs', 1)
                    sem = row[f'{metric}_std'] / np.sqrt(max(1, n))
                    row[f'{metric}_sem'] = float(sem)
                rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir,
                            f'realtime_tracking_n{n_nodes}_lambda{lambda_val}.csv')
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved: {csv_path}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    return csv_path


NMI_RC = {
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
    'font.size':         6,
    'axes.labelsize':    7,
    'axes.titlesize':    7,
    'xtick.labelsize':   6,
    'ytick.labelsize':   6,
    'legend.fontsize':   6,
    'axes.linewidth':    0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size':  2.5,
    'ytick.major.size':  2.5,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
}

COLORS = {
    'Adaptive TD':   '#0072B2',
    'Adaptive Katz': '#D55E00',
    'Adaptive TIA':  '#009E73',
    'GNN-RL':        '#CC79A7',
}
MARKERS = {
    'Adaptive TD':   'o',
    'Adaptive Katz': 's',
    'Adaptive TIA':  '^',
    'GNN-RL':        'D',
}
METHOD_ORDER = ['Adaptive TD', 'Adaptive Katz', 'Adaptive TIA', 'GNN-RL']

CONFIGS      = ['BA-random', 'BA-localized', 'WS-random', 'WS-localized']
COL_TITLES   = ['BA-Random', 'BA-Localized', 'WS-Random', 'WS-Localized']
PANEL_LABELS = list('abcdefghijklmnop')

METRIC_META = {
    'distances':         'Distance to Nearest Target',
    'local_betweenness': 'Local Betweenness Centrality',
    'cost_benefit':      'Cost-Benefit Ratio (ΔLCC / ΔPC)',
    'isolated_ratio':    'Isolated Target Ratio',
}


def _make_legend_handles():
    return [
        mlines.Line2D([], [],
                      color=COLORS[m], marker=MARKERS[m],
                      markersize=4, linewidth=0.8,
                      markerfacecolor=COLORS[m],
                      markeredgecolor='white', markeredgewidth=0.3,
                      label=m)
        for m in METHOD_ORDER
    ]


def plot_metric_2x2(metric_key: str, stats_results: Dict,
                    num_bins: int, n_nodes: int, lambda_val: float,
                    save_dir: str) -> None:
    mpl.rcParams.update(NMI_RC)

    ylabel    = METRIC_META[metric_key]
    time_pts  = np.linspace(0, 1, num_bins)

    LEG_H = 0.22
    PAN_H = 1.55
    FIG_W = 7.087

    fig = plt.figure(figsize=(FIG_W, LEG_H + 2 * PAN_H + 0.15))
    gs  = fig.add_gridspec(
        3, 2,
        height_ratios=[LEG_H, PAN_H, PAN_H],
        hspace=0.46, wspace=0.35,
        left=0.09, right=0.99,
        top=0.97,  bottom=0.10,
    )

    ax_leg = fig.add_subplot(gs[0, :])
    ax_leg.set_axis_off()
    ax_leg.legend(
        handles=_make_legend_handles(),
        loc='center', ncol=len(METHOD_ORDER),
        frameon=False, fontsize=6,
        handlelength=1.8, handletextpad=0.4,
        columnspacing=1.2, borderpad=0,
    )

    panel_idx = 0
    for row in range(2):
        for col in range(2):
            config = CONFIGS[panel_idx]
            title  = COL_TITLES[panel_idx]
            label  = PANEL_LABELS[panel_idx]
            panel_idx += 1

            ax = fig.add_subplot(gs[row + 1, col])

            if config not in stats_results:
                ax.set_visible(False)
                continue

            for method in METHOD_ORDER:
                md = stats_results[config].get(method, {})
                mean_key = f'{metric_key}_mean'
                std_key  = f'{metric_key}_std'
                if mean_key not in md:
                    continue
                mean = md[mean_key]
                std  = md[std_key]
                n    = md.get('n_runs', 1)
                sem  = std / np.sqrt(max(1, n))

                ax.plot(time_pts, mean,
                        color=COLORS[method], marker=MARKERS[method],
                        markevery=10, markersize=2.5, linewidth=0.8,
                        markerfacecolor=COLORS[method],
                        markeredgecolor='white', markeredgewidth=0.3)
                ax.fill_between(time_pts, mean - sem, mean + sem,
                                color=COLORS[method], alpha=0.15, linewidth=0)

            ax.set_title(title, fontsize=7, pad=3)
            ax.set_xlabel('Normalised Game Time', fontsize=7, labelpad=2)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=7, labelpad=3)
            ax.tick_params(which='both', direction='in',
                           top=True, right=True, labelsize=6, pad=2)
            ax.set_xlim(0, 1)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

            ax.text(-0.16 if col == 0 else -0.12, 1.08, label,
                    transform=ax.transAxes,
                    fontsize=8, fontweight='bold', va='top', ha='left')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        stem = os.path.join(save_dir,
                            f'realtime_{metric_key}_n{n_nodes}_lambda{lambda_val}')
        fig.savefig(stem + '.pdf')
        fig.savefig(stem + '.png')
        print(f"  Figure saved: {stem}.pdf / .png")

    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    try:
        mp.set_start_method('fork')
    except (RuntimeError, ValueError):
        pass

    os.makedirs(Config.SAVE_DIR, exist_ok=True)

    print("=" * 80)
    print("Real-time Tracking Analysis: Adaptive Heuristics vs GNN-RL (Parallel)")
    print("=" * 80)
    print(f"  N_NODES          : {Config.N_NODES}")
    print(f"  TARGET_RATIO     : {Config.TARGET_RATIO:.2%}")
    print(f"  LAMBDA           : {Config.LAMBDA}")
    print(f"  SIMULATION_TIMES : {Config.SIMULATION_TIMES}")
    print(f"  NUM_BINS         : {Config.NUM_BINS}")
    print(f"  WORKERS          : {Config.NUM_WORKERS}")
    print(f"  METRICS          : {METRICS}")
    print("=" * 80)

    all_raw = defaultdict(
        lambda: defaultdict(
            lambda: {m: [] for m in METRICS}
        )
    )

    CONFIGS_RUN = [
        ('BA', 'random'),
        ('BA', 'localized'),
        ('WS', 'random'),
        ('WS', 'localized'),
    ]

    pool = mp.Pool(
        processes=Config.NUM_WORKERS,
        initializer=_init_worker,
        initargs=(Config.MODEL_PATH,)
    )

    t0 = time.time()
    try:
        for gtype, tdist in CONFIGS_RUN:
            config_key = f"{gtype}-{tdist}"
            tasks = [
                (sim_idx, gtype, tdist, Config.GRAPH_PARAMS,
                 Config.N_NODES, Config.TARGET_RATIO,
                 Config.LAMBDA, Config.NUM_BINS)
                for sim_idx in range(Config.SIMULATION_TIMES)
            ]
            print(f"\nRunning {config_key} ({len(tasks)} simulations)...")
            for result in tqdm(
                    pool.imap_unordered(_worker_task, tasks),
                    total=len(tasks), desc=config_key):
                for method_name, metric_dict in result.items():
                    for metric, arr in metric_dict.items():
                        all_raw[config_key][method_name][metric].append(arr)

    except KeyboardInterrupt:
        print("\nInterrupted — terminating workers.")
        pool.terminate()
        sys.exit(1)
    finally:
        pool.close()
        pool.join()

    print(f"\nSimulations done in {(time.time() - t0)/60:.1f} min")

    print("\nComputing statistics...")
    stats_results = {}
    for config_key in all_raw:
        stats_results[config_key] = {}
        for method in all_raw[config_key]:
            n_runs = len(all_raw[config_key][method][METRICS[0]])
            stats_results[config_key][method] = {'n_runs': n_runs}
            for metric in METRICS:
                arrays = all_raw[config_key][method][metric]
                if arrays:
                    arr = np.array(arrays)
                    stats_results[config_key][method][f'{metric}_mean'] = np.mean(arr, axis=0)
                    stats_results[config_key][method][f'{metric}_std']  = np.std(arr,  axis=0)
                else:
                    stats_results[config_key][method][f'{metric}_mean'] = np.zeros(Config.NUM_BINS)
                    stats_results[config_key][method][f'{metric}_std']  = np.zeros(Config.NUM_BINS)

    print("\nSaving CSV...")
    save_to_csv(
        stats_results, Config.NUM_BINS, Config.SAVE_DIR,
        Config.N_NODES, Config.LAMBDA
    )

    print("\nPlotting...")
    for metric in METRICS:
        plot_metric_2x2(
            metric_key=metric,
            stats_results=stats_results,
            num_bins=Config.NUM_BINS,
            n_nodes=Config.N_NODES,
            lambda_val=Config.LAMBDA,
            save_dir=Config.SAVE_DIR,
        )

    print("\n" + "=" * 80)
    print("✓ Real-time Tracking Analysis Completed!")
    print("=" * 80)