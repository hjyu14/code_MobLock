import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from typing import Optional, Tuple, Dict, Any, List
import torch
from torch_geometric.data import Data
import math


class TargetIsolationEnvDual(gym.Env):

    metadata = {'render_modes': ['human']}

    def __init__(
            self,
            graph: nx.Graph,
            target_nodes: List[int],
            max_steps: Optional[int] = None,
            move_prob: float = 0.0,
            attack_per_step: int = 1,
    ):
        super().__init__()

        self.original_graph = graph.copy()
        self.target_nodes = set(target_nodes)
        self.non_target_nodes = set(graph.nodes()) - self.target_nodes
        self.n_nodes = graph.number_of_nodes()
        self.n_targets = len(target_nodes)
        self.n_non_targets = len(self.non_target_nodes)

        self.node_list = sorted(list(graph.nodes()))
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}
        self.idx_to_node = {idx: node for idx, node in enumerate(self.node_list)}

        self.move_prob = move_prob
        self.attack_per_step = attack_per_step

        self.max_steps = max_steps if max_steps else len(self.non_target_nodes)

        self.action_space = spaces.Discrete(self.n_nodes)

        self.observation_space = spaces.Dict({
            "x": spaces.Box(low=0, high=np.inf, shape=(self.n_nodes, 3), dtype=np.float32),
            "global_x": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        })

        self.current_graph = None
        self.removed_nodes = None
        self.steps = 0
        self.rounds = 0
        self.current_attack_count = 0

        self.initial_lcc_size = 0
        self.cumulative_anc = 0.0
        self.anc_count = 0

        self.rng = np.random.RandomState()

    def update_graph(self, graph: nx.Graph, target_nodes: List[int],
                     move_prob: float = None, attack_per_step: int = None):

        self.original_graph = graph.copy()
        self.target_nodes = set(target_nodes)
        self.non_target_nodes = set(graph.nodes()) - self.target_nodes
        self.n_nodes = graph.number_of_nodes()
        self.n_targets = len(target_nodes)
        self.n_non_targets = len(self.non_target_nodes)

        self.node_list = sorted(list(graph.nodes()))
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}
        self.idx_to_node = {idx: node for idx, node in enumerate(self.node_list)}

        self.max_steps = len(self.non_target_nodes)

        if move_prob is not None:
            self.move_prob = move_prob
        if attack_per_step is not None:
            self.attack_per_step = attack_per_step

        self.action_space = spaces.Discrete(self.n_nodes)
        self.observation_space = spaces.Dict({
            "x": spaces.Box(low=0, high=np.inf, shape=(self.n_nodes, 3), dtype=np.float32),
            "global_x": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        })

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Data, Dict]:

        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.current_graph = self.original_graph.copy()
        self.removed_nodes = set()
        self.steps = 0
        self.rounds = 0
        self.current_attack_count = 0

        self.target_nodes = set([n for n in self.target_nodes if n in self.original_graph])
        self.non_target_nodes = set(self.original_graph.nodes()) - self.target_nodes

        self.initial_lcc_size = self._get_lcc_size()
        self.cumulative_anc = 0.0
        self.anc_count = 0

        data = self._get_pyg_data()
        info = self._get_info()
        info['action_mask'] = self._get_action_mask()

        return data, info

    def step(self, action: int) -> Tuple[Data, float, bool, bool, Dict]:

        if isinstance(action, (np.ndarray, torch.Tensor)):
            action = int(action.item()) if hasattr(action, 'item') else int(action)

        self.steps += 1

        if action < 0 or action >= self.n_nodes:
            reward = 0.0
            terminated = False
            truncated = self.steps >= self.max_steps
            data = self._get_pyg_data()
            info = self._get_info()
            info['action_mask'] = self._get_action_mask()
            info['invalid_action'] = True
            return data, reward, terminated, truncated, info

        node_id = self.idx_to_node[action]

        if node_id in self.target_nodes or node_id in self.removed_nodes or node_id not in self.current_graph:
            reward = 0.0
            terminated = False
            truncated = self.steps >= self.max_steps
            data = self._get_pyg_data()
            info = self._get_info()
            info['action_mask'] = self._get_action_mask()
            info['invalid_action'] = True
            return data, reward, terminated, truncated, info

        self.removed_nodes.add(node_id)
        self.current_graph.remove_node(node_id)
        self.current_attack_count += 1

        current_lcc_ratio = self._get_lcc_size() / max(1, self.initial_lcc_size)
        self.cumulative_anc += current_lcc_ratio
        self.anc_count += 1

        targets_in_lcc = self._count_targets_in_lcc()

        if targets_in_lcc == 0:
            reward = self._calculate_final_reward()
            terminated = True
            truncated = False
            data = self._get_pyg_data()
            info = self._get_info()
            info['action_mask'] = self._get_action_mask()
            info['success'] = True
            return data, reward, terminated, truncated, info

        if self.current_attack_count >= self.attack_per_step:
            self._execute_target_movement()
            self.current_attack_count = 0
            self.rounds += 1

        reward = 0.0
        terminated = False
        truncated = self.steps >= self.max_steps

        data = self._get_pyg_data()
        info = self._get_info()
        info['action_mask'] = self._get_action_mask()
        info['invalid_action'] = False

        return data, reward, terminated, truncated, info

    def _execute_target_movement(self):
        if self.move_prob <= 0:
            return

        movements = []
        claimed_positions = set()

        for target in list(self.target_nodes):
            if target not in self.current_graph:
                continue

            if self.rng.random() > self.move_prob:
                continue

            neighbors = list(self.current_graph.neighbors(target))
            valid_neighbors = [
                n for n in neighbors
                if n not in self.removed_nodes
                   and n not in self.target_nodes
                   and n not in claimed_positions
            ]

            if len(valid_neighbors) == 0:
                continue

            new_pos = self.rng.choice(valid_neighbors)
            movements.append((target, new_pos))
            claimed_positions.add(new_pos)

        for old_pos, new_pos in movements:
            self.target_nodes.remove(old_pos)
            self.target_nodes.add(new_pos)
            self.non_target_nodes.add(old_pos)
            self.non_target_nodes.discard(new_pos)


    def _calculate_final_reward(self) -> float:
        pc = len(self.removed_nodes) / max(1, self.n_non_targets)
        avg_anc = self.cumulative_anc / max(1, self.anc_count)
        return (1.0 - pc) + avg_anc

    def _get_lcc_size(self) -> int:
        if self.current_graph.number_of_nodes() == 0:
            return 0
        try:
            return len(max(nx.connected_components(self.current_graph), key=len))
        except ValueError:
            return 0

    def _count_targets_in_lcc(self) -> int:
        if self.current_graph.number_of_nodes() == 0:
            return 0
        try:
            lcc = max(nx.connected_components(self.current_graph), key=len)
            return sum(1 for t in self.target_nodes if t in lcc)
        except ValueError:
            return 0

    def _get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_nodes, dtype=np.float32)
        for idx, node_id in enumerate(self.node_list):
            if (node_id not in self.target_nodes and
                    node_id not in self.removed_nodes and
                    node_id in self.current_graph):
                mask[idx] = 1.0
        return mask

    def _get_pyg_data(self) -> Data:
        local_features, global_features = self._compute_split_features()

        edges = []
        for u, v in self.current_graph.edges():
            if u in self.node_to_idx and v in self.node_to_idx:
                idx_u = self.node_to_idx[u]
                idx_v = self.node_to_idx[v]
                edges.append([idx_u, idx_v])
                edges.append([idx_v, idx_u])

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        action_mask = self._get_action_mask()

        data = Data(
            x=torch.tensor(local_features, dtype=torch.float32),
            global_x=torch.tensor(global_features, dtype=torch.float32).unsqueeze(0),
            edge_index=edge_index,
            action_mask=torch.tensor(action_mask, dtype=torch.float32),
            num_nodes=self.n_nodes
        )
        return data

    def _compute_split_features(self) -> Tuple[np.ndarray, np.ndarray]:
        local_feats = np.zeros((self.n_nodes, 3), dtype=np.float32)

        urgency = self.current_attack_count / max(1, self.attack_per_step)
        global_feats = np.array([self.move_prob, urgency], dtype=np.float32)

        distances = self._compute_distances_to_targets()
        if self.current_graph.number_of_nodes() > 0:
            degrees = dict(self.current_graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
        else:
            max_degree = 1
            degrees = {}
        log_max_degree = math.log(max_degree + 1)

        for idx in range(self.n_nodes):
            node_id = self.idx_to_node[idx]

            local_feats[idx, 0] = 1.0 if node_id in self.target_nodes else 0.0

            if node_id in self.removed_nodes or node_id not in self.current_graph:
                local_feats[idx, 1] = 0.0
                local_feats[idx, 2] = 0.0
            else:
                dist = distances.get(node_id, float('inf'))
                local_feats[idx, 1] = 1.0 / (dist + 1.0)

                degree = degrees.get(node_id, 0)
                log_degree = math.log(degree + 1)
                local_feats[idx, 2] = log_degree / max(log_max_degree, 1e-6)

        return local_feats, global_feats

    def _compute_distances_to_targets(self) -> Dict[int, float]:
        from collections import deque

        distances = {node: float('inf') for node in self.current_graph.nodes()}
        queue = deque()

        for target in self.target_nodes:
            if target in self.current_graph:
                distances[target] = 0
                queue.append(target)

        while queue:
            current = queue.popleft()
            current_dist = distances[current]

            for neighbor in self.current_graph.neighbors(current):
                if distances[neighbor] > current_dist + 1:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

        return distances

    def _get_info(self) -> Dict[str, Any]:
        targets_in_lcc = self._count_targets_in_lcc()
        pc = len(self.removed_nodes) / max(1, self.n_non_targets)
        avg_anc = self.cumulative_anc / max(1, self.anc_count) if self.anc_count > 0 else 1.0

        return {
            'targets_in_lcc': targets_in_lcc,
            'nodes_removed': len(self.removed_nodes),
            'steps': self.steps,
            'rounds': self.rounds,
            'lcc_size': self._get_lcc_size(),
            'pc': pc,
            'avg_anc': avg_anc,
            'n_nodes': self.n_nodes,
            'n_targets': self.n_targets,
            'move_prob': self.move_prob,
            'attack_per_step': self.attack_per_step,
            'current_attack_count': self.current_attack_count,
        }


class DynamicGraphEnvWrapperDual(gym.Wrapper):

    def __init__(self, env_config: dict, rank: int = 0):
        from graph_generator_moving import DynamicGraphGenerator

        self.env_config = env_config
        self.rank = rank
        self.episode_count = 0

        self.move_prob_range = env_config.get('move_prob_range', (0.0, 0.0))
        self.attack_ratio_range = env_config.get('attack_ratio_range', (0.05, 0.5))

        self.graph_generator = DynamicGraphGenerator(
            n_nodes_range=env_config['n_nodes_range'],
            target_ratio_range=env_config['target_ratio_range'],
            graph_types=env_config.get('graph_types'),
            target_distributions=env_config.get('target_distributions'),
            ba_m_range=env_config.get('ba_m_range'),
            ws_k_range=env_config.get('ws_k_range'),
            ws_beta_range=env_config.get('ws_beta_range'),
            seed=env_config['base_seed'] + rank * 10000
        )

        self.rng = np.random.RandomState(env_config['base_seed'] + rank * 10000)

        graph, targets, metadata = self.graph_generator.generate_graph_and_targets(
            episode_seed=env_config['base_seed'] + rank * 10000
        )

        move_prob, attack_per_step = self._sample_dynamic_params(graph, targets)

        base_env = TargetIsolationEnvDual(
            graph=graph,
            target_nodes=targets,
            move_prob=move_prob,
            attack_per_step=attack_per_step
        )
        super().__init__(base_env)

        self.current_metadata = metadata
        self.current_metadata['move_prob'] = move_prob
        self.current_metadata['attack_per_step'] = attack_per_step

    def _sample_dynamic_params(self, graph: nx.Graph, targets: List[int]) -> Tuple[float, int]:
        move_prob = self.rng.uniform(*self.move_prob_range)

        n_targets = len(targets)

        attack_ratio = self.rng.uniform(*self.attack_ratio_range)
        attack_per_step = max(1, int(attack_ratio * n_targets))

        return move_prob, attack_per_step

    def reset(self, **kwargs):
        self.episode_count += 1

        episode_seed = self.env_config['base_seed'] + self.rank * 10000 + self.episode_count
        graph, targets, metadata = self.graph_generator.generate_graph_and_targets(
            episode_seed=episode_seed
        )

        self.rng = np.random.RandomState(episode_seed)
        move_prob, attack_per_step = self._sample_dynamic_params(graph, targets)

        self.env.update_graph(graph, targets, move_prob, attack_per_step)
        self.current_metadata = metadata
        self.current_metadata['move_prob'] = move_prob
        self.current_metadata['attack_per_step'] = attack_per_step

        data, info = self.env.reset(**kwargs)
        info['graph_metadata'] = self.current_metadata

        return data, info

    def step(self, action):
        data, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info['graph_metadata'] = self.current_metadata
        return data, reward, terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        return self.env._get_action_mask()