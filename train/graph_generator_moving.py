import numpy as np
import networkx as nx
from typing import Tuple, List, Dict, Optional
from enum import Enum
import random


class GraphType(Enum):
    BA = 'ba'
    WS = 'ws'


class TargetDistribution(Enum):
    RANDOM = 'random'
    LOCALIZED = 'localized'


class DynamicGraphGenerator:

    def __init__(
            self,
            n_nodes_range: Tuple[int, int] = (50, 200),
            target_ratio_range: Tuple[float, float] = (0.01, 0.1),
            graph_types: List[GraphType] = None,
            target_distributions: List[TargetDistribution] = None,
            ba_m_range: Tuple[int, int] = (2, 4),
            ws_k_range: Tuple[int, int] = (4, 8),
            ws_beta_range: Tuple[float, float] = (0.1, 0.3),
            seed: int = None
    ):
        self.n_nodes_range = n_nodes_range
        self.target_ratio_range = target_ratio_range

        if graph_types is None:
            self.graph_types = [GraphType.BA, GraphType.WS]
        else:
            self.graph_types = [GraphType(gt) if isinstance(gt, str) else gt for gt in graph_types]

        if target_distributions is None:
            self.target_distributions = [TargetDistribution.RANDOM, TargetDistribution.LOCALIZED]
        else:
            self.target_distributions = [TargetDistribution(td) if isinstance(td, str) else td for td in target_distributions]

        self.ba_m_range = ba_m_range
        self.ws_k_range = ws_k_range
        self.ws_beta_range = ws_beta_range

        self.rng = np.random.RandomState(seed)

    def generate_graph_and_targets(
            self,
            episode_seed: int = None
    ) -> Tuple[nx.Graph, List[int], Dict]:
        if episode_seed is not None:
            self.rng = np.random.RandomState(episode_seed)
            random.seed(episode_seed)

        n_nodes = self.rng.randint(self.n_nodes_range[0], self.n_nodes_range[1] + 1)
        target_ratio = self.rng.uniform(*self.target_ratio_range)
        graph_type = self.rng.choice(self.graph_types)
        target_dist = self.rng.choice(self.target_distributions)

        if graph_type == GraphType.BA:
            m = self.rng.randint(self.ba_m_range[0], self.ba_m_range[1] + 1)
            m = min(m, n_nodes - 1)
            G = nx.barabasi_albert_graph(n_nodes, m, seed=episode_seed)
            graph_params = {'type': 'BA', 'm': m}
        else:
            k = self.rng.randint(self.ws_k_range[0] // 2, self.ws_k_range[1] // 2 + 1) * 2
            k = max(2, min(k, n_nodes - 1))
            beta = self.rng.uniform(*self.ws_beta_range)
            G = nx.watts_strogatz_graph(n_nodes, k, beta, seed=episode_seed)
            graph_params = {'type': 'WS', 'k': k, 'beta': beta}

        if not nx.is_connected(G):
            lcc = max(nx.connected_components(G), key=len)
            G = nx.convert_node_labels_to_integers(G.subgraph(lcc).copy())
            n_nodes = G.number_of_nodes()

        n_targets = max(1, int(n_nodes * target_ratio))

        if target_dist == TargetDistribution.RANDOM:
            targets = self._generate_random_targets(G, n_targets)
        else:
            targets = self._generate_localized_targets(G, n_targets, episode_seed)

        metadata = {
            'n_nodes': n_nodes,
            'n_targets': len(targets),
            'target_ratio': len(targets) / n_nodes,
            'graph_type': graph_type.value,
            'target_distribution': target_dist.value,
            **graph_params
        }

        return G, targets, metadata

    def _generate_random_targets(self, G: nx.Graph, n_targets: int) -> List[int]:
        nodes = list(G.nodes())
        return list(self.rng.choice(nodes, size=min(n_targets, len(nodes)), replace=False))

    def _generate_localized_targets(
            self,
            G: nx.Graph,
            n_targets: int,
            seed: int = None,
            radius: float = 0.05,
            max_radius: float = 1.0,
            step: float = 0.05
    ) -> List[int]:
        pos = nx.spring_layout(G, seed=seed)
        nodes = list(G.nodes())

        center_node = self.rng.choice(nodes)
        center_x, center_y = pos[center_node]
        current_radius = radius

        localized_candidate = []
        while current_radius < max_radius:
            localized_candidate = [
                node for node, (x, y) in pos.items()
                if (x - center_x) ** 2 + (y - center_y) ** 2 < current_radius ** 2
            ]
            if len(localized_candidate) >= n_targets:
                break
            current_radius += step

        if len(localized_candidate) >= n_targets:
            indices = self.rng.choice(len(localized_candidate), size=n_targets, replace=False)
            return [localized_candidate[i] for i in indices]
        else:
            remaining = n_targets - len(localized_candidate)
            other_nodes = list(set(nodes) - set(localized_candidate))
            if remaining > 0 and other_nodes:
                extra_indices = self.rng.choice(len(other_nodes), size=min(remaining, len(other_nodes)), replace=False)
                extra = [other_nodes[i] for i in extra_indices]
                return localized_candidate + extra
            return localized_candidate