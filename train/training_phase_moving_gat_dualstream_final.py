import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ppo_gnn_moving_gat_dualstream_final import Args, train

args = Args()

args.n_nodes_range = (50, 100)
args.target_ratio_range = (0.01, 0.10)
args.graph_types = ['ba', 'ws']
args.target_distributions = ['localized', 'random']
args.ba_m_range = (2, 4)
args.ws_k_range = (4, 8)
args.ws_beta_range = (0.1, 0.3)

args.move_prob_range = (0.0, 1.0)
args.attack_ratio_range = (0.05, 0.5)

args.total_timesteps = 4_000_000
args.learning_rate = 3e-4
args.num_envs = 16
args.num_steps = 512
args.num_minibatches = 8
args.update_epochs = 5
args.clip_coef = 0.2
args.ent_coef = 0.02
args.vf_coef = 0.5
args.max_grad_norm = 0.5

args.hidden_dim = 128
args.num_gnn_layers = 3
args.gat_heads = 4

args.local_dim = 3
args.global_dim = 2
args.context_dim = 32

args.seed = 42
args.cuda = True
args.exp_name = "gnn_ppo_dual_stream_n50-100"

args.use_global_stream = True
args.use_attention     = True
args.use_virtual_node  = True

save_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    agent, save_path = train(args, save_dir=save_dir)