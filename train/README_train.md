# `train/`: MobileIsolator agent training

This directory contains the code, hyperparameter configuration, 
and the released pre-trained weights of the **MobileIsolator** 
agent used throughout the paper *Containment of Escaping Targets 
in Complex Networks*.

MobileIsolator is a Proximal-Policy-Optimization (PPO) agent 
whose policy network is a dual-stream Graph Attention Network 
(GAT). It is trained from scratch on randomly sampled synthetic 
networks (Barabási–Albert and Watts–Strogatz) with randomly 
sampled target-placement and dynamics parameters, and is then 
evaluated zero-shot on the experiments in Figs 3–6 and the 
Supplementary Figures.


## Files

| File | Role |
|------|------|
| `graph_generator_moving.py` | Generates synthetic training environments — BA and WS networks with random or localized target placement. |
| `env_moving_dualstream_final.py` | Gym-style RL environment that wraps the graph generator and exposes step / reset / observation logic. |
| `ppo_gnn_moving_gat_dualstream_final.py` | Defines the `DualStreamAgent` policy network (the GAT-based architecture), the `Args` dataclass holding all training hyperparameters, and the `train(...)` function that runs PPO. |
| `training_phase_moving_gat_dualstream_final.py` | Entry-point script. Instantiates `Args`, populates every hyperparameter from Supplementary Table 4 (see below), and calls `train(...)`. **Edit this file** to change training settings. |
| `gnn_ppo_dual_stream_n50-100__seed42__1769653775/` | The released pre-trained MobileIsolator (the `Full` model), including the weights (`model.pt`), training-curve plot, and TensorBoard logs. |


## Released pre-trained model

The released full model is stored in:

```
train/gnn_ppo_dual_stream_n50-100__seed42__1769653775/model.pt
```

This single weights file is loaded by every comparison and 
rendering script in the repository that requires MobileIsolator's 
policy (Figs 3–6 and SM Figs 1, 5, 6, 8, 9, and SM Tables 1–3). 
The folder name encodes the experiment name 
(`gnn_ppo_dual_stream_n50-100`), the seed (`seed42`), and the 
Unix timestamp at which training completed (`1769653775`); see 
`training_phase_moving_gat_dualstream_final.py` for how this name 
is composed.

The three ablation variants of the model used by **SM Tables 1–3** 
are not stored in `train/` — they live alongside the ablation 
experiment at `fig_sm/sm_table1_3/`. See 
`fig_sm/sm_table1_3/README.md` for details.


## Supplementary Table 4: Hyperparameters and training-distribution configurations

Supplementary Table 4 of the paper lists the complete set of 
parameters used to train MobileIsolator, organised into three 
groups:

- **Environment & Training Distribution** — the synthetic graph 
  topologies (BA / WS), the target placement strategy (random / 
  localized), the target ratio, the target movement probability 
  $p$, and the per-step removal ratio $r$ used to generate 
  training episodes.
- **Network Architecture** — the structural choices of the 
  `DualStreamAgent` policy network: feature dimensions, number 
  of GNN layers, number of GAT attention heads, etc.
- **PPO Hyperparameters** — the learning rate, rollout length, 
  number of parallel environments, clip coefficient, optimisation 
  epochs, and other standard PPO settings.

The values in Supplementary Table 4 can be cross-checked against 
the source code as follows:

| Table 4 section | Source of truth in this directory |
|-----------------|-----------------------------------|
| Environment & Training Distribution | The `args.*_range`, `args.graph_types`, and `args.target_distributions` lines of `training_phase_moving_gat_dualstream_final.py`; the sampling logic itself is in `graph_generator_moving.py`. |
| Network Architecture | The `args.hidden_dim`, `args.num_gnn_layers`, `args.gat_heads`, `args.local_dim`, `args.global_dim`, `args.context_dim` lines of `training_phase_moving_gat_dualstream_final.py`; the architectural assembly itself is in the `DualStreamAgent` class of `ppo_gnn_moving_gat_dualstream_final.py`. |
| PPO Hyperparameters | The `args.total_timesteps`, `args.learning_rate`, `args.num_envs`, `args.num_steps`, `args.num_minibatches`, `args.update_epochs`, `args.clip_coef`, `args.ent_coef`, `args.vf_coef`, and `args.max_grad_norm` lines of `training_phase_moving_gat_dualstream_final.py`; the PPO update logic itself is in the `train(...)` function of `ppo_gnn_moving_gat_dualstream_final.py`. |

The committed values in these files are the **authoritative** 
ones: they match Supplementary Table 4 exactly and were used to 
produce the released 
`gnn_ppo_dual_stream_n50-100__seed42__1769653775/model.pt`.


## Retraining MobileIsolator

The released weights are sufficient to reproduce every figure 
and table in the paper; retraining is only needed if you want 
to validate the full pipeline end-to-end.

To retrain the full MobileIsolator with the exact published 
hyperparameters:

```bash
python training_phase_moving_gat_dualstream_final.py
```

This writes a new sub-directory next to the script, named 
`{exp_name}__seed{seed}__{timestamp}/`, containing the trained 
weights (`model.pt`), training-curve plot 
(`training_curves.png`), and TensorBoard event logs. The timestamp 
suffix ensures repeated training runs do not overwrite each 
other.

Training uses CUDA by default (`args.cuda = True`). To train on 
CPU, set `args.cuda = False` near the bottom of the script.

### Retraining ablation variants (used by SM Tables 1–3)

Three boolean flags at the bottom of the training script control 
which architectural components are enabled:

```python
args.use_global_stream = True
args.use_attention     = True
args.use_virtual_node  = True
```

Setting any one of these to `False` produces the corresponding 
ablation variant. The three released ablation models in 
`fig_sm/sm_table1_3/` were each trained by flipping exactly one 
of these flags to `False` while keeping the other two at `True`.


## Platform notes

The training scripts run on Linux, macOS, and Windows. CUDA is 
strongly recommended; on a typical single-GPU workstation, the 
full $4 \times 10^6$-step run completes in roughly a day.


## Requirements

Training uses the project-wide Python environment specified in 
the top-level `requirements.txt`. The specific packages used 
here are:

- `numpy`, `pandas`, `tqdm`
- `networkx`
- `torch` (with CUDA support recommended)
- `torch-geometric`
- `tensorboard` (only for inspecting training curves; not needed 
  for inference)


## Citation

If you use the released model weights or retrain MobileIsolator 
from this code, please cite the main paper.
