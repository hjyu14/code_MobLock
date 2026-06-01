import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv, GATv2Conv, global_mean_pool, global_max_pool


@dataclass
class Args:
    exp_name: str = "gnn_ppo_dual_stream"
    seed: int = 42
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "gnn_ppo"
    wandb_entity: str = None

    n_nodes_range: Tuple[int, int] = (50, 100)
    target_ratio_range: Tuple[float, float] = (0.01, 0.10)
    graph_types: List[str] = None
    target_distributions: List[str] = None
    ba_m_range: Tuple[int, int] = (2, 4)
    ws_k_range: Tuple[int, int] = (4, 8)
    ws_beta_range: Tuple[float, float] = (0.1, 0.3)
    move_prob_range: Tuple[float, float] = (0.0, 1.0)
    attack_ratio_range: Tuple[float, float] = (0.05, 0.5)

    total_timesteps: int = 4_000_000
    learning_rate: float = 3e-4
    num_envs: int = 16
    num_steps: int = 512
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    hidden_dim: int = 128
    num_gnn_layers: int = 3
    gat_heads: int = 4

    local_dim: int = 3
    global_dim: int = 2
    context_dim: int = 32

    use_global_stream: bool = True
    use_attention:     bool = True
    use_virtual_node:  bool = True

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


class VirtualNodeGATSAGE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3,
                 gat_heads: int = 4,
                 use_attention: bool = True,
                 use_virtual_node: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gat_heads = gat_heads
        self.use_attention   = use_attention
        self.use_virtual_node = use_virtual_node

        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        if use_virtual_node:
            self.virtual_node_embedding = nn.Parameter(torch.zeros(1, hidden_dim))
            nn.init.xavier_uniform_(self.virtual_node_embedding)

        self.conv_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for layer_idx in range(num_layers):
            if use_attention and layer_idx == 0:
                self.conv_layers.append(
                    GATv2Conv(hidden_dim, hidden_dim, heads=gat_heads, concat=False)
                )
            else:
                self.conv_layers.append(SAGEConv(hidden_dim, hidden_dim))

            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        if use_virtual_node:
            self.vn_mlp = nn.ModuleList()
            for _ in range(num_layers):
                self.vn_mlp.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_graphs = batch.max().item() + 1

        h = self.feature_encoder(x)

        if self.use_virtual_node:
            vn_embedding = self.virtual_node_embedding.expand(num_graphs, -1).clone()

        for layer_idx in range(self.num_layers):
            h = self.conv_layers[layer_idx](h, edge_index)
            h = self.layer_norms[layer_idx](h)
            h = torch.relu(h)

            if self.use_virtual_node:
                pooled = global_mean_pool(h, batch)
                vn_embedding = vn_embedding + self.vn_mlp[layer_idx](pooled)

        mean_pool = global_mean_pool(h, batch)
        max_pool  = global_max_pool(h, batch)
        if self.use_virtual_node:
            graph_embedding = mean_pool + max_pool + vn_embedding
        else:
            graph_embedding = mean_pool + max_pool

        return h, graph_embedding


class DualStreamAgent(nn.Module):
    def __init__(self,
                 local_dim: int = 3,
                 global_dim: int = 2,
                 hidden_dim: int = 128,
                 context_dim: int = 32,
                 num_gnn_layers: int = 3,
                 gat_heads: int = 4,
                 use_global_stream: bool = True,
                 use_attention:     bool = True,
                 use_virtual_node:  bool = True):
        super().__init__()
        self.use_global_stream = use_global_stream

        self.gnn = VirtualNodeGATSAGE(
            input_dim=local_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            gat_heads=gat_heads,
            use_attention=use_attention,
            use_virtual_node=use_virtual_node,
        )

        if use_global_stream:
            self.context_encoder = nn.Sequential(
                nn.Linear(global_dim, context_dim),
                nn.ReLU(),
                nn.Linear(context_dim, context_dim)
            )
            fusion_dim = hidden_dim + context_dim
        else:
            fusion_dim = hidden_dim

        self.actor_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.critic_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)

    def forward(self, batch_data: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, batch, global_x = (batch_data.x, batch_data.edge_index,
                                           batch_data.batch, batch_data.global_x)

        node_emb, graph_emb = self.gnn(x, edge_index, batch)

        if self.use_global_stream:
            context_emb      = self.context_encoder(global_x)
            context_expanded = context_emb[batch]
            actor_input  = torch.cat([node_emb,  context_expanded], dim=1)
            critic_input = torch.cat([graph_emb, context_emb],      dim=1)
        else:
            actor_input  = node_emb
            critic_input = graph_emb

        logits = self.actor_head(actor_input)
        values = self.critic_head(critic_input)

        return logits, values

    def get_value(self, batch_data: Batch) -> torch.Tensor:
        _, values = self.forward(batch_data)
        return values

    def get_action_and_value(
            self,
            batch_data: Batch,
            action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(batch_data)
        logits = logits.squeeze(-1)
        values = values.squeeze(-1)

        batch_idx = batch_data.batch
        num_graphs = batch_idx.max().item() + 1

        action_mask = batch_data.action_mask

        masked_logits = logits.clone()
        masked_logits[action_mask == 0] = float('-inf')

        actions_list = []
        log_probs_list = []
        entropies_list = []

        for graph_id in range(num_graphs):
            node_mask = (batch_idx == graph_id)
            graph_logits = masked_logits[node_mask]

            valid_mask = action_mask[node_mask]
            if valid_mask.sum() == 0:
                a = torch.tensor(0, device=logits.device)
                log_p = torch.tensor(0.0, device=logits.device)
                ent = torch.tensor(0.0, device=logits.device)
            else:
                probs = Categorical(logits=graph_logits)

                if action is None:
                    a = probs.sample()
                else:
                    a = action[graph_id]

                log_p = probs.log_prob(a)
                ent = probs.entropy()

            actions_list.append(a)
            log_probs_list.append(log_p)
            entropies_list.append(ent)

        actions = torch.stack(actions_list)
        log_probs = torch.stack(log_probs_list)
        entropies = torch.stack(entropies_list)

        return actions, log_probs, entropies, values


class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, device: torch.device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.obs: List[List[Data]] = [[None] * num_envs for _ in range(num_steps)]
        self.actions = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)

    def insert(self, step: int, obs_list: List[Data], actions: torch.Tensor,
               logprobs: torch.Tensor, rewards: torch.Tensor,
               dones: torch.Tensor, values: torch.Tensor):
        for env_id, obs in enumerate(obs_list):
            self.obs[step][env_id] = obs

        self.actions[step] = actions
        self.logprobs[step] = logprobs
        self.rewards[step] = rewards
        self.dones[step] = dones
        self.values[step] = values

    def get_batch(self):
        flat_obs = []
        for step in range(self.num_steps):
            for env_id in range(self.num_envs):
                flat_obs.append(self.obs[step][env_id])

        flat_actions = self.actions.reshape(-1)
        flat_logprobs = self.logprobs.reshape(-1)
        flat_values = self.values.reshape(-1)

        return flat_obs, flat_actions, flat_logprobs, flat_values


def compute_gae(rewards, values, dones, next_value, next_done, gamma, gae_lambda):
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]

        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

    returns = advantages + values
    return advantages, returns


def train(args: Args, save_dir: str = "./runs"):
    run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"
    save_path = os.path.join(save_dir, run_name)
    os.makedirs(save_path, exist_ok=True)

    writer = SummaryWriter(os.path.join(save_path, "tensorboard"))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    print(f"Batch size: {args.batch_size}")
    print(f"Minibatch size: {args.minibatch_size}")
    print(f"Num iterations: {args.num_iterations}")

    from graph_generator_moving import GraphType, TargetDistribution
    from env_moving_dualstream_final import DynamicGraphEnvWrapperDual

    graph_types = None
    if args.graph_types:
        graph_types = [GraphType(gt) for gt in args.graph_types]

    target_distributions = None
    if args.target_distributions:
        target_distributions = [TargetDistribution(td) for td in args.target_distributions]

    env_config = {
        'n_nodes_range': args.n_nodes_range,
        'target_ratio_range': args.target_ratio_range,
        'graph_types': graph_types,
        'target_distributions': target_distributions,
        'ba_m_range': args.ba_m_range,
        'ws_k_range': args.ws_k_range,
        'ws_beta_range': args.ws_beta_range,
        'base_seed': args.seed,
        'move_prob_range': args.move_prob_range,
        'attack_ratio_range': args.attack_ratio_range,
    }

    print(f"Creating {args.num_envs} environments (5D features)...")
    print(f"Move Prob Range: {args.move_prob_range}")
    print(f"Attack Ratio Range: {args.attack_ratio_range}")
    envs = [DynamicGraphEnvWrapperDual(env_config, i) for i in range(args.num_envs)]

    agent = DualStreamAgent(
        local_dim=args.local_dim,
        global_dim=args.global_dim,
        hidden_dim=args.hidden_dim,
        context_dim=args.context_dim,
        num_gnn_layers=args.num_gnn_layers,
        gat_heads=args.gat_heads,
        use_global_stream=args.use_global_stream,
        use_attention=args.use_attention,
        use_virtual_node=args.use_virtual_node,
    ).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    buffer = RolloutBuffer(args.num_steps, args.num_envs, device)

    print("Initializing environments...")
    next_obs_list = []
    next_done = torch.zeros(args.num_envs, device=device)

    for env_id, env in enumerate(envs):
        obs, info = env.reset(seed=args.seed + env_id)
        next_obs_list.append(obs)

    global_step = 0
    start_time = time.time()
    episode_pcs = []
    episode_ancs = []
    episode_successes = []

    print("Starting training...")
    print("=" * 70)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(args.num_steps):
            global_step += args.num_envs

            batch_data = Batch.from_data_list(next_obs_list).to(device)

            with torch.no_grad():
                actions, logprobs, _, values = agent.get_action_and_value(batch_data)

            next_obs_list_new = []
            rewards = []
            dones = []

            for env_id, env in enumerate(envs):
                action = actions[env_id].item()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                rewards.append(reward)
                dones.append(float(done))

                if done:
                    if 'pc' in info:
                        episode_pcs.append(info['pc'])
                    if 'avg_anc' in info:
                        episode_ancs.append(info['avg_anc'])
                    success = info.get('targets_in_lcc', 1) == 0
                    episode_successes.append(float(success))

                    obs, info = env.reset()

                    if len(episode_pcs) % 10 == 0 and len(episode_pcs) > 0:
                        recent_sr = np.mean(episode_successes[-100:]) if episode_successes else 0
                        print(f"Step {global_step:,} | Episodes: {len(episode_pcs)} | "
                              f"Mean pc: {np.mean(episode_pcs[-100:]):.4f} | "
                              f"Mean ANC: {np.mean(episode_ancs[-100:]):.4f} | "
                              f"Success Rate: {recent_sr:.2%}")

                next_obs_list_new.append(obs)

            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device)

            buffer.insert(
                step=step,
                obs_list=next_obs_list,
                actions=actions,
                logprobs=logprobs,
                rewards=rewards_tensor,
                dones=next_done,
                values=values
            )

            next_obs_list = next_obs_list_new
            next_done = dones_tensor

        with torch.no_grad():
            batch_data = Batch.from_data_list(next_obs_list).to(device)
            next_value = agent.get_value(batch_data).squeeze(-1)

        advantages, returns = compute_gae(
            buffer.rewards, buffer.values, buffer.dones,
            next_value, next_done, args.gamma, args.gae_lambda
        )

        flat_obs, flat_actions, flat_logprobs, flat_values = buffer.get_batch()
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs = [flat_obs[i] for i in mb_inds]
                mb_batch = Batch.from_data_list(mb_obs).to(device)

                mb_actions = flat_actions[mb_inds]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_batch, mb_actions)

                mb_logprobs = flat_logprobs[mb_inds]
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                mb_returns = b_returns[mb_inds]
                if args.clip_vloss:
                    mb_values_old = flat_values[mb_inds]
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values_old + torch.clamp(
                        newvalue - mb_values_old,
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred = flat_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        if len(episode_pcs) > 0:
            writer.add_scalar("charts/mean_pc", np.mean(episode_pcs[-100:]), global_step)
        if len(episode_ancs) > 0:
            writer.add_scalar("charts/mean_anc", np.mean(episode_ancs[-100:]), global_step)
        if len(episode_successes) > 0:
            writer.add_scalar("charts/success_rate", np.mean(episode_successes[-100:]), global_step)

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)

        if iteration % 10 == 0:
            print(f"Iteration {iteration}/{args.num_iterations} | SPS: {sps}")

    model_path = os.path.join(save_path, "model.pt")
    torch.save({
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
    }, model_path)
    print(f"Model saved to {model_path}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if len(episode_pcs) > 0:
        axes[0].plot(episode_pcs, alpha=0.3, label='pc')
        window = min(50, max(1, len(episode_pcs) // 10))
        if window > 1:
            moving_avg = np.convolve(episode_pcs, np.ones(window) / window, mode='valid')
            axes[0].plot(range(window - 1, len(episode_pcs)), moving_avg, label=f'MA({window})')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('pc')
        axes[0].set_title('Removal Ratio (pc)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    if len(episode_ancs) > 0:
        axes[1].plot(episode_ancs, alpha=0.3, label='ANC')
        window = min(50, max(1, len(episode_ancs) // 10))
        if window > 1:
            moving_avg = np.convolve(episode_ancs, np.ones(window) / window, mode='valid')
            axes[1].plot(range(window - 1, len(episode_ancs)), moving_avg, label=f'MA({window})')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('ANC')
        axes[1].set_title('Average Normalized Connectivity')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    if len(episode_successes) > 0:
        window = min(50, max(1, len(episode_successes) // 10))
        if window > 1:
            success_ma = np.convolve(episode_successes, np.ones(window) / window, mode='valid')
            axes[2].plot(range(window - 1, len(episode_successes)), success_ma, label=f'MA({window})')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Success Rate')
        axes[2].set_title('Success Rate')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curves.png'), dpi=150)
    plt.close()

    writer.close()

    print("=" * 70)
    print("Training Complete!")
    print(f"Model saved at: {save_path}")
    print("=" * 70)

    return agent, save_path