"""SAC + ALDA (Associative Latent DisentAnglement), Batra & Sukhatme 2024.

Pixel DMC training, zero-shot eval on the Distracting Control Suite.
Defaults follow the paper's hyperparameter table (Appendix).
"""
import argparse
import copy
import csv
import os
import time
from collections import defaultdict, deque
from types import SimpleNamespace

import numpy as np
import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as D
from PIL import Image
os.environ.setdefault("MUJOCO_GL", "egl")  # headless; must be set before dm_control is imported
from dm_control import suite  # noqa: E402
from dm_control.suite.wrappers import pixels  # noqa: E402


# ---------------------------------------------------------------- env
class DMCEnv:
    """Pixel DMC: action repeat + frame stack, obs uint8 (3k, H, W)."""

    def __init__(self, domain, task, seed, action_repeat, frame_stack, size=64, distract=None):
        render_kwargs = dict(height=size, width=size, camera_id=2 if domain == "quadruped" else 0)
        if distract is None:
            env = suite.load(domain, task, task_kwargs={"random": seed})
            env = pixels.Wrapper(env, pixels_only=True, render_kwargs=render_kwargs)
        else:
            from distracting_control import suite as dsuite
            env = dsuite.load(domain, task, task_kwargs={"random": seed},
                              render_kwargs=render_kwargs, pixels_only=True, **distract)
        spec = env.action_spec()
        assert np.allclose(spec.minimum, -1) and np.allclose(spec.maximum, 1)
        self.env, self.action_repeat, self.ac_dim = env, action_repeat, spec.shape[0]
        self.frames = deque(maxlen=frame_stack)

    def reset(self):
        frame = self.env.reset().observation["pixels"].transpose(2, 0, 1).copy()
        for _ in range(self.frames.maxlen):
            self.frames.append(frame)
        return np.concatenate(self.frames)

    def step(self, ac):
        rew = 0.0
        for _ in range(self.action_repeat):
            ts = self.env.step(ac)
            rew += ts.reward or 0.0
            if ts.last():
                break
        self.frames.append(ts.observation["pixels"].transpose(2, 0, 1).copy())
        return np.concatenate(self.frames), rew, ts.discount == 0, ts.last()


class ReplayBuffer:
    """Stores only the newest frame per step; stacks are rebuilt at sample time (~4x less RAM)."""

    def __init__(self, capacity, frame_shape, ac_dim, frame_stack):
        self.frames = np.zeros((capacity, *frame_shape), np.uint8)       # newest frame of ob
        self.next_frames = np.zeros((capacity, *frame_shape), np.uint8)  # newest frame of next_ob
        self.t = np.zeros(capacity, np.int64)                            # step index within episode
        self.acs = np.zeros((capacity, ac_dim), np.float32)
        self.rews = np.zeros(capacity, np.float32)
        self.dones = np.zeros(capacity, np.float32)
        self.k, self.c = frame_stack, frame_shape[0]
        self.capacity, self.size, self.ptr = capacity, 0, 0

    def insert(self, ob, ac, rew, next_ob, done, t):
        i = self.ptr
        self.frames[i], self.next_frames[i] = ob[-self.c:], next_ob[-self.c:]
        self.t[i], self.acs[i], self.rews[i], self.dones[i] = t, ac, rew, done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        # ponytail: right after ring wrap the oldest few samples may stack frames already overwritten; negligible
        idx = np.random.randint(0, self.size, batch_size)
        back = np.maximum(np.arange(1 - self.k, 1)[None], -self.t[idx, None])  # clamp at episode start, like reset()
        obs = self.frames[(idx[:, None] + back) % self.capacity]                 # (B, k, C, H, W)
        next_obs = np.concatenate([obs[:, 1:], self.next_frames[idx, None]], 1)
        B = batch_size
        out = dict(obs=obs.reshape(B, -1, *obs.shape[-2:]), next_obs=next_obs.reshape(B, -1, *obs.shape[-2:]),
                   acs=self.acs[idx], rews=self.rews[idx], dones=self.dones[idx])
        return {k: torch.as_tensor(v, device=device) for k, v in out.items()}


# ---------------------------------------------------------------- networks
def build_mlp(in_dim, out_dim, n_layers=2, size=1024):
    layers, d = [], in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, size), nn.GELU()]
        d = size
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


# Encoder/decoder from QLAE (Hsu et al. 2023, App. C.3); ALDA swaps every leaky ReLU for GeLU.
WIDTHS = (32, 64, 128, 256)  # 64x64 -> 4x4


class Encoder(nn.Module):
    """f_theta: one RGB frame -> n_z continuous latents.
    4 blocks of [conv3 s1, conv3 s1, conv4 s2], each conv -> GeLU -> InstanceNorm; then 2x dense-256 ReLU -> affine."""

    def __init__(self, n_z):
        super().__init__()
        layers, c = [], 3
        for w in WIDTHS:
            for k, s in ((3, 1), (3, 1), (4, 2)):
                layers += [nn.Conv2d(c, w, k, s, 1), nn.GELU(), nn.InstanceNorm2d(w, affine=True)]
                c = w
        self.net = nn.Sequential(*layers, nn.Flatten(), nn.Linear(c * 16, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, n_z))

    def forward(self, x):
        return self.net(x)


class StyleLayer(nn.Module):
    """Transposed conv -> GeLU -> AdaIN, with per-channel scale/bias projected from w."""

    def __init__(self, c_in, c_out, k, s, w_dim=256):
        super().__init__()
        self.conv = nn.ConvTranspose2d(c_in, c_out, k, s, 1)
        self.norm = nn.InstanceNorm2d(c_out)
        self.style = nn.Linear(w_dim, 2 * c_out)
        with torch.no_grad():  # scale starts at 1, bias at 0 (StyleGAN convention; init not given in C.3)
            self.style.bias.copy_(torch.cat([torch.ones(c_out), torch.zeros(c_out)]))

    def forward(self, x, w):
        scale, bias = self.style(w)[..., None, None].chunk(2, dim=1)
        return scale * self.norm(F.gelu(self.conv(x))) + bias


class Decoder(nn.Module):
    """g_phi: StyleGAN-like. z -> 2x dense-256 ReLU = w; learned 256x4x4 input (init 0.1);
    4 blocks of [style3 s1, style3 s1, style4 s2] at widths 256,128,64,32; 1x1 conv -> 3x64x64."""

    def __init__(self, n_z):
        super().__init__()
        self.mapping = nn.Sequential(nn.Linear(n_z, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.const = nn.Parameter(torch.full((1, WIDTHS[-1], 4, 4), 0.1))
        layers, c = [], WIDTHS[-1]
        for width in WIDTHS[::-1]:
            for k, s in ((3, 1), (3, 1), (4, 2)):
                layers.append(StyleLayer(c, width, k, s))
                c = width
        self.layers = nn.ModuleList(layers)
        self.out = nn.Conv2d(c, 3, 1)

    def forward(self, z):
        w = self.mapping(z)
        x = self.const.expand(z.shape[0], -1, -1, -1)
        for layer in self.layers:
            x = layer(x, w)
        return self.out(x)


class AssociativeLatent(nn.Module):
    """l_psi: per-dim scalar codebooks with softmax(-beta*L1) retrieval (paper Eq. 6 / Alg. 2)."""

    def __init__(self, n_z, n_v, beta):
        super().__init__()
        self.values = nn.Parameter(torch.linspace(-1, 1, n_v).repeat(n_z, 1))  # (n_z, n_v)
        self.beta = beta

    def forward(self, z):  # (N, n_z) -> (N, n_z)
        w = torch.softmax(-self.beta * (z[..., None] - self.values).abs(), dim=-1)
        return (w * self.values).sum(-1)


class HistoryEncoder(nn.Module):
    """h_gamma: 1D conv over the k per-frame latent vectors (Fig. 2 "1D CNN")."""

    def __init__(self, n_z, k, channels=64):
        super().__init__()
        self.out_dim = channels * (k - 2)
        self.net = nn.Sequential(nn.Conv1d(n_z, channels, 2), nn.GELU(), nn.Conv1d(channels, channels, 2), nn.GELU(),
                                 nn.Flatten())

    def forward(self, z_d):  # (B, k, n_z)
        return self.net(z_d.transpose(1, 2))


class TanhGaussianPolicy(nn.Module):
    def __init__(self, ob_dim, ac_dim, n_layers=2, size=1024):
        super().__init__()
        self.net = build_mlp(ob_dim, 2 * ac_dim, n_layers, size)

    def forward(self, obs) -> D.Distribution:
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        std = log_std.clamp(-20, 2).exp()
        base = D.Independent(D.Normal(mean, std), 1)
        return D.TransformedDistribution(base, [D.TanhTransform(cache_size=1)])


# ---------------------------------------------------------------- agent
class Critic(nn.Module):
    """1D CNN -> linear (z_Q) -> double-Q MLPs. The 1D CNN is trained only through J(Q)."""

    def __init__(self, n_z, k, ac_dim, feature_dim, num_critics):
        super().__init__()
        self.history = HistoryEncoder(n_z, k)
        self.head = nn.Linear(self.history.out_dim, feature_dim)
        self.qs = nn.ModuleList([build_mlp(feature_dim + ac_dim, 1) for _ in range(num_critics)])

    def forward(self, h, acs):  # h: 1D CNN output
        x = torch.cat([self.head(h), acs], dim=-1)
        return torch.stack([q(x).squeeze(-1) for q in self.qs])  # (num_critics, B)


class Actor(nn.Module):
    """linear (z_pi) -> policy MLP, on top of the critic's (detached) 1D CNN output."""

    def __init__(self, in_dim, ac_dim, feature_dim):
        super().__init__()
        self.head = nn.Linear(in_dim, feature_dim)
        self.policy = TanhGaussianPolicy(feature_dim, ac_dim)

    def forward(self, h):
        return self.policy(self.head(h))


class ALDAAgent(nn.Module):
    def __init__(
        self, ac_dim,
        frame_stack=3, n_latents=12, n_values=12, beta=100.0, feature_dim=50,
        discount=0.99, init_temperature=0.1, num_critics=2, target_update_rate=0.005,
        lr=1e-3, alpha_lr=1e-4, weight_decay=0.1, actor_update_freq=2, target_update_freq=2,
    ):
        super().__init__()
        self.k, self.n_z = frame_stack, n_latents
        self.encoder, self.decoder = Encoder(n_latents), Decoder(n_latents)
        self.latent = AssociativeLatent(n_latents, n_values, beta)
        self.critic = Critic(n_latents, frame_stack, ac_dim, feature_dim, num_critics)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor = Actor(self.critic.history.out_dim, ac_dim, feature_dim)
        self.log_alpha = nn.Parameter(torch.tensor(float(np.log(init_temperature))))
        self.target_entropy = -ac_dim

        # Critic and ALDA losses touch disjoint params, so one optimizer on their sum is equivalent to two.
        # Weight decay (lambda_theta, lambda_phi) only on encoder/decoder, decoupled (AdamW, as in QLAE):
        # coupled L2 of 0.1 in Adam swamps the reconstruction gradient and collapses the encoder to a constant.
        self.model_opt = torch.optim.AdamW([
            {"params": self.encoder.parameters(), "weight_decay": weight_decay},
            {"params": self.decoder.parameters(), "weight_decay": weight_decay},
            {"params": [*self.latent.parameters(), *self.critic.parameters()], "weight_decay": 0.0},
        ], lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.discount, self.tau = discount, target_update_rate
        self.actor_update_freq, self.target_update_freq = actor_update_freq, target_update_freq

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def encode(self, obs):
        """uint8 (B, 3k, H, W) -> frames (Bk, 3, H, W) in [-0.5, 0.5], z_cont (Bk, n_z). Framestack folded into batch."""
        x = obs.float().div(255).sub(0.5).view(-1, 3, *obs.shape[-2:])
        return x, self.encoder(x)

    def history(self, z_d, critic):
        return critic.history(z_d.view(-1, self.k, self.n_z))

    @torch.no_grad()
    def act(self, ob, sample=True):
        _, z = self.encode(torch.as_tensor(ob, device=self.log_alpha.device)[None])
        dist = self.actor(self.history(self.latent(z), self.critic))
        ac = dist.sample() if sample else torch.tanh(dist.base_dist.mean)
        return ac[0].cpu().numpy()

    def update(self, batch, step):
        obs, acs, rews, next_obs, dones = (batch[k] for k in ("obs", "acs", "rews", "next_obs", "dones"))

        # Gradient routing follows Fig. 2:
        #   J(Q): Q MLP -> z_Q linear -> 1D CNN -> latent model (codebook), stops before z_cont.
        #   J(ALDA): decoder -> z_cont straight-through (bypasses the latent model) -> encoder; plus L_commit.
        #   No L_quantize: the codebook ("memories") is optimized only by the task loss.
        x, z_cont = self.encode(obs)
        z_d = self.latent(z_cont.detach())
        commit_loss = F.mse_loss(z_cont, z_d.detach())
        recon_loss = F.mse_loss(self.decoder(z_cont + (z_d - z_cont).detach()), x)

        h = self.history(z_d, self.critic)
        with torch.no_grad():
            next_z_d = self.latent(self.encode(next_obs)[1])
            next_dist = self.actor(self.history(next_z_d, self.critic))
            next_acs = next_dist.sample()
            next_q = self.target_critic(self.history(next_z_d, self.target_critic), next_acs).min(0).values
            next_q = next_q - self.alpha * next_dist.log_prob(next_acs)
            target = rews + self.discount * (1 - dones) * next_q
        q = self.critic(h, acs)
        critic_loss = ((q - target[None]) ** 2).mean()

        self.model_opt.zero_grad()
        (critic_loss + commit_loss + recon_loss).backward()
        self.model_opt.step()
        info = {"critic_loss": critic_loss.item(), "q": q.mean().item(),
                "commit_loss": commit_loss.item(), "recon_loss": recon_loss.item()}

        if step % self.actor_update_freq == 0:
            info.update(self.update_actor(h.detach()))
        if step % self.target_update_freq == 0:
            self.update_targets()
        return info

    def update_actor(self, h):
        dist = self.actor(h)
        acs = dist.rsample()
        log_prob = dist.log_prob(acs)
        q = self.critic(h, acs).min(0).values
        actor_loss = (self.alpha.detach() * log_prob - q).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = (self.alpha * (-log_prob.detach() - self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        return {"actor_loss": actor_loss.item(), "entropy": -log_prob.mean().item(), "alpha": self.alpha.item()}

    @torch.no_grad()
    def latent_stats(self, frames):
        """uint8 (N, 3, H, W) -> how far encoder outputs sit from the memories, i.e. how much association acts."""
        dev, values = self.log_alpha.device, self.latent.values
        zc, zd, mse = [], [], []
        for chunk in torch.as_tensor(frames).split(256):
            x, z = self.encode(chunk.to(dev))
            d = self.latent(z)
            zc.append(z), zd.append(d), mse.append(((self.decoder(d) - x) ** 2).mean((1, 2, 3)))
        zc, zd = torch.cat(zc), torch.cat(zd)
        p = F.one_hot((zc[..., None] - values).abs().argmin(-1), values.shape[1]).float().mean(0)  # (n_z, n_v)
        return {
            "assoc_l1": (zc - zd).abs().mean().item(),  # mean |z_cont - z_d|: shift applied by association
            "z_out_of_range": ((zc < values.min(-1).values) | (zc > values.max(-1).values)).float().mean().item(),
            "code_usage_entropy": (-(p * p.clamp_min(1e-12).log()).sum(-1).mean() / np.log(values.shape[1])).item(),
            "recon_mse": torch.cat(mse).mean().item(),
        }

    @torch.no_grad()
    def recon_grid(self, frames):
        """uint8 (n, 3, H, W) -> image: originals (top row) over reconstructions."""
        x, z = self.encode(torch.as_tensor(frames, device=self.log_alpha.device))
        return to_grid(torch.stack([x, self.decoder(self.latent(z))]))

    @torch.no_grad()
    def traversal_grid(self, frame):
        """Paper Fig. 5: row i sweeps latent i over its sorted codebook, others fixed at z_d(frame)."""
        _, z = self.encode(torch.as_tensor(frame[None], device=self.log_alpha.device))
        vals = self.latent.values.sort(-1).values  # (n_z, n_v)
        n_z, n_v = vals.shape
        zs = self.latent(z)[0].repeat(n_z, n_v, 1)  # (n_z, n_v, n_z)
        idx = torch.arange(n_z)
        zs[idx, :, idx] = vals
        return to_grid(self.decoder(zs.view(-1, n_z)).view(n_z, n_v, *frame.shape))

    @torch.no_grad()
    def update_targets(self):
        for t, s in zip(self.target_critic.parameters(), self.critic.parameters()):
            t.lerp_(s, self.tau)


# ---------------------------------------------------------------- logging / eval
def to_grid(imgs):
    """(rows, cols, 3, H, W) in [-0.5, 0.5] -> uint8 (rows*H, cols*W, 3)."""
    r, c, _, h, w = imgs.shape
    return ((imgs.clamp(-0.5, 0.5) + 0.5) * 255).byte().permute(0, 3, 1, 4, 2).reshape(r * h, c * w, 3).cpu().numpy()


TRAIN_FIELDS = ["env_step", "episode_return", "fps",
                "critic_loss", "q", "commit_loss", "recon_loss", "actor_loss", "entropy", "alpha"]
EVAL_FIELDS = ["env_step", "env", "return_mean", "return_std",
               "assoc_l1", "z_out_of_range", "code_usage_entropy", "recon_mse"]


class Logger:
    """CSVs in the run dir are the source of truth for report figures; TensorBoard mirrors them live.
    Images (reconstructions, latent traversals) are overwritten as PNGs and kept per step in TensorBoard."""

    def __init__(self, run_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.run_dir, self.tb, self.writers = run_dir, SummaryWriter(run_dir), {}

    def row(self, name, fields, data):
        if name not in self.writers:
            f = open(os.path.join(self.run_dir, f"{name}.csv"), "w", buffering=1)
            self.writers[name] = csv.DictWriter(f, fields, restval="")
            self.writers[name].writeheader()
        self.writers[name].writerow(data)
        prefix = f"{name}_{data['env']}" if "env" in data else name
        for k, v in data.items():
            if isinstance(v, float):
                self.tb.add_scalar(f"{prefix}/{k}", v, data["env_step"])

    def image(self, name, img, step):
        Image.fromarray(img).save(os.path.join(self.run_dir, f"{name}.png"))
        self.tb.add_image(name, img, step, dataformats="HWC")


def evaluate(agent, env, n_episodes):
    returns, frames = [], []
    for _ in range(n_episodes):
        ob, done, ret = env.reset(), False, 0.0
        while not done:
            frames.append(ob[-3:])
            ob, rew, _, done = env.step(agent.act(ob, sample=False))
            ret += rew
        returns.append(ret)
    frames = np.stack(frames)
    return {"return_mean": float(np.mean(returns)), "return_std": float(np.std(returns)),
            **agent.latent_stats(frames)}, frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", help="YAML config, e.g. configs/walker_walk.yaml")
    p.add_argument("overrides", nargs="*", help="key=value, dotted for nesting: seed=2 agent.beta=10")
    cli = p.parse_args()
    with open(cli.config) as f:
        cfg = yaml.safe_load(f)
    for kv in cli.overrides:
        key, val = kv.split("=", 1)
        *parents, leaf = key.split(".")
        d = cfg
        for k in parents:
            d = d[k]
        d[leaf] = yaml.safe_load(val)
    args = SimpleNamespace(**cfg)

    domain, task = args.env_id.split("-", 1)
    ar = args.action_repeat or (2 if args.env_id == "finger-spin" else 4)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = DMCEnv(domain, task, args.seed, ar, args.frame_stack)
    eval_envs = {"train": DMCEnv(domain, task, args.seed + 100, ar, args.frame_stack)}
    for name, distract in (args.eval_envs or {}).items():
        eval_envs[name] = DMCEnv(domain, task, args.seed + 100, ar, args.frame_stack,
                                 distract=dict(background_dataset_path=args.davis_path, **distract))

    agent = ALDAAgent(env.ac_dim, args.frame_stack, **args.agent).to(device)
    buffer = ReplayBuffer(args.buffer_size, (3, 64, 64), env.ac_dim, args.frame_stack)

    run_dir = os.path.join(args.run_dir, f"{args.env_id}_s{args.seed}_{int(time.time())}")
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    log = Logger(run_dir)

    def run_eval(env_step):
        for name, e in eval_envs.items():
            stats, frames = evaluate(agent, e, args.eval_episodes)
            log.row("eval", EVAL_FIELDS, {"env_step": env_step, "env": name, **stats})
            log.image(f"recon_{name}", agent.recon_grid(frames[np.linspace(0, len(frames) - 1, 8).astype(int)]), env_step)
            if name == "train":
                log.image("traversal", agent.traversal_grid(frames[len(frames) // 2]), env_step)
            print(f"[eval] step {env_step} {name}: " + " ".join(f"{k}={v:.3g}" for k, v in stats.items()), flush=True)
        torch.save(agent.state_dict(), os.path.join(run_dir, "agent.pt"))

    ob, t, ep_ret, ep_info, start = env.reset(), 0, 0.0, defaultdict(list), time.time()
    for step in range(args.total_env_steps // ar):
        env_step = step * ar
        if env_step % args.eval_every == 0:
            run_eval(env_step)

        ac = np.random.uniform(-1, 1, env.ac_dim).astype(np.float32) if step < args.random_steps else agent.act(ob)
        next_ob, rew, terminated, truncated = env.step(ac)
        buffer.insert(ob, ac, rew, next_ob, float(terminated), t)  # truncation does not cut the bootstrap
        ob, t, ep_ret = next_ob, t + 1, ep_ret + rew
        if terminated or truncated:
            row = {"env_step": env_step, "episode_return": float(ep_ret), "fps": env_step / (time.time() - start),
                   **{k: float(np.mean(v)) for k, v in ep_info.items()}}
            log.row("train", TRAIN_FIELDS, row)
            print(" ".join(f"{k}={v:.3g}" for k, v in row.items()), flush=True)
            ob, t, ep_ret = env.reset(), 0, 0.0
            ep_info.clear()

        if step >= args.random_steps:
            for k, v in agent.update(buffer.sample(args.batch_size, device), step).items():
                ep_info[k].append(v)

    run_eval(args.total_env_steps)
    log.tb.close()


if __name__ == "__main__":
    main()
