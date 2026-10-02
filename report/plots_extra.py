"""Extra report figures: training dynamics (train.csv), OOD retention and association-vs-gap scatter (eval.csv),
and reconstruction/traversal evolution over training (TensorBoard image history).
Run: uv run --with pandas --with matplotlib python report/plots_extra.py [ru]
`ru` writes the same figures with Russian labels as *_ru.pdf."""
import glob
import io
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RU = sys.argv[1:] == ["ru"]
SUF = "_ru" if RU else ""


def L(en, ru):
    return ru if RU else en


RUNS = sorted(glob.glob("runs/main1m/walker-walk_s*"))
OUT = "report/figures"
SEED_COLORS = {"1": "#1f77b4", "2": "#d62728"}
ENV_COLORS = {"color": "#ff7f0e", "dcs": "#2ca02c"}
ENV_LABELS = {"color": L("Color shift", "Сдвиг цвета"), "dcs": L("DistractingCS (easy)", "DistractingCS (лёгкий)")}
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def seed_of(run):
    return run.split("_s")[1][0]


tr = pd.concat([pd.read_csv(f"{r}/train.csv").assign(seed=seed_of(r)) for r in RUNS])
ev = pd.concat([pd.read_csv(f"{r}/eval.csv").assign(seed=seed_of(r)) for r in RUNS])

# ---- training dynamics: one row per episode, losses averaged over the episode's updates
panels = [("episode_return", L("Train episode return", "Награда на обучении"), False),
          ("critic_loss", L("Critic loss", "Ошибка критика"), True),
          ("q", L("Mean Q", "Среднее Q"), False), ("actor_loss", L("Actor loss", "Ошибка актора"), False),
          ("recon_loss", L("Reconstruction loss", "Ошибка реконструкции"), True),
          ("commit_loss", L("Commitment loss", "Ошибка привязки"), True),
          ("entropy", L("Policy entropy", "Энтропия политики"), False),
          ("alpha", L(r"Temperature $\alpha$", r"Температура $\alpha$"), True)]
fig, axs = plt.subplots(2, 4, figsize=(6.75, 3.0))
for ax, (col, label, log) in zip(axs.flat, panels):
    for seed, d in tr.dropna(subset=[col]).groupby("seed"):
        ax.plot(d.env_step / 1e6, d[col].rolling(20, min_periods=1).mean(), color=SEED_COLORS[seed], lw=1,
                label=L(f"seed {seed}", f"сид {seed}"))
    ax.set_title(label, fontsize=8)
    if log:
        ax.set_yscale("log")
for ax in axs[1]:
    ax.set_xlabel(L("Env steps (M)", "Шаги среды, млн"))
axs[0, 0].legend(frameon=False, fontsize=6)
fig.tight_layout()
fig.savefig(f"{OUT}/training_dynamics{SUF}.pdf")

# ---- OOD retention over training and codebook usage
wide = ev.pivot_table(index=["seed", "env_step"], columns="env", values=["return_mean", "code_usage_entropy"])
fig, axs = plt.subplots(1, 2, figsize=(6.75, 2.0))
for env in ("color", "dcs"):
    ratio = (wide["return_mean", env] / wide["return_mean", "train"]).unstack("seed")
    ratio = ratio[ratio.index >= 100_000].rolling(5, min_periods=1).mean()
    x = ratio.index / 1e6
    axs[0].plot(x, ratio.mean(axis=1), color=ENV_COLORS[env], label=ENV_LABELS[env], lw=1.2)
    axs[0].fill_between(x, ratio.min(axis=1), ratio.max(axis=1), color=ENV_COLORS[env], alpha=0.2, lw=0)
axs[0].axhline(1, color="gray", lw=0.6, ls="--")
axs[0].set_ylabel(L("OOD / train return", "Награда OOD / обучение"))
axs[0].legend(frameon=False, fontsize=6)
for env, c in (("train", "#1f77b4"), *ENV_COLORS.items()):
    e = wide["code_usage_entropy", env].unstack("seed").rolling(5, min_periods=1).mean()
    axs[1].plot(e.index / 1e6, e.mean(axis=1), color=c, lw=1.2, label={"train": L("Train", "Обучение"), **ENV_LABELS}[env])
    axs[1].fill_between(e.index / 1e6, e.min(axis=1), e.max(axis=1), color=c, alpha=0.2, lw=0)
axs[1].set_ylabel(L("Codebook usage entropy\n(normalized)", "Энтропия использования\nкодовой книги (норм.)"))
axs[1].legend(frameon=False, fontsize=6)
for ax in axs:
    ax.set_xlabel(L("Environment steps (M)", "Шаги среды, млн"))
fig.tight_layout()
fig.savefig(f"{OUT}/retention_entropy{SUF}.pdf")

# ---- research question: per-evaluation association shift vs return gap, steps >= 200k
late = ev[ev.env_step >= 200_000].pivot_table(index=["seed", "env_step"], columns="env",
                                              values=["return_mean", "assoc_l1"]).reset_index()
fig, axs = plt.subplots(1, 2, figsize=(6.75, 2.2), sharey=True)
for ax, env in zip(axs, ("color", "dcs")):
    shift = late["assoc_l1", env] - late["assoc_l1", "train"]
    gap = late["return_mean", "train"] - late["return_mean", env]
    sc = ax.scatter(shift, gap, c=late["env_step"] / 1e6, cmap="viridis", s=8, vmin=0.2, vmax=1.0)
    k, b = np.polyfit(shift, gap, 1)
    xs = np.linspace(shift.min(), shift.max(), 2)
    ax.plot(xs, k * xs + b, color="k", lw=0.8)
    ax.set_title(f"{ENV_LABELS[env]}  (r = {np.corrcoef(shift, gap)[0, 1]:.2f}, n = {len(gap)})", fontsize=8)
    ax.set_xlabel(L(r"assoc. shift OOD $-$ train", r"ассоц. сдвиг: OOD $-$ обучение"))
axs[0].set_ylabel(L("Return gap (train $-$ OOD)", "Разрыв награды (обучение $-$ OOD)"))
fig.colorbar(sc, ax=axs, label=L("Env steps (M)", "Шаги среды, млн"), fraction=0.03)
fig.savefig(f"{OUT}/rq_scatter{SUF}.pdf", bbox_inches="tight")


# ---- image evolution from TensorBoard (seed 1)
def tb_images(run, tag):
    ea = EventAccumulator(glob.glob(f"{run}/events.out.tfevents.*")[0], size_guidance={"images": 0})
    ea.Reload()
    return {e.step: np.array(Image.open(io.BytesIO(e.encoded_image_string))) for e in ea.Images(tag)}


STEPS = [0, 100_000, 300_000, 600_000, 1_000_000]
run = RUNS[0]
for tag in ("recon_train", "recon_dcs", "recon_color"):
    imgs = tb_images(run, tag)
    fig, axs = plt.subplots(len(STEPS), 1, figsize=(3.3, 0.85 * len(STEPS)))
    for ax, s in zip(axs, STEPS):
        ax.imshow(imgs[s])
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_ylabel(L(f"{s / 1e6:g}M", f"{s / 1e6:g} млн".replace(".", ",")), rotation=0, ha="right", va="center")
    fig.tight_layout(h_pad=0.2)
    fig.savefig(f"{OUT}/{tag}_evolution{SUF}.pdf", dpi=200)

imgs = tb_images(run, "traversal")
fig, axs = plt.subplots(1, 3, figsize=(6.75, 2.4))
for ax, s in zip(axs, (100_000, 500_000, 1_000_000)):
    ax.imshow(imgs[s])
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(L(f"{s / 1e6:g}M steps", f"{s / 1e6:g} млн шагов".replace(".", ",")), fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/traversal_evolution{SUF}.pdf", dpi=200)
