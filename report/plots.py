"""Report figures from runs/main1m/*/eval.csv. Run: uv run --with pandas --with matplotlib python report/plots.py [ru]
`ru` writes the same figures with Russian labels as *_ru.pdf."""
import glob
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RU = sys.argv[1:] == ["ru"]
SUF = "_ru" if RU else ""


def tr(en, ru):
    return ru if RU else en


RUNS = sorted(glob.glob("runs/main1m/walker-walk_s*"))
OUT = "report/figures"
ENVS = {"train": tr("Train (DMC)", "Обучение (DMC)"), "color": tr("Color shift", "Сдвиг цвета"),
        "dcs": tr("DistractingCS (easy)", "DistractingCS (лёгкий)")}
COLORS = {"train": "#1f77b4", "color": "#ff7f0e", "dcs": "#2ca02c"}
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

ev = pd.concat([pd.read_csv(f"{r}/eval.csv").assign(seed=r.split("_s")[1][0]) for r in RUNS])


def smooth(s, k=5):
    return s.rolling(k, min_periods=1).mean()


def curves(ax, col, ylabel):
    for env, label in ENVS.items():
        d = ev[ev.env == env].pivot(index="env_step", columns="seed", values=col).apply(smooth)
        x = d.index / 1e6
        ax.plot(x, d.mean(axis=1), color=COLORS[env], label=label, lw=1.2)
        ax.fill_between(x, d.min(axis=1), d.max(axis=1), color=COLORS[env], alpha=0.2, lw=0)
    ax.set_xlabel(tr("Environment steps (M)", "Шаги среды, млн"))
    ax.set_ylabel(ylabel)


fig, ax = plt.subplots(figsize=(3.3, 2.2))
curves(ax, "return_mean", tr("Episode return", "Суммарная награда"))
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6)
fig.tight_layout()
fig.savefig(f"{OUT}/returns{SUF}.pdf")
fig.savefig(f"{OUT}/returns{SUF}.png", dpi=200)  # for the README

fig, axs = plt.subplots(1, 3, figsize=(6.75, 1.9))
for ax, (col, lab) in zip(axs, [("assoc_l1", tr(r"$|z_{cont}-z_d|$ (assoc. shift)", r"$|z_{cont}-z_d|$ (ассоц. сдвиг)")),
                                ("z_out_of_range", tr("frac. $z_{cont}$ outside codebook", "доля $z_{cont}$ вне кодовой книги")),
                                ("recon_mse", tr("Reconstruction MSE", "MSE реконструкции"))]):
    curves(ax, col, lab)
axs[2].set_yscale("log")
axs[0].legend(frameon=False, fontsize=6)
fig.tight_layout()
fig.savefig(f"{OUT}/latent_stats{SUF}.pdf")

# research question: does the association shift on OOD frames track the return gap? per eval point, steps >= 200k
late = ev[ev.env_step >= 200_000].pivot_table(index=["seed", "env_step"], columns="env", values=["return_mean", "assoc_l1"])
for env in ("color", "dcs"):
    gap = late["return_mean", "train"] - late["return_mean", env]
    shift = late["assoc_l1", env] - late["assoc_l1", "train"]
    ratio = late["assoc_l1", env] / late["assoc_l1", "train"]
    print(f"{env}: assoc ratio ood/train = {ratio.mean():.2f}, pearson(shift, return gap) = {np.corrcoef(shift, gap)[0, 1]:.2f}, n={len(gap)}")

for r in RUNS:
    s = r.split("_s")[1][0]
    for name in ("traversal", "recon_train", "recon_dcs", "recon_color"):
        shutil.copy(f"{r}/{name}.png", f"{OUT}/{name}_s{s}.png")

summary = ev[ev.env_step >= 900_000].groupby(["env", "seed"]).return_mean.mean().unstack()
print(summary.round(1), "\nmean over seeds:\n", summary.mean(axis=1).round(1), "\nretained vs train:\n", (summary.mean(axis=1) / summary.mean(axis=1)["train"]).round(2))
