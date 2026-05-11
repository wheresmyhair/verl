"""Visualize per-prompt CV distribution across the 27-cell matrix.

Per cell, for each prompt we compute CV = std/mean of its n=16
response lengths. The figure shows the *distribution* of those CVs
across all prompts in that cell.

Two figures:
  1. cv_distribution_grid.png — small-multiples grid (model × dataset),
     each panel a violin/histogram of per-prompt CVs.
  2. cv_summary_box.png — one boxplot per cell, sorted by median CV,
     for at-a-glance comparison.

Per-prompt CV interprets directly to "if we use the median of past
samples to predict the next sample's length, what's the relative
error we should expect?". Low CV → predictor useful. High CV →
even a perfect rank-correlation predictor produces ±X% absolute
errors.
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/home/user/profiling_rlpipe/rollout_stats")
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

MODELS_ORDER = [
    "Qwen3-1.7B", "Qwen3-8B", "Qwen3-14B",
    "Qwen3-30B-A3B", "Qwen3-32B", "DS-R1-Distill-Llama-70B",
]
DATASETS_ORDER = [
    "dapo-math-17k", "aime-24", "math-500",
    "livecodebench", "codecontests",
]


def load_cv_per_prompt(resp_path: Path) -> list[float]:
    """Per-prompt CV = std/mean across the 16 samples of that prompt."""
    by_prompt: dict[int, list[int]] = {}
    with resp_path.open() as f:
        for line in f:
            r = json.loads(line)
            by_prompt.setdefault(r["prompt_id"], []).append(int(r["response_tokens"]))
    cvs = []
    for lengths in by_prompt.values():
        if len(lengths) < 2:
            continue
        m = statistics.fmean(lengths)
        if m <= 0:
            continue
        sd = statistics.pstdev(lengths)
        cvs.append(sd / m)
    return cvs


def fig_grid():
    fig, axes = plt.subplots(
        len(MODELS_ORDER), len(DATASETS_ORDER),
        figsize=(17, 16),
    )
    for i, model in enumerate(MODELS_ORDER):
        for j, ds in enumerate(DATASETS_ORDER):
            ax = axes[i, j]
            resp = ROOT / model / ds / "responses.jsonl"
            if not resp.exists():
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, fontsize=18, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                cvs = load_cv_per_prompt(resp)
                if not cvs:
                    continue
                ax.hist(cvs, bins=25, range=(0, 1.0),
                        color="#4A7AB7", edgecolor="black", linewidth=0.4)
                med = statistics.median(cvs)
                ax.axvline(med, color="red", linestyle="--", linewidth=1)
                ax.axvline(0.20, color="gray", linestyle=":", linewidth=0.6)
                ax.text(
                    0.97, 0.95,
                    f"med={med:.2f}\nprompts={len(cvs)}",
                    transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.78, edgecolor="none"),
                )
            ax.set_xlim(0, 1.0)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.tick_params(axis="x", labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(alpha=0.18, linewidth=0.4)
            if i == 0:
                ax.set_title(ds, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{model}\ncount", fontsize=8)
            if i == len(MODELS_ORDER) - 1:
                ax.set_xlabel("per-prompt CV", fontsize=8)
    fig.suptitle(
        "Per-prompt CV distribution (each prompt: CV across its 16 samples)\n"
        "red dashed = cell median; gray dotted = CV 0.20 reference",
        fontsize=11,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    out = OUT_DIR / "cv_distribution_grid.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def fig_box():
    cells = []
    for model in MODELS_ORDER:
        for ds in DATASETS_ORDER:
            resp = ROOT / model / ds / "responses.jsonl"
            if resp.exists():
                cvs = load_cv_per_prompt(resp)
                if cvs:
                    cells.append((model, ds, cvs))
    cells.sort(key=lambda x: statistics.median(x[2]))

    labels = [f"{m} / {d}" for m, d, _ in cells]
    data = [c[2] for c in cells]

    fig, ax = plt.subplots(figsize=(10, 9))
    bp = ax.boxplot(
        data, vert=False, widths=0.7,
        patch_artist=True, showfliers=True, flierprops=dict(marker=".", markersize=2),
    )
    for patch, (m, _d, _c) in zip(bp["boxes"], cells):
        # Color by model family.
        if m.startswith("Qwen3-"):
            patch.set_facecolor("#A8C7E2")
        elif m.startswith("DS-"):
            patch.set_facecolor("#E2A8A8")
        elif "MoE" in m or "A3B" in m:
            patch.set_facecolor("#A8E2B0")
    ax.set_yticks(range(1, len(labels) + 1))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("per-prompt CV (std/mean of 16 sample lengths)")
    ax.set_xlim(0, 1.0)
    ax.set_title(
        "Per-prompt response-length CV across 27 (model × dataset) cells\n"
        "sorted by median CV; lower = same prompt produces more consistent lengths"
    )
    ax.axvline(0.20, color="gray", linestyle=":", linewidth=1)
    ax.text(0.20, len(labels) + 0.3, " CV=0.20 reference", fontsize=8, color="gray")
    fig.tight_layout()
    out = OUT_DIR / "cv_summary_box.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def fig_cv_vs_mean():
    """Per-prompt scatter: x = mean response length of that prompt,
    y = CV of its 16 samples. One sub-plot per cell, with axes."""
    fig, axes = plt.subplots(
        len(MODELS_ORDER), len(DATASETS_ORDER),
        figsize=(17, 16),
    )
    for i, model in enumerate(MODELS_ORDER):
        for j, ds in enumerate(DATASETS_ORDER):
            ax = axes[i, j]
            resp = ROOT / model / ds / "responses.jsonl"
            if not resp.exists():
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, fontsize=18, color="gray")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            by_prompt: dict[int, list[int]] = {}
            with resp.open() as f:
                for line in f:
                    r = json.loads(line)
                    by_prompt.setdefault(r["prompt_id"], []).append(int(r["response_tokens"]))
            xs, ys = [], []
            for lengths in by_prompt.values():
                if len(lengths) < 2: continue
                m = statistics.fmean(lengths)
                if m <= 0: continue
                xs.append(m)
                ys.append(statistics.pstdev(lengths) / m)
            ax.scatter(xs, ys, s=5, alpha=0.45, color="#4A7AB7")
            ax.set_xlim(0, 17000)
            ax.set_ylim(0, 1.0)
            ax.axhline(0.20, color="gray", linestyle=":", linewidth=0.6)
            ax.axvline(16384, color="red", linestyle=":", linewidth=0.6)
            ax.set_xticks([0, 4000, 8000, 12000, 16000])
            ax.set_xticklabels(["0", "4K", "8K", "12K", "16K"], fontsize=7)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(alpha=0.2, linewidth=0.4)
            if i == 0:
                ax.set_title(ds, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{model}\nCV", fontsize=8)
            if i == len(MODELS_ORDER) - 1:
                ax.set_xlabel("mean length", fontsize=8)
    fig.suptitle(
        "Per-prompt: mean response length (x, tokens) vs sampling CV (y)\n"
        "each dot = 1 prompt's mean over 16 samples; gray dotted = CV 0.20; red dotted = 16K cap",
        fontsize=11,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    out = OUT_DIR / "cv_vs_mean_scatter.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def fig_response_length_grid():
    """6×5 grid: per-cell histogram of *all* response tokens with
    16K cap line and clip% annotation."""
    fig, axes = plt.subplots(
        len(MODELS_ORDER), len(DATASETS_ORDER),
        figsize=(17, 16),
    )
    for i, model in enumerate(MODELS_ORDER):
        for j, ds in enumerate(DATASETS_ORDER):
            ax = axes[i, j]
            resp = ROOT / model / ds / "responses.jsonl"
            if not resp.exists():
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, fontsize=18, color="gray")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            tokens = []
            clip = 0
            with resp.open() as f:
                for line in f:
                    r = json.loads(line)
                    t = int(r["response_tokens"])
                    tokens.append(t)
                    if t >= 16384:
                        clip += 1
            n = len(tokens)
            ax.hist(
                tokens, bins=40, range=(0, 16384),
                color="#5C8DC9", edgecolor="black", linewidth=0.3,
            )
            ax.axvline(16384, color="red", linestyle=":", linewidth=0.7)
            med = sorted(tokens)[n // 2]
            ax.axvline(med, color="orange", linestyle="--", linewidth=0.8)
            ax.set_xlim(0, 16384)
            ax.set_xticks([0, 4000, 8000, 12000, 16000])
            ax.set_xticklabels(["0", "4K", "8K", "12K", "16K"], fontsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.text(
                0.97, 0.95,
                f"med={med}\nclip={clip / n * 100:.1f}%\nN={n}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.78, edgecolor="none"),
            )
            if i == 0:
                ax.set_title(ds, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{model}\ncount", fontsize=8)
            if i == len(MODELS_ORDER) - 1:
                ax.set_xlabel("response tokens", fontsize=8)
    fig.suptitle(
        "Response-length distribution per cell (all 16 samples × all prompts)\n"
        "orange dashed = median; red dotted = 16K cap; clip% = fraction hitting cap",
        fontsize=11,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    out = OUT_DIR / "response_length_grid.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def fig_response_length_cdf():
    """Single-plot CDF, one curve per cell, color-coded by model and
    line-style by dataset. Shows tail shape comparison directly."""
    fig, ax = plt.subplots(figsize=(11, 7))
    model_colors = {
        "Qwen3-1.7B":              "#5dade2",
        "Qwen3-8B":                "#3498db",
        "Qwen3-14B":               "#2874a6",
        "Qwen3-30B-A3B":           "#27ae60",
        "Qwen3-32B":               "#1b4f72",
        "DS-R1-Distill-Llama-70B": "#c0392b",
    }
    ds_styles = {
        "dapo-math-17k":  "-",
        "math-500":       "--",
        "aime-24":        "-.",
        "livecodebench":  ":",
        "codecontests":   (0, (3, 1, 1, 1)),  # dash-dot-dot
    }
    for model in MODELS_ORDER:
        for ds in DATASETS_ORDER:
            resp = ROOT / model / ds / "responses.jsonl"
            if not resp.exists():
                continue
            tokens = []
            with resp.open() as f:
                for line in f:
                    tokens.append(int(json.loads(line)["response_tokens"]))
            tokens.sort()
            n = len(tokens)
            xs = tokens
            ys = [(i + 1) / n for i in range(n)]
            ax.plot(
                xs, ys,
                color=model_colors[model],
                linestyle=ds_styles[ds],
                linewidth=1.0,
                alpha=0.85,
            )
    ax.axvline(16384, color="red", linestyle=":", linewidth=0.7)
    ax.set_xlim(0, 16384)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16384])
    ax.set_xticklabels(["0", "2K", "4K", "6K", "8K", "10K", "12K", "14K", "16K"])
    ax.set_xlabel("response tokens", fontsize=11)
    ax.set_ylabel("cumulative fraction of samples", fontsize=11)
    ax.set_title(
        "Response-length CDF per (model × dataset) cell\n"
        "color = model family; line style = dataset",
        fontsize=11,
    )
    ax.grid(alpha=0.25)

    # Two legends.
    from matplotlib.lines import Line2D
    model_handles = [
        Line2D([0], [0], color=c, linewidth=2, label=m)
        for m, c in model_colors.items()
    ]
    ds_handles = [
        Line2D([0], [0], color="black", linestyle=s, linewidth=1.5, label=d)
        for d, s in ds_styles.items()
    ]
    leg1 = ax.legend(handles=model_handles, loc="lower right",
                     bbox_to_anchor=(1.0, 0.0), fontsize=9, title="model")
    ax.add_artist(leg1)
    ax.legend(handles=ds_handles, loc="lower right",
              bbox_to_anchor=(1.0, 0.32), fontsize=9, title="dataset")

    fig.tight_layout()
    out = OUT_DIR / "response_length_cdf.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def fig_clip_ratio_heatmap():
    """Compact 6×5 heatmap of clip%."""
    import numpy as np
    grid = np.full((len(MODELS_ORDER), len(DATASETS_ORDER)), np.nan)
    for i, model in enumerate(MODELS_ORDER):
        for j, ds in enumerate(DATASETS_ORDER):
            resp = ROOT / model / ds / "responses.jsonl"
            if not resp.exists():
                continue
            n = clip = 0
            with resp.open() as f:
                for line in f:
                    n += 1
                    if int(json.loads(line)["response_tokens"]) >= 16384:
                        clip += 1
            if n:
                grid[i, j] = clip / n * 100

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=80, aspect="auto")
    ax.set_xticks(range(len(DATASETS_ORDER)))
    ax.set_xticklabels(DATASETS_ORDER, rotation=20, ha="right")
    ax.set_yticks(range(len(MODELS_ORDER)))
    ax.set_yticklabels(MODELS_ORDER)
    for i in range(len(MODELS_ORDER)):
        for j in range(len(DATASETS_ORDER)):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=11, color="gray")
            else:
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                        color="white" if v > 35 else "black", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("clip ratio = fraction of samples hitting 16K cap (%)", fontsize=9)
    ax.set_title("Truncation rate per (model × dataset) cell", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "clip_ratio_heatmap.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def main():
    fig_grid()
    fig_box()
    fig_cv_vs_mean()
    fig_response_length_grid()
    fig_response_length_cdf()
    fig_clip_ratio_heatmap()


if __name__ == "__main__":
    main()
