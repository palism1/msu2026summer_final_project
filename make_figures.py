#!/usr/bin/env python3
"""Render the five presentation figures from results/summary/ and results/ overlays.

Torch-free. Runs on system python3 (3.9). Outputs PNGs to docs/figures/.
Figure list: docs/PRESENTATION_OUTLINE.md, "Figures to build".
"""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"

# Categorical slots in fixed order (validated): blue, orange, aqua, yellow.
C_SAMH = "#2a78d6"
C_UNET = "#eb6834"
C_SAMB = "#1baf7a"
C_MEDSAM = "#eda100"

MODELS = [
    ("SAM-ViT-H + LoRA", "SAM-H + LoRA", C_SAMH),
    ("U-Net (ResNet-34)", "U-Net", C_UNET),
    ("SAM-ViT-B + LoRA", "SAM-B + LoRA", C_SAMB),
    ("MedSAM-ViT-B + LoRA", "MedSAM + LoRA", C_MEDSAM),
]

SPLIT_COLS = ["seen_kvasir", "seen_clinicdb", "cvc_300", "cvc_colondb", "etis_larib"]
SPLIT_LABELS = ["Kvasir\n(seen)", "ClinicDB\n(seen)", "CVC-300\n(unseen)",
                "ColonDB\n(unseen)", "ETIS-Larib\n(unseen)"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial"],
    "text.color": INK,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 200,
})


def load_summary():
    rows = {}
    with open(ROOT / "results" / "summary" / "summary_by_model.csv") as f:
        for r in csv.DictReader(f):
            rows[r["model"]] = r
    return rows


# ---------------------------------------------------------------- fig 1: radar
def fig1_radar(S):
    n = len(SPLIT_COLS)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])

    fig, ax = plt.subplots(figsize=(7.6, 7.0), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9"], fontsize=8, color=MUTED)
    ax.set_rlabel_position(90)
    ax.set_xticks(ang)
    ax.set_xticklabels(SPLIT_LABELS, fontsize=10.5, color=INK2)
    ax.tick_params(axis="x", pad=14)
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines["polar"].set_color(GRID)
    ax.set_axisbelow(True)

    for key, label, col in MODELS:
        vals = [float(S[key][c + "_mean"]) for c in SPLIT_COLS]
        vals_c = vals + vals[:1]
        ax.plot(ang_c, vals_c, color=col, linewidth=2, marker="o", markersize=4.5,
                markerfacecolor=col, markeredgecolor=SURFACE, markeredgewidth=1,
                zorder=3, label=label)
        ax.fill(ang_c, vals_c, color=col, alpha=0.06, zorder=2)

    # Direct labels, one model per axis, so no two labels collide.
    label_axis = {"SAM-ViT-H + LoRA": "etis_larib", "U-Net (ResNet-34)": "cvc_colondb",
                  "SAM-ViT-B + LoRA": "cvc_300", "MedSAM-ViT-B + LoRA": "etis_larib"}
    offsets = {"SAM-ViT-H + LoRA": (10, 10), "U-Net (ResNet-34)": (-6, -16),
               "SAM-ViT-B + LoRA": (10, -2), "MedSAM-ViT-B + LoRA": (2, -18)}
    for key, label, col in MODELS:
        c = label_axis[key]
        v = float(S[key][c + "_mean"])
        ax.annotate(label, xy=(ang[SPLIT_COLS.index(c)], v), xytext=offsets[key],
                    textcoords="offset points", fontsize=9.5, color=col,
                    fontweight="bold")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False,
              fontsize=9, handlelength=1.4, columnspacing=1.2)
    fig.suptitle("Per-split mean Dice (3 seeds)", fontsize=13, color=INK, y=0.97)
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.92)
    fig.savefig(OUT / "fig1_radar.png")
    plt.close(fig)


# ----------------------------------------------------- fig 2: gap delta bars
def fig2_gap_bars(S):
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    order = sorted(MODELS, key=lambda m: float(S[m[0]]["generalization_gap_dice_mean"]))
    y = np.arange(len(order))
    for i, (key, label, col) in enumerate(order):
        gap = float(S[key]["generalization_gap_dice_mean"])
        sd = float(S[key]["generalization_gap_dice_std"])
        ax.barh(i, gap, height=0.58, color=col, zorder=3)
        ax.errorbar(gap, i, xerr=sd, fmt="none", ecolor=INK2, elinewidth=1,
                    capsize=3, zorder=4)
        ax.annotate("%.3f" % gap, xy=(gap + sd + 0.004, i), va="center",
                    fontsize=10, color=INK, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([m[1] for m in order], fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.19)
    ax.set_xlabel("Generalization drop: seen mDice − unseen mDice", fontsize=10)
    ax.xaxis.grid(color=GRID, linewidth=0.8, zorder=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    ax.set_title("How much each model loses on unseen datasets", fontsize=13,
                 color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_gap_bars.png")
    plt.close(fig)


# ------------------------------------------------- fig 3: LoRA architecture
def fig3_lora_arch():
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    def box(x, y, w, h, fc, ec, text, tc=INK, fs=9, lw=1.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, col=INK2, lw=1.4):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=12, color=col, lw=lw))

    gray_fc, gray_ec = "#eceae4", "#c3c2b7"

    box(2, 20, 10, 10, "#ffffff", BASE, "Image\n1024²", INK2)
    arrow(13, 25, 17, 25)

    # Frozen ViT encoder: four blocks; Q/V in blocks carry the LoRA bypass.
    for i in range(4):
        x = 18 + i * 11
        box(x, 18, 9, 14, gray_fc, gray_ec, "ViT\nblock", INK2)
        if i < 3:
            arrow(x + 9.6, 25, x + 10.6, 25)
        # LoRA bypass under each block
        box(x + 0.7, 6, 7.6, 7, "#e3edf9", C_SAMH, "LoRA\nA·B  r=4", "#1c5cab", fs=7.5)
        arrow(x + 2.4, 18, x + 2.4, 13.6, C_SAMH, 1.2)
        arrow(x + 6.4, 13.6, x + 6.4, 18, C_SAMH, 1.2)

    ax.text(39.5, 35.5, "SAM ViT encoder: frozen", ha="center", fontsize=10,
            color=INK2)
    ax.text(39.5, 2.2, "LoRA adapters on Q and V only: trained, r=4, α=8",
            ha="center", fontsize=10, color="#1c5cab", fontweight="bold")

    arrow(62.6, 25, 67, 25)
    box(68, 18, 13, 14, "#e3edf9", C_SAMH, "Light CNN\ndecoder\n(trained)", "#1c5cab", fs=9)
    arrow(81.6, 25, 86, 25)
    box(87, 20, 11, 10, "#ffffff", BASE, "Mask\nlogits", INK2)

    ax.text(50, 45.5, "830K trainable parameters = 3.4% of U-Net's 24.4M. No prompts at inference.",
            ha="center", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_lora_arch.png")
    plt.close(fig)


# --------------------------------------------------- fig 4: PraNet splits
def fig4_pranet_split():
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 54)
    ax.axis("off")

    def box(x, y, w, h, fc, ec, lines, tc=INK, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, lines, ha="center", va="center", fontsize=fs, color=tc)

    def arrow(x0, y0, x1, y1, col=INK2):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=12, color=col, lw=1.4))

    blue_fc, blue_ec = "#e3edf9", C_SAMH

    box(3, 34, 20, 12, blue_fc, blue_ec, "Kvasir-SEG\n1,000 images", "#1c5cab")
    box(3, 16, 20, 12, blue_fc, blue_ec, "CVC-ClinicDB\n612 images", "#1c5cab")

    box(34, 24, 22, 14, C_SAMH, C_SAMH, "TRAIN\n900 Kvasir\n+ 550 ClinicDB\n= 1,450", "#ffffff")
    arrow(23.8, 40, 33, 33)
    arrow(23.8, 22, 33, 29)

    # Seen test boxes
    box(66, 40, 30, 9, "#ffffff", BASE, "Kvasir test: 100 img (seen)", INK2)
    box(66, 29, 30, 9, "#ffffff", BASE, "ClinicDB test: 62 img (seen)", INK2)
    # Unseen test boxes
    ue = "#b3540f"
    box(66, 14, 30, 9, "#fdeee6", C_UNET, "CVC-ColonDB: 380 img (unseen)", ue)
    box(66, 3, 30, 9, "#fdeee6", C_UNET, "ETIS-Larib: 196 img (unseen)", ue)
    box(66, -8, 30, 9, "#fdeee6", C_UNET, "CVC-300: 60 img (unseen)", ue)

    arrow(23.8, 42, 65, 44.5)
    arrow(23.8, 20, 65, 33.5)
    ax.text(81, 25.2, "Never seen in training: three whole datasets held out",
            ha="center", fontsize=8.5, color=ue, style="italic")

    ax.set_ylim(-10, 56)
    ax.text(50, 53.5, "PraNet protocol: train on two datasets, test on five",
            ha="center", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_pranet_split.png")
    plt.close(fig)


# -------------------------------------------- fig 5: qualitative overlay grid
def _panel(img, third, with_title):
    """Crop panel `third` (0..2) from a 3-panel overlay strip."""
    w, h = img.size
    x0, x1 = int(third * w / 3), int((third + 1) * w / 3)
    y0 = 30 if with_title else 56
    return img.crop((x0, y0, x1, h - 14))


def _thumb(img, third):
    p = _panel(img, third, False).convert("L").resize((48, 48))
    return np.asarray(p, dtype=float)


def fig5_qual_grid():
    udir = ROOT / "results" / "unet" / "seed42"
    sdir = ROOT / "results" / "sam_vit_h" / "seed42"
    ucases = sorted(udir.glob("overlay_cvc_colondb_*.png"))
    scases = sorted(sdir.glob("overlay_cvc_colondb_*.png"))
    uimgs = [Image.open(p) for p in ucases]
    simgs = [Image.open(p) for p in scases]

    # Match SAM strips to U-Net strips by the image panel (indices may differ).
    uthumbs = [_thumb(i, 0) for i in uimgs]
    sthumbs = [_thumb(i, 0) for i in simgs]
    match = {}
    for ui, ut in enumerate(uthumbs):
        errs = [float(np.mean((ut - st) ** 2)) for st in sthumbs]
        match[ui] = int(np.argmin(errs))
        print("unet case %02d -> sam case %02d (mse %.1f)" % (ui, match[ui], min(errs)))

    rows = [2, 1, 0]  # both-succeed anchor, partial U-Net failure, U-Net collapse
    # Per-case Dice, transcribed from the overlay title strips.
    dice = {2: (0.965, 0.963), 1: (0.560, 0.955), 0: (0.114, 0.846)}
    fig, axes = plt.subplots(len(rows), 4, figsize=(10.4, 2.62 * len(rows)))
    col_titles = ["Image", "Ground truth", "U-Net", "SAM-H + LoRA"]
    for r, ui in enumerate(rows):
        si = match[ui]
        panels = [
            _panel(uimgs[ui], 0, False),
            _panel(uimgs[ui], 1, False),
            _panel(uimgs[ui], 2, False),
            _panel(simgs[si], 2, False),
        ]
        for c, p in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(p)
            ax.axis("off")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11, color=INK, pad=8,
                             fontweight="bold")
            if c >= 2:
                d = dice[ui][c - 2]
                ax.text(0.06, 0.94, "Dice %.3f" % d, transform=ax.transAxes,
                        fontsize=10, color="#ffffff", fontweight="bold", va="top")
    fig.suptitle("CVC-ColonDB (unseen): where the specialist breaks and the adapted "
                 "foundation model holds", fontsize=12, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT / "fig5_qual_grid.png")
    plt.close(fig)


if __name__ == "__main__":
    S = load_summary()
    fig1_radar(S)
    fig2_gap_bars(S)
    fig3_lora_arch()
    fig4_pranet_split()
    fig5_qual_grid()
    print("Wrote figures to", OUT)
