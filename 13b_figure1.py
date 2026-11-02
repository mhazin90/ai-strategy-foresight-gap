import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, FancyArrowPatch, PathPatch
from matplotlib.path import Path

os.makedirs("figures/labeled", exist_ok=True)
os.makedirs("figures/clean", exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans"})

C_OI = "#0072B2"
C_OF = "#E69F00"
C_FI = "#009E73"
C_MED = "#CC79A7"
C_GREY = "#9E9E9E"
C_LINE = "#BDBDBD"
C_TEXT = "#555555"

BOXES = [
    ("Documents", "25 national AI strategies\n15 first + 10 second generation", "#F2F2F2", C_GREY),
    ("Paragraphs", "6,184 paragraphs\nEnglish, French, Portuguese", "#F2F2F2", C_GREY),
    ("Automated coding", "Coder A Qwen2.5 · Coder C Llama 3.1\nCoder B embeddings", "#E3EFF8", C_OI),
    ("Foresight detector", "437 keyword candidates\nchecked by two LLMs", "#FDF0DA", C_OF),
    ("Measures", "Link densities and mediation\ncomparison of generations", "#E2F3EE", C_FI),
]
X0, W, GAP, Y0, H = 3.5, 15.0, 4.5, 34.0, 22.0

for clean in [False, True]:
    folder = "figures/clean" if clean else "figures/labeled"
    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64)
    ax.set_aspect("equal")
    ax.axis("off")

    for k, (title, subtitle, fill, edge) in enumerate(BOXES):
        x = X0 + k * (W + GAP)
        cx, cy = x + W / 2, Y0 + 14.5
        ax.add_patch(FancyBboxPatch((x, Y0), W, H, boxstyle="round,pad=0,rounding_size=1.2", fc=fill, ec=edge, lw=1.2))

        if k == 0:
            for j in [2, 1, 0]:
                ax.add_patch(Rectangle((cx - 3.9 + 0.9 * j, cy - 3.1 - 0.9 * j + 0.9), 5.4, 7.2, fc="white", ec=C_GREY, lw=0.9))
            for t in range(4):
                ax.plot([cx - 3.2, cx + 0.8], [cy + 2.6 - t * 1.4] * 2, color=C_LINE, lw=0.8)
        if k == 1:
            ax.add_patch(Rectangle((cx - 3.5, cy - 4.2), 7, 8.4, fc="white", ec=C_GREY, lw=0.9))
            lines = [(3.2, 2.8, C_LINE), (2.4, 2.8, C_LINE), (1.6, 1.2, C_LINE), (0.2, 2.8, C_OI), (-0.6, 0.6, C_OI), (-2.0, 2.8, C_LINE), (-2.8, 2.8, C_LINE), (-3.6, 1.0, C_LINE)]
            for y, right, color in lines:
                ax.plot([cx - 2.8, cx + right], [cy + y] * 2, color=color, lw=1.0)
        if k == 2:
            for offset, color, letter in [(-3.4, C_OI, "A"), (0.0, C_GREY, "B"), (3.4, "#56B4E9", "C")]:
                ax.add_patch(Circle((cx + offset, cy + 1.2), 1.5, fc=color, ec="white", lw=0.8))
                if not clean:
                    ax.text(cx + offset, cy + 1.2, letter, ha="center", va="center", fontsize=6, color="white", fontweight="bold")
            for t, right in enumerate([2.8, 1.6, 2.2]):
                ax.plot([cx - 2.8, cx + right], [cy - 1.8 - t * 1.0] * 2, color=C_OI, lw=1.0, alpha=0.6)
        if k == 3:
            ax.add_patch(Polygon([(cx - 4, cy + 3.0), (cx + 4, cy + 3.0), (cx + 0.9, cy - 0.8), (cx + 0.9, cy - 3.4), (cx - 0.9, cy - 3.4), (cx - 0.9, cy - 0.8)], closed=True, fc=C_OF, ec="none", alpha=0.85))
            for dx in [-3.6, -2.4, -1.2, 0.0, 1.2, 2.4, 3.6]:
                ax.add_patch(Circle((cx + dx, cy + 4.6), 0.45, fc=C_GREY, ec="none"))
            ax.add_patch(Circle((cx, cy - 4.6), 0.6, fc=C_OF, ec="none"))
        if k == 4:
            ax.plot([cx - 3.8, cx + 4.0], [cy - 3.8] * 2, color=C_GREY, lw=0.9)
            for bx, height, color in [(-3.0, 7.0, C_OI), (-0.8, 3.2, C_OF), (1.4, 3.8, C_FI)]:
                ax.add_patch(Rectangle((cx + bx, cy - 3.8), 1.8, height, fc=color, ec="none"))

        if not clean:
            ax.text(cx, Y0 + 7.0, title, ha="center", va="center", fontsize=7.5, fontweight="bold", color="#222222")
            ax.text(cx, Y0 + 3.2, subtitle, ha="center", va="center", fontsize=5.2, linespacing=1.35, color=C_TEXT)

        if k < len(BOXES) - 1:
            ax.add_patch(FancyArrowPatch((x + W + 0.6, Y0 + H / 2), (x + W + GAP - 0.6, Y0 + H / 2), arrowstyle="-|>", mutation_scale=9, color="#7F7F7F", lw=1.2))

    ax.add_patch(FancyBboxPatch((X0, 26.5), 93, 4.5, boxstyle="round,pad=0,rounding_size=1.0", fc="#FAFAFA", ec=C_LINE, lw=0.9, ls=(0, (3, 2))))
    if not clean:
        ax.text(50, 28.75, "Validation   ·   agreement between coders   ·   evidence quotes   ·   expectations set in advance   ·   strict and broad foresight bounds   ·   rarefaction", ha="center", va="center", fontsize=5.6, color=C_TEXT)

    O, I, F = (36, 19), (64, 19), (50, 9)
    ax.plot([O[0], I[0]], [O[1], I[1]], color=C_OI, lw=2.2, zorder=1)
    ax.plot([O[0], F[0]], [O[1], F[1]], color=C_OF, lw=2.2, zorder=1)
    ax.plot([F[0], I[0]], [F[1], I[1]], color=C_FI, lw=2.2, zorder=1)
    arc = Path([(O[0], O[1] - 3.6), (50, -9), (I[0], I[1] - 3.6)], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    ax.add_patch(PathPatch(arc, fc="none", ec=C_MED, lw=1.4, ls=(0, (3, 2)), zorder=0))
    for (px, py), fill, edge in [(O, "#E3EFF8", C_OI), (I, "#E3EFF8", C_OI), (F, "#FDF0DA", C_OF)]:
        ax.add_patch(Circle((px, py), 3.6, fc=fill, ec=edge, lw=1.4, zorder=2))

    if not clean:
        ax.text(O[0], O[1], "Objectives", ha="center", va="center", fontsize=5.6, fontweight="bold", zorder=3)
        ax.text(I[0], I[1], "Instruments", ha="center", va="center", fontsize=5.6, fontweight="bold", zorder=3)
        ax.text(F[0], F[1], "Foresight", ha="center", va="center", fontsize=5.6, fontweight="bold", zorder=3)
        ax.text(50, 20.6, r"$\rho^{OI}$", ha="center", va="bottom", fontsize=8, color=C_OI)
        ax.text(41.5, 13.2, r"$\rho^{OF}$", ha="right", va="center", fontsize=8, color=C_OF)
        ax.text(58.5, 13.2, r"$\rho^{FI}$", ha="left", va="center", fontsize=8, color=C_FI)
        ax.text(58.5, 3.0, r"$\mu$", ha="left", va="center", fontsize=8, color=C_MED)
        ax.text(3.5, 15.5, "Link density  " + r"$\rho$" + "\nshare of possible pairs\nthat a strategy connects", ha="left", va="center", fontsize=5.6, color=C_TEXT, linespacing=1.4)
        ax.text(96.5, 15.5, "Foresight mediation  " + r"$\mu$" + "\nshare of objective–instrument\nlinks anchored in foresight", ha="right", va="center", fontsize=5.6, color=C_TEXT, linespacing=1.4)
        ax.text(1.0, 61.5, "a", fontsize=11, fontweight="bold", va="center")
        ax.text(1.0, 23.5, "b", fontsize=11, fontweight="bold", va="center")

    fig.savefig(f"{folder}/fig1_design.png", dpi=600)
    plt.close(fig)

print("saved:", sorted(f for f in os.listdir("figures/labeled") if f.startswith("fig1")))
