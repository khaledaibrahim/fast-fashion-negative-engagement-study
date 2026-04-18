from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a PDF appendix of Study 1 figures.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "figures" / "study1_figures_appendix.pdf"),
        help="Path to output PDF.",
    )
    args = parser.parse_args()

    figure_paths = [
        FIGURES_DIR / "fig1_sentiment_distribution.png",
        FIGURES_DIR / "fig2_sentiment_vs_rating.png",
        FIGURES_DIR / "fig3_lda_heatmap.png",
        FIGURES_DIR / "fig4_topic_distribution.png",
        FIGURES_DIR / "fig5_aspect_distribution.png",
        FIGURES_DIR / "fig6_sentiment_trend.png",
        FIGURES_DIR / "fig7_emotion_prevalence.png",
        FIGURES_DIR / "fig8_topic_coherence_comparison.png",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        for figure_path in figure_paths:
            if not figure_path.exists():
                continue
            image = mpimg.imread(figure_path)
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.imshow(image)
            ax.axis("off")
            fig.suptitle(figure_path.stem.replace("_", " ").title(), fontsize=18, fontweight="bold")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Wrote PDF to {output_path}")


if __name__ == "__main__":
    main()
