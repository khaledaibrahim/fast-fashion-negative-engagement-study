from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "reports"
DEFAULT_SOURCE = REPORTS_DIR / "study1_manuscript_support.md"
DEFAULT_OUTPUT = OUTPUTS_DIR / "study1_manuscript_support.docx"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Study 1 manuscript support Word document.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Source markdown file.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output .docx path.")
    parser.add_argument(
        "--downloads-copy",
        default=str(Path.home() / "Downloads" / "study1_manuscript_support.docx"),
        help="Optional copy path for the generated Word document.",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    downloads_copy = Path(args.downloads_copy).resolve() if args.downloads_copy else None

    output.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "/opt/anaconda3/bin/pandoc",
            str(source),
            "--from",
            "gfm",
            "--to",
            "docx",
            "--resource-path",
            str(source.parent),
            "--output",
            str(output),
        ],
        check=True,
    )

    if downloads_copy:
        downloads_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output, downloads_copy)

    print(f"Wrote Word document to {output}")
    if downloads_copy:
        print(f"Copied Word document to {downloads_copy}")


if __name__ == "__main__":
    main()
