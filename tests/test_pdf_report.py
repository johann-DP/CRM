import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from PyPDF2 import PdfReader
from phase4 import build_pdf_report, concat_pdf_reports


def test_build_pdf_report(tmp_path):
    out_dir = tmp_path / "out"
    (out_dir / "pca").mkdir(parents=True)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(out_dir / "pca" / "pca_scatter_2d.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    fig.savefig(out_dir / "pca" / "pca_cluster_grid.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot([1, 0], [0, 1])
    fig.savefig(out_dir / "pca" / "pca_analysis_summary.png")
    plt.close(fig)

    comp_dir = out_dir / "comparisons" / "v1" / "pca"
    comp_dir.mkdir(parents=True)
    fig, ax = plt.subplots()
    ax.plot([1, 0], [0, 1])
    fig.savefig(comp_dir / "pca_scatter_2d.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    fig.savefig(comp_dir / "pca_cluster_grid.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(comp_dir / "pca_analysis_summary.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.imshow([[0, 1], [1, 0]])
    ax.axis("off")
    fig.savefig(out_dir / "general_heatmap.png")
    plt.close(fig)

    pdf_path = tmp_path / "report.pdf"
    build_pdf_report(out_dir, pdf_path, ["main", "v1"], {})

    assert pdf_path.exists() and pdf_path.stat().st_size > 0
    reader = PdfReader(str(pdf_path))
    assert len(reader.pages) == 14


def _make_simple_pdf(path: Path, text: str) -> None:
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center")
        pdf.savefig(fig)
        plt.close(fig)


def test_concat_pdf_reports(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    names = [
        "phase4_report_raw.pdf",
        "phase4_report_cleaned_1.pdf",
        "phase4_report_cleaned_3_univ.pdf",
        "phase4_report_cleaned_3_multi.pdf",
    ]

    for i, name in enumerate(names):
        _make_simple_pdf(out_dir / name, str(i))

    seg_dir = out_dir / "old" / "segments"
    seg_dir.mkdir(parents=True)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    fig.savefig(seg_dir / "seg.png")
    plt.close(fig)

    final_pdf = tmp_path / "final.pdf"
    concat_pdf_reports(out_dir, final_pdf)

    assert final_pdf.exists() and final_pdf.stat().st_size > 0
    reader = PdfReader(str(final_pdf))
    assert len(reader.pages) == len(names) + 2
