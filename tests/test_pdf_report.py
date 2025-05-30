import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from PyPDF2 import PdfReader
from phase4_functions import export_report_to_pdf
from phase4 import concat_pdf_reports


def test_export_report_to_pdf(tmp_path):
    pdf_path = tmp_path / "report.pdf"
    figs = {}
    for idx, name in enumerate(
        ["pca_scatter_2d", "pca_cluster_grid", "pca_analysis_summary"]
    ):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [idx, idx + 1])
        figs[f"main_{name}"] = fig
    export_report_to_pdf(figs, {}, pdf_path)

    assert pdf_path.exists() and pdf_path.stat().st_size > 0
    reader = PdfReader(str(pdf_path))
    assert len(reader.pages) >= 3


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
