import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from matplotlib.backends.backend_pdf import PdfPages
from phase4 import build_pdf_report, concat_pdf_reports


def test_build_pdf_report(tmp_path):
    out_dir = tmp_path / "out"
    (out_dir / "pca").mkdir(parents=True)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(out_dir / "pca" / "plot.png")
    plt.close(fig)

    comp_dir = out_dir / "comparisons" / "v1" / "pca"
    comp_dir.mkdir(parents=True)
    fig, ax = plt.subplots()
    ax.plot([1, 0], [0, 1])
    fig.savefig(comp_dir / "plot.png")
    plt.close(fig)

    pdf_path = tmp_path / "report.pdf"
    build_pdf_report(out_dir, pdf_path, ["main", "v1"], {})

    assert pdf_path.exists() and pdf_path.stat().st_size > 0
    reader = PdfReader(str(pdf_path))
    assert len(reader.pages) >= 5


def test_concat_pdf_reports(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    names = [
        "phase4_report_raw.pdf",
        "phase4_report_cleaned_1.pdf",
        "phase4_report_cleaned_3_univ.pdf",
        "phase4_report_cleaned_3_multi.pdf",
    ]

    for name in names:
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        with PdfPages(out_dir / name) as pdf:
            pdf.savefig(fig)
        plt.close(fig)

    seg_dir = out_dir / "old" / "segments"
    seg_dir.mkdir(parents=True)
    fig, ax = plt.subplots()
    ax.plot([1, 2], [2, 1])
    fig.savefig(seg_dir / "seg.png")
    plt.close(fig)

    final_pdf = tmp_path / "final.pdf"
    concat_pdf_reports(out_dir, final_pdf)

    reader = PdfReader(str(final_pdf))
    assert len(reader.pages) == 6
