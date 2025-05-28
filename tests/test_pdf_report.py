import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from phase4 import build_pdf_report


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
