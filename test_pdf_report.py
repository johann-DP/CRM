import pandas as pd
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader

from pdf_report import export_report_to_pdf


def test_export_report_to_pdf(tmp_path):
    figs = {}
    # simple plot
    fig1, ax1 = plt.subplots()
    ax1.plot([0, 1], [0, 1])
    figs["diag"] = fig1

    # second figure
    fig2, ax2 = plt.subplots()
    ax2.bar([0, 1, 2], [1, 2, 3])
    figs["bar"] = fig2

    table = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=["row1", "row2"])
    tables = {"metrics": table}

    pdf_path = tmp_path / "report.pdf"
    out = export_report_to_pdf(figs, tables, pdf_path)
    assert out.exists()

    reader = PdfReader(str(out))
    assert len(reader.pages) == 1 + len(figs) + len(tables)
