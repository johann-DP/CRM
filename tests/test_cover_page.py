from pathlib import Path
from PyPDF2 import PdfReader

from phase4.create_cover import create_cover_pdf, TITLE


def test_create_cover_pdf(tmp_path: Path) -> None:
    pdf_path = create_cover_pdf(tmp_path / "cover.pdf", author="Tester", subtitle="Sub")
    assert pdf_path.exists() and pdf_path.stat().st_size > 0

    reader = PdfReader(str(pdf_path))
    info = reader.metadata
    assert info.title == TITLE
    assert info.author == "Tester"
    assert len(reader.pages) == 1
