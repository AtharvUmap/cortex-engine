import logging
import sys
from unittest.mock import MagicMock, patch

from PIL import Image, ImageDraw

from langchain_core.documents import Document

from src.document_loader import (
    _ocr_pdf,
    _ocr_pdf_with_tesseract,
    _ocr_pdf_with_vision,
    load_documents,
)


# --- Test helpers ---

def _create_dummy_pdf(path):
    """Create a minimal valid PDF file at the given path."""
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 12 Tf 100 700 Td (Hello PDF) Tj ET\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000360 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n441\n%%EOF\n"
    )
    with open(path, "wb") as f:
        f.write(pdf_content)


def _create_image_with_text(path, text="Receipt Total $42.99"):
    """Create a PNG/JPG image with visible text for OCR testing.
    Uses Pillow to draw text onto a white background.
    """
    img = Image.new("RGB", (400, 200), color="white")
    draw = ImageDraw.Draw(img)
    # Use default font — no external font files needed
    draw.text((50, 80), text, fill="black")
    img.save(path)


# --- PDF tests (uses PyPDFDirectoryLoader — fast, exact) ---

def test_load_pdf_returns_list_of_documents(tmp_path):
    """Loading a directory with a PDF should return a list of Document objects."""
    _create_dummy_pdf(tmp_path / "test.pdf")

    docs = load_documents(str(tmp_path))

    assert isinstance(docs, list)
    assert len(docs) > 0
    assert all(isinstance(doc, Document) for doc in docs)


def test_load_empty_directory_returns_empty_list(tmp_path, caplog):
    """Loading an empty directory should return an empty list and log a warning."""
    with caplog.at_level(logging.WARNING):
        docs = load_documents(str(tmp_path))

    assert docs == []
    assert any("No documents found" in record.message for record in caplog.records)


def test_load_ignores_non_supported_files(tmp_path):
    """Text files and other unsupported types should be ignored."""
    (tmp_path / "notes.txt").write_text("not a pdf or image")

    docs = load_documents(str(tmp_path))

    assert docs == []


# --- Image OCR tests (uses UnstructuredFileLoader with hi_res) ---

def test_load_png_image(tmp_path):
    """The loader should be able to process PNG image files via OCR."""
    _create_image_with_text(tmp_path / "receipt.png")

    docs = load_documents(str(tmp_path))

    assert isinstance(docs, list)
    assert len(docs) > 0
    assert all(isinstance(doc, Document) for doc in docs)


def test_load_image_extracts_text(tmp_path):
    """OCR should extract readable text from an image file."""
    _create_image_with_text(tmp_path / "receipt.png", text="Invoice Amount 99")

    docs = load_documents(str(tmp_path))

    # The extracted text should contain at least part of the text in the image
    combined_text = " ".join(doc.page_content for doc in docs).lower()
    assert "invoice" in combined_text or "amount" in combined_text or "99" in combined_text


def test_load_jpg_image(tmp_path):
    """The loader should be able to process JPG image files via OCR."""
    _create_image_with_text(tmp_path / "scan.jpg", text="Scanned Document")

    docs = load_documents(str(tmp_path))

    assert isinstance(docs, list)
    assert len(docs) > 0


# --- Mixed file type tests ---

def test_load_mixed_file_types(tmp_path):
    """The loader should handle a directory with both PDFs and images."""
    _create_dummy_pdf(tmp_path / "notes.pdf")
    _create_image_with_text(tmp_path / "receipt.png")

    docs = load_documents(str(tmp_path))

    # Should return documents from both files
    assert len(docs) >= 2


# --- Apple Vision OCR tests (Ticket 19) ---

@patch("src.document_loader._ocr_pdf_with_vision")
def test_low_text_pdf_triggers_vision_ocr(mock_ocr, tmp_path):
    """A PDF whose text layer is below MIN_PDF_CHARS must hand off to Vision OCR."""
    pdf_path = tmp_path / "scan.pdf"
    _create_dummy_pdf(pdf_path)  # yields ~9 chars, well below MIN_PDF_CHARS=50
    mock_ocr.return_value = Document(
        page_content="ocr'd text",
        metadata={"source": str(pdf_path)},
    )

    docs = load_documents(str(tmp_path))

    mock_ocr.assert_called_once()
    called_path = mock_ocr.call_args.args[0]
    assert str(called_path) == str(pdf_path)
    assert len(docs) == 1
    assert docs[0].page_content == "ocr'd text"


@patch("src.document_loader._ocr_pdf_with_vision")
@patch("src.document_loader.PyPDFLoader")
def test_high_text_pdf_skips_vision_ocr(mock_loader_cls, mock_ocr, tmp_path):
    """A PDF with extractable text above MIN_PDF_CHARS must NOT call Vision OCR."""
    pdf_path = tmp_path / "essay.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 placeholder")

    long_doc = Document(
        page_content="lorem ipsum " * 20,  # ~240 chars, well above threshold
        metadata={"source": str(pdf_path)},
    )
    mock_loader_cls.return_value.load.return_value = [long_doc]

    docs = load_documents(str(tmp_path))

    mock_ocr.assert_not_called()
    assert len(docs) == 1
    assert docs[0].page_content.startswith("lorem ipsum")


def test_vision_ocr_returns_document_with_pdf_path(tmp_path, monkeypatch):
    """_ocr_pdf_with_vision must return a Document with OCR text + source metadata."""
    pdf_path = tmp_path / "visa.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 placeholder")

    fake_image = MagicMock(name="page_image")
    fake_page = MagicMock()
    fake_page.render.return_value.to_pil.return_value = fake_image

    fake_pdf = MagicMock()
    fake_pdf.__len__.return_value = 1
    fake_pdf.__getitem__.return_value = fake_page

    fake_pdfium = MagicMock()
    fake_pdfium.PdfDocument.return_value = fake_pdf

    # Real ocrmac returns [(text, confidence, bbox), ...] per page
    fake_ocrmac_sub = MagicMock()
    fake_ocrmac_sub.text_from_image.return_value = [
        ("Atharv US Visa B-1/B-2", 0.99, (0.0, 0.0, 1.0, 1.0)),
        ("expires 2028", 0.98, (0.0, 0.0, 1.0, 1.0)),
    ]
    fake_ocrmac_pkg = MagicMock()
    fake_ocrmac_pkg.ocrmac = fake_ocrmac_sub

    monkeypatch.setitem(sys.modules, "pypdfium2", fake_pdfium)
    monkeypatch.setitem(sys.modules, "ocrmac", fake_ocrmac_pkg)
    monkeypatch.setitem(sys.modules, "ocrmac.ocrmac", fake_ocrmac_sub)

    doc = _ocr_pdf_with_vision(pdf_path)

    assert isinstance(doc, Document)
    assert "expires 2028" in doc.page_content
    assert doc.metadata["source"] == str(pdf_path)
    fake_ocrmac_sub.text_from_image.assert_called_once_with(fake_image)


def test_vision_ocr_continues_on_per_page_error(tmp_path, monkeypatch, caplog):
    """If ocrmac raises on one page, the remaining pages must still be OCR'd."""
    pdf_path = tmp_path / "multipage.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 placeholder")

    img_a, img_b, img_c = MagicMock(name="img_a"), MagicMock(name="img_b"), MagicMock(name="img_c")
    page_a, page_b, page_c = MagicMock(), MagicMock(), MagicMock()
    page_a.render.return_value.to_pil.return_value = img_a
    page_b.render.return_value.to_pil.return_value = img_b
    page_c.render.return_value.to_pil.return_value = img_c

    fake_pdf = MagicMock()
    fake_pdf.__len__.return_value = 3
    fake_pdf.__getitem__.side_effect = lambda i: [page_a, page_b, page_c][i]

    fake_pdfium = MagicMock()
    fake_pdfium.PdfDocument.return_value = fake_pdf

    def text_side_effect(image):
        if image is img_a:
            return [("page A content", 0.99, (0, 0, 1, 1))]
        if image is img_c:
            return [("page C content", 0.99, (0, 0, 1, 1))]
        raise RuntimeError("Vision crashed on page B")

    fake_ocrmac_sub = MagicMock()
    fake_ocrmac_sub.text_from_image.side_effect = text_side_effect
    fake_ocrmac_pkg = MagicMock()
    fake_ocrmac_pkg.ocrmac = fake_ocrmac_sub

    monkeypatch.setitem(sys.modules, "pypdfium2", fake_pdfium)
    monkeypatch.setitem(sys.modules, "ocrmac", fake_ocrmac_pkg)
    monkeypatch.setitem(sys.modules, "ocrmac.ocrmac", fake_ocrmac_sub)

    with caplog.at_level(logging.WARNING):
        doc = _ocr_pdf_with_vision(pdf_path)

    assert "page A content" in doc.page_content
    assert "page C content" in doc.page_content
    assert "page B content" not in doc.page_content
    # The skipped page should have produced a warning that mentions the page index.
    assert any("page 2" in record.message.lower() for record in caplog.records)


# --- Cross-platform OCR dispatch tests (Ticket 19.5) ---

def test_ocr_pdf_uses_vision_when_modules_available(tmp_path, monkeypatch):
    """When ocrmac and pypdfium2 import cleanly, the dispatcher must pick Vision."""
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 placeholder")

    # Both modules present in sys.modules → import inside dispatcher succeeds
    monkeypatch.setitem(sys.modules, "ocrmac", MagicMock())
    monkeypatch.setitem(sys.modules, "ocrmac.ocrmac", MagicMock())
    monkeypatch.setitem(sys.modules, "pypdfium2", MagicMock())

    with patch("src.document_loader._ocr_pdf_with_vision") as vision_mock, \
         patch("src.document_loader._ocr_pdf_with_tesseract") as tess_mock:
        vision_mock.return_value = Document(
            page_content="vision result",
            metadata={"source": str(pdf_path)},
        )

        result = _ocr_pdf(pdf_path)

    vision_mock.assert_called_once_with(pdf_path)
    tess_mock.assert_not_called()
    assert result.page_content == "vision result"


def test_ocr_pdf_falls_back_to_tesseract_when_ocrmac_missing(tmp_path, monkeypatch):
    """If ocrmac can't be imported, dispatcher must route to Tesseract."""
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 placeholder")

    # sys.modules[name] = None forces ImportError on `import name`
    monkeypatch.setitem(sys.modules, "ocrmac", None)
    monkeypatch.setitem(sys.modules, "ocrmac.ocrmac", None)

    with patch("src.document_loader._ocr_pdf_with_vision") as vision_mock, \
         patch("src.document_loader._ocr_pdf_with_tesseract") as tess_mock:
        tess_mock.return_value = Document(
            page_content="tesseract result",
            metadata={"source": str(pdf_path)},
        )

        result = _ocr_pdf(pdf_path)

    tess_mock.assert_called_once_with(pdf_path)
    vision_mock.assert_not_called()
    assert result.page_content == "tesseract result"


@patch("src.document_loader.UnstructuredFileLoader")
def test_tesseract_helper_concatenates_pages(mock_loader_cls, tmp_path):
    """_ocr_pdf_with_tesseract returns one Document with joined text + source."""
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 placeholder")

    mock_loader_cls.return_value.load.return_value = [
        Document(page_content="page one text", metadata={"page": 0}),
        Document(page_content="page two text", metadata={"page": 1}),
    ]

    result = _ocr_pdf_with_tesseract(pdf_path)

    mock_loader_cls.assert_called_once_with(str(pdf_path), strategy="hi_res")
    assert isinstance(result, Document)
    assert "page one text" in result.page_content
    assert "page two text" in result.page_content
    assert result.metadata["source"] == str(pdf_path)


@patch("src.document_loader._ocr_pdf")
@patch("src.document_loader.PyPDFLoader")
def test_high_text_pdf_skips_ocr_dispatcher(mock_loader_cls, mock_dispatch, tmp_path):
    """Regression: PDFs with extractable text above MIN_PDF_CHARS skip OCR entirely."""
    pdf_path = tmp_path / "essay.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 placeholder")

    long_doc = Document(
        page_content="lorem ipsum " * 20,
        metadata={"source": str(pdf_path)},
    )
    mock_loader_cls.return_value.load.return_value = [long_doc]

    docs = load_documents(str(tmp_path))

    mock_dispatch.assert_not_called()
    assert len(docs) == 1
    assert docs[0].page_content.startswith("lorem ipsum")
