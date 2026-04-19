import logging
from PIL import Image, ImageDraw

from langchain_core.documents import Document

from src.document_loader import load_documents


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
