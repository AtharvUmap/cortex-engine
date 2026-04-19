import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader

# Set up a logger for this module to track loading activity
logger = logging.getLogger(__name__)

# Image extensions that need OCR processing
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Minimum total characters a PDF must yield via the direct text-layer reader
# before we consider it "successfully" extracted. Below this, we assume the
# PDF uses an unsupported encoding (e.g. CJK 90ms-RKSJ-H on US visas) and
# fall back to OCR. Tuned empirically: a single-page PDF with any real
# content comfortably exceeds this, while broken extractions yield <5 chars.
MIN_PDF_CHARS = 50


def _load_pdf_with_fallback(pdf_path: Path) -> list:
    """Load a single PDF, falling back to OCR if direct text extraction fails.

    Strategy:
      1. Try PyPDFLoader first — fast, exact, reads the text layer directly.
      2. If the extracted text is suspiciously short (empty, garbage, or
         decoded as unsupported CJK encoding), re-process the file with
         UnstructuredFileLoader(strategy="hi_res") which rasterizes each
         page and runs Tesseract OCR on the resulting images.

    This hybrid approach keeps normal PDFs (tickets, I-20s) fast while
    still reading tricky PDFs (US visas, scanned docs) accurately.
    """
    # Attempt 1: fast text-layer extraction
    try:
        fast_loader = PyPDFLoader(str(pdf_path))
        fast_docs = fast_loader.load()
    except Exception as e:
        # If PyPDF blows up entirely, skip straight to OCR
        logger.warning("PyPDFLoader failed on %s: %s — falling back to OCR", pdf_path.name, e)
        fast_docs = []

    # Measure total extracted content across all pages
    total_chars = sum(len(d.page_content.strip()) for d in fast_docs)

    if fast_docs and total_chars >= MIN_PDF_CHARS:
        # Looks like a normal text PDF — use the fast result as-is
        return fast_docs

    # Attempt 2: OCR fallback for PDFs with missing/unreadable text layers
    logger.warning(
        "Text extraction yielded only %d chars for %s — falling back to OCR (hi_res).",
        total_chars,
        pdf_path.name,
    )
    ocr_loader = UnstructuredFileLoader(str(pdf_path), strategy="hi_res")
    return ocr_loader.load()


def load_documents(dir_path: str) -> list:
    """Load all documents from the given directory.

    Strategy per file type:
    - PDFs: try direct text extraction first (PyPDFLoader); if that yields
      almost nothing (unsupported encoding, scanned PDF, etc.), fall back
      to OCR via UnstructuredFileLoader(strategy="hi_res").
    - Images (.png/.jpg/.jpeg): always processed with OCR.

    Args:
        dir_path: Path to directory containing document files.

    Returns:
        A list of LangChain Document objects.
    """
    docs = []
    dir_pathlib = Path(dir_path)

    # Step 1: Load each PDF individually so we can apply the fallback per-file.
    # Batch-loading via PyPDFDirectoryLoader would hide which PDF failed and
    # make per-file OCR fallback awkward.
    pdf_files = [f for f in dir_pathlib.rglob("*") if f.is_file() and f.suffix.lower() == ".pdf"]
    for pdf_path in pdf_files:
        docs.extend(_load_pdf_with_fallback(pdf_path))

    # Step 2: Images always go through OCR — they have no text layer.
    image_files = [
        f for f in dir_pathlib.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    for image_path in image_files:
        loader = UnstructuredFileLoader(str(image_path), strategy="hi_res")
        docs.extend(loader.load())

    # Warn if nothing was found across both loaders
    if not docs:
        logger.warning("No documents found in directory: %s", dir_path)

    return docs
