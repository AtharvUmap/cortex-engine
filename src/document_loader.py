import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Below this we assume the PDF text layer is broken (e.g. unsupported CJK
# encoding on US visa PDFs) and fall back to OCR. Tuned empirically: a real
# single-page PDF comfortably exceeds 50 chars; broken extractions yield <5.
MIN_PDF_CHARS = 50

# PDFs are defined in 72-point units, so scale 300/72 produces a ~300 DPI
# bitmap — the resolution Apple Vision OCRs reliably without runaway memory.
_OCR_RENDER_SCALE = 300 / 72


def _flatten_ocr_annotations(annotations) -> str:
    """ocrmac.text_from_image returns [(text, confidence, bbox), ...].
    Collapse it into a single newline-joined string."""
    parts = []
    for ann in annotations:
        if isinstance(ann, str):
            parts.append(ann)
        elif ann:
            parts.append(str(ann[0]))
    return "\n".join(parts)


def _ocr_pdf_with_vision(pdf_path: Path) -> Document:
    """OCR a PDF page-by-page using Apple Vision Framework via ocrmac.

    Renders each page through pypdfium2 (no system-level Poppler dependency),
    then hands the image to Apple's native Vision OCR. Per-page errors are
    logged and skipped so one bad page never costs an entire document.
    """
    import pypdfium2 as pdfium
    from ocrmac import ocrmac

    pdf = pdfium.PdfDocument(str(pdf_path))
    page_texts = []
    for page_index in range(len(pdf)):
        try:
            page = pdf[page_index]
            image = page.render(scale=_OCR_RENDER_SCALE).to_pil()
            annotations = ocrmac.text_from_image(image)
            page_texts.append(_flatten_ocr_annotations(annotations))
        except Exception as e:
            logger.warning(
                "Vision OCR failed on page %d of %s: %s — skipping page",
                page_index + 1,
                pdf_path.name,
                e,
            )
            continue

    return Document(
        page_content="\n\n".join(page_texts),
        metadata={"source": str(pdf_path)},
    )


def _ocr_pdf_with_tesseract(pdf_path: Path) -> Document:
    """Cross-platform OCR fallback when Apple Vision isn't available.

    Routes through UnstructuredFileLoader(strategy="hi_res") — the same
    Tesseract path Cortex Engine used before Ticket 19. Returns a single
    Document with the same shape as _ocr_pdf_with_vision so the caller
    doesn't need to know which backend ran.
    """
    loader = UnstructuredFileLoader(str(pdf_path), strategy="hi_res")
    page_docs = loader.load()
    return Document(
        page_content="\n\n".join(d.page_content for d in page_docs),
        metadata={"source": str(pdf_path)},
    )


# Module-level flag so the "Vision unavailable" warning only fires once
# per process — otherwise it would spam the logs on every PDF that hit OCR.
_vision_unavailable_warned = False


def _ocr_pdf(pdf_path: Path) -> Document:
    """Pick an OCR backend by platform capability and run it.

    Apple Vision (ocrmac + pypdfium2) is preferred — far more accurate on
    visa-style scans and ~10× faster on Apple Silicon. On systems where
    ocrmac can't be installed (Linux/Windows), drop back to the Tesseract
    path so PDFs still ingest, just with lower OCR quality.
    """
    global _vision_unavailable_warned
    try:
        from ocrmac import ocrmac  # noqa: F401
        import pypdfium2  # noqa: F401
    except ImportError:
        if not _vision_unavailable_warned:
            logger.warning(
                "Apple Vision OCR is not available on this platform — "
                "falling back to Tesseract for PDF OCR. Install ocrmac on "
                "macOS for higher-accuracy scans."
            )
            _vision_unavailable_warned = True
        return _ocr_pdf_with_tesseract(pdf_path)
    return _ocr_pdf_with_vision(pdf_path)


def _load_pdf_with_fallback(pdf_path: Path) -> list:
    """Load a single PDF, falling back to Vision OCR if direct extraction fails.

    Strategy:
      1. PyPDFLoader — fast, exact text-layer read.
      2. If extracted text is suspiciously short (empty, garbage, or
         decoded as unsupported CJK encoding), re-process with
         _ocr_pdf_with_vision which rasterizes each page and runs Apple
         Vision Framework OCR.
    """
    try:
        fast_loader = PyPDFLoader(str(pdf_path))
        fast_docs = fast_loader.load()
    except Exception as e:
        logger.warning("PyPDFLoader failed on %s: %s — falling back to OCR", pdf_path.name, e)
        fast_docs = []

    total_chars = sum(len(d.page_content.strip()) for d in fast_docs)

    if fast_docs and total_chars >= MIN_PDF_CHARS:
        return fast_docs

    logger.warning(
        "Text extraction yielded only %d chars for %s — falling back to OCR.",
        total_chars,
        pdf_path.name,
    )
    return [_ocr_pdf(pdf_path)]


def load_documents(dir_path: str) -> list:
    """Load all documents from the given directory.

    Strategy per file type:
    - PDFs: try direct text extraction (PyPDFLoader); if that yields almost
      nothing, fall back to Apple Vision OCR via _ocr_pdf_with_vision.
    - Images (.png/.jpg/.jpeg): processed with UnstructuredFileLoader
      (Tesseract). Vision OCR is wired only for the PDF path.

    Args:
        dir_path: Path to directory containing document files.

    Returns:
        A list of LangChain Document objects.
    """
    docs = []
    dir_pathlib = Path(dir_path)

    pdf_files = [f for f in dir_pathlib.rglob("*") if f.is_file() and f.suffix.lower() == ".pdf"]
    for pdf_path in pdf_files:
        docs.extend(_load_pdf_with_fallback(pdf_path))

    image_files = [
        f for f in dir_pathlib.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    for image_path in image_files:
        loader = UnstructuredFileLoader(str(image_path), strategy="hi_res")
        docs.extend(loader.load())

    if not docs:
        logger.warning("No documents found in directory: %s", dir_path)

    return docs
