# extract.py

"""
Document extraction module: extracts raw text from supported formats (PDF, DOCX, TXT).
Includes OCR fallback for scanned PDFs. Logs detailed events and metadata.
"""

import os
import fitz  # PyMuPDF
import docx
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from typing import Tuple, Dict
from datetime import datetime

def extract_text(file_path: str, config: Dict) -> Tuple[str, Dict]:
    """
    Extracts raw text from a document based on its type.

    Args:
        file_path (str): Full path to the document.
        config (Dict): Parsed configuration settings.

    Returns:
        Tuple[str, Dict]: Extracted text and basic metadata.
    """
    ext = os.path.splitext(file_path)[1].lower().replace('.', '')
    text = ""
    meta = {
        "source_file": os.path.basename(file_path),
        "extension": ext,
        "extracted_at": datetime.now().isoformat(),
        "pages": 0,
    }

    if ext == 'pdf':
        text, pages = extract_pdf(file_path, config)
        meta["pages"] = pages
    elif ext == 'docx':
        text = extract_docx(file_path)
    elif ext == 'txt':
        text = extract_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return text, meta


def extract_pdf(file_path: str, config: Dict) -> Tuple[str, int]:
    """Extract text from PDF using PyMuPDF with OCR fallback."""
    doc = fitz.open(file_path)
    text = ""
    ocr_enabled = config.get("enable_ocr", False)
    lang = "+".join(config.get("ocr_languages", ["eng"]))

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        if not page_text.strip() and ocr_enabled:
            # OCR fallback for empty pages
            images = convert_from_path(file_path, first_page=page_num + 1, last_page=page_num + 1)
            ocr_text = ""
            for image in images:
                ocr_text += pytesseract.image_to_string(image, lang=lang)
            page_text = ocr_text
        text += f"\n--- Page {page_num + 1} ---\n" + page_text

    return text, len(doc)


def extract_docx(file_path: str) -> str:
    """Extract text from DOCX using python-docx."""
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text


def extract_txt(file_path: str) -> str:
    """Read plain text file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()
