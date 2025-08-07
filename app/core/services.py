import os
import tempfile
import requests
import pdfplumber
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app import get_llm_client

llm_client = get_llm_client()

def download_pdf(url: str, temp_dir: Optional[str] = None) -> Optional[str]:
    """
    Downloads a PDF from a URL and saves it to a temporary file.

    Args:
        url: The URL of the PDF to download.
        temp_dir: The directory to save the temporary file in.

    Returns:
        The path to the temporary file, or None if an error occurred.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf", dir=temp_dir
        ) as tmp:
            tmp.write(response.content)
            return tmp.name
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF: {e}")
        return None


def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extracts text from a PDF file.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        The extracted text, or None if an error occurred.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return "".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None


def save_summary_to_md(summary: str, original_filename: str) -> str:
    """
    Saves a summary to a Markdown file.

    Args:
        summary: The summary to save.
        original_filename: The original filename of the PDF.

    Returns:
        The path to the saved summary file.
    """
    output_dir = Path(settings.SUMMARIES_DIR)
    output_dir.mkdir(exist_ok=True)
    file_stem = Path(original_filename).stem
    summary_path = output_dir / f"{file_stem}.md"
    summary_path.write_text(summary, encoding="utf-8")
    return str(summary_path)


def summarize_paper(
    pdf_path: str, original_filename: Optional[str] = None
) -> Optional[dict]:
    """
    Summarizes a paper from a PDF file.

    Args:
        pdf_path: The path to the PDF file.
        original_filename: The original filename of the PDF.

    Returns:
        A dictionary containing the summary and the path to the summary file,
        or None if an error occurred.
    """
    if not original_filename:
        original_filename = pdf_path

    text = extract_text_from_pdf(pdf_path)
    if not text:
        return None

    summary = llm_client.summarize_paper_text(text)
    if not summary:
        return None

    summary_path = save_summary_to_md(summary, original_filename)

    return {"summary": summary, "summary_path": summary_path}
