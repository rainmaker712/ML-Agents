import os
import tempfile
import requests
import pdfplumber
import re
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


def extract_paper_title(text: str) -> str:
    """
    PDF 텍스트에서 논문 제목을 추출합니다.
    
    Args:
        text: PDF에서 추출된 텍스트
        
    Returns:
        추출된 논문 제목
    """
    # 첫 번째 페이지의 텍스트를 가져옴 (보통 제목이 첫 페이지에 있음)
    lines = text.split('\n')
    
    # 빈 줄을 제거하고 의미있는 텍스트만 필터링
    meaningful_lines = [line.strip() for line in lines if line.strip() and len(line.strip()) > 5]
    
    if not meaningful_lines:
        return "Unknown_Paper"
    
    # 논문 제목을 찾기 위한 전략
    # 1. 첫 번째 의미있는 줄이 제목일 가능성이 높음
    # 2. 제목은 보통 첫 페이지 상단에 위치
    # 3. 제목은 보통 여러 줄에 걸쳐 있을 수 있음
    
    title_candidates = []
    for i, line in enumerate(meaningful_lines[:10]):  # 첫 10줄만 확인
        # 제목 후보 조건: 길이가 적당하고, 특정 키워드가 포함되지 않은 경우
        if (10 < len(line) < 200 and 
            not any(keyword in line.lower() for keyword in ['abstract', 'introduction', 'author', 'university', 'email', '@'])):
            title_candidates.append((i, line))
    
    # 제목 선택
    if title_candidates:
        title = title_candidates[0][1]
    else:
        title = meaningful_lines[0]
    
    # 제목을 간단한 파일명으로 변환
    # 특수문자 제거, 공백을 언더스코어로 변환, 길이 제한
    clean_title = re.sub(r'[^\w\s-]', '', title)  # 특수문자 제거
    clean_title = re.sub(r'[-\s]+', '_', clean_title)  # 공백과 하이픈을 언더스코어로
    clean_title = clean_title.strip('_')  # 앞뒤 언더스코어 제거
    
    # 길이 제한 (파일명이 너무 길어지지 않도록)
    if len(clean_title) > 30:
        clean_title = clean_title[:80].rsplit('_', 1)[0]
    
    # 빈 문자열이면 기본값 사용
    if not clean_title:
        clean_title = "Unknown_Paper"
    
    return clean_title


def save_summary_to_md(summary: str, original_filename: str, paper_title: str = None) -> str:
    """
    Saves a summary to a Markdown file.

    Args:
        summary: The summary to save.
        original_filename: The original filename of the PDF.
        paper_title: Extracted paper title for filename.

    Returns:
        The path to the saved summary file.
    """
    output_dir = Path(settings.SUMMARIES_DIR)
    output_dir.mkdir(exist_ok=True)
    
    if paper_title:
        # 논문 제목을 기반으로 파일명 생성
        filename = f"{paper_title}.md"
    else:
        # 기존 방식: 원본 파일명 사용
        file_stem = Path(original_filename).stem
        filename = f"{file_stem}.md"
    
    summary_path = output_dir / filename
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

    # 논문 제목 추출
    paper_title = extract_paper_title(text)
    print(f"📖 추출된 논문 제목: {paper_title}")

    summary = llm_client.summarize_paper_text(text)
    if not summary:
        return None

    summary_path = save_summary_to_md(summary, original_filename, paper_title)

    return {"summary": summary, "summary_path": summary_path}
