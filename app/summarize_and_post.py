import sys
from pathlib import Path
import os

from app.core.services import download_pdf, summarize_paper


def main():
    if len(sys.argv) < 2:
        print("사용법: python -m app.summarize_and_post <PDF 파일 경로 또는 URL>")
        sys.exit(1)

    input_path = sys.argv[1]

    if input_path.startswith("http"):
        print("📥 PDF 다운로드 중...")
        pdf_path = download_pdf(input_path)
        if not pdf_path:
            print("❌ PDF 다운로드에 실패했습니다.")
            sys.exit(1)
        original_filename = input_path
    else:
        pdf_path = input_path
        original_filename = os.path.basename(pdf_path)

    print(f"📄 처리할 PDF: {pdf_path}")
    print("🤖 논문 요약 중...")
    result = summarize_paper(pdf_path, original_filename)

    if result:
        print("\n✅ 처리 완료!")
        print(f"📁 요약 파일이 저장된 경로: {result['summary_path']}")
        print(f"📝 파일명: {os.path.basename(result['summary_path'])}")
        print("\n📋 요약 내용:")
        print("=" * 50)
        print(result["summary"])
        print("=" * 50)
    else:
        print("\n❌ 요약 생성에 실패했습니다.")

    if input_path.startswith("http") and os.path.exists(pdf_path):
        os.unlink(pdf_path)


if __name__ == "__main__":
    main()
