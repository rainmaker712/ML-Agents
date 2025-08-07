import sys
from pathlib import Path
import os

from app.core.services import download_pdf, summarize_paper


def main():
    if len(sys.argv) < 2:
        print("사용법: python summarize_and_post.py <PDF 파일 경로 또는 URL>")
        sys.exit(1)

    input_path = sys.argv[1]

    if input_path.startswith("http"):
        print("PDF 다운로드 중...")
        pdf_path = download_pdf(input_path)
        if not pdf_path:
            print("PDF 다운로드에 실패했습니다.")
            sys.exit(1)
        original_filename = input_path
    else:
        pdf_path = input_path
        original_filename = os.path.basename(pdf_path)

    print("논문 요약 중...")
    result = summarize_paper(pdf_path, original_filename)

    if result:
        print("\n✅ 처리 완료!")
        print(f"📄 요약 파일이 '{result['summary_path']}'에 저장되었습니다.")
        print("\n📋 요약:")
        print(result["summary"])
    else:
        print("\n❌ 요약 생성에 실패했습니다.")

    if input_path.startswith("http") and os.path.exists(pdf_path):
        os.unlink(pdf_path)


if __name__ == "__main__":
    main()
