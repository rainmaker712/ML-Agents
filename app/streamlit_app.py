import streamlit as st
import os
import tempfile
from pathlib import Path

from app.core.services import download_pdf, summarize_paper, extract_text_from_pdf
from app import get_llm_client
from app.core.config import settings

# 페이지 설정
st.set_page_config(
    page_title="논문 요약 플랫폼",
    page_icon="📚",
    layout="wide",
)

# LLM 클라이언트 초기화
try:
    llm_client = get_llm_client()
except Exception as e:
    st.error(f"LLM 클라이언트 초기화 실패: {e}")
    llm_client = None


def main():
    st.title("📚 논문 요약 플랫폼")
    st.markdown("PDF 논문을 업로드하거나 URL을 입력하여 자동으로 요약을 생성합니다.")

    # 사이드바 설정
    st.sidebar.header("설정")

    # LLM 상태 표시
    if llm_client:
        provider_info = llm_client.get_provider_info()
        st.sidebar.success(f"✅ LLM 연결됨: {provider_info.get('provider', 'N/A')}")
        st.sidebar.info(f"모델: {provider_info.get('model', 'N/A')}")
    else:
        st.sidebar.error("❌ LLM 연결 실패")

    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["📁 파일 업로드", "🔗 URL 입력", "⚙️ 설정"])

    with tab1:
        st.header("PDF 파일 업로드")
        uploaded_file = st.file_uploader("PDF 파일을 선택하세요", type=["pdf"])

        if uploaded_file is not None:
            if st.button("요약 생성", key="upload_button", type="primary"):
                with st.spinner("PDF를 처리하고 있습니다..."):
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    try:
                        result = summarize_paper(tmp_path, uploaded_file.name)
                        if result:
                            st.success("요약 생성 완료!")
                            st.info(
                                f"요약 파일이 `{result['summary_path']}`에 저장되었습니다."
                            )

                            st.subheader("📋 생성된 요약")
                            st.markdown(result["summary"])

                            st.download_button(
                                label="📥 요약 다운로드 (Markdown)",
                                data=result["summary"],
                                file_name=f"{Path(uploaded_file.name).stem}_summary.md",
                                mime="text/markdown",
                            )
                        else:
                            st.error("요약 생성에 실패했습니다.")
                    finally:
                        os.unlink(tmp_path)

    with tab2:
        st.header("PDF URL 입력")
        url = st.text_input("PDF URL을 입력하세요")

        if st.button("요약 생성", key="url_button", type="primary"):
            if url:
                with st.spinner("PDF를 다운로드하고 처리하고 있습니다..."):
                    pdf_path = download_pdf(url)
                    if pdf_path:
                        try:
                            result = summarize_paper(pdf_path, url)
                            if result:
                                st.success("요약 생성 완료!")
                                st.info(
                                    f"요약 파일이 `{result['summary_path']}`에 저장되었습니다."
                                )

                                st.subheader("📋 생성된 요약")
                                st.markdown(result["summary"])

                                st.download_button(
                                    label="📥 요약 다운로드 (Markdown)",
                                    data=result["summary"],
                                    file_name=f"{Path(url).stem}_summary.md",
                                    mime="text/markdown",
                                    key="url_download",
                                )
                            else:
                                st.error("요약 생성에 실패했습니다.")
                        finally:
                            os.unlink(pdf_path)
            else:
                st.error("URL을 입력해주세요.")

    with tab3:
        st.header("설정")

        st.subheader("🔑 API 키 상태")
        if settings.OPENAI_API_KEY or settings.LOCAL_LLM_BASE_URL:
            st.success("✅ LLM API 키 설정됨")
        else:
            st.error("❌ LLM API 키 미설정")
            st.info(
                "환경변수 OPENAI_API_KEY 또는 LOCAL_LLM_* 설정을 확인해주세요."
            )

        if llm_client:
            st.subheader("🤖 LLM 제공자 정보")
            provider_info = llm_client.get_provider_info()
            st.json(provider_info)


if __name__ == "__main__":
    main()
