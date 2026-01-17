import streamlit as st
import requests
import os
from typing import List, Dict, Optional, Any
from pathlib import Path

st.set_page_config(
    page_title="WhoLM - AI-Powered Video & Document Q&A",
    layout="wide",
    initial_sidebar_state="expanded"
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

class WhoLMClient:
    def __init__(self, base_url: str = BACKEND_URL):
        self.base_url = base_url

    def get_upload_url(self, filename: str, content_type: str) -> Dict[str, Any]:
        data = {"filename": filename, "content_type": content_type}
        response = requests.post(f"{self.base_url}/upload/get-url", json=data)
        return response.json()

    def process_uploaded_file(self, content_id: str, name: str) -> Dict[str, Any]:
        data = {"content_id": content_id, "name": name}
        response = requests.post(f"{self.base_url}/upload/process", json=data)
        return response.json()

    def ask_question(self, question: str, session_id: str = None) -> Dict[str, Any]:
        data = {"question": question}
        if session_id:
            data["session_id"] = session_id
        response = requests.post(f"{self.base_url}/chat", json=data)
        return response.json()

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        response = requests.get(f"{self.base_url}/chat/history/{session_id}")
        return response.json()

    def get_uploaded_content(self) -> List[Dict[str, Any]]:
        response = requests.get(f"{self.base_url}/content")
        return response.json()

    def delete_content(self, content_id: str) -> Dict[str, Any]:
        response = requests.delete(f"{self.base_url}/content/{content_id}")
        return response.json()

    def upload_youtube_video(self, youtube_url: str) -> Dict[str, Any]:
        data = {"youtube_url": youtube_url}
        response = requests.post(f"{self.base_url}/upload/youtube", json=data)
        return response.json()

def main():
    client = WhoLMClient()

    st.title("WhoLM - AI-Powered Video & Document Q&A")
    st.markdown("""
    Upload videos or documents to cloud storage and ask questions about their content using advanced AI.
    """)

    with st.sidebar:
        st.header("Upload Content")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["mp4", "avi", "mov", "pdf", "txt", "docx"]
        )

        if uploaded_file is not None:
            content_type = uploaded_file.type

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Upload as Video", use_container_width=True):
                    with st.spinner("Uploading..."):
                        try:
                            url_response = client.get_upload_url(uploaded_file.name, content_type)
                            if not url_response.get("upload_url"):
                                st.error("Failed to get upload URL")
                            else:
                                upload_response = requests.put(
                                    url_response["upload_url"],
                                    data=uploaded_file.getvalue(),
                                    headers={"Content-Type": content_type}
                                )

                                if upload_response.status_code == 200:
                                    with st.spinner("Processing video..."):
                                        process_response = client.process_uploaded_file(
                                            url_response["content_id"],
                                            uploaded_file.name
                                        )
                                        if process_response.get("success"):
                                            st.success("Video uploaded and processing started!")
                                            st.rerun()
                                        else:
                                            st.error(f"Processing failed: {process_response.get('error', 'Unknown error')}")
                                else:
                                    st.error("Failed to upload file to cloud storage")

                        except Exception as e:
                            st.error(f"Failed to upload video: {str(e)}")

            with col2:
                if st.button("Upload as Document", use_container_width=True):
                    with st.spinner("Uploading..."):
                        try:
                            url_response = client.get_upload_url(uploaded_file.name, content_type)
                            if not url_response.get("upload_url"):
                                st.error("Failed to get upload URL")
                            else:
                                upload_response = requests.put(
                                    url_response["upload_url"],
                                    data=uploaded_file.getvalue(),
                                    headers={"Content-Type": content_type}
                                )

                                if upload_response.status_code == 200:
                                    with st.spinner("Processing document..."):
                                        process_response = client.process_uploaded_file(
                                            url_response["content_id"],
                                            uploaded_file.name
                                        )
                                        if process_response.get("success"):
                                            st.success("Document uploaded and processing started!")
                                            st.rerun()
                                        else:
                                            st.error(f"Processing failed: {process_response.get('error', 'Unknown error')}")
                                else:
                                    st.error("Failed to upload file to cloud storage")

                        except Exception as e:
                            st.error(f"Failed to upload document: {str(e)}")

        st.divider()

        st.header("Settings")
        backend_url = st.text_input("Backend URL", value=BACKEND_URL)
        if backend_url != BACKEND_URL:
            st.session_state.client = WhoLMClient(backend_url)

        st.header("Chat Sessions")
        if st.button("New Chat Session"):
            st.session_state.session_id = None
            st.session_state.messages = []
            st.rerun()

    st.header("Ask Questions")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("View Sources"):
                        for source in message["sources"]:
                            st.write(f"**{source.get('type', 'Source')}:** {source.get('content', '')[:200]}...")

    if prompt := st.chat_input("Ask a question about your uploaded content..."):
        if not st.session_state.messages:
            st.session_state.messages = []

        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.write(prompt)

        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = client.ask_question(prompt, st.session_state.session_id)

                        if response.get("success"):
                            answer = response.get("answer", "No answer provided")
                            sources = response.get("sources", [])
                            session_id = response.get("session_id")

                            if session_id and not st.session_state.session_id:
                                st.session_state.session_id = session_id

                            st.write(answer)

                            if sources:
                                with st.expander("View Sources"):
                                    for source in sources:
                                        source_type = source.get("type", "Unknown")
                                        content = source.get("content", "")
                                        metadata = source.get("metadata", {})

                                        st.subheader(f"{source_type.title()}")
                                        st.write(content[:500] + "..." if len(content) > 500 else content)

                                        if metadata:
                                            st.json(metadata)

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": sources
                            })

                        else:
                            error_msg = response.get("error", "Unknown error occurred")
                            st.error(f"Error: {error_msg}")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"Error: {error_msg}"
                            })

                    except Exception as e:
                        st.error(f"Failed to get response: {str(e)}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error: {str(e)}"
                        })

    st.divider()
    st.header("Uploaded Content")

    try:
        content_list = client.get_uploaded_content()
        if content_list:
            cols = st.columns(3)
            for i, content in enumerate(content_list):
                with cols[i % 3]:
                    with st.container():
                        content_type = content.get("type", "Unknown")
                        name = content.get("name", "Unnamed")
                        upload_time = content.get("upload_time", "Unknown")
                        status = content.get("status", "Unknown")

                        st.markdown(f"**{name}**")
                        st.caption(f"Status: {status}")
                        st.caption(f"Type: {content_type}")
                        st.caption(f"Uploaded: {upload_time}")

                        if st.button("Delete", key=content.get("id"), use_container_width=True):
                            try:
                                delete_response = client.delete_content(content.get("id"))
                                if delete_response.get("success"):
                                    st.success("Content deleted!")
                                    st.rerun()
                                else:
                                    st.error("Failed to delete content")
                            except Exception as e:
                                st.error(f"Error deleting content: {str(e)}")
        else:
            st.info("No content uploaded yet.")

    except Exception as e:
        st.warning(f"Could not load content list: {str(e)}")

    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
        WhoLM - AI-Powered Video and Document Analysis
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()