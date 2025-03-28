from RAG_modules.base import RetrievalChain
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class JSONRetrievalChain(RetrievalChain):
    def __init__(self, file_path):
        self.file_path = file_path

    def load_documents(self, file_path):
        loader = JSONLoader(
            file_path=file_path,
            jq_schema=".[]",
            content_key="text",
            text_content=True,
            metadata_func=lambda sample, _: sample.get("metadata", {})
        )
        docs = loader.load()
        if not docs:
            raise ValueError("❌ 문서 로딩 결과가 비어 있습니다. JSON 파일 경로 또는 jq_schema 확인 필요.")
        return docs

    def create_text_splitter(self):
        return RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
