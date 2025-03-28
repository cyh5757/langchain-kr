import os
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_teddynote import logging
from langchain_teddynote.community.pinecone import (
    create_index,
    upsert_documents_parallel_dense_only,
)

# ✅ 사용자 정의 preprocess_documents 함수 수정본 (None → 빈 문자열)
def preprocess_documents(split_docs, metadata_keys=["source", "page"], min_length=2, use_basename=False):
    contents = []
    metadatas = {key: [] for key in metadata_keys}
    for doc in split_docs:
        content = doc.page_content.strip()
        if content and len(content) >= min_length:
            contents.append(content)
            for k in metadata_keys:
                value = doc.metadata.get(k, "")
                if value is None:
                    value = ""
                if k == "source" and use_basename:
                    value = os.path.basename(value)
                try:
                    metadatas[k].append(int(value))
                except (ValueError, TypeError):
                    metadatas[k].append(str(value))
    return contents, metadatas

# 1. 환경 변수 로드
load_dotenv()
logging.langsmith("Json2pinecone")

# 2. JSON 문서 로딩
json_file = "Snack_data/snack_vector.json"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

def metadata_fn(sample, default_metadata):
    return sample.get("metadata", {})

loader = JSONLoader(
    file_path=json_file,
    jq_schema=".[]",
    content_key="text",
    text_content=False,
    metadata_func=metadata_fn
)
split_docs = loader.load_and_split(text_splitter)

# 3. 전처리
contents, metadatas = preprocess_documents(
    split_docs=split_docs,
    metadata_keys=["snack_name", "category", "company"],
    min_length=5,
    use_basename=False,
)

# 4. Pinecone 인덱스 생성
pc_index = create_index(
    api_key=os.environ["PINECONE_API_KEY"],
    index_name=os.getenv("PINECONE_INDEX_NAME", "snack-db"),
    dimension=3072,
    metric="dotproduct",
)

# 5. Embedding
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 6. 업서트 실행
upsert_documents_parallel_dense_only(
    index=pc_index,
    namespace="snack-rag-namespace",
    contents=contents,
    metadatas=metadatas,
    embedder=openai_embeddings,
    batch_size=8, #TPM에 맞춰서 알아서
    max_workers=30,
)

print("✅ Pinecone 하이브리드 업로드 완료")
