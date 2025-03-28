import os
import json
import uuid
import concurrent.futures
from typing import List, Any

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_teddynote import logging

# LangSmith 추적 설정
logging.langsmith("Json2pinecone")

# 1. 환경변수 로드
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "snack-db")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
NAMESPACE = "snack-rag-namespace"

# 환경 변수 확인
if not PINECONE_API_KEY:
    raise ValueError("❌ 환경변수 'PINECONE_API_KEY'가 설정되지 않았습니다.")

# 2. JSON 데이터를 LangChain Document 객체로 변환
def load_documents(filepath="Snack_data/snack_data.json") -> List[Document]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = [
        Document(
            page_content=doc["page_content"],
            metadata=doc["metadata"]
        )
        for doc in data
    ]
    return documents

# 3. Pinecone 인덱스 가져오기 또는 생성 (최초 1회만 삭제)
def get_index(delete_first=False):
    pc = Pinecone(api_key=PINECONE_API_KEY)


    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"📌 인덱스 '{PINECONE_INDEX_NAME}' 생성 중...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=3072,  # text-embedding-3-large 모델의 출력 차원
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
        )
        print(f"✅ 인덱스 생성 완료: {PINECONE_INDEX_NAME}")

    return pc.Index(PINECONE_INDEX_NAME)

# 4. 문서 벡터 임베딩 및 Pinecone 업로드
def process_batch(batch: List[Document], embeddings: OpenAIEmbeddings, index: Any):
    texts = [doc.page_content for doc in batch]
    metadatas = [doc.metadata for doc in batch]
    vectors = embeddings.embed_documents(texts)

    index.upsert(
        vectors=[
            (str(uuid.uuid4()), vector, metadata)
            for vector, metadata in zip(vectors, metadatas)
        ],
        namespace=NAMESPACE
    )

# 5. 메인 실행 로직
if __name__ == "__main__":
    print("🚀 과자 정보 벡터 저장 프로세스 시작...")

    try:
        print("📄 문서 로드 중...")
        documents = load_documents()

        print("📌 Pinecone 인덱스 준비 중...")
        index = get_index(delete_first=False)  # 필요시 True로 변경

        print("🔗 OpenAI 임베딩 모델 초기화 중...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # 3072차원 출력

        BATCH_SIZE = 64
        batches = [documents[i:i + BATCH_SIZE] for i in range(0, len(documents), BATCH_SIZE)]

        print("🚀 벡터 업로드 중...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(process_batch, batch, embeddings, index)
                for batch in batches
            ]
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                pass

        print("✅ 전체 벡터 저장 완료: Pinecone 업로드 성공")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
