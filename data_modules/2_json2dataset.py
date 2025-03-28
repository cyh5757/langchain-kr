import time
import openai
import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential

from langchain_community.document_loaders import JSONLoader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.extractor import KeyphraseExtractor
from ragas.testset.docstore import InMemoryDocumentStore

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import LLMResult, Generation

import os
import asyncio

loader = JSONLoader(
    file_path="C:/Users/MAIN/Workplace/LLM/langchain-kr/langchain-kr/Snack_data/snack_data.json",
    jq_schema=".[].page_content",
    text_content=True,
)

docs = loader.load()
for doc in docs:
    doc.metadata["filename"] = doc.metadata.get("source", "snack_data.json")



# 데이터셋 생성기
generator_llm = ChatOpenAI(model="gpt-4o-mini")
# 데이터셋 비평기
critic_llm = ChatOpenAI(model="gpt-4o-mini")
# 문서 임베딩
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 텍스트 분할기를 설정합니다.
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# LangChain의 ChatOpenAI 모델을 LangchainLLMWrapper로 감싸 Ragas와 호환되게 만듭니다.
langchain_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

# 주요 구문 추출기를 초기화합니다. 위에서 정의한 LLM을 사용합니다.
keyphrase_extractor = KeyphraseExtractor(llm=langchain_llm)

# ragas_embeddings 생성
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

# InMemoryDocumentStore를 초기화합니다.
# 이는 문서를 메모리에 저장하고 관리하는 저장소입니다.
docstore = InMemoryDocumentStore(
    splitter=splitter,
    embeddings=ragas_embeddings,
    extractor=keyphrase_extractor,
)

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    ragas_embeddings,
    docstore=docstore,
)

# 질문 유형별 분포 결정
# simple: 간단한 질문, reasoning: 추론이 필요한 질문, multi_context: 여러 맥락을 고려해야 하는 질문, conditional: 조건부 질문
distributions = {simple: 0.4, reasoning: 0.2, multi_context: 0.2, conditional: 0.2}

# 테스트셋 생성
# docs: 문서 데이터, 10: 생성할 질문의 수, distributions: 질문 유형별 분포, with_debugging_logs: 디버깅 로그 출력 여부
testset = generator.generate_with_langchain_docs(
    documents=docs[:5],
    test_size=10,
    distributions=distributions,
    with_debugging_logs=True,
    raise_exceptions=False,
)

# 생성된 테스트셋을 pandas DataFrame으로 변환
test_df = testset.to_pandas()

# -------------------------
# 6. 결과 저장
# -------------------------
test_df = testset.to_pandas()
test_df.to_csv("C:/Users/MAIN/Workplace/LLM/langchain-kr/langchain-kr/Snack_data/ragas_synthetic_dataset.csv", index=False)
print("✅ 테스트셋 저장 완료: ragas_synthetic_dataset.csv")