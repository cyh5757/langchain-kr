import os
import time
import pandas as pd
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.extractor import KeyphraseExtractor
from ragas.testset.docstore import InMemoryDocumentStore

from langchain_community.document_loaders import JSONLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.schema import Document

# ------------------------------
# 1. 문서 로드 및 설정
# ------------------------------
loader = JSONLoader(
    file_path="Snack_data/snack_vector.json",
    jq_schema=".[]",            # 리스트 안의 객체들 하나씩 처리
    content_key="text",         # 이 키의 값을 page_content로 사용
    text_content=True,          # 이 값이 문자열임을 명시
    metadata_func=lambda sample, _: sample.get("metadata", {})
)

docs = loader.load()
docs = docs[:20]
for doc in docs:
    doc.metadata["filename"] = doc.metadata.get("source", "snack_vector.json")

print(f"✅ 전체 문서 수: {len(docs)}")

# ------------------------------
# 2. RAGAS 설정
# ------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
generator_llm = ChatOpenAI(model="gpt-4o")
critic_llm = ChatOpenAI(model="gpt-4o")
langchain_llm = LangchainLLMWrapper(generator_llm)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

keyphrase_extractor = KeyphraseExtractor(llm=langchain_llm)
docstore = InMemoryDocumentStore(
    splitter=splitter,
    embeddings=ragas_embeddings,
    extractor=keyphrase_extractor,
)
generator = TestsetGenerator.from_langchain(
    generator_llm, critic_llm, ragas_embeddings, docstore=docstore
)

# ------------------------------
# 3. 반복 생성 설정
# ------------------------------
distributions = {
    simple: 0.3,
    reasoning: 0.3,
    multi_context: 0.2,
    conditional: 0.2,
}

batch_size = 20
total_test_size = 5
results = []

print("🚀 테스트셋 생성 시작...")

for i in range(0, len(docs), batch_size):
    doc_batch = docs[i:i + batch_size]
    if not doc_batch:
        continue

    try:
        testset = generator.generate_with_langchain_docs(
            documents=doc_batch,
            test_size=min(batch_size, total_test_size - len(results)),
            distributions=distributions,
            with_debugging_logs=True,
            raise_exceptions=False,
        )
        test_df = testset.to_pandas()
        results.append(test_df)

        print(f"✅ Batch {i // batch_size + 1}: {len(test_df)}개 생성됨")
        time.sleep(1.5)  # ⏱️ OpenAI TPM 제한 회피용 딜레이

        # 중간 저장
        combined_df = pd.concat(results, ignore_index=True)
        combined_df.to_csv("Snack_data/ragas_synthetic_dataset.csv", index=False)

        # 최대 수량 도달하면 중단
        if len(combined_df) >= total_test_size:
            break

    except Exception as e:
        print(f"❌ 오류 발생 (Batch {i // batch_size + 1}): {e}")
        time.sleep(5)

print("🎉 전체 생성 완료 ✅")
