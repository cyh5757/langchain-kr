import os
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
import time
from tqdm import tqdm

# 사용자 정의 전처리 함수
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

# Pinecone 인덱스 생성 함수
def create_index(api_key, index_name, dimension, metric="dotproduct"):
    pc = Pinecone(api_key=api_key)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    return pc.Index(index_name)

# 문서 업서트 함수 (dense only)
def upsert_documents_parallel_dense_only(index, namespace, contents, metadatas, embedder, batch_size=100):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import itertools

    keys = list(metadatas.keys())

    def chunks(iterable, size):
        it = iter(iterable)
        chunk = list(itertools.islice(it, size))
        while chunk:
            yield chunk
            chunk = list(itertools.islice(it, size))

    def process_batch(batch):
        context_batch = [contents[i] for i in batch]
        metadata_batches = {key: [metadatas[key][i] for i in batch] for key in keys}

        batch_result = [
            {
                "context": context[:1000],
                **{key: metadata_batches[key][j] for key in keys},
            }
            for j, context in enumerate(context_batch)
        ]

        import secrets
        def generate_hash():
            return secrets.token_hex(12)

        ids = [generate_hash() for _ in range(len(batch))]
        dense_embeds = embedder.embed_documents(context_batch)

        vectors = [
            {
                "id": _id,
                "values": dense,
                "metadata": metadata,
            }
            for _id, dense, metadata in zip(ids, dense_embeds, batch_result)
        ]

        return index.upsert(vectors=vectors, namespace=namespace, async_req=False)

    batches = list(chunks(range(len(contents)), batch_size))

    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Pinecone 업서트 중"):
            future.result()

# 실행 시작
load_dotenv()

# 1. JSON 문서 로드
json_file = "Snack_data/snack_vector.json"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
loader = JSONLoader(
    file_path=json_file,
    jq_schema=".[]",
    content_key="text",
    text_content=False,
    metadata_func=lambda sample, _: sample.get("metadata", {})
)
split_docs = loader.load_and_split(text_splitter)

# 2. 전처리
contents, metadatas = preprocess_documents(
    split_docs=split_docs,
    metadata_keys=["snack_name", "category", "company"],
    min_length=5,
    use_basename=False
)

# 3. Pinecone 인덱스 생성
pc_index = create_index(
    api_key=os.environ["PINECONE_API_KEY"],
    index_name=os.getenv("PINECONE_INDEX_NAME", "snack-db"),
    dimension=3072,
    metric="dotproduct",
)

# 4. 임베딩 생성
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 5. 업서트
upsert_documents_parallel_dense_only(
    index=pc_index,
    namespace="snack-rag-namespace",
    contents=contents,
    metadatas=metadatas,
    embedder=openai_embeddings,
    batch_size=8,
)

print("✅ Pinecone 업로드 완료")