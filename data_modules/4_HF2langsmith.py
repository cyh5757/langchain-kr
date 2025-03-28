import os
from dotenv import load_dotenv
import pandas as pd
from datasets import load_dataset
from langsmith import Client
from langchain_teddynote import logging

# 1. 환경변수 및 로깅 설정
load_dotenv()
logging.langsmith("HF2langsmith")

# 2. HuggingFace 데이터셋 로드
dataset = load_dataset(
    "meto/ragas-test-dataset",
    token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)

huggingface_df = dataset["korean_v1"].to_pandas()

# 3. LangSmith 클라이언트 설정
client = Client()
dataset_name = "RAG_EVAL_DATASET"

# 4. LangSmith 데이터셋 생성 or 불러오기
def create_dataset(client, dataset_name, description=None):
    for ds in client.list_datasets():
        if ds.name == dataset_name:
            print(f"📦 기존 데이터셋 '{dataset_name}' 불러옴")
            return ds
    print(f"🆕 새로운 데이터셋 '{dataset_name}' 생성")
    return client.create_dataset(dataset_name=dataset_name, description=description)

dataset = create_dataset(client, dataset_name)

# 5. LangSmith에 예제 업로드
inputs = [{"question": q} for q in huggingface_df["question"]]
outputs = [{"answer": a} for a in huggingface_df["ground_truth"]]


client.create_examples(
    inputs=inputs,
    outputs=outputs,
    dataset_id=dataset.id,
)

print("✅ LangSmith 데이터셋 업로드 완료")
