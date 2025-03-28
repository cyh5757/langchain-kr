# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
from langchain_teddynote import logging
import pandas as pd
from datasets import load_dataset, Dataset
from langsmith import Client
import os



load_dotenv()
logging.langsmith("HF2langsmith")


# huggingface Dataset에서 repo_id로 데이터셋 다운로드
dataset = load_dataset(
    "meto/ragas-test-dataset",  # 데이터셋 이름
    token=os.environ["HUGGINGFACEHUB_API_TOKEN"],  # private 데이터인 경우 필요합니다.
)

# 데이터셋에서 split 기준으로 조회
huggingface_df = dataset["korean_v1"].to_pandas()

client = Client()
dataset_name = "RAG_EVAL_DATASET"



def create_dataset(client, dataset_name, description=None):
    for dataset in client.list_datasets():
        if dataset.name == dataset_name:
            return dataset

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=description,
    )
    return dataset


dataset = create_dataset(client, dataset_name)

