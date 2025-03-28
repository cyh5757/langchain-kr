from dotenv import load_dotenv
from langchain_teddynote import logging
import pandas as pd
import os
from langchain_teddynote.translate import Translator
from datasets import Dataset

# 프로젝트 이름을 입력합니다.
logging.langsmith("dataset2HF")
load_dotenv()


df = pd.read_csv("Snack_data/ragas_synthetic_dataset.csv")
df.head()

# api키 설정
deepl_api_key = os.getenv("DEEPL_API_KEY")

# 객체 생성
translator = Translator(deepl_api_key, "EN", "KO")

# 번역 실행
translated_text = translator("hello, nice to meet you")


from tqdm import tqdm

# 번역
for i, row in tqdm(df.iterrows(), total=len(df), desc="번역 진행 중"):
    df.loc[i, "question_translated"] = translator(row["question"])
    df.loc[i, "ground_truth_translated"] = translator(row["ground_truth"])

# question, ground_truth 열을 삭제하고 question_translated, ground_truth_translated 열의 이름을 변경합니다.
df.drop(columns=["question", "ground_truth"], inplace=True)
df.rename(
    columns={
        "question_translated": "question",
        "ground_truth_translated": "ground_truth",
    },
    inplace=True,
)


# 번역한 데이터셋을 저장합니다.
df.to_csv("Snack_data/ragas_synthetic_dataset_translated.csv", index=False)

dataset = Dataset.from_pandas(df)



# pandas DataFrame을 Hugging Face Dataset으로 변환
dataset = Dataset.from_pandas(df)

# 데이터셋 이름 설정 (원하는 이름으로 변경하세요)
dataset_name = "meto/ragas-test-dataset"

# 데이터셋 업로드
dataset.push_to_hub(
    dataset_name,
    private=True,  # private=False로 설정하면 공개 데이터셋이 됩니다.
    split="korean_v1",  # 데이터셋 split 이름 입력
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)