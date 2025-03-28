from baseline_RAG import JSONRAGRetrievalChain
from langchain_openai import ChatOpenAI
from langsmith.evaluation import evaluate, LangChainStringEvaluator
import time
import openai

# ✅ JSONRAG 객체에만 LLM 지정
rag = JSONRAGRetrievalChain(
    "Snack_data/snack_vector.json",
    ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
).create_chain()

retriever = rag.create_retriever()
chain = rag.create_chain(retriever)

# ✅ RAG 평가 함수 (context + answer 반환)
def context_answer_rag_answer(inputs: dict):
    context = retriever.invoke(inputs["question"])
    return {
        "context": "\n".join([doc.page_content for doc in context]),
        "answer": chain.invoke(inputs["question"]),
        "query": inputs["question"],
    }

# ✅ 평가자에서도 명시적으로 LLM 지정 (중요!)
evaluator_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

cot_qa_evaluator = LangChainStringEvaluator(
    "cot_qa",
    config={"llm": evaluator_llm},
    prepare_data=lambda run, example: {
        "prediction": run.outputs["answer"],
        "reference": run.outputs["context"],
        "input": example.inputs["question"],
    },
)

context_qa_evaluator = LangChainStringEvaluator(
    "context_qa",
    config={"llm": evaluator_llm},
    prepare_data=lambda run, example: {
        "prediction": run.outputs["answer"],
        "reference": run.outputs["context"],
        "input": example.inputs["question"],
    },
)

# ✅ RateLimitError 자동 재시도 evaluate 함수
def safe_evaluate(*args, max_retries=5, wait_time=10, **kwargs):
    for i in range(max_retries):
        try:
            return evaluate(*args, **kwargs)
        except openai.RateLimitError as e:
            print(f"⏳ Rate limit hit. Sleeping {wait_time}s... (retry {i+1}/{max_retries})")
            time.sleep(wait_time)
        except Exception as e:
            print(f"❌ 예외 발생: {e}")
            raise e
    raise RuntimeError("Too many retries due to rate limits.")

# ✅ 평가 실행
dataset_name = "RAG_EVAL_DATASET"

safe_evaluate(
    context_answer_rag_answer,
    data=dataset_name,
    evaluators=[cot_qa_evaluator, context_qa_evaluator],
    experiment_prefix="RAG_EVAL",
    metadata={
        "variant": "모든 평가자에 gpt-3.5-turbo 명시적 지정",
    },
)
