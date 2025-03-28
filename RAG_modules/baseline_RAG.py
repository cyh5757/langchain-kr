from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_pinecone import Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from pinecone import Pinecone as PineconeClient
from langchain_pinecone import PineconeVectorStore
from operator import itemgetter
import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod

load_dotenv()

class JSONRAGRetrievalChain(ABC):
    def __init__(self, file_path: str, llm=None, **kwargs):
        self.file_path = file_path
        self.llm = llm or ChatOpenAI(model_name="gpt-4o", temperature=0)
        self.k = kwargs.get("k", 5)

        # ✅ 환경 변수 로드
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "snack-db")
        self.namespace = os.getenv("PINECONE_NAMESPACE", "snack-rag-namespace")
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

        if not all([self.api_key, self.index_name, self.environment]):
            raise ValueError("❌ Pinecone 관련 환경변수(PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_ENVIRONMENT)가 모두 필요합니다.")

        self.embedding_model = "text-embedding-3-large"
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)

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

    def split_documents(self, docs, text_splitter):
        return text_splitter.split_documents(docs)

    def create_vectorstore(self, split_docs):
        pc = PineconeClient(api_key=self.api_key, environment=self.environment)
        index = pc.Index(self.index_name)

        vectorstore = PineconeVectorStore(
            index=index,
            embedding=self.embeddings,
            namespace=self.namespace,
            text_key="context",
        )
        return vectorstore

    def create_retriever(self, vectorstore):
        retriever = vectorstore.as_retriever()
        retriever.search_kwargs["k"] = self.k
        return retriever

    def create_model(self):
        return self.llm

    def create_prompt(self):
        return PromptTemplate.from_template(
            """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 

#Context: 
{context}

#Question:
{question}

#Answer:"""
        )

    def create_chain(self):
        docs = self.load_documents(self.file_path)
        text_splitter = self.create_text_splitter()
        split_docs = self.split_documents(docs, text_splitter)
        self.vectorstore = self.create_vectorstore(split_docs)
        self.retriever = self.create_retriever(self.vectorstore)
        model = self.create_model()
        prompt = self.create_prompt()

        self.chain = (
            {
                "question": itemgetter("question"),
                "context": itemgetter("context"),
            }
            | prompt
            | model
            | StrOutputParser()
        )
        return self

    def invoke(self, question: str):
        context_docs = self.retriever.invoke(question)
        context_text = "\n".join([doc.page_content for doc in context_docs])
        return self.chain.invoke({"question": question, "context": context_text})
