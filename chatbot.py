from dotenv import load_dotenv
import os
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
import streamlit as st
from vars import QNAS, PROMPT_TEMPLATE, INDEX_NAME, MIN_SCORE
import time


class Retriever:
    def __init__(self, texts):
        self.texts = texts
        self.tokenized_corpus = [doc.split() for doc in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def get_sparse(self, query, top_k=3):
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25 = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[
            :top_k
        ]
        return [self.texts[i] for i, _ in top_bm25]


class VectorDB:
    def __init__(self, api_key, index_name):
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)

    def get_dense(self, embeddings, query, min_score=0.5, top_k=3):
        query_vector = embeddings.embed_query(query)
        results = self.index.query(
            vector=query_vector, top_k=top_k, include_metadata=True
        )
        return [
            match["metadata"]["answer"]
            for match in results["matches"]
            if match["score"] >= min_score
        ]


class RAGChatbot:
    def __init__(
        self,
        upstage_api,
        pinecone_api,
        index_name,
        prompt_template,
        qnas,
        min_score=0.5,
    ):
        self.embeddings = UpstageEmbeddings(
            api_key=upstage_api, model="solar-embedding-1-large"
        )
        self.vector_db = VectorDB(pinecone_api, index_name)
        self.texts = [f"{q[0]}, {q[1]}" for q in qnas]
        self.retriever = Retriever(self.texts)
        self.llm = ChatUpstage(
            model="solar-pro", api_key=upstage_api, temperature=0.7, max_tokens=1000
        )
        self.prompt_template = prompt_template
        self.min_score = min_score

    def generate_response(self, user_query):
        sparse_contexts = self.retriever.get_sparse(user_query)
        dense_contexts = self.vector_db.get_dense(
            self.embeddings, user_query, self.min_score
        )
        final_contexts = list(dict.fromkeys(sparse_contexts + dense_contexts))
        context = "\n\n".join(final_contexts)
        prompt = self.prompt_template.format(context=context, question=user_query)
        response = self.llm.invoke(prompt)
        return response.content


# --- Streamlit 앱 ---
load_dotenv()
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

rag_bot = RAGChatbot(
    UPSTAGE_API_KEY,
    PINECONE_API_KEY,
    INDEX_NAME,
    PROMPT_TEMPLATE,
    QNAS,
    min_score=MIN_SCORE,
)

st.set_page_config(page_title="Perso.ai Chatbot", layout="centered", page_icon="💬")

# 타이틀 추가
st.title("💬 Perso.ai Chatbot")
st.markdown("**이스트소프트 Perso.ai**에 대해 무엇이든 물어보세요!")
st.divider()

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def stream_response(response_text):
    for char in response_text:
        yield char
        time.sleep(0.01)


user_query = st.chat_input("질문을 입력하세요")
if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})
    bot_reply = rag_bot.generate_response(user_query)
    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg is st.session_state["messages"][-1]:
            st.write_stream(stream_response(msg["content"]))
        else:
            st.markdown(msg["content"])
