from dotenv import load_dotenv
import os
from langchain_upstage import UpstageEmbeddings
from vars import QNAS, INDEX_NAME
from pinecone import Pinecone, ServerlessSpec


class EmbeddingPreparer:
    def __init__(self, qnas, embedding_model):
        self.qnas = qnas
        self.embedding_model = embedding_model

    def build_texts_and_metadata(self):
        texts, metadata = [], []
        for idx, (question, answer) in enumerate(self.qnas):
            texts.append(f"{question}, {answer}")
            metadata.append({"question": question, "answer": answer, "index": idx})
        return texts, metadata

    def create_embeddings(self, texts):
        import numpy as np

        print("임베딩 생성 중...")
        dense_doc_vectors = np.array(self.embedding_model.embed_documents(texts))
        print(f"임베딩 완료. 차원: {dense_doc_vectors.shape}")
        return dense_doc_vectors


class PineconeIndexer:
    def __init__(self, api_key, index_name, dimension):

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        if self.index_name not in self.pc.list_indexes().names():
            print("인덱스 생성 중...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print("인덱스 생성 완료!")
        self.index = self.pc.Index(self.index_name)

    def upload_vectors(self, dense_vectors, metadata, batch_size=100):
        print("Pinecone에 업로드 중...")
        for i in range(0, len(dense_vectors), batch_size):
            end_idx = min(i + batch_size, len(dense_vectors))
            vectors_to_upsert = [
                {
                    "id": str(j),
                    "values": dense_vectors[j].tolist(),
                    "metadata": metadata[j],
                }
                for j in range(i, end_idx)
            ]
            self.index.upsert(vectors=vectors_to_upsert)
            print(f"{end_idx}/{len(dense_vectors)} 업로드 완료...")
        print("모든 데이터 업로드 완료!")

    def print_stats(self):
        stats = self.index.describe_index_stats()
        print(f"총 벡터 개수: {stats['total_vector_count']}")


# 환경변수와 모델 세팅
load_dotenv()
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

EMBED_DIM = 4096

embeddings = UpstageEmbeddings(api_key=UPSTAGE_API_KEY, model="solar-embedding-1-large")

# 1. 데이터와 임베딩
prep = EmbeddingPreparer(QNAS, embeddings)
texts, metadata = prep.build_texts_and_metadata()
dense_doc_vectors = prep.create_embeddings(texts)

# 2. Pinecone 업로드
indexer = PineconeIndexer(PINECONE_API_KEY, INDEX_NAME, EMBED_DIM)
indexer.upload_vectors(dense_doc_vectors, metadata)
indexer.print_stats()
