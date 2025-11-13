import numpy as np
from dotenv import load_dotenv
import os
from langchain_upstage import UpstageEmbeddings
from pinecone import Pinecone
from dataset import dataset

load_dotenv()
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "estsoft"

embeddings = UpstageEmbeddings(api_key=UPSTAGE_API_KEY, model="solar-embedding-1-large")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# 테스트할 파라미터들
min_score_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
top_k_candidates = [3, 5]

best_f1 = 0
best_params = {}

print("🔍 하이퍼파라미터 튜닝 시작...\n")

for min_score in min_score_candidates:
    for top_k in top_k_candidates:
        true_positives = 0
        total_retrieved = 0
        total_relevant = 0

        # 평가 루프
        for case in dataset:
            query = case["query"]
            labels = set(case["labels"])

            query_vector = embeddings.embed_query(query)
            results = index.query(
                vector=query_vector, top_k=top_k, include_metadata=True
            )

            retrieved_indices = set()
            for match in results["matches"]:
                if match["score"] >= min_score and "index" in match["metadata"]:
                    retrieved_indices.add(match["metadata"]["index"])

            tp = len(retrieved_indices & labels)
            true_positives += tp
            total_retrieved += len(retrieved_indices)
            total_relevant += len(labels)

        # 지표 계산
        precision = true_positives / total_retrieved if total_retrieved > 0 else 0
        recall = true_positives / total_relevant if total_relevant > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print(
            f"MIN_SCORE={min_score:.1f}, K={top_k} → P:{precision:.3f}, R:{recall:.3f}, F1:{f1:.3f}"
        )

        # 최고 성능 기록
        if f1 > best_f1:
            best_f1 = f1
            best_params = {
                "min_score": min_score,
                "top_k": top_k,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

print("\n" + "=" * 50)
print("🏆 최적 파라미터")
print("=" * 50)
print(f"MIN_SCORE: {best_params['min_score']}")
print(f"TOP_K: {best_params['top_k']}")
print(f"Precision: {best_params['precision']:.3f}")
print(f"Recall: {best_params['recall']:.3f}")
print(f"F1-score: {best_params['f1']:.3f}")
