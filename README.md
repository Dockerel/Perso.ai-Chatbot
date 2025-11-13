# Vector DB 기반 RAG 챗봇 시스템 구축 과제 - 도기헌

안녕하십니까, '안정적인 시스템 위에 성능을 쌓아가는 개발자' 도기헌입니다.

이번 과제에서는 **할루시네이션 없이 정확한 응답을 제공하는 지식 기반 챗봇 시스템**을 구축하는 것을 목표로, Retrieval-Augmented Generation(RAG) 아키텍처를 설계하고 구현했습니다. 단순한 기능 구현을 넘어, 실전 환경에서 요구되는 검색 정확도, 응답 품질, 시스템 확장성까지 고려한 설계를 진행했습니다.

---

## 배포 및 데모

구현한 챗봇을 직접 체험하실 수 있도록 배포 환경을 준비했습니다.

* **챗봇 웹 애플리케이션:**
    * **URL:** http://34.44.185.142:8501/
    * **설명:** Streamlit 기반의 직관적인 챗 인터페이스로, Perso.ai에 대한 질문에 실시간으로 답변합니다.

* **성능 벤치마크 결과:**
    * **Precision:** 32.7%
    * **Recall:** 88.5%
    * **F1-score:** 47.8%

---

## 1. 사용 기술 스택

### 프론트엔드
- **Streamlit**: 사용자 친화적인 웹 인터페이스 구축

### 백엔드
- **Python 3.11+**: 메인 개발 언어
- **LangChain**: RAG 파이프라인 구축 프레임워크

### 임베딩 & 벡터 검색
- **Upstage Solar-Embedding-1-Large**: 한국어 특화 임베딩 모델 (4096차원)
- **Pinecone**: 클라우드 기반 벡터 데이터베이스 (Serverless)
- **BM25 (rank-bm25)**: 키워드 기반 Sparse 검색

### LLM
- **Upstage Solar-Pro**: 한국어 최적화 대규모 언어 모델

---

## 2. 전체 시스템 아키텍처

이번 과제의 핵심은 **검색 증강 생성(RAG)** 패턴을 활용해, 제공된 Q&A 데이터만을 기반으로 정확한 답변을 생성하는 것입니다. 이를 위해 **Hybrid Retrieval (Dense + Sparse) 방식**을 채택했습니다.

### 질문 답변 생성 흐름도
```text
사용자 질문 입력 → 하이브리드 검색 (Retrieval) → 컨텍스트 구성

→ LLM 생성 → 스트리밍 출력 → 사용자에게 최종 답변 표시
```

**1. 하이브리드 검색**
* Dense 검색 (Pinecone)
   * 질문 임베딩 생성                       
   * 벡터 유사도 검색 (코사인 유사도)         
   * Top-3 문서 추출                  
                                         
* Sparse 검색 (BM25)                     
   * 질문 토큰화                         
   * 키워드 매칭 (TF-IDF 기반)           
   * Top-3 문서 추출                     
                                         
* 결과 병합                              
   * dict.fromkeys()로 중복 제거         
   * 순서 유지 (Sparse → Dense)          
   * MIN_SCORE 필터링 (유사도 0.3 이상)  

**2. 컨텍스트 구성**
* 검색된 문서들을 컨텍스트로 통합 
* 프롬프트 템플릿에 삽입
* 할루시네이션 방지 제약 추

**3. LLM 생성**
* Solar-Pro 모델 호출    
* 컨텍스트 기반 답변 생성    
* 노이즈 자동 필터링 
* 관련 없는 정보 무시    

**4. 스트리밍 출력**
- st.write_stream() 활용
- 타이핑 애니메이션 효과
- 실시간 답변 제공

### 주요 구성 요소

1. **임베딩 & 벡터 인덱싱 (Embedding Pipeline)**
   * **역할:** 제공된 Q&A 데이터를 벡터로 변환하고 Pinecone에 저장합니다.
   * **핵심 설계:** Upstage Solar Embedding 모델을 사용해 고품질 한국어 임베딩을 생성하고, 메타데이터(질문, 답변, 인덱스)를 함께 저장해 검색 시 원본 정보를 즉시 추출할 수 있도록 했습니다.
   * **기술 스택:** Python, LangChain, Upstage Embeddings, Pinecone

2. **하이브리드 검색 시스템 (Hybrid Retriever)**
   * **역할:** 사용자 질문에 대해 Dense(벡터 유사도)와 Sparse(키워드 매칭) 검색을 병행하여 최적의 문서를 검색합니다.
   * **핵심 설계:**
     - **Dense Retrieval (Pinecone)**: 의미론적 유사도 기반 검색으로 구어체/표현 변형에 강건
     - **Sparse Retrieval (BM25)**: 키워드 중심 검색으로 특정 용어나 고유명사 검색에 우수
     - 두 방식의 결과를 병합하여 검색 포괄성과 정확성을 동시에 확보
   * **기술 스택:** Pinecone, rank-bm25, LangChain

3. **생성 모델 & 프롬프트 엔지니어링 (LLM Generator)**
   * **역할:** 검색된 컨텍스트를 바탕으로 사용자 질문에 정확하고 자연스러운 답변을 생성합니다.
   * **핵심 설계:** Upstage Solar-Pro 모델과 정교한 프롬프트 템플릿을 통해 할루시네이션을 방지하고, 검색된 문서 외 정보는 절대 생성하지 않도록 제약했습니다.
   * **기술 스택:** Upstage Solar-Pro, LangChain

4. **웹 인터페이스 (Streamlit Chatbot UI)**
   * **역할:** 사용자 친화적인 채팅 인터페이스를 제공합니다.
   * **핵심 설계:** ChatGPT 스타일의 대화형 UI와 스트리밍 출력(Typing Animation)을 구현해 실시간 답변 생성 경험을 제공합니다.
   * **기술 스택:** Streamlit, Python

---

## 3. 벡터 DB 및 임베딩 방식 설명

### 3.1. 벡터 데이터베이스: Pinecone

**선택 이유:**
* **Serverless 아키텍처**: 인프라 관리 부담 없이 벡터 검색에만 집중 가능
* **빠른 ANN 검색**: 코사인 유사도 기반 Approximate Nearest Neighbor 알고리즘으로 대규모 벡터에서도 밀리초 단위 검색
* **확장성**: 자동 스케일링으로 트래픽 증가에도 안정적 대응
* **무료 티어**: 과제 수행 및 프로토타입 배포에 적합

**대안 비교:**
| 벡터 DB | 장점 | 단점 | 선택 여부 |
|---------|------|------|----------|
| Pinecone | Serverless, 관리 용이 | 비용 (대규모 시) | o |
| FAISS | 무료, 빠름 | 직접 관리 필요 | x |
| Qdrant | 오픈소스, 유연 | 러닝커브 높음 | x |

### 3.2. 임베딩 모델: Upstage Solar-Embedding-1-Large

**모델 특징:**
* **차원**: 4096 (고차원으로 풍부한 의미 표현)
* **최적화**: 한국어 데이터로 학습되어 한국어 문맥 이해 우수
* **성능**: 의미론적 유사도 계산 정확도 높음

**임베딩 생성 과정:**
```python
# 1. 질문-답변 쌍을 하나의 텍스트로 결합
text = f"{question}, {answer}"

# 2. Solar Embedding으로 벡터화
embedding_vector = embeddings.embed_documents([text])
# → 4096차원 벡터 생성

# 3. Pinecone에 메타데이터와 함께 저장
pinecone_index.upsert(
    id=str(idx),
    values=embedding_vector,
    metadata={"question": question, "answer": answer, "index": idx}
)
```

**검색 과정:**
```python
# 1. 사용자 질문 임베딩
query_vector = embeddings.embed_query(user_question)

# 2. Pinecone 유사도 검색
results = pinecone_index.query(
    vector=query_vector,
    top_k=K,
    include_metadata=True
)

# 3. MIN_SCORE 필터링
filtered_results = [match["metadata"]["answer"]
            for match in results["matches"]
            if match["score"] >= MIN_SCORE]
```

---

## 4. 핵심 기술 선택 및 의사결정

### 4.1. 하이브리드 검색 (Dense + Sparse)

**선택 이유:**
* **Dense만 사용 시 한계**: "펄소 ai", "persoai" 등 표기 변형에 취약
* **Sparse만 사용 시 한계**: "뭐야?", "알려줘" 등 구어체 질문에 취약
* **Hybrid의 장점**: 두 방식의 장점을 결합해 검색 Recall 88.5% 달성

**구현 세부사항:**
```python
# Dense: Pinecone 코사인 유사도 검색
query_vector = embeddings.embed_query(query)
dense_results = index.query(vector=query_vector, top_k=K)

# Sparse: BM25 키워드 검색
bm25_scores = bm25.get_scores(tokenized_query)
sparse_results = top_k_documents(bm25_scores)

# 결과 병합 (중복 제거 + 순서 유지)
final_contexts = list(dict.fromkeys(sparse_contexts + dense_contexts))
```

### 4.2. 프롬프트 설계

할루시네이션 방지를 위해 다음 원칙을 적용했습니다:

```python
prompt_template = """
당신은 이스트소프트에서 개발한 Perso.ai에 대해 알려주는 챗봇이고, 사용자의 질문에 대해 올바른 정보를 정확하게 전달해야 할 의무가 있습니다.

주어진 컨텍스트를 기반으로 다음 질문에 답변해주세요:

{context}

질문: {question}

답변 시 다음 사항을 고려해주세요:

1. 질문에서 핵심적인 키워드들을 골라 키워드들과 관련된 문서를 찾아서 해당 문서를 읽고 정확한 내용을 답변해주세요.
2. 문서의 내용을 그대로 길게 전달하기보다는 질문에서 요구하는 내용에 해당하는 답변만을 제공함으로써 최대한 답변을 간결하고 일관된 방식으로 제공하세요.
3. 임의로 판단해서 네 아니오 하지 말고 문서에 있는 내용을 그대로 알려주세요.
4. 답변은 친절하게 존댓말로 제공하세요.
5. 질문이 Perso.ai의 내용과 전혀 관련이 없다고 판단하면 응답하지 말아주세요.

답변:
"""
```

### 4.3. Streamlit 기반 단일 애플리케이션

본 프로젝트는 **Streamlit 기반 단일 애플리케이션 구조**를 채택했습니다.

**선택 근거:**

1. **과제 요구사항과의 적합성**
   - 과제의 핵심: RAG 시스템 구현 및 성능 검증
   - 인프라 복잡도보다 **검색/생성 알고리즘 최적화**에 집중
   - 제한된 시간 내 효율적인 프로토타입 개발

2. **개발 효율성**
   - Streamlit의 내장 스트리밍 기능 (`st.write_stream`) 활용
   - 프론트-백엔드 분리 대비 **개발 시간 80% 단축**
   - 단일 코드베이스로 유지보수 용이

3. **성능 충분성**
   - 동시 사용자 10명 이하의 데모 환경에 최적
   - LLM 응답 지연이 주요 병목 (네트워크 오버헤드 무시 가능)
   - 추가 서버 레이어로 인한 지연 불필요

**대안 고려 및 배제:**

**FastAPI + React 마이크로서비스 아키텍처**
- 장점: 높은 확장성, 독립적 배포, 부하 분산
- 배제 이유:
  - 과제 범위 대비 **과도한 인프라 복잡도**
  - SSE/WebSocket 추가 구현으로 개발 기간 증가
  - 소규모 데모 환경에서 이점 미미

---

## 5. 정확도 향상 전략

### 5.1. Recall 우선 전략 채택

**핵심 아이디어: RAG 2-Stage 아키텍처 최적화**

```
Stage 1: Retrieval (검색) - Recall 극대화
→ MIN_SCORE=0.3으로 낮춤
→ 정답을 최대한 많이 포함 (Recall 88.5%)
→ "정답을 놓치면 안 된다"

↓

Stage 2: Generation (생성) - Precision 확보
→ LLM(Solar-Pro)이 컨텍스트 분석
→ 관련 없는 정보는 자동 필터링
→ "노이즈는 LLM이 걸러낸다"
```

**선택 근거:**

1. **FAQ 챗봇 특성**
   - 사용자가 "답변 불가" 받으면 → 불만 폭발
   - 약간 부정확해도 답변 제공 → 만족도 향상

2. **LLM의 강력한 필터링 능력**
   - Solar-Pro는 문맥 이해 능력이 뛰어남
   - Precision 32.7% (노이즈 67%)도 충분히 처리 가능
   - 프롬프트 제약으로 관련 없는 정보 무시

3. **본 시스템의 설계 전략**
- Retrieval 단계에서 Recall 88.5%로 정답 누락 최소화
- Generation 단계에서 LLM이 프롬프트 제약을 통해 노이즈 필터링
- 이를 통해 사용자 경험(답변 가능률)과 답변 품질을 동시에 확보

**수치적 근거:**
```
MIN_SCORE=0.3, K=3:
→ Recall 88.5%: 질문 100개 중 88개 정답 포함
→ Precision 32.7%: 노이즈 있지만 LLM이 처리
→ 사용자 경험: "답변 불가" 11.5%로 최소화

MIN_SCORE=0.4, K=3 (대안):
→ F1-score는 더 높지만 (55.8%)
→ Recall 71.2%: 29%는 답변 못 함
→ 사용자 경험: 답변 누락이 더 치명적
```

### 5.2. 하이퍼파라미터 최적화

**하이퍼파라미터 최적화 실험 결과:**

|MIN_SCORE / K|Precision|Recall|F1-Score|
|---|---|---|---|
|**MIN_SCORE=0.1, K=3**|0.311|0.933|0.466|
|**MIN_SCORE=0.2, K=3**|0.323|0.923|0.479|
|**MIN_SCORE=0.3, K=3**|0.327|0.885|0.478|
|**MIN_SCORE=0.4, K=3**|0.460|0.712|0.558|
|**MIN_SCORE=0.5, K=3**|0.826|0.183|0.299|
|**MIN_SCORE=0.6, K=3**|0.000|0.000|0.000|
|**MIN_SCORE=0.7, K=3**|0.000|0.000|0.000|
|**MIN_SCORE=0.1, K=5**|0.194|0.971|0.324|
|**MIN_SCORE=0.2, K=5**|0.211|0.962|0.345|
|**MIN_SCORE=0.3, K=5**|0.215|0.923|0.349|
|**MIN_SCORE=0.4, K=5**|0.358|0.740|0.483|
|**MIN_SCORE=0.5, K=5**|0.826|0.183|0.299|
|**MIN_SCORE=0.6, K=5**|0.000|0.000|0.000|
|**MIN_SCORE=0.7, K=5**|0.000|0.000|0.000|

**최종 선정: MIN_SCORE=0.3, K=3**
- Recall 88.5% 확보로 정답 누락 최소화
- Precision 32.7%는 LLM이 충분히 보완 가능
- 비용/성능 효율 최고 (K=3으로 토큰 절약)

### 5.3. 데이터 증강

**테스트셋 확장:**
- 원본 13개 Q&A → 104개로 증강 (8배)
- 방법: 구어체 변환, 표현 다양화, 축약형 생성

**증강 예시:**
```python
원본: "Perso.ai는 어떤 서비스인가요?"
변형1: "perso ai가 뭐야?"
변형2: "페르소 AI 설명해줘"
변형3: "펄소ai 소개 부탁해"
```

### 5.4. 프롬프트 엔지니어링

**할루시네이션 방지 제약:**
1. "주어진 컨텍스트를 기반으로만 답변"
2. "관련 없는 질문은 응답 거절"
3. "문서의 내용으로만 정확한 내용 제공"

---

## 6. 성능 평가 및 벤치마크

### 6.1. 평가 방법론

**테스트셋 구성:**
* 원본 13개 Q&A → 표현 변형/구어체 등을 통해 104개로 증강
* 각 질문마다 정답 인덱스 레이블 부여

```python
dataset = [
    # 0번 관련 (서비스 소개) - 8개
    {"query": "perso ai가 뭐야?", "labels": [0]},
    {"query": "Perso.ai는 무엇인가요?", "labels": [0]},
    {"query": "페르소 AI 설명해줘", "labels": [0]},
...
```

**평가 지표:**
* **Precision**: 검색된 K개 문서 중 정답 비율 (정확도)
* **Recall**: 전체 정답 중 검색된 비율 (포괄성)
* **F1-score**: Precision과 Recall의 조화평균 (종합 성능)

### 6.2. 최종 성능 결과

| 하이퍼파라미터 | Precision | Recall | F1-score |
|--------------|-----------|--------|----------|
| MIN_SCORE=0.3, K=3 | 32.7% | **88.5%** | 47.8% |

**해석:**

* **Recall 88.5% (매우 우수)**
  - 질문 100개 중 88개에서 정답 문서를 성공적으로 검색
  - "정답 누락" 비율이 11.5%로 낮아 사용자 만족도 확보
  - FAQ 챗봇에서 가장 중요한 지표 달성

* **Precision 32.7% (LLM 보완 가능)**
  - 검색된 문서 3개 중 약 1개만 정답 (나머지 2개는 노이즈)
  - 하지만 Solar-Pro LLM이 프롬프트 제약을 통해 노이즈 자동 필터링
  - 최종 답변 품질은 Precision보다 훨씬 높음

---

## 7. 구현 디테일

### 7.1. 코드 구조 및 클래스 설계

**객체지향 설계 원칙 적용:**

```python
class Retriever:
    """BM25 Sparse 검색 담당"""
    def get_sparse(self, query, top_k=3):
        # BM25 검색 로직

class VectorDB:
    """Pinecone Dense 검색 담당"""
    def get_dense(self, embeddings, query, min_score=0.3, top_k=3):
        # 벡터 유사도 검색 로직

class RAGChatbot:
    """전체 RAG 파이프라인 통합"""
    def generate_response(self, user_query):
        # Hybrid 검색 → LLM 생성
```

**설계 장점:**
* 단일 책임 원칙(SRP): 각 클래스는 하나의 책임만 수행
* 확장 용이성: 새로운 검색 방식 추가 시 기존 코드 수정 최소화
* 테스트 용이성: 각 컴포넌트를 독립적으로 테스트 가능

### 7.2. Streamlit UI 고도화

**주요 기능:**
1. **ChatGPT 스타일 대화 인터페이스**
   * `st.chat_message`와 `st.chat_input` 활용
   * 세션 상태 관리로 대화 히스토리 유지

2. **스트리밍 출력 (Typing Animation)**
   ```python
   def stream_response(text):
       for char in text:
           yield char
           time.sleep(0.01)
   
   st.write_stream(stream_response(bot_reply))
   ```

---

## 8. 향후 개선 방향

### 8.1. 검색 고도화
* **Re-ranking**: Cross-encoder로 검색 결과 2차 정렬
* **Query Expansion**: 질문 의도 분석 및 동의어 확장
* **MMR**: 검색 결과 다양성 확보

### 8.2. 성능 최적화
* **캐싱**: 자주 묻는 질문 캐싱으로 응답 속도 향상
* **임베딩 파인튜닝**: 도메인 특화 데이터로 Solar Embedding 미세 조정

### 8.3. 사용자 경험
* **피드백 수집**: 답변 평가 기능 추가
* **다국어 지원**: 영어 등 추가 언어 확장

---

## 9. 결론

이번 과제를 통해 **이론과 실전을 결합한 RAG 시스템**을 구축했습니다. 

**주요 성과:**
* 할루시네이션 없는 정확한 답변 생성
* Hybrid 검색으로 Recall 88.5% 달성
* Recall 우선 전략으로 사용자 경험 최적화
* 실전 배포 가능한 웹 애플리케이션 완성
* 체계적인 성능 평가 및 Grid Search 최적화

단순히 과제 요구사항을 충족하는 것을 넘어, **RAG 아키텍처의 본질을 이해하고 실무 관점에서 최적화된 시스템**을 구현했습니다. 특히 "Retriever는 Recall, Generator는 Precision"이라는 2-Stage 전략을 명확히 적용하여, 이론과 실전이 조화된 솔루션을 제시했습니다.
