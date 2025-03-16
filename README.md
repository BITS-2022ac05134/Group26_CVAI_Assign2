# Financial RAG Chatbot

## Objective
The **Financial RAG (Retrieval-Augmented Generation) chatbot** is designed to answer **finance-related questions** using company **financial statements**. It leverages a **Small Open-Source Language Model (SLM)** and a **Multi-Stage Retrieval** approach to ensure **accurate and contextually relevant responses**.

---

## Core Components

### 1. Small Language Model (SLM) Selection

**Model Used:** `FLAN-T5 Small (Google)`

#### Why This Model?
- **Open-source & lightweight** → Reduces memory and storage usage.
- **Fine-tuned for instruction following** → Ideal for Q&A tasks.
- **Supports financial queries** with structured text input-output generation.

#### Decision Rationale
Originally, **Falcon-7B** was considered, but it was **too large for local execution**. **FLAN-T5 Small** was chosen because:
- It fits within local system constraints.
- It provides reasonable accuracy for text generation.
- It processes financial text efficiently.

---

## 2. Retrieval-Augmented Generation (RAG) Strategy

### Basic RAG vs. Advanced RAG (Multi-Stage Retrieval)

#### Basic RAG (Single-Stage Retrieval)
- Uses **FAISS (Facebook AI Similarity Search)** to retrieve **top similar financial text chunks** based on embeddings.
- Queries are vectorized using **MiniLM embeddings**, and the closest matches are retrieved.
- The top match is fed into **FLAN-T5 Small** for response generation.

#### Advanced RAG (Multi-Stage Retrieval)
- Uses a **hybrid retrieval** approach:
  - **BM25 (Lexical Search)** → Finds the **most relevant** financial text based on **word overlap**.
  - **FAISS (Semantic Search)** → Finds the **most similar** financial text based on **meaning**.
- The **best-ranked results** from both are combined and **re-ranked** before passing to the language model.
- Users can **toggle between Basic & Advanced RAG** using the **Streamlit UI**.

#### Business Impact
- Improves response relevance by combining **keyword-based** & **semantic search**.
- Ensures factual consistency by using **verified financial document chunks**.

---

## 3. Confidence Score Calculation

### How is Confidence Calculated?
- **BM25 Score + FAISS Similarity Score** → Converted into a **percentage (0-100%)**.

### Why Does It Sometimes Show Low Confidence?
- **FAISS similarity scores** range from **0 to 1**, and moderate matches may lead to lower confidence values.
- **BM25 scores** depend on **query-document token overlap**, which may be lower for financial texts.

### Improvement Ideas
- Adjust weighting or normalize scores for better scaling.
- Introduce contextual ranking instead of direct FAISS scores.

### Final Display
Only the **top-ranked financial statement chunk** is shown with its **confidence percentage**.

---

## Future Enhancements & Improvements

### 1. Improving Confidence Score Accuracy
- Introduce contextual ranking: Instead of direct FAISS scores, use a **ranking model** to weigh results.
- Normalize confidence: Adjust BM25 and FAISS scoring scale to better reflect relevance.

### 2. Using a More Advanced SLM
- Upgrade to `FLAN-T5 Base` for **better text understanding** while keeping it lightweight.
- Consider `Mistral 7B` (quantized version) for **more powerful response generation**.

### 3. Enhancing UI & User Experience
- Add Streamlit caching → Speeds up repeated queries.
- Enable PDF & Excel support → Allows financial statements in multiple formats.

### 4. Model Fine-Tuning for Finance
- Train MiniLM embeddings on financial data to improve retrieval accuracy.
- Fine-tune FLAN-T5 on **financial Q&A datasets** for better domain-specific response quality.

### 5. Integrate with External APIs
- Connect with `Yahoo Finance` or `EDGAR SEC API` for real-time financial data.
- Add `GPT-4 API fallback` for complex queries beyond local model capabilities.

---

## Business Impact Summary
- Reduces dependency on large proprietary AI models → 100% **open-source & cost-effective**.
- Provides factual, document-backed financial insights → **Increases trust & accuracy**.
- Optimized for local execution → Works on **standard hardware** without **cloud costs**.
- Future scalability → Can integrate **better models, real-time APIs, and UI enhancements**.

### Next Steps
Implement **confidence score improvements** & **model fine-tuning** for **higher accuracy**.

