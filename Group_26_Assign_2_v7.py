# Group_26 Conversational AI Assignment 2
# NAVEEN S VIJAPURAPU 2022ac05134
# POPURI SAI KRISHNA 2023aa08053
# PRAWALIKA K 2023aa05465
# REKAPALLI TARAKARAM 2023a05651
# VADLAPATLA MONI SRI VALLI 2023aa05468


import os
import faiss
import torch
import streamlit as st
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer
from rank_bm25 import BM25Okapi
from scipy.special import softmax  # For better confidence score normalization
from transformers import AutoModelForSeq2SeqLM

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use CPU to avoid GPU issues

# Load Embedding Model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Small Open-Source Language Model (Lighter Model)
####### tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
####### model = AutoModelForCausalLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")


# Pre-processing Function to parse financial XML files
def parse_financial_xml(file):
    tree = ET.parse(file)
    root = tree.getroot()
    namespace = {"bse": "http://www.bseindia.com/xbrl/fin/2020-03-31/in-bse-fin"}

    keys_of_interest = [
        "RevenueFromOperations",
        "ProfitBeforeTax",
        "ProfitLossForPeriod",
        "ComprehensiveIncomeForThePeriod",
        "EmployeeBenefitExpense",
        "DepreciationDepletionAndAmortisationExpense",
        "FinanceCosts",
        "OtherExpenses",
        "PaidUpValueOfEquityShareCapital",
        "BasicEarningsLossPerShareFromContinuingOperations",
    ]

    extracted_data = []
    for key in keys_of_interest:
        element = root.find(f".//bse:{key}", namespace)
        if element is not None:
            extracted_data.append(f"{key.replace('bse:', '')}: {element.text}")

    return "\n".join(extracted_data)


# Function to process financial documents
def process_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(docs)
    embeddings = embed_model.encode(chunks, convert_to_tensor=True)
    tokenized_chunks = [chunk.split() for chunk in chunks]  # For BM25
    bm25 = BM25Okapi(tokenized_chunks)
    return chunks, embeddings, bm25


# Store embeddings in FAISS
def create_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.cpu().numpy())
    return index


######
# Note: Advance RAG Technique for Group 26 is 1 - Multi Stage Retrieval
######


# Multi-Stage Retrieval (BM25 + FAISS) with Confidence Score Calculation
def retrieve_documents(query, index, chunks, bm25, use_multistage):
    tokenized_query = query.split()

    # BM25 retrieval (only if Multi-Stage retrieval is enabled)
    if use_multistage:
        bm25_scores = bm25.get_scores(tokenized_query)
        top_bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:10]
    else:
        top_bm25_indices = []

    # FAISS retrieval
    query_embedding = embed_model.encode([query], convert_to_tensor=True)
    distances, faiss_indices = index.search(query_embedding.cpu().numpy(), k=3)

    final_indices = list(set(top_bm25_indices + list(faiss_indices[0])))[:3]

    # Normalize BM25 scores (softmax for better distribution)
    if use_multistage:
        bm25_scores_array = [bm25_scores[i] for i in final_indices]
        normalized_bm25_scores = softmax(bm25_scores_array)
    else:
        normalized_bm25_scores = [0] * len(final_indices)

    # Normalize FAISS distances (convert to similarity)
    max_faiss_distance = max(distances[0]) if len(distances[0]) > 0 else 1
    normalized_faiss_scores = softmax(
        [
            1 - (distances[0][i] / (max_faiss_distance + 1e-6))
            for i in range(len(final_indices))
        ]
    )

    # Adjust weights dynamically (give more weight to the more confident retrieval)
    weight_bm25 = 0.5 if use_multistage else 0.0  # BM25 only if enabled
    weight_faiss = 1.0 - weight_bm25  # FAISS gets higher weight if BM25 is disabled

    # Compute final confidence score (weighted sum with dynamic weighting)
    confidence_scores = [
        weight_bm25 * bm25_score + weight_faiss * faiss_score
        for bm25_score, faiss_score in zip(
            normalized_bm25_scores, normalized_faiss_scores
        )
    ]

    # Select the best chunk with highest confidence score
    best_idx = confidence_scores.index(max(confidence_scores))
    return (
        chunks[final_indices[best_idx]],
        confidence_scores[best_idx] * 100,
    )  # Convert to percentage


# Guardrail: Input Validation
def validate_query(query):
    if len(query) < 5 or not any(
        keyword in query.lower()
        for keyword in [
            "revenue",
            "profit",
            "earnings",
            "margin",
            "financial",
            "statement",
        ]
    ):
        return "Invalid query: Please ask a relevant financial question."
    return None


# Generate response using the small LLM
def generate_response(context, query):
    prompt = f"\nContext: {context}\nQuestion: {query}\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_new_tokens=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)


# Streamlit UI Development
def main():
    st.title("Financial RAG Chatbot")
    uploaded_file = st.file_uploader("Upload Financial Statement (XML)", type=["xml"])

    if uploaded_file is not None:
        financial_text = parse_financial_xml(uploaded_file)
        text_chunks, embeddings, bm25 = process_documents(financial_text)
        faiss_index = create_faiss_index(embeddings)

        # Toggle for Multi-Stage Retrieval
        use_multistage = st.checkbox(
            "Enable Multi-Stage Retrieval (BM25 + FAISS)", value=True
        )

        query = st.text_input("Ask a financial question:")

        if query:
            validation_error = validate_query(query)
            if validation_error:
                st.write("**Error:**", validation_error)
            else:
                best_chunk, confidence = retrieve_documents(
                    query, faiss_index, text_chunks, bm25, use_multistage
                )
                response = generate_response(best_chunk, query)

                # Display results
                st.write("**Answer:**", response)
                st.write(f"**Confidence Score:** {confidence:.2f}%")


if __name__ == "__main__":
    main()
