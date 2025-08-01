"""
Main orchestration of the RAG pipeline.
Demonstrates how all the modules work together to create a complete RAG system.
This file can be run interactively to see each step of the process.
"""

import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Import our custom modules
from document_loader import (
    load_markdown_document,
    create_processed_chunks,
)
from chunking import chunk_markdown_with_langchain
from openai_services import embed_chunks_with_openai, generate_response_with_context
from vector_search import vector_similarity_search

# Load environment variables and initialize OpenAI client
load_dotenv()
client = OpenAI()

# --------------------------------------------------------------
# Optional: Convert PDF to Markdown (uncomment to use)
# --------------------------------------------------------------

# from document_loader import convert_pdf_to_markdown
# pdf_path = "archive/2408.09869v5.pdf"
# markdown_file = convert_pdf_to_markdown(pdf_path, "pdf_output.md")

# --------------------------------------------------------------
# Step 1: Document Loading
# --------------------------------------------------------------
print("📁 STEP 1: DOCUMENT LOADING")
print("=" * 50)

filename = "pdf_output.md"
print(f"Loading markdown document: {filename}")
document = load_markdown_document(filename)
print(f"✅ Loaded document with {len(document.page_content)} characters")

# --------------------------------------------------------------
# Step 2: Document Chunking
# --------------------------------------------------------------
print("\n🔀 STEP 2: DOCUMENT CHUNKING")
print("=" * 50)

print("Chunking document into 1000-character pieces...")
chunks = chunk_markdown_with_langchain(document, chunk_size=1000, chunk_overlap=200)
processed_chunks = create_processed_chunks(chunks, filename)
print(f"✅ Created {len(processed_chunks)} processed chunks")

# Show chunk statistics
chunk_lengths = [len(chunk["text"]) for chunk in processed_chunks]
print(f"Average chunk length: {np.mean(chunk_lengths):.0f} characters")
print(f"Min chunk length: {min(chunk_lengths)}")
print(f"Max chunk length: {max(chunk_lengths)}")

# --------------------------------------------------------------
# Step 3: Embedding Generation
# --------------------------------------------------------------
print("\n🧠 STEP 3: EMBEDDING GENERATION")
print("=" * 50)

print("Embedding chunks with OpenAI text-embedding-3-small...")
embedded_chunks = embed_chunks_with_openai(processed_chunks, client)
print("✅ Generated embeddings for {len(embedded_chunks)} chunks")

# Create DataFrame for easy manipulation
df = pd.DataFrame(embedded_chunks)
print(f"Created DataFrame with {len(df)} embedded chunks")

print(df.iloc[0]["text"])

# --------------------------------------------------------------
# Step 4: Search Quality Testing
# --------------------------------------------------------------
print("\n🔍 STEP 4: SEARCH QUALITY TESTING")
print("=" * 50)

test_queries = [
    "What is the main idea of Docling?",
    "How does Docling process PDF documents?",
    "What AI models does Docling use?",
    "How to install and use Docling?",
]

print("Testing vector similarity search...")
for query in test_queries:
    print(f"\nQuery: '{query}'")
    print("-" * 40)

    results = vector_similarity_search(query, df, client, k=3)

    for i, (idx, row) in enumerate(results.iterrows(), 1):
        score = row["similarity_score"]
        text_preview = row["text"][:100].replace("\n", " ")
        source = row["metadata"]["source"]
        chunk_id = row["metadata"]["chunk_id"]
        chunk_size = row["metadata"]["chunk_size"]

        print(f"  Result {i} (Score: {score:.4f}): {text_preview}...")
        print(f"    Source: {source}, Chunk ID: {chunk_id}, Size: {chunk_size} chars")

# Save embeddings for later use
df.to_csv("data/langchain_simple_chunks_with_embeddings.csv", index=False)
print("\nSaved embeddings to data/langchain_simple_chunks_with_embeddings.csv")

# --------------------------------------------------------------
# Step 5: Simple RAG Demonstration
# --------------------------------------------------------------
print("\nSTEP 5: RAG DEMONSTRATION")
print("=" * 40)

# Simple test query
query = "What is Docling and what does it do?"
print(f"Query: {query}")

# Retrieve relevant chunks
search_results = vector_similarity_search(query, df, client, k=5)
retrieved_texts = search_results["text"].tolist()

print(f"\nRetrieved {len(retrieved_texts)} chunks:")
for i, (idx, row) in enumerate(search_results.iterrows(), 1):
    score = row["similarity_score"]
    chunk_id = row["metadata"]["chunk_id"]
    print(f"  {i}. Chunk {chunk_id} (score: {score:.3f})")

# Generate response
response = generate_response_with_context(query, retrieved_texts, client)

print("\nResponse:")
print(response)
