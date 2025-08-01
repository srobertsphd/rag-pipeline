"""
Clean RAG pipeline execution without verbose print statements.
Contains the same functionality as main.py but with minimal output.
"""

import pandas as pd
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
# Step 1: Document Loading
# --------------------------------------------------------------
print("STEP 1: DOCUMENT LOADING")

filename = "pdf_output.md"
document = load_markdown_document(filename)

# --------------------------------------------------------------
# Step 2: Document Chunking
# --------------------------------------------------------------
print("STEP 2: DOCUMENT CHUNKING")

chunks = chunk_markdown_with_langchain(document, chunk_size=1000, chunk_overlap=200)
processed_chunks = create_processed_chunks(chunks, filename)

# --------------------------------------------------------------
# Step 3: Embedding Generation
# --------------------------------------------------------------
print("STEP 3: EMBEDDING GENERATION")

embedded_chunks = embed_chunks_with_openai(processed_chunks, client)
df = pd.DataFrame(embedded_chunks)

# --------------------------------------------------------------
# Step 4: Search Quality Testing
# --------------------------------------------------------------
print("STEP 4: SEARCH QUALITY TESTING")

test_queries = [
    "What is the main idea of Docling?",
    "How does Docling process PDF documents?",
    "What AI models does Docling use?",
    "How to install and use Docling?",
]

for query in test_queries:
    results = vector_similarity_search(query, df, client, k=3)

# Save embeddings
df.to_csv("data/langchain_simple_chunks_with_embeddings.csv", index=False)

# --------------------------------------------------------------
# Step 5: RAG Demonstration
# --------------------------------------------------------------
print("STEP 5: RAG DEMONSTRATION")

query = "What is Docling and what does it do?"
search_results = vector_similarity_search(query, df, client, k=5)
retrieved_texts = search_results["text"].tolist()
response = generate_response_with_context(query, retrieved_texts, client)

print("Response:")
print(response.choices[0].message.content)
