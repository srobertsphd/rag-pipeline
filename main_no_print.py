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

type(document)

print(document.model_dump_json(indent=4))
print(document.page_content)
doc_dict = document.model_dump()

doc_dict["page_content"]


# --------------------------------------------------------------
# Step 2: Document Chunking
# --------------------------------------------------------------
print("STEP 2: DOCUMENT CHUNKING")

chunks = chunk_markdown_with_langchain(document, chunk_size=2500, chunk_overlap=1000)
processed_chunks = create_processed_chunks(chunks, filename)

pd_temp = pd.DataFrame(processed_chunks)
pd_temp.to_csv("data/langchain_simple_chunks_BEFORE_embeddings.csv", index=False)

print(pd_temp["metadata"][0]["chunk_size"])


# Function to print character count for each chunk
def print_chunk_sizes(dataframe):
    print(f"Character count for all {len(dataframe)} chunks:")
    for i, row in dataframe.iterrows():
        print(f"Chunk {i + 1}: {row['metadata']['chunk_size']} characters")


# Print character count for all chunks
print_chunk_sizes(pd_temp)

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
    # "What is the main idea of Docling?",
    # "How does Docling process PDF documents?",
    # "What AI models does Docling use?",
    # "How to install and use Docling?",
    "tell me how to make a peanut butter and jelly sandwich"
]

for query in test_queries:
    results = vector_similarity_search(query, df, client, k=10)

# Save embeddings
df.to_csv("data/langchain_simple_chunks_with_embeddings.csv", index=False)

# --------------------------------------------------------------
# Step 5: RAG Demonstration
# --------------------------------------------------------------
print("STEP 5: RAG DEMONSTRATION")

query = "tell me how to make a peanut butter and jelly sandwich"
search_results = vector_similarity_search(query, df, client, k=5)
retrieved_texts = search_results["text"].tolist()
response = generate_response_with_context(query, retrieved_texts, client)

print("Response:")
print(response.choices[0].message.content)

response.choices[0].model_dump()
