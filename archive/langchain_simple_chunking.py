import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
client = OpenAI()


def load_markdown_document(file_path):
    """Load the markdown file as a LangChain Document."""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Create a Document object with metadata
    doc = Document(
        page_content=content, metadata={"source": file_path, "type": "markdown"}
    )
    return doc


def chunk_markdown_with_langchain(document, chunk_size=1000, chunk_overlap=200):
    """
    Split markdown document into chunks using LangChain's RecursiveCharacterTextSplitter.

    Args:
        document: LangChain Document object
        chunk_size: Target chunk size in characters
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of Document chunks
    """
    # Create text splitter optimized for markdown
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Split on paragraphs first, then sentences, then words
        separators=[
            "\n\n",  # Paragraphs (double newline)
            "\n",  # Single newlines
            "## ",  # Markdown headers level 2
            "### ",  # Markdown headers level 3
            ". ",  # Sentences
            " ",  # Words
            "",  # Characters (fallback)
        ],
        length_function=len,
        is_separator_regex=False,
    )

    # Split the document
    chunks = text_splitter.split_documents([document])

    return chunks


def create_processed_chunks(chunks, source_file):
    """
    Convert LangChain chunks to our standard format.

    Args:
        chunks: List of LangChain Document chunks
        source_file: Source file name/path

    Returns:
        List of dictionaries with text and simplified metadata
    """
    processed_chunks = []

    for i, chunk in enumerate(chunks):
        text = chunk.page_content

        processed_chunk = {
            "text": text,
            "metadata": {
                "source": source_file,
                "chunk_id": i,
                "chunk_size": len(text),
            },
        }
        processed_chunks.append(processed_chunk)

    return processed_chunks


def embed_chunks_with_openai(chunks, client):
    """
    Embed each chunk using OpenAI's text-embedding-3-small model.

    Args:
        chunks: List of dictionaries with 'text' and 'metadata' keys
        client: OpenAI client instance

    Returns:
        List of dictionaries with added 'embed' key containing the embedding vector
    """
    embedded_chunks = []

    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i + 1}/{len(chunks)}")

        # Get embedding for the chunk text
        response = client.embeddings.create(
            model="text-embedding-3-small", input=chunk["text"]
        )

        # Extract the embedding vector
        embedding_vector = response.data[0].embedding

        # Create new chunk dict with embedding
        embedded_chunk = {
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "embed": embedding_vector,
        }

        embedded_chunks.append(embedded_chunk)

    return embedded_chunks


def cosine_similarity_openai(query_vector, chunk_vectors):
    """Calculate cosine similarity for OpenAI embeddings (already normalized)."""
    similarities = np.dot(chunk_vectors, query_vector)
    return similarities


def embed_query(query_text, client):
    """Embed a single query using OpenAI's text-embedding-3-small model."""
    response = client.embeddings.create(
        model="text-embedding-3-small", input=query_text
    )
    return response.data[0].embedding


def vector_similarity_search(query_text, df, client, k=5):
    """Perform vector similarity search to find the most relevant chunks."""
    # Embed the query
    print(f"Embedding query: '{query_text}'")
    query_embedding = embed_query(query_text, client)

    # Convert embeddings to numpy arrays
    chunk_embeddings = np.array(df["embed"].tolist())
    query_embedding_array = np.array(query_embedding)

    # Calculate cosine similarities (OpenAI embeddings are already normalized)
    similarities = cosine_similarity_openai(query_embedding_array, chunk_embeddings)

    # Add similarity scores to a copy of the dataframe
    df_with_scores = df.copy()
    df_with_scores["similarity_score"] = similarities

    # Sort by similarity score (descending) and get top k results
    top_results = df_with_scores.nlargest(k, "similarity_score")

    return top_results[["text", "metadata", "similarity_score"]]


def generate_response_with_context(user_prompt, retrieved_chunks, client):
    """
    Generate a response using OpenAI's Chat Completions API with retrieved context.

    Args:
        user_prompt: The user's question/prompt
        retrieved_chunks: List of text chunks from similarity search
        client: OpenAI client instance

    Returns:
        str: The model's response
    """
    # Combine retrieved chunks into context
    context = "\n\n".join(retrieved_chunks)

    # Create system message with context
    system_message = f"""You are a helpful assistant that answers questions based on the provided context from documents. 
Use only the information from the context to answer questions. If you're unsure or the context doesn't contain the relevant information, say so clearly.

Context:
{context}"""

    # Create messages for chat completion
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]

    try:
        # Call OpenAI Chat Completions API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,  # Low temperature for more focused responses
            max_tokens=4000,  # Adjust as needed
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating response: {str(e)}"


# --------------------------------------------------------------
# Main execution
# --------------------------------------------------------------
filename = "pdf_output.md"
# Load the markdown document
print("Loading markdown document...")
document = load_markdown_document(filename)
print(f"Loaded document with {len(document.page_content)} characters")


# Process with chunk size (500)
print(f"\n{'=' * 60}")
print("PROCESSING WITH 500 CHARACTER CHUNKS")
print("=" * 60)

chunks = chunk_markdown_with_langchain(document, chunk_size=500, chunk_overlap=100)
processed_chunks = create_processed_chunks(chunks, filename)

print(f"Created {len(processed_chunks)} processed chunks")

# Embed the chunks
print("\nEmbedding chunks...")
embedded_chunks = embed_chunks_with_openai(processed_chunks, client)

# Create DataFrame
df = pd.DataFrame(embedded_chunks)
print(f"Created DataFrame with {len(df)} embedded chunks")

# Test search only (without RAG response)
test_queries = [
    "What is the main idea of Docling?",
    "How does Docling process PDF documents?",
    "What AI models does Docling use?",
    "How to install and use Docling?",
]

print(f"\n{'=' * 60}")
print("TESTING SEARCH QUALITY")
print("=" * 60)

for query in test_queries:
    print(f"\nQuery: '{query}'")
    print("-" * 50)

    results = vector_similarity_search(query, df, client, k=3)

    for i, (idx, row) in enumerate(results.iterrows(), 1):
        score = row["similarity_score"]
        text_preview = row["text"][:150].replace("\n", " ")
        source = row["metadata"]["source"]
        chunk_id = row["metadata"]["chunk_id"]
        chunk_size = row["metadata"]["chunk_size"]

        print(f"Result {i} (Score: {score:.4f}): {text_preview}...")
        print(f"  Source: {source}, Chunk ID: {chunk_id}, Size: {chunk_size} chars")
        print()

# Save results
df.to_csv("data/langchain_simple_chunks_with_embeddings.csv", index=False)
print(f"\nSaved embeddings to data/langchain_simple_chunks_with_embeddings.csv")

# --------------------------------------------------------------
# RAG Step-by-Step Testing
# --------------------------------------------------------------

# Test RAG with step-by-step execution
rag_test_queries = [
    "What is Docling and what does it do?",
    "How does Docling process PDF documents? Explain the pipeline.",
    "What AI models does Docling use and what are they for?",
    "How do I install and get started with Docling?",
    "What are the main features and capabilities of Docling?",
]

print(f"\n{'=' * 80}")
print("TESTING RAG STEP-BY-STEP WITH RESPONSES")
print("=" * 80)

for query in rag_test_queries:
    print(f"\n{'=' * 60}")
    print(f"QUERY: '{query}'")
    print("=" * 60)

    # Step 1: Retrieve relevant chunks
    print("🔍 Step 1: Retrieving relevant chunks...")
    search_results = vector_similarity_search(query, df, client, k=3)

    # Extract text content for context
    retrieved_texts = search_results["text"].tolist()
    print(f"✅ Retrieved {len(retrieved_texts)} chunks")

    # Show retrieved chunks
    print(f"\n📊 RETRIEVED CHUNKS:")
    print("-" * 40)
    for i, (idx, row) in enumerate(search_results.iterrows(), 1):
        score = row["similarity_score"]
        chunk_id = row["metadata"]["chunk_id"]
        chunk_size = row["metadata"]["chunk_size"]
        text_preview = row["text"][:100].replace("\n", " ")
        print(f"  Chunk {i}: ID {chunk_id}, Score {score:.4f}, Size {chunk_size} chars")
        print(f"    Preview: {text_preview}...")
        print()

    # Step 2: Generate response with context
    print("🤖 Step 2: Generating response with context...")
    response = generate_response_with_context(query, retrieved_texts, client)

    # Display the response
    print(f"\n📝 FINAL RESPONSE:")
    print("-" * 40)
    print(response)

    print("\n" + "=" * 80)
