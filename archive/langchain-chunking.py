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


def chunk_markdown_with_langchain(
    document, chunk_size=1000, chunk_overlap=200, min_chunk_size=500
):
    """
    Split markdown document into chunks using LangChain's RecursiveCharacterTextSplitter.

    Args:
        document: LangChain Document object
        chunk_size: Target chunk size in characters
        chunk_overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum chunk size - smaller chunks will be aggregated

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
    initial_chunks = text_splitter.split_documents([document])

    # Enforce minimum chunk size by aggregating small chunks
    final_chunks = enforce_minimum_chunk_size(
        initial_chunks, min_chunk_size, chunk_size
    )

    return final_chunks


def enforce_minimum_chunk_size(chunks, min_size=500, max_size=1500):
    """
    Aggregate chunks that are below the minimum size.

    Args:
        chunks: List of Document chunks
        min_size: Minimum chunk size in characters
        max_size: Maximum size to avoid when combining chunks

    Returns:
        List of Document chunks that meet minimum size requirements
    """
    if not chunks:
        return chunks

    aggregated_chunks = []
    current_chunk_content = ""
    current_chunk_metadata = chunks[0].metadata.copy()

    for i, chunk in enumerate(chunks):
        chunk_text = chunk.page_content

        # Add to current aggregate
        if current_chunk_content:
            current_chunk_content += "\n\n" + chunk_text
        else:
            current_chunk_content = chunk_text
            current_chunk_metadata = chunk.metadata.copy()

        current_size = len(current_chunk_content)
        is_last_chunk = i == len(chunks) - 1

        # Check if adding next chunk would exceed max size
        next_would_exceed_max = False
        if not is_last_chunk:
            next_chunk_size = len(chunks[i + 1].page_content)
            next_would_exceed_max = (
                current_size + next_chunk_size + 2
            ) > max_size  # +2 for \n\n

        # Finalize chunk if: meets min size AND (is last OR next would exceed max)
        should_finalize = current_size >= min_size and (
            is_last_chunk or next_would_exceed_max
        )

        if should_finalize:
            aggregated_chunk = Document(
                page_content=current_chunk_content, metadata=current_chunk_metadata
            )
            aggregated_chunks.append(aggregated_chunk)
            current_chunk_content = ""

    # Handle any remaining content (edge case)
    if current_chunk_content:
        aggregated_chunk = Document(
            page_content=current_chunk_content, metadata=current_chunk_metadata
        )
        aggregated_chunks.append(aggregated_chunk)

    # Report aggregation results
    under_min = sum(1 for chunk in chunks if len(chunk.page_content) < min_size)
    final_under_min = sum(
        1 for chunk in aggregated_chunks if len(chunk.page_content) < min_size
    )

    print(f"Aggregation results:")
    print(f"  Original chunks: {len(chunks)} ({under_min} under {min_size} chars)")
    print(
        f"  Final chunks: {len(aggregated_chunks)} ({final_under_min} under {min_size} chars)"
    )

    return aggregated_chunks


def create_processed_chunks(chunks, source_file):
    """
    Convert LangChain chunks to our standard format.

    Args:
        chunks: List of LangChain Document chunks
        source_file: Original source file name

    Returns:
        List of dictionaries with text and metadata
    """
    processed_chunks = []

    for i, chunk in enumerate(chunks):
        # Extract section title if available (look for markdown headers)
        text = chunk.page_content
        title = None

        # Try to find a section title in the chunk
        lines = text.split("\n")
        for line in lines:
            if line.startswith("## ") or line.startswith("### "):
                title = line.strip("# ").strip()
                break

        processed_chunk = {
            "text": text,
            "metadata": {
                "source": source_file,
                "chunk_id": i,
                "title": title,
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


# --------------------------------------------------------------
# Main execution
# --------------------------------------------------------------

# Load the markdown document
print("Loading markdown document...")
document = load_markdown_document("pdf_output.md")
print(f"Loaded document with {len(document.page_content)} characters")

# Try different chunk sizes
chunk_sizes = [500, 1000]

for chunk_size in chunk_sizes:
    print(f"\n{'=' * 60}")
    print(f"TESTING CHUNK SIZE: {chunk_size} characters")
    print("=" * 60)

    # Chunk the document
    chunks = chunk_markdown_with_langchain(
        document,
        chunk_size=chunk_size,
        chunk_overlap=100,  # 100 character overlap
        min_chunk_size=500,  # Ensure minimum chunk size
    )

    print(f"Created {len(chunks)} chunks")

    # Show chunk statistics
    chunk_lengths = [len(chunk.page_content) for chunk in chunks]
    print(f"Average chunk length: {np.mean(chunk_lengths):.0f} characters")
    print(f"Min chunk length: {min(chunk_lengths)}")
    print(f"Max chunk length: {max(chunk_lengths)}")

    # Show first few chunks
    print(f"\nSample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i + 1} ({len(chunk.page_content)} chars) ---")
        print(chunk.page_content[:200] + "...")

# Process with optimal chunk size (1000)
print(f"\n{'=' * 60}")
print("PROCESSING WITH 1000 CHARACTER CHUNKS")
print("=" * 60)

chunks = chunk_markdown_with_langchain(
    document, chunk_size=500, chunk_overlap=100, min_chunk_size=400
)
processed_chunks = create_processed_chunks(chunks, "pdf_output.md")

print(f"Created {len(processed_chunks)} processed chunks")

# Embed the chunks
print("\nEmbedding chunks...")
embedded_chunks = embed_chunks_with_openai(processed_chunks, client)

# Create DataFrame
df = pd.DataFrame(embedded_chunks)
print(f"Created DataFrame with {len(df)} embedded chunks")

# Test search
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

    results = vector_similarity_search(query, df, client, k=10)

    for i, (idx, row) in enumerate(results.iterrows(), 1):
        score = row["similarity_score"]
        text_preview = row["text"][:150].replace("\n", " ")
        title = row["metadata"].get("title", "No Title")

        print(f"Result {i} (Score: {score:.4f}): {text_preview}...")
        print(f"  Section: {title}")
        print()

# Save results
df.to_csv("data/langchain_500_chunks_with_embeddings.csv", index=False)
print(f"\nSaved embeddings to data/langchain_500_chunks_with_embeddings.csv")
