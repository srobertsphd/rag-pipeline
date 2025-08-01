"""
Vector similarity search functionality.
Handles cosine similarity calculations and vector-based document retrieval.
"""

import numpy as np


def cosine_similarity_openai(query_vector, chunk_vectors):
    """
    Calculate cosine similarity for OpenAI embeddings (already normalized).

    Since OpenAI embeddings are pre-normalized to unit length,
    cosine similarity = dot product (much faster!)
    """
    similarities = np.dot(chunk_vectors, query_vector)
    return similarities


def vector_similarity_search(query_text, df, client, k=5):
    """
    Perform vector similarity search to find the most relevant chunks.

    Args:
        query_text: The search query
        df: DataFrame containing embedded chunks
        client: OpenAI client instance
        k: Number of top results to return

    Returns:
        DataFrame with top k most similar chunks and their similarity scores
    """
    # Import here to avoid circular imports
    from openai_services import embed_query

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
