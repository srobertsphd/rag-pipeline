"""
OpenAI API integration for embeddings and chat completions.
Handles all interactions with OpenAI's embedding and chat completion APIs.
"""


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


def embed_query(query_text, client):
    """Embed a single query using OpenAI's text-embedding-3-small model."""
    response = client.embeddings.create(
        model="text-embedding-3-small", input=query_text
    )
    return response.data[0].embedding


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

        return response

    except Exception as e:
        return f"Error generating response: {str(e)}"
