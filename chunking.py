"""
LangChain text processing and chunking functionality.
Handles intelligent text splitting using LangChain's RecursiveCharacterTextSplitter.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter


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
