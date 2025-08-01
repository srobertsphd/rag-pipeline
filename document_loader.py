"""
Document loading and data structure management.
Handles file I/O and converting between different document formats.
"""

from langchain_core.documents import Document
from docling.document_converter import DocumentConverter


def convert_pdf_to_markdown(pdf_path, output_path="pdf_output.md"):
    """
    Convert PDF to Markdown using Docling.

    Args:
        pdf_path: Path to the PDF file
        output_path: Path for the output Markdown file

    Returns:
        Path to the generated Markdown file
    """
    # Initialize Docling converter
    converter = DocumentConverter()

    # Convert PDF
    result = converter.convert(pdf_path)

    # Export to Markdown
    markdown_content = result.document.export_to_markdown()

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"✅ PDF converted to Markdown: {output_path}")
    return output_path


def load_markdown_document(file_path):
    """Load the markdown file as a LangChain Document."""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Create a Document object with metadata
    doc = Document(
        page_content=content, metadata={"source": file_path, "type": "markdown"}
    )
    return doc


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
