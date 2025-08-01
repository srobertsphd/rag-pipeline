from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Sample markdown content with short and long paragraphs
sample_content = """
## Introduction

This is a short paragraph.

This is another short paragraph that's also brief.

## Methods

This is a much longer paragraph that contains significantly more content and goes into detail about the methodology used in this research. It discusses various aspects of the approach and provides comprehensive information about the techniques employed. This paragraph alone might be close to or exceed our minimum chunk size requirements.

This is a medium-length paragraph that contains a moderate amount of content. It provides some detail but isn't as comprehensive as the longer paragraphs we've seen before.

## Results

Short result paragraph.

Another brief result.

## Discussion

This is an extremely long paragraph that goes into extensive detail about the implications of the research findings. It covers multiple aspects of the results, discusses their significance in the broader context of the field, analyzes potential limitations of the study, and explores future directions for research. This paragraph contains substantial content that would definitely exceed most minimum chunk size requirements and demonstrates how the text splitter handles longer content sections. The discussion continues with analysis of the methodology and its effectiveness in achieving the research objectives.

Final short conclusion.
"""


def analyze_chunking_behavior(content, chunk_size, chunk_overlap=100):
    """Analyze how RecursiveCharacterTextSplitter handles different paragraph sizes."""

    print(f"\n{'=' * 60}")
    print(f"ANALYZING CHUNK SIZE: {chunk_size} characters")
    print(f"CHUNK OVERLAP: {chunk_overlap} characters")
    print("=" * 60)

    # Create document
    doc = Document(page_content=content)

    # Set up text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",  # Paragraphs first
            "\n",  # Then lines
            "## ",  # Headers
            ". ",  # Sentences
            " ",  # Words
            "",  # Characters
        ],
        length_function=len,
    )

    # Split the document
    chunks = text_splitter.split_documents([doc])

    print(f"Created {len(chunks)} chunks")

    # Analyze each chunk
    for i, chunk in enumerate(chunks):
        text = chunk.page_content
        size = len(text)

        # Count how many paragraphs are in this chunk
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        print(f"\n--- Chunk {i + 1} ({size} chars, {len(paragraphs)} paragraphs) ---")

        # Show first line to identify content
        first_line = text.split("\n")[0]
        print(f"Starts with: {first_line}")

        # Show if it's combining multiple paragraphs
        if len(paragraphs) > 1:
            print(f"✅ COMBINED {len(paragraphs)} paragraphs to reach target size")
            for j, p in enumerate(paragraphs):
                print(f"  Para {j + 1}: {len(p)} chars - {p[:50]}...")
        else:
            print(f"Single paragraph: {size} chars")

        print(f"Preview: {text[:100]}...")


def test_minimum_chunk_size_enforcement():
    """Test if we can enforce a true minimum chunk size."""

    print(f"\n{'=' * 80}")
    print("TESTING MINIMUM CHUNK SIZE ENFORCEMENT")
    print("=" * 80)

    # Try a very aggressive approach to ensure minimum sizes
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Target size
        chunk_overlap=50,  # Small overlap
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    doc = Document(page_content=sample_content)
    chunks = text_splitter.split_documents([doc])

    print(f"With target 800 chars, got {len(chunks)} chunks:")

    under_500 = 0
    for i, chunk in enumerate(chunks):
        size = len(chunk.page_content)
        status = "✅ Good size" if size >= 500 else f"❌ Too small ({size} < 500)"
        if size < 500:
            under_500 += 1
        print(f"Chunk {i + 1}: {size} chars - {status}")

    print(f"\nChunks under 500 chars: {under_500}/{len(chunks)}")


def custom_minimum_chunk_aggregator(chunks, min_size=500):
    """Custom function to ensure all chunks meet minimum size by combining small ones."""

    print(f"\n{'=' * 60}")
    print("CUSTOM MINIMUM SIZE ENFORCER")
    print("=" * 60)

    aggregated_chunks = []
    current_chunk_text = ""
    current_chunk_size = 0

    for i, chunk in enumerate(chunks):
        chunk_text = chunk.page_content
        chunk_size = len(chunk_text)

        # Add to current chunk
        if current_chunk_text:
            current_chunk_text += "\n\n" + chunk_text
        else:
            current_chunk_text = chunk_text
        current_chunk_size = len(current_chunk_text)

        # Check if we should finalize this chunk
        is_last_chunk = i == len(chunks) - 1
        next_would_exceed_max = False

        if not is_last_chunk:
            next_chunk_size = len(chunks[i + 1].page_content)
            next_would_exceed_max = (
                current_chunk_size + next_chunk_size
            ) > 1500  # Max size limit

        # Finalize chunk if: meets min size AND (is last OR next would be too big)
        if current_chunk_size >= min_size and (is_last_chunk or next_would_exceed_max):
            aggregated_chunks.append(
                {
                    "text": current_chunk_text,
                    "size": current_chunk_size,
                    "original_chunks": i - len([c for c in aggregated_chunks]) + 1,
                }
            )
            current_chunk_text = ""
            current_chunk_size = 0

    # Handle any remaining text
    if current_chunk_text:
        aggregated_chunks.append(
            {
                "text": current_chunk_text,
                "size": current_chunk_size,
                "original_chunks": len(chunks) - len(aggregated_chunks) + 1,
            }
        )

    print(
        f"Converted {len(chunks)} original chunks into {len(aggregated_chunks)} aggregated chunks"
    )

    for i, chunk in enumerate(aggregated_chunks):
        status = "✅ Good" if chunk["size"] >= min_size else "❌ Still too small"
        print(
            f"Aggregated chunk {i + 1}: {chunk['size']} chars from {chunk['original_chunks']} original chunks - {status}"
        )

    return aggregated_chunks


# Run the analysis
print("TESTING LANGCHAIN'S AUTOMATIC AGGREGATION BEHAVIOR")

# Test different chunk sizes
for size in [500, 1000, 1500]:
    analyze_chunking_behavior(sample_content, size)

# Test minimum size enforcement
test_minimum_chunk_size_enforcement()

# Test custom aggregation
doc = Document(page_content=sample_content)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,  # Intentionally small to create many chunks
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)
small_chunks = text_splitter.split_documents([doc])

print(f"\nStarting with {len(small_chunks)} small chunks:")
for i, chunk in enumerate(small_chunks):
    print(f"  Chunk {i + 1}: {len(chunk.page_content)} chars")

aggregated = custom_minimum_chunk_aggregator(small_chunks, min_size=500)
