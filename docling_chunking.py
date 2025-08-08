from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
import textwrap

load_dotenv()


# Initialize OpenAI client (make sure you have OPENAI_API_KEY in your environment variables)
client = OpenAI()


# Create a proper OpenAI tokenizer class that implements BaseTokenizer
class OpenAITokenizer(BaseTokenizer):
    """OpenAI tokenizer implementation using tiktoken."""

    model_name: str = "gpt-4"
    max_tokens: int = 8191  # Default for text-embedding-3-large
    encoding: object = None  # Declare the encoding field

    def model_post_init(self, __context):
        """Initialize the tokenizer after Pydantic validation."""
        self.encoding = tiktoken.encoding_for_model(self.model_name)

    def count_tokens(self, text: str) -> int:
        """Get number of tokens for given text."""
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def get_max_tokens(self) -> int:
        """Get maximum number of tokens allowed."""
        return self.max_tokens

    def get_tokenizer(self):
        """Get underlying tokenizer object."""
        return self.encoding


# Initialize the OpenAI tokenizer
tokenizer = OpenAITokenizer(model_name="gpt-4", max_tokens=8191)
MAX_TOKENS = 8191  # text-embedding-3-large's maximum context length


# --------------------------------------------------------------
# Extract the data
# --------------------------------------------------------------

converter = DocumentConverter()
result = converter.convert("https://arxiv.org/pdf/2408.09869")


# --------------------------------------------------------------
# Apply hybrid chunking
# --------------------------------------------------------------

chunker = HybridChunker(
    tokenizer=tokenizer,  # Using our OpenAITokenizer instance
    max_tokens=MAX_TOKENS,
    merge_peers=True,
)

chunk_iter = chunker.chunk(dl_doc=result.document)
chunks = list(chunk_iter)

len(chunks)

processed_chunks = [
    {
        "text": chunk.text,
        "metadata": {
            "filename": chunk.meta.origin.filename,
            "page_numbers": [
                page_no
                for page_no in sorted(
                    set(
                        prov.page_no
                        for item in chunk.meta.doc_items
                        for prov in item.prov
                    )
                )
            ]
            or None,
            "title": chunk.meta.headings[0] if chunk.meta.headings else None,
        },
    }
    for chunk in chunks
]


# Set the width for text wrapping (adjust as needed)
width = 80

for chunk in processed_chunks:
    print(f"Character count: {len(chunk['text'])}")

    # Wrap the text to the specified width
    wrapped_text = textwrap.fill(chunk["text"], width=width)
    print(wrapped_text)

    print("#" * width)
