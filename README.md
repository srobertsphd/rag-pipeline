# Modular RAG (Retrieval-Augmented Generation) System

A comprehensive, educational implementation of a RAG pipeline using Docling, LangChain, OpenAI, and vector similarity search. This project is designed to help students understand the components and data flow of modern RAG systems through clean, modular code.

## 🎯 Overview

This project demonstrates how to build a complete RAG system by breaking it down into clear, understandable modules. Each module handles a specific aspect of the pipeline, making it easy to learn, modify, and extend.

### What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that combines:
- **Retrieval**: Finding relevant information from a knowledge base
- **Generation**: Using that information to generate accurate, contextual responses

Our pipeline: `Document → Chunking → Embedding → Search → Response`

## 📁 Project Structure

```
├── README.md                    # This comprehensive guide
├── document_loader.py           # 📁 Document loading and data handling
├── chunking.py                  # 🔀 LangChain text processing
├── openai_services.py           # 🧠 OpenAI API integration
├── vector_search.py             # 🔍 Vector similarity search
├── main.py                      # 💬 Interactive orchestration
├── langchain_simple_chunking.py # Original monolithic version
├── pdf_output.md               # Sample document (Docling technical paper)
└── data/                       # Generated embeddings and results
```

## 🧩 Module Architecture

### **1. `document_loader.py` - Document & Data Handling**
**Purpose:** File I/O and data structure management

```python
def convert_pdf_to_markdown(pdf_path, output_path="pdf_output.md")
def load_markdown_document(file_path)
def create_processed_chunks(chunks, source_file)
```

**What it does:**
- Converts PDF documents to Markdown using Docling's DocumentConverter
- Loads markdown files into LangChain Document format
- Converts LangChain chunks into standardized dictionaries
- Handles metadata management (source, chunk_id, chunk_size)
- Provides clean data structures for the pipeline

**Key Concepts for Students:**
- Data normalization and standardization
- Metadata preservation across transformations
- Clean interfaces between system components

---

### **2. `chunking.py` - LangChain Text Processing**
**Purpose:** Intelligent text splitting using LangChain's algorithms

```python
def chunk_markdown_with_langchain(document, chunk_size=1000, chunk_overlap=200)
```

**What it does:**
- Uses RecursiveCharacterTextSplitter for smart text division
- Preserves semantic boundaries (paragraphs, sentences, words)
- Handles markdown-specific separators (headers, sections)
- Maintains document structure and context

**Chunking Strategy (in order of preference):**
1. `\n\n` - Paragraphs (preserves complete thoughts)
2. `\n` - Single newlines (maintains line structure)
3. `## `, `### ` - Markdown headers (keeps sections together)
4. `. ` - Sentences (natural language boundaries)
5. ` ` - Words (fallback for long content)
6. `""` - Characters (last resort)

**Key Concepts for Students:**
- Semantic vs. arbitrary chunking
- Trade-offs between chunk size and context preservation
- How text splitters maintain document structure

---

### **3. `openai_services.py` - OpenAI API Integration**
**Purpose:** All interactions with OpenAI's embedding and chat APIs

```python
def embed_chunks_with_openai(chunks, client)
def embed_query(query_text, client)  
def generate_response_with_context(user_prompt, retrieved_chunks, client)
```

**What it does:**
- **Embedding Generation**: Converts text to numerical vectors using `text-embedding-3-small`
- **Query Embedding**: Creates searchable vectors for user questions
- **Response Generation**: Uses `gpt-4o-mini` for contextual chat completions
- **Error Handling**: Graceful API failure management

**Embedding Details:**
- Model: `text-embedding-3-small` (1536 dimensions)
- Pre-normalized vectors (cosine similarity = dot product)
- Batch processing for efficiency
- Consistent model usage for queries and chunks

**Chat Completion Setup:**
- Model: `gpt-4o-mini` (cost-effective, high-quality)
- System prompt engineering for RAG context
- Temperature: 0.1 (focused, consistent responses)
- Max tokens: 4000 (comprehensive answers)

**Key Concepts for Students:**
- Vector embeddings and semantic similarity
- API design patterns and error handling
- Prompt engineering for RAG systems
- Cost-effective model selection

---

### **4. `vector_search.py` - Vector Similarity Search**
**Purpose:** Finding relevant chunks using cosine similarity

```python
def cosine_similarity_openai(query_vector, chunk_vectors)
def vector_similarity_search(query_text, df, client, k=5)
```

**What it does:**
- **Optimized Similarity**: Leverages OpenAI's pre-normalized embeddings
- **Efficient Search**: Uses NumPy for fast vector operations
- **Ranking**: Returns top-k most relevant chunks
- **Score Transparency**: Provides similarity scores for analysis

**Mathematical Foundation:**
```python
# For normalized vectors: cosine_similarity = dot_product
similarities = np.dot(chunk_vectors, query_vector)
```

**Search Pipeline:**
1. Embed the user query
2. Calculate similarities with all chunk embeddings
3. Rank by similarity score (higher = more relevant)
4. Return top-k results with metadata

**Key Concepts for Students:**
- Vector similarity mathematics
- Computational efficiency optimizations
- Information retrieval ranking
- NumPy operations for ML

---

### **5. `main.py` - Interactive Orchestration**
**Purpose:** Demonstrates the complete RAG pipeline step-by-step

**Educational Flow:**
```python
# Step 1: Document Loading
document = load_markdown_document(filename)

# Step 2: Document Chunking  
chunks = chunk_markdown_with_langchain(document)
processed_chunks = create_processed_chunks(chunks, filename)

# Step 3: Embedding Generation
embedded_chunks = embed_chunks_with_openai(processed_chunks, client)

# Step 4: Search Quality Testing
results = vector_similarity_search(query, df, client, k=3)

# Step 5: Complete RAG Pipeline
response = generate_response_with_context(query, retrieved_texts, client)
```

**What it demonstrates:**
- End-to-end RAG pipeline execution
- Module integration patterns
- Error handling and logging
- Performance monitoring
- Result analysis and interpretation

**Key Concepts for Students:**
- System integration and orchestration
- Pipeline debugging and monitoring
- Data flow through complex systems
- Interactive development patterns

## 🔄 Data Flow Architecture

```
📄 PDF Document (2408.09869v5.pdf)
         ↓
📁 document_loader.py: load_markdown_document()
         ↓ 
📋 LangChain Document Object
         ↓
🔀 chunking.py: chunk_markdown_with_langchain()
         ↓
📑 List of Text Chunks
         ↓
📁 document_loader.py: create_processed_chunks()
         ↓
📊 Standardized Chunk Dictionaries
         ↓
🧠 openai_services.py: embed_chunks_with_openai()
         ↓
🔢 Vector Embeddings (1536-dimensional)
         ↓
💾 Pandas DataFrame Storage
         ↓
❓ User Query → 🧠 openai_services.py: embed_query()
         ↓
🔍 vector_search.py: vector_similarity_search()
         ↓
📋 Top-K Relevant Chunks
         ↓
💬 openai_services.py: generate_response_with_context()
         ↓
✨ Final AI Response
```

## 🚀 Getting Started

### Prerequisites

```bash
# Install dependencies
uv sync

# Or with pip
pip install docling langchain-core langchain-text-splitters openai pandas numpy python-dotenv
```

### Environment Setup

Create a `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Quick Start

```python
# Interactive execution
exec(open('main.py').read())

# Or run individual modules
from document_loader import load_markdown_document
from chunking import chunk_markdown_with_langchain
from openai_services import embed_chunks_with_openai
from vector_search import vector_similarity_search
```

## 📚 Educational Learning Path

### For Students New to RAG:

#### **Level 1: Understanding Components**
1. **Start with `document_loader.py`**
   - Learn about data preparation and standardization
   - Understand metadata management
   - Practice file I/O patterns

2. **Explore `chunking.py`**
   - Understand text splitting strategies
   - Learn about semantic boundaries
   - Experiment with different chunk sizes

#### **Level 2: AI Integration**
3. **Study `openai_services.py`**
   - Learn about embedding generation
   - Understand API integration patterns
   - Practice prompt engineering

4. **Examine `vector_search.py`**
   - Learn vector similarity concepts
   - Understand search and ranking
   - Practice NumPy operations

#### **Level 3: System Integration**
5. **Run `main.py`**
   - See complete pipeline in action
   - Understand system orchestration
   - Learn debugging techniques

### **Learning Exercises**

#### **Beginner:**
- Modify chunk sizes and observe effects on search quality
- Try different queries and analyze similarity scores
- Experiment with different OpenAI models

#### **Intermediate:**
- Implement alternative similarity metrics
- Add new metadata fields to chunks
- Create custom evaluation metrics

#### **Advanced:**
- Implement hybrid search (keyword + semantic)
- Add document filtering capabilities
- Optimize for different document types

## 🔧 Configuration Options

### **Chunking Parameters**
```python
chunk_markdown_with_langchain(
    document,
    chunk_size=1000,    # Target characters per chunk
    chunk_overlap=200   # Overlap between adjacent chunks
)
```

**Recommendations:**
- **Small documents**: 300-500 characters
- **Medium documents**: 500-1000 characters  
- **Large documents**: 1000-1500 characters
- **Overlap**: 10-20% of chunk_size

### **Search Parameters**
```python
vector_similarity_search(
    query_text,
    df, 
    client,
    k=5    # Number of chunks to retrieve
)
```

**Recommendations:**
- **Simple queries**: k=3-5
- **Complex queries**: k=5-10
- **Analysis tasks**: k=10-20

### **Response Generation**
```python
client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.1,     # Lower = more focused
    max_tokens=4000      # Response length limit
)
```

## 📊 Performance Considerations

### **Embedding Costs** (OpenAI text-embedding-3-small)
- Cost: ~$0.00002 per 1K tokens
- Speed: ~1000 chunks/minute
- Storage: 1536 float32 per chunk

### **Chat Completion Costs** (GPT-4o-mini)
- Cost: ~$0.00015 per 1K input tokens
- Cost: ~$0.0006 per 1K output tokens
- Speed: ~20-50 tokens/second

### **Optimization Tips**
- Cache embeddings in CSV/database
- Batch API calls when possible
- Use appropriate chunk sizes for your use case
- Monitor API usage and costs

## 🧪 Testing and Evaluation

### **Sample Queries for Testing**
```python
test_queries = [
    "What is Docling and what does it do?",
    "How does Docling process PDF documents?", 
    "What AI models does Docling use?",
    "How to install and use Docling?",
    "What are the main features of Docling?"
]
```

### **Evaluation Metrics**
- **Similarity Scores**: Higher scores indicate better matches
- **Response Quality**: Factual accuracy and completeness
- **Relevance**: How well chunks answer the question
- **Coverage**: Whether important information is retrieved

### **Quality Indicators**
- Similarity scores > 0.5 are typically good matches
- Similarity scores > 0.7 are very strong matches
- Scores < 0.3 may indicate poor chunk relevance

## 🔍 Troubleshooting

### **Common Issues**

#### **Low Similarity Scores**
- **Cause**: Chunks too small or query mismatch
- **Solution**: Increase chunk size or refine queries

#### **Irrelevant Results**
- **Cause**: Poor chunking or broad queries
- **Solution**: Adjust chunking strategy or be more specific

#### **API Errors**
- **Cause**: Rate limits or invalid API keys
- **Solution**: Check credentials and implement rate limiting

#### **Memory Issues**
- **Cause**: Too many embeddings in memory
- **Solution**: Process in batches or use disk storage

### **Debugging Tips**
- Check similarity scores for retrieved chunks
- Examine chunk content and metadata
- Test individual modules in isolation
- Monitor API usage and responses

## 🚀 Extension Ideas

### **Advanced Features**
- **Hybrid Search**: Combine semantic and keyword search
- **Multi-document**: Handle multiple source documents
- **Streaming**: Real-time response generation
- **Evaluation**: Automated quality assessment

### **Alternative Models**
- **Embeddings**: Try `text-embedding-3-large` for better quality
- **Chat**: Experiment with `gpt-4` for complex reasoning
- **Local**: Use Ollama for on-premises deployment

### **Production Enhancements**
- **Caching**: Redis for embedding storage
- **Database**: PostgreSQL with pgvector
- **API**: FastAPI for web service deployment
- **Monitoring**: Logging and metrics collection

## 📖 Additional Resources

### **Key Concepts Explained**
- **Vector Embeddings**: Numerical representations of text meaning
- **Cosine Similarity**: Measure of vector alignment (0-1 scale)
- **Semantic Search**: Finding meaning-based matches vs. keyword matching
- **Context Window**: Amount of text an AI model can process at once
- **Temperature**: Controls randomness in AI responses (0=deterministic, 1=creative)

### **Further Reading**
- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [Docling Documentation](https://docling-project.github.io/)
- [RAG Papers and Research](https://arxiv.org/abs/2005.11401)

## 🤝 Contributing

This project is designed for educational purposes. Students and educators are encouraged to:
- Experiment with different configurations
- Add new features and modules
- Share improvements and extensions
- Create tutorials and documentation

## 📄 License

This project is provided for educational use. Please respect API terms of service and usage limits.

---

**Happy Learning! 🎉**

*This modular RAG system provides a solid foundation for understanding and building production-ready retrieval-augmented generation applications.*