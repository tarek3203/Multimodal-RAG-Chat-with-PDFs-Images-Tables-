# 🤖 Multimodal PDF RAG Chatbot

A powerful conversational AI chatbot that can analyze PDF documents using advanced multimodal capabilities. The system extracts and understands text, tables, and images from PDFs, then allows natural language querying through a local RAG (Retrieval Augmented Generation) system.

## ✨ Features

### 🔍 Advanced PDF Processing
- **Text Extraction**: Direct text extraction and OCR capabilities
- **Table Analysis**: Smart table detection and structured data extraction
- **Image Understanding**: Vision AI analysis of charts, diagrams, and embedded images
- **Multimodal Integration**: Unified understanding across all content types

### 💬 Intelligent Chat Interface
- **Context-Aware Responses**: Maintains conversation history and context
- **Source Attribution**: Tracks which documents and content types inform each answer
- **Multimodal Retrieval**: Searches across text, table, and image content simultaneously
- **Natural Language Processing**: Understands complex queries about document content

### 🏠 Local-First Architecture
- **FAISS Vector Store**: Local vector database with no external dependencies
- **Privacy Focused**: Documents never leave your system for vector storage
- **Fast Performance**: Local embeddings and retrieval for quick responses
- **Scalable**: Handle multiple documents with efficient indexing

## 🏗️ System Architecture

```
PDF Upload → Unstructured Processing → Content Analysis → Vector Storage → Chat Interface
     ↓              ↓                      ↓               ↓              ↓
[PDF Files] → [Text/Tables/Images] → [AI Summaries] → [FAISS Index] → [Groq LLM]
```

### Core Components

1. **PDF Processor** (`services/pdf_processor.py`)
   - Uses Unstructured library for comprehensive extraction
   - Separates text, tables, and images
   - Generates AI-powered summaries for each content type

2. **Multimodal RAG System** (`services/rag_system.py`) 
   - FAISS vector store for local document indexing
   - Semantic search across all content types
   - Context synthesis and response generation

3. **Embedding Service** (`vector_services/embeddings.py`)
   - HuggingFace sentence transformers for embeddings
   - Document preparation and linking
   - Mac M1 optimized (CPU-based for stability)

4. **Prompt Management** (`prompts.py`)
   - Centralized prompt templates
   - Optimized for different content types
   - Context-aware conversation handling

## 🚀 Quick Start

### Prerequisites

**System Dependencies** (install via your system package manager):
```bash
# macOS
brew install poppler tesseract libmagic

# Ubuntu/Debian
sudo apt-get install poppler-utils tesseract-ocr libmagic-dev

# Windows - see unstructured documentation for setup
```

**API Keys Required**:
- **GROQ_API_KEY** (required) - Get from [console.groq.com](https://console.groq.com)
- **GOOGLE_API_KEY** or **OPENAI_API_KEY** (recommended) - For image analysis

### Installation

1. **Clone and Setup Environment**
   ```bash
   git clone <your-repo-url>
   cd multimodal-pdf-rag-chatbot
   python -m venv venv
   
   # Activate virtual environment
   # Windows: venv\Scripts\activate
   # macOS/Linux: source venv/bin/activate
   
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**
   
   Create a `.env` file in the project root:
   ```env
   # Required for text generation and summarization
   GROQ_API_KEY=your_groq_api_key_here
   
   # Vision capabilities (choose one or both)
   GOOGLE_API_KEY=your_google_api_key_here      # Recommended: Cheaper, fast
   OPENAI_API_KEY=your_openai_api_key_here      # Fallback: More expensive
   
   # Optional: Customize models
   GROQ_MODEL=llama-3.1-8b-instant
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   MAX_PDF_PAGES=20
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

The app will open at `http://localhost:8501`

## 📱 How to Use

### 1. Upload Documents
- Use the **sidebar** to upload PDF files
- Click "🚀 Process Files" to analyze uploaded documents
- Monitor processing progress and content extraction details

### 2. Chat with Your Documents
- Ask questions in the **main chat area**
- The system will search across all content types (text, tables, images)
- Get responses with source attribution and content type information

### 3. Example Queries
```
"What are the main findings in the research paper?"
"Show me the financial data from the quarterly report"
"What does the diagram on page 5 illustrate?"
"Compare the revenue figures across different years"
"Summarize the key points from all uploaded documents"
```

## ⚙️ Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | - | **Required** - Groq API key for text generation |
| `GOOGLE_API_KEY` | - | Google API key for Gemini vision analysis |
| `OPENAI_API_KEY` | - | OpenAI API key (fallback for vision) |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model for text generation |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `MAX_PDF_PAGES` | `20` | Maximum pages to process per PDF |
| `CHUNK_SIZE` | `1000` | Text chunk size for vector storage |
| `VISION_MODEL_PRIORITY` | `gemini,openai` | Order to try vision APIs |

### Advanced Configuration

Edit `config.py` to customize:
- Vector store settings
- Processing limits  
- Model configurations
- File paths and directories

## 🧠 AI Models Used

### Text Generation & Summarization
- **Groq Llama-3.1 8B**: Fast, efficient text generation and summarization
- **Cost**: ~$0.05 per 1M tokens (very affordable)

### Vision Analysis (Choose One)
- **Google Gemini 1.5 Flash**: Preferred for image analysis
  - **Cost**: ~$0.075 per 1M tokens (27x cheaper than OpenAI)
  - **Speed**: Very fast processing
  - **Capability**: Excellent for charts, diagrams, and document images

- **OpenAI GPT-4V**: Fallback option
  - **Cost**: ~$2.50 per 1M tokens
  - **Capability**: Highest quality vision analysis

### Embeddings
- **HuggingFace Sentence Transformers**: Local, free embeddings
- **Model**: `all-MiniLM-L6-v2` (384 dimensions, good performance)

## 📊 System Monitoring

The sidebar shows real-time system status:
- **Documents**: Number of processed documents in vector store
- **Content Types**: Breakdown of text, tables, and images
- **API Status**: Which APIs are configured and available
- **Conversation Length**: Current chat history size

## 🔧 Troubleshooting

### Common Issues

1. **"No vision model available"**
   - Add `GOOGLE_API_KEY` or `OPENAI_API_KEY` to your `.env` file
   - Image analysis will be limited without vision APIs

2. **PDF processing fails**
   - Ensure system dependencies are installed (`poppler`, `tesseract`)
   - Check PDF file isn't password protected or corrupted

3. **Out of memory errors**
   - Reduce `MAX_PDF_PAGES` in config
   - Process smaller files or fewer files at once

4. **Slow performance**
   - Vision analysis can be slow - consider using Gemini over OpenAI
   - Reduce concurrent processing in config

### Performance Tips

- **Use Google Gemini** for vision analysis (much faster and cheaper than OpenAI)
- **Process documents in batches** rather than all at once
- **Clear old documents** periodically to reduce vector store size
- **Monitor API usage** to avoid rate limits

## 📁 Project Structure

```
multimodal-pdf-rag-chatbot/
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration management
├── prompts.py                      # Centralized prompt templates
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (create this)
├── services/
│   ├── pdf_processor.py           # Multimodal PDF processing
│   └── rag_system.py             # Enhanced RAG pipeline
├── vector_services/
│   ├── embeddings.py              # Embedding service
│   └── faiss_manager.py          # FAISS vector store management
├── vector_storage/                # Local vector database (auto-created)
└── models/                       # Model cache (auto-created)
```

## 🔄 Upgrading from Previous Versions

If you have an existing system:

1. **Backup your data** (vector_storage folder)
2. **Update dependencies**: `pip install -r requirements.txt`
3. **Add new environment variables** to your `.env` file
4. **Clear existing documents** if you encounter compatibility issues

The new system maintains FAISS compatibility but uses enhanced document processing.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Unstructured.io** for comprehensive PDF processing
- **LangChain** for the RAG framework
- **Meta AI** for Llama models via Groq
- **Google** for Gemini vision capabilities
- **Streamlit** for the user interface

---

**Built with ❤️ for document intelligence and conversational AI**