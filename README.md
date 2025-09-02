# 🤖 Multimodal RAG Chat with PDFs, Images & Tables

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39%2B-red)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-green)](https://langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A state-of-the-art **Multimodal RAG (Retrieval-Augmented Generation)** chatbot that intelligently processes and understands PDFs containing text, images, tables, and complex layouts. Built with Streamlit, LangChain, and powered by multiple AI models for comprehensive document analysis.

## 🌟 Key Features

### � **Advanced PDF Processing**
- **Text Extraction**: Natural language content with context preservation
- **Table Recognition**: Automatic detection and structured extraction of tables
- **Image Analysis**: OCR and visual understanding of charts, diagrams, and images
- **Layout Awareness**: Maintains document structure and relationships

### 🧠 **Intelligent RAG System**
- **Vector Storage**: FAISS-based similarity search for fast retrieval
- **Semantic Search**: Context-aware document chunk retrieval
- **Conversation Memory**: Maintains chat context across interactions
- **Smart Chunking**: Optimal text segmentation for better retrieval

### 💬 **Interactive Chat Interface**
- **Streaming Responses**: Real-time response generation (100 WPM)
- **Document Context**: Answers based on uploaded PDF content
- **Conversation History**: Maintains context across multiple exchanges
- **Error Handling**: Robust fallback mechanisms

### 🔧 **AI-Powered Analysis**
- **Groq Integration**: Lightning-fast text generation with Llama models
- **Vision Models**: Google Gemini & OpenAI GPT-4V for image understanding
- **Multi-Modal Understanding**: Combines text, images, and tables for comprehensive answers

## 🚀 Quick Start

### Prerequisites

**System Requirements:**
- Python 3.8+
- macOS, Linux, or Windows
- 4GB+ RAM recommended
- Internet connection for AI APIs

**macOS System Dependencies:**
```bash
# Install system dependencies via Homebrew
brew install poppler tesseract libmagic pkg-config
```

### 1. Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/tarek3203/Multimodal-RAG-Chatbot.git
cd Multimodal-RAG-Chatbot

# Create virtual environment named 'tariq'
python -m venv tariq

# Activate virtual environment
# macOS/Linux:
source tariq/bin/activate
# Windows:
# tariq\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 3. Configure API Keys

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
nano .env  # or use any text editor
```

**Required API Keys:**
- **Groq API Key** (Required): Get from [Groq Console](https://console.groq.com/)
- **Google API Key** (Optional): Get from [Google AI Studio](https://aistudio.google.com/)
- **OpenAI API Key** (Optional): Get from [OpenAI Platform](https://platform.openai.com/)

### 4. Launch the Application

```bash
# Run the Streamlit app
streamlit run app.py
```

**The app will open in your browser at `http://localhost:8501`**

## 📖 Usage Guide

### Step 1: Upload Documents
1. Use the sidebar to upload PDF files
2. Files are automatically processed upon upload
3. Progress indicators show processing status
4. Processed documents appear in the "Processed Documents" list

### Step 2: Chat with Your Documents
1. Type questions in the chat input
2. The system retrieves relevant content from your PDFs
3. Responses stream in real-time with context from your documents
4. Ask follow-up questions - the system maintains conversation context

### Step 3: Manage Your Session
- **Clear Chat**: Reset conversation history
- **Clear Docs**: Remove all processed documents
- **Upload More**: Add additional PDFs anytime

## 🏗️ Project Structure

```
Multimodal-RAG-Chatbot/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration settings
├── prompts.py                  # AI prompt templates
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── README.md                  # This file
├── setup.sh                   # Automated setup script
│
├── services/                   # Core processing services
│   ├── __init__.py
│   ├── pdf_processor.py        # PDF multimodal processing
│   └── rag_system.py          # RAG implementation
│
├── vector_services/           # Vector storage & embeddings
│   ├── __init__.py
│   ├── embeddings.py          # Embedding service
│   └── faiss_manager.py       # FAISS vector store
│
├── data/                      # Document storage
│   └── processed_pdf/         # Processed document cache
│
├── vector_storage/            # Vector database files
├── models/                    # Model cache
└── tariq/                     # Virtual environment
```

## ⚙️ Configuration

### Environment Variables (.env)

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional (for enhanced image analysis)
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Model Settings
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Processing Limits
MAX_PDF_PAGES=20
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Supported Models

**Text Generation (Groq):**
- `llama-3.3-70b-versatile` (Recommended)
- `llama-3.1-8b-instant`
- `mixtral-8x7b-32768`

**Vision Analysis:**
- Google Gemini 1.5 Pro (Primary)
- OpenAI GPT-4V (Fallback)

**Embeddings:**
- `sentence-transformers/all-MiniLM-L6-v2` (Default)
- `sentence-transformers/all-mpnet-base-v2`

## 🔧 Advanced Features

### Streaming Response Control
Adjust typing speed in `app.py`:
```python
# Current: 100 WPM
delay = min(words_in_chunk * 0.06, 0.25)

# For 80 WPM: 0.075
# For 120 WPM: 0.05
```

### Custom Prompt Templates
Modify AI behavior in `prompts.py`:
- RAG query prompts
- General chat prompts
- Document summarization prompts

### Vector Store Configuration
Adjust retrieval settings in `config.py`:
- Chunk size and overlap
- FAISS index parameters
- Embedding model selection

## 🛠️ Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure virtual environment is activated
source tariq/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**2. Missing System Dependencies (macOS)**
```bash
# Install missing system libraries
brew install poppler tesseract libmagic
```

**3. API Key Issues**
- Verify keys in `.env` file
- Check API key validity
- Ensure sufficient API credits

**4. Memory Issues**
- Reduce `MAX_PDF_PAGES` in `.env`
- Use smaller embedding models
- Process fewer documents simultaneously

**5. Slow Performance**
- Check internet connection
- Use faster Groq models
- Reduce chunk size

### Debug Mode
Enable verbose logging by setting in `rag_system.py`:
```python
verbose=True  # Line 74
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links & Resources

- **Groq API**: [console.groq.com](https://console.groq.com/)
- **Google AI Studio**: [aistudio.google.com](https://aistudio.google.com/)
- **LangChain Documentation**: [docs.langchain.com](https://docs.langchain.com/)
- **Streamlit Documentation**: [docs.streamlit.io](https://docs.streamlit.io/)

## 🙏 Acknowledgments

- **LangChain** for the RAG framework
- **Groq** for lightning-fast inference
- **Unstructured** for comprehensive PDF processing
- **FAISS** for efficient vector similarity search
- **Streamlit** for the intuitive web interface

---

**Built with ❤️ for intelligent document interaction**

*For questions, issues, or feature requests, please open an issue on GitHub.*
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