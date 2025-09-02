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

I welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 🔗 Links & Resources

- **Groq API**: [console.groq.com](https://console.groq.com/)
- **Google AI Studio**: [aistudio.google.com](https://aistudio.google.com/)
- **LangChain Documentation**: [docs.langchain.com](https://docs.langchain.com/)
- **Streamlit Documentation**: [docs.streamlit.io](https://docs.streamlit.io/)

##  Acknowledgments

- **LangChain** for the RAG framework
- **Groq** for lightning-fast inference
- **Unstructured** for comprehensive PDF processing
- **FAISS** for efficient vector similarity search
- **Streamlit** for the intuitive web interface

---

*For questions, issues, or feature requests, please open an issue on GitHub.*
   - Centralized prompt templates
   - Optimized for different content types
   - Context-aware conversation handling

