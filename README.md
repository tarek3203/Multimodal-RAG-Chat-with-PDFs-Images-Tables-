# 🤖 AI Chatbot with RAG (Retrieval Augmented Generation)

A modern conversational AI chatbot that can chat normally and analyze PDF documents using advanced RAG techniques. Features a Claude-like interface where users can attach documents directly in the conversation.

## ✨ Features

- **💬 Normal Conversation**: Chat with AI without any documents
- **📄 PDF Analysis**: Upload PDFs and ask questions about their content
- **🔍 Advanced Processing**: Extracts text, images, tables from PDFs using GPT-4 Vision
- **🧠 Conversational Memory**: Maintains context throughout the conversation
- **⚡ Fast Responses**: Uses Groq for lightning-fast inference
- **🎯 Smart Interface**: Claude-like chat interface with document attachment

## 🛠️ Technology Stack

- **Frontend**: Streamlit for modern web UI
- **LLM**: Groq Mixtral-8x7B for fast chat responses
- **Vision**: OpenAI GPT-4o for PDF image processing
- **Vector Store**: Pinecone for document embeddings and retrieval
- **PDF Processing**: PyPDF2 + pdf2image for comprehensive extraction

## 📦 Installation

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd rag-chatbot
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the root directory:

```env
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# Pinecone Configuration
PINECONE_INDEX_NAME=rag-chatbot-index
```

### 3. Get API Keys

1. **OpenAI API Key**: https://platform.openai.com/api-keys
2. **Groq API Key**: https://console.groq.com/keys
3. **Pinecone API Key**: https://app.pinecone.io/

### 4. Setup Pinecone Index

1. Go to https://app.pinecone.io/
2. Create a new index with these settings:
   - **Name**: `rag-chatbot-index` (or match your `.env` file)
   - **Dimensions**: `1536` (for OpenAI embeddings)
   - **Metric**: `cosine`
   - **Cloud**: Choose your preferred region

## 🚀 Usage

### Start the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### How to Use

1. **Normal Chat**: Just type your message and press Enter - chat with the AI normally
2. **Upload Documents**: 
   - Click "📎 Attach Documents" above the chat input
   - Upload one or more PDF files
   - Click "📤 Upload & Process Files"
3. **Ask Questions**: Once documents are uploaded, ask questions about their content
4. **Mixed Conversation**: You can switch between normal chat and document questions seamlessly

### Example Interactions

```
User: Hello! How are you?
AI: Hello! I'm doing great, thank you for asking! How can I help you today?

User: [uploads a research paper PDF]
AI: Great! I've processed your document. You can now ask me questions about its content.

User: What is the main conclusion of this paper?
AI: Based on the document, the main conclusion is... [analysis based on PDF content]

User: Thanks! Can you also tell me about the weather today?
AI: I don't have access to current weather data, but I can help you with...
```

## 🔧 Configuration

### PDF Processing Settings

The app automatically:
- Tries text extraction first (fast)
- Falls back to vision processing for image-based PDFs
- Processes up to 15 pages with vision (configurable in `pdf_processor.py`)
- Optimizes images for better performance

### Memory Management

- Keeps last 10 conversation exchanges in memory
- Uses last 3 exchanges for conversation context
- Automatically manages context window size

## 🐛 Troubleshooting

### Common Issues

1. **"st.experimental_rerun not found"**
   - Fixed in the updated code (now uses `st.rerun()`)

2. **LangChain deprecation warnings**
   - Updated to use modern LangChain patterns without deprecated memory classes

3. **PDF processing fails**
   - Check if you have the required system dependencies for pdf2image
   - On macOS: `brew install poppler`
   - On Ubuntu: `sudo apt-get install poppler-utils`
   - On Windows: Download poppler binary

4. **Pinecone connection errors**
   - Verify your API key and index name in `.env`
   - Ensure your index has the correct dimensions (1536)

5. **OpenAI API errors**
   - Check your API key and billing status
   - Ensure you have access to GPT-4o model

## 🔄 Recent Updates

- ✅ Fixed `st.experimental_rerun()` deprecation
- ✅ Replaced deprecated LangChain memory with simple conversation management
- ✅ Added Claude-like interface with document attachment in chat
- ✅ Improved PDF processing with intelligent fallback
- ✅ Updated to latest model versions (GPT-4o, Mixtral-8x7B)
- ✅ Enhanced error handling and user feedback

## 📝 File Structure

```
rag-chatbot/
├── app.py                 # Main Streamlit application
├── rag_system.py         # RAG logic and conversation management
├── pdf_processor.py      # PDF text and vision extraction
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
└── README.md            # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 💡 Tips for Best Results

1. **For better PDF processing**: Use PDFs with clear, readable text
2. **For vision processing**: Ensure images in PDFs are high quality
3. **For conversations**: Be specific in your questions about documents
4. **For performance**: Clear chat memory periodically for very long conversations

---

*Built with ❤️ using Streamlit, LangChain, OpenAI, Groq, and Pinecone*