# RAG PDF Chatbot

A conversational AI chatbot that processes PDF documents and answers questions based on their content using Retrieval Augmented Generation (RAG).

## Features

- **PDF Processing**: Extract text, images, and tables from PDFs using LLM vision capabilities
- **Conversational Memory**: Maintains chat history for context-aware responses
- **Multiple PDF Support**: Upload and process multiple PDF documents
- **Real-time Chat**: Streamlit-based chat interface
- **Advanced RAG**: Uses Pinecone vector database and Groq/OpenAI models

## Setup Instructions

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd rag_chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt