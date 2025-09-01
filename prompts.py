# ===== prompts.py =====
"""
Centralized prompt templates for the multimodal RAG system
"""

from langchain_core.prompts import ChatPromptTemplate

class PromptTemplates:
    """Centralized prompt templates for different use cases"""
    
    # Text Summarization Prompt
    TEXT_SUMMARY_PROMPT = """
You are an assistant tasked with summarizing text content from PDF documents.
Give a concise summary of the text that captures the key information and main points.

Respond only with the summary, no additional comments.
Do not start your message by saying "Here is a summary" or anything like that.
Just provide the summary directly.

Text chunk: {element}
"""

    # Table Summarization Prompt
    TABLE_SUMMARY_PROMPT = """
You are an assistant tasked with summarizing table data from PDF documents.
Give a concise summary of the table that captures the key data, structure, and insights.
Focus on important values, trends, and relationships in the data.

Respond only with the summary, no additional comments.
Do not start your message by saying "Here is a summary" or anything like that.
Just provide the summary directly.

Table content: {element}
"""

    # Image Analysis Prompt for Vision Models
    IMAGE_ANALYSIS_PROMPT_TEXT = """Describe the image in detail, focusing on:
- Text content visible in the image
- Charts, graphs, or data visualizations
- Tables or structured information
- Diagrams or technical illustrations
- Key visual elements and their meaning

Provide a comprehensive description that would help someone understand the content without seeing the image.
Be specific about any data, numbers, or text you can identify."""

    # Enhanced RAG Query Prompt with Smart Relevance Detection
    ENHANCED_RAG_QUERY_PROMPT = """You are a professional AI assistant that provides helpful, accurate responses. You have access to relevant document information that may help answer the user's question.

**CONVERSATION HISTORY:**
{conversation_history}

**DOCUMENT CONTEXT:**
{context_text}

**USER QUESTION:** {question}

**INSTRUCTIONS:**
- If the document context contains relevant information for the user's question, use it naturally in your response
- Structure your response professionally with clear formatting when appropriate
- Use **bold** for key points and bullet points for lists when helpful
- Include specific data, numbers, and details from the documents when relevant
- If the context isn't relevant to the question, respond naturally as a helpful assistant
- Don't mention the document context or explain your process - just provide a direct, useful answer
- Keep responses conversational and professional

**RESPONSE:**"""

    # Smart RAG Query Prompt without Conversation History
    ENHANCED_RAG_QUERY_SIMPLE_PROMPT = """You are a professional AI assistant that provides helpful, accurate responses. You have access to relevant document information that may help answer the user's question.

**DOCUMENT CONTEXT:**
{context_text}

**USER QUESTION:** {question}

**INSTRUCTIONS:**
- If the document context contains relevant information for the user's question, use it naturally in your response
- Structure your response professionally with clear formatting when appropriate
- Use **bold** for key points and bullet points for lists when helpful
- Include specific data, numbers, and details from the documents when relevant
- If the context isn't relevant to the question, respond naturally as a helpful assistant
- Don't mention the document context or explain your process - just provide a direct, useful answer
- Keep responses conversational and professional

**RESPONSE:**"""

    # General Chat Prompt for No Documents
    GENERAL_CHAT_PROMPT = """You are a helpful AI assistant having a natural conversation.

**CONVERSATION HISTORY:**
{conversation_context}

**USER MESSAGE:** {message}

**INSTRUCTIONS:**
- Respond naturally and conversationally
- Reference previous conversation when relevant
- Be helpful and informative
- If asked about documents, explain that no documents are currently loaded
- Maintain a friendly, professional tone

**RESPONSE:**"""

    @classmethod
    def get_text_summary_prompt(cls):
        """Get prompt for text summarization"""
        return ChatPromptTemplate.from_template(cls.TEXT_SUMMARY_PROMPT)
    
    @classmethod
    def get_table_summary_prompt(cls):
        """Get prompt for table summarization"""
        return ChatPromptTemplate.from_template(cls.TABLE_SUMMARY_PROMPT)
    
    @classmethod
    def get_image_analysis_prompt(cls):
        """Get multimodal prompt for image analysis with proper message format"""
        messages = [
            (
                "user",
                [
                    {"type": "text", "text": cls.IMAGE_ANALYSIS_PROMPT_TEXT},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{image}"}
                    }
                ]
            )
        ]
        return ChatPromptTemplate.from_messages(messages)
    
    @classmethod
    def get_enhanced_rag_query_prompt(cls):
        """Get enhanced prompt for RAG-based query answering with conversation history"""
        return ChatPromptTemplate.from_template(cls.ENHANCED_RAG_QUERY_PROMPT)
    
    @classmethod
    def get_enhanced_rag_query_simple_prompt(cls):
        """Get enhanced prompt for RAG-based query answering without conversation history"""
        return ChatPromptTemplate.from_template(cls.ENHANCED_RAG_QUERY_SIMPLE_PROMPT)
    
    @classmethod
    def get_general_chat_prompt(cls):
        """Get prompt for general conversation"""
        return ChatPromptTemplate.from_template(cls.GENERAL_CHAT_PROMPT)