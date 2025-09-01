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

    # RAG Query Processing Prompt
    RAG_QUERY_PROMPT = """
You are a helpful AI assistant analyzing PDF documents. Answer the user's question based on the provided context from the documents.

CONVERSATION HISTORY:
{conversation_context}

DOCUMENT CONTEXT:
{document_context}

QUESTION: {question}

Instructions:
- Answer based on the document context provided
- Be specific and cite relevant information from the documents
- If referencing tables or charts, explain the data clearly
- If the answer isn't in the documents, say so clearly
- Maintain conversation continuity with previous exchanges
- Be concise but thorough

Answer:"""

    # Multimodal Context Enhancement Prompt
    MULTIMODAL_CONTEXT_PROMPT = """
Based on the following extracted content from a PDF document, provide context for answering questions:

TEXT CONTENT:
{text_content}

TABLE DATA:
{table_content}

IMAGE DESCRIPTIONS:
{image_content}

Synthesize this information to provide comprehensive context about the document's content.
Focus on key facts, data points, and relationships between different elements.
"""

    # Multimodal RAG Query Processing Prompt (for mixed content with images)
    MULTIMODAL_RAG_QUERY_PROMPT = """Answer the question based only on the following context, which can include text, tables, and images.

Context: {context_text}
Question: {question}

Instructions:
- Use all the provided context including any visual information from images
- Be specific about data, numbers, and relationships
- If referencing visual elements, describe what you see
- Provide a comprehensive answer based on the multimodal context

Answer:"""

    # Chat Without Documents Prompt
    GENERAL_CHAT_PROMPT = """
You are a helpful AI assistant having a natural conversation.

CONVERSATION HISTORY:
{conversation_context}

MESSAGE: {message}

Instructions:
- Respond naturally and conversationally
- Reference previous conversation when relevant
- Be helpful and informative
- If asked about documents, explain that no documents are currently loaded

Response:"""

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
    def get_rag_query_prompt(cls):
        """Get prompt for RAG-based query answering"""
        return ChatPromptTemplate.from_template(cls.RAG_QUERY_PROMPT)
    
    @classmethod
    def get_multimodal_rag_query_prompt(cls):
        """Get prompt for multimodal RAG-based query answering"""
        return ChatPromptTemplate.from_template(cls.MULTIMODAL_RAG_QUERY_PROMPT)
    
    @classmethod
    def get_multimodal_context_prompt(cls):
        """Get prompt for multimodal context synthesis"""
        return ChatPromptTemplate.from_template(cls.MULTIMODAL_CONTEXT_PROMPT)
    
    @classmethod
    def get_general_chat_prompt(cls):
        """Get prompt for general conversation"""
        return ChatPromptTemplate.from_template(cls.GENERAL_CHAT_PROMPT)