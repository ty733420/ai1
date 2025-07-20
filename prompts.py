"""
prompts.py
Prompt generation utilities for LLM tasks: summarization and question answering.
"""

def get_summary_prompt(document_text: str, length: str, style: str) -> str:
    """
    Generate a detailed prompt for summarizing document_text.
    - length: 'short', 'medium', or 'long'
    - style: 'executive_summary', 'bullet_points', or 'narrative'
    """
    length_instructions = {
        'short': "Keep the summary concise, no more than 3-4 sentences.",
        'medium': "Provide a moderately detailed summary, about 1-2 paragraphs.",
        'long': "Provide a comprehensive summary, covering all key points in detail."
    }
    style_instructions = {
        'executive_summary': "Format as an executive summary for a business audience.",
        'bullet_points': "Format the summary as clear, concise bullet points.",
        'narrative': "Write the summary in a narrative, story-like style."
    }
    length_text = length_instructions.get(length, "Provide a summary of appropriate length.")
    style_text = style_instructions.get(style, "Format the summary in a clear and readable way.")
    prompt = (
        f"You are an expert summarizer. Your task is to summarize the following document.\n"
        f"{length_text} {style_text}\n"
        f"Document:\n" +
        f"""{document_text}""" +
        "\nSummary:"
    )
    return prompt

def get_qa_prompt(document_text: str, question: str) -> str:
    """
    Generate a prompt for answering a question strictly based on the provided document_text.
    The LLM must not use any outside knowledge or make up information.
    """
    prompt = (
        "You are a helpful assistant. Answer the following question using only the information in the provided document. "
        "If the answer is not present in the document, reply with 'The answer is not available in the provided document.'\n"
        "Document:\n"
        f"""{document_text}""" +
        "\nQuestion: " + question +
        "\nAnswer:"
    )
    return prompt
