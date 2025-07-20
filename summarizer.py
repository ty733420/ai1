"""
summarizer.py
Main application for document summarization and question answering using AIAgent and prompt utilities.
"""


from agent import AIAgent
import prompts
import requests
from bs4 import BeautifulSoup
import urllib.parse
import re
import config
logger = config.logger

def summarize_document(document_text: str, agent_instance: AIAgent, length: str = 'medium', style: str = 'narrative') -> str:
    """Summarize a document using the LLM agent and a generated prompt."""
    try:
        logger.info(f"Generating summary... (length: {length}, style: {style})")
        prompt = prompts.get_summary_prompt(document_text, length, style)
        summary = agent_instance.generate_text(prompt)
        logger.debug(f"Summary generated: {summary[:200]}{'...' if len(summary) > 200 else ''}")
        return summary
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return "[Error: Could not generate summary.]"

def answer_question_about_document(document_text: str, question: str, agent_instance: AIAgent) -> str:
    """Answer a question about a document using the LLM agent and a grounded prompt."""
    try:
        logger.info(f"Answering question based on document: {question}")
        prompt = prompts.get_qa_prompt(document_text, question)
        answer = agent_instance.generate_text(prompt)
        logger.debug(f"Answer generated: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        return answer
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return "[Error: Could not answer question.]"


# Utility function to fetch and clean text from a URL
def fetch_text_from_url(url: str) -> str:
    """
    Fetches the main text content from a web page at the given URL.
    Returns cleaned text or an empty string on error.
    """
    logger.info(f"Fetching text from URL: {url}")
    try:
        response = requests.get(url, timeout=15)
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching URL: {e}")
        return ""
    if response.status_code != 200:
        logger.warning(f"Failed to retrieve URL (status code {response.status_code}): {url}")
        return ""
    soup = BeautifulSoup(response.text, 'html.parser')
    # Remove script, style, and non-content elements
    for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form', 'noscript']):
        tag.decompose()
    # Extract text from common content tags
    content_tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
    if content_tags:
        text = '\n'.join(tag.get_text(separator=' ', strip=True) for tag in content_tags)
    else:
        text = soup.get_text(separator='\n', strip=True)
    # Clean up excessive newlines and whitespace
    text = re.sub(r'\n+', '\n', text)
    text = text.strip()
    logger.debug(f"Fetched text length: {len(text)}")
    return text


if __name__ == "__main__":
    logger.info("=== URL Document Summarizer & QA ===")
    agent = AIAgent()

    while True:
        url = input("Enter a URL to summarize (or 'quit' to exit): ").strip()
        if url.lower() == 'quit':
            logger.info("Goodbye!")
            break
        # Basic URL validation
        parsed = urllib.parse.urlparse(url)
        if not (parsed.scheme in {"http", "https"} and parsed.netloc):
            logger.warning("Invalid URL. Please enter a valid http(s) URL.")
            continue
        fetched_text = fetch_text_from_url(url)
        if not fetched_text:
            logger.warning("Could not retrieve content from URL. Please try another.")
            continue
        logger.info("Successfully retrieved content. Summarizing...")
        summary = summarize_document(fetched_text, agent, length='medium', style='narrative')
        print("\n--- Summary ---\n")
        print(summary)
        print("\nYou can now ask questions about this document.")
        print("Type 'exit' or 'quit' to return to URL prompt.\n")
        while True:
            user_question = input("Enter your question (or 'exit' to quit): ")
            if user_question.strip().lower() in {"exit", "quit"}:
                logger.info("Returning to URL prompt.")
                break
            answer = answer_question_about_document(fetched_text, user_question, agent)
            print(f"Answer: {answer}\n")
