"""
summarizer.py
Main application for document summarization and question answering using AIAgent and prompt utilities.
"""

from agent import AIAgent
import prompts

def summarize_document(document_text: str, agent_instance: AIAgent, length: str = 'medium', style: str = 'narrative') -> str:
    """Summarize a document using the LLM agent and a generated prompt."""
    try:
        print(f"Generating summary... (length: {length}, style: {style})")
        prompt = prompts.get_summary_prompt(document_text, length, style)
        summary = agent_instance.generate_text(prompt)
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "[Error: Could not generate summary.]"

def answer_question_about_document(document_text: str, question: str, agent_instance: AIAgent) -> str:
    """Answer a question about a document using the LLM agent and a grounded prompt."""
    try:
        print("Answering question based on document...")
        prompt = prompts.get_qa_prompt(document_text, question)
        answer = agent_instance.generate_text(prompt)
        return answer
    except Exception as e:
        print(f"Error answering question: {e}")
        return "[Error: Could not answer question.]"

if __name__ == "__main__":
    print("\n=== Document Summarizer & QA Demo ===\n")
    agent = AIAgent()

    # Sample multi-paragraph document (approx. 300 words)
    document_text = (
        """
        The rapid advancement of artificial intelligence (AI) has transformed industries across the globe. In recent years, organizations have increasingly adopted AI-driven solutions to optimize operations, enhance customer experiences, and unlock new business opportunities. For example, in healthcare, AI-powered diagnostic tools assist doctors in identifying diseases earlier and more accurately, while predictive analytics help hospitals manage resources efficiently. In the financial sector, AI algorithms detect fraudulent transactions and automate trading, leading to improved security and efficiency.

        Education has also seen significant changes due to AI. Adaptive learning platforms personalize educational content to suit individual student needs, resulting in better engagement and outcomes. Meanwhile, AI chatbots provide instant support to students and educators alike, streamlining administrative tasks and improving communication. However, the widespread adoption of AI raises important ethical considerations, such as data privacy, algorithmic bias, and the potential displacement of jobs. Policymakers and industry leaders are working together to establish guidelines and best practices to ensure responsible AI development and deployment.

        Looking ahead, experts predict that AI will continue to evolve, with advancements in natural language processing, computer vision, and autonomous systems. These innovations are expected to drive further economic growth and societal change. Nevertheless, it is crucial to address the challenges associated with AI to maximize its benefits while minimizing risks. Ongoing research, collaboration, and public dialogue will play a vital role in shaping the future of artificial intelligence for the betterment of humanity.
        """
    )

    # Demonstrate summarization with different options
    print("\n--- Short Executive Summary ---")
    print(summarize_document(document_text, agent, length='short', style='executive_summary'))

    print("\n--- Medium Bullet-Point Summary ---")
    print(summarize_document(document_text, agent, length='medium', style='bullet_points'))

    print("\n--- Long Narrative Summary ---")
    print(summarize_document(document_text, agent, length='long', style='narrative'))

    # Interactive QA loop
    print("\n=== Document Question Answering ===")
    print("You can now ask questions about the sample document.")
    print("Type 'exit' or 'quit' to end the session.\n")
    while True:
        user_question = input("Enter your question (or 'exit' to quit): ")
        if user_question.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        answer = answer_question_about_document(document_text, user_question, agent)
        print(f"Answer: {answer}\n")
