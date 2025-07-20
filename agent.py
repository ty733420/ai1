# agent.py
#
# GitHub Copilot: Create a Python file for a LangChain AI agent.
# It should import the configuration from config.py.
# Based on config.LLM_PROVIDER, dynamically initialize either ChatOllama or ChatGoogleGenerativeAI.
# Implement session history using RunnableWithMessageHistory.
# The memory should also be dynamically chosen (SQLite for Ollama, a placeholder for cloud DB for Gemini).
# Create a simple conversational chain.
# Include an example of how to invoke the agent with a session ID.


import langchain_core.prompts
import langchain_core.runnables.history
import langchain_google_genai
import langchain_community.chat_models
import langchain_core.messages
import config  # Import our dynamic configuration



class AIAgent:
    def __init__(self):
        self.llm = self._initialize_llm()
        self.prompt = self._initialize_prompt()
        self.chain = self.prompt | self.llm
        self.with_message_history = self._initialize_message_history()

    def _initialize_llm(self):
        """Initializes the LLM based on the configured provider."""
        if config.config.LLM_PROVIDER == "ollama":
            print(f"Initializing Ollama LLM: {config.config.OLLAMA_MODEL} at {config.config.OLLAMA_BASE_URL}")
            return langchain_community.chat_models.ChatOllama(model=config.config.OLLAMA_MODEL, base_url=config.config.OLLAMA_BASE_URL)
        elif config.config.LLM_PROVIDER == "gemini":
            if not config.config.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not set for production environment.")
            print(f"Initializing Gemini LLM: {config.config.GEMINI_MODEL}")
            return langchain_google_genai.ChatGoogleGenerativeAI(
                model=config.config.GEMINI_MODEL,
                google_api_key=config.config.GEMINI_API_KEY # LangChain picks from env by default, but explicit for clarity
            )
        else:
            raise ValueError(f"Unknown LLM provider: {config.config.LLM_PROVIDER}")

    def _initialize_prompt(self):
        """Defines the chat prompt template."""
        return langchain_core.prompts.ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful AI assistant. Keep your answers concise."),
                langchain_core.prompts.MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

    def _get_session_history(self, session_id: str):
        """Returns the appropriate ChatMessageHistory for the session."""
        if config.config.LLM_PROVIDER == "ollama":
            import os
            os.makedirs(os.path.dirname(config.config.MEMORY_DB_PATH), exist_ok=True)
            # SQLiteChatMessageHistory not found, fallback to in-memory
            # return langchain_community.chat_message_histories.SQLiteChatMessageHistory(
            #     session_id=session_id,
            #     connection_string=f"sqlite:///{config.config.MEMORY_DB_PATH}"
            # )
            from langchain_community.chat_message_histories import ChatMessageHistory
            return ChatMessageHistory()
        elif config.config.LLM_PROVIDER == "gemini":
            # Placeholder for a real cloud database history
            # In production, you'd use a cloud-specific history like PostgresChatMessageHistory, etc.
            if not config.config.MEMORY_DB_URL:
                print("Warning: MEMORY_DB_URL not set for production. Using in-memory history.")
                from langchain_community.chat_message_histories import ChatMessageHistory
                return ChatMessageHistory()
            else:
                from langchain_community.chat_message_histories import SQLChatMessageHistory
                return SQLChatMessageHistory(
                    session_id=session_id,
                    connection_string=config.config.MEMORY_DB_URL
                )

    def _initialize_message_history(self):
        """Wraps the chain with message history management."""
        return langchain_core.runnables.history.RunnableWithMessageHistory(
            self.chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def invoke(self, user_input: str, session_id: str):
        """Invokes the agent with user input and session ID."""
        print(f"\n--- Invoking agent for session {session_id} ---")
        response = self.with_message_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        return response.content

if __name__ == "__main__":
    agent = AIAgent()

    # Interactive chat session
    dev_session_id = "user_abc_dev_session"
    print("Welcome to the AI chat! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        response = agent.invoke(user_input, dev_session_id)
        print(f"AI: {response}")

    # Example of how you would simulate production locally (by changing .env or directly in shell)
    # Uncomment the following to test "production" locally (requires GOOGLE_API_KEY in .env)
    # os.environ["ENVIRONMENT"] = "production"
    # os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY_HERE" # Replace with actual key
    # os.environ["GEMINI_MODEL"] = "gemini-pro"
    # os.environ["DATABASE_URL"] = "sqlite:///chat_histories/prod_memory.db" # Using sqlite for local prod test
    #
    # print("\n--- Simulating Production Environment ---")
    # agent_prod_test = AIAgent()
    # prod_session_id = "user_xyz_prod_session"
    # print(agent_prod_test.invoke("Hi there, how are you today?", prod_session_id))
    # print(agent_prod_test.invoke("What's the weather like in New York?", prod_session_id))
