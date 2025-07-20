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

    def generate_text(self, user_prompt: str) -> str:
        """Generate text using Gemini (production) or Ollama (development) based on environment."""
        try:
            if self.production_env:
                # Gemini
                response = self.llm.generate_content(user_prompt)
                return response.text
            else:
                # Ollama
                response = self.llm.chat(
                    model=self.current_model,
                    messages=[{'role': 'user', 'content': user_prompt}]
                )
                return response['message']['content']
        except Exception as e:
            print(f"Error generating text: {e}")
            return "Sorry, there was an error generating a response."
    def __init__(self):
        self._initialize_llm()
        self.prompt = self._initialize_prompt()
        self.chain = self.prompt | self.llm
        self.with_message_history = self._initialize_message_history()

    def _initialize_llm(self):
        """Initializes self.llm and self.current_model for Gemini or Ollama based on environment."""
        import os
        self.production_env = os.getenv("ENVIRONMENT", "development").lower() == "production"
        if self.production_env:
            # Google Gemini setup
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set for production environment.")
            genai.configure(api_key=api_key)
            self.current_model = os.getenv("GEMINI_MODEL", "gemini-pro")
            self.llm = genai.GenerativeModel(self.current_model)
        else:
            # Ollama setup
            import ollama
            ollama_host = os.getenv("OLLAMA_BASE_URL", "http://192.168.1.100:11434")
            self.current_model = os.getenv("OLLAMA_MODEL", "gemma3")
            self.llm = ollama.Client(host=ollama_host)

    def generate_text(self, user_prompt: str) -> str:
        """Generate text using Gemini (production) or Ollama (development) based on environment."""
        try:
            if self.production_env:
                # Gemini
                response = self.llm.generate_content(user_prompt)
                return response.text
            else:
                # Ollama
                response = self.llm.chat(
                    model=self.current_model,
                    messages=[{'role': 'user', 'content': user_prompt}]
                )
                return response['message']['content']
        except Exception as e:
            print(f"Error generating text: {e}")
            return "Sorry, there was an error generating a response."

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
