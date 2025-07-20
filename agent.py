# agent.py

import langchain_core.prompts
import langchain_core.runnables.history
import langchain_google_genai
import langchain_community.chat_models
import langchain_core.messages
import config

logger = config.logger

class AIAgent:

    def generate_text(self, user_prompt: str) -> str:
        """Generate text using Gemini (production) or Ollama (development) based on environment."""
        logger.debug(f"generate_text called with prompt: {user_prompt[:100]}{'...' if len(user_prompt) > 100 else ''}")
        try:
            if self.production_env:
                logger.info("Using Gemini LLM for text generation.")
                response = self.llm.generate_content(user_prompt)
                logger.debug(f"Gemini response: {getattr(response, 'text', str(response))[:100]}")
                return response.text
            else:
                logger.info("Using Ollama LLM for text generation.")
                response = self.llm.chat(
                    model=self.current_model,
                    messages=[{'role': 'user', 'content': user_prompt}]
                )
                logger.debug(f"Ollama response: {str(response)[:100]}")
                return response['message']['content']
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return "Sorry, there was an error generating a response."
    def __init__(self):
        logger.info("Initializing AIAgent...")
        self._initialize_llm()
        self.prompt = self._initialize_prompt()
        self.chain = self.prompt | self.langchain_llm
        self.with_message_history = self._initialize_message_history()

    def _initialize_llm(self):
        """Initializes both LangChain and direct API LLMs for Gemini or Ollama based on environment."""
        import os
        self.production_env = os.getenv("ENVIRONMENT", "development").lower() == "production"
        logger.debug(f"Production environment: {self.production_env}")
        if self.production_env:
            logger.info("Setting up Gemini LLM...")
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY not set for production environment.")
                raise ValueError("GEMINI_API_KEY not set for production environment.")
            genai.configure(api_key=api_key)
            self.current_model = os.getenv("GEMINI_MODEL", "gemini-pro")
            self.llm = genai.GenerativeModel(self.current_model)
            import langchain_google_genai
            self.langchain_llm = langchain_google_genai.ChatGoogleGenerativeAI(
                model=self.current_model,
                google_api_key=api_key
            )
        else:
            logger.info("Setting up Ollama LLM...")
            import ollama
            ollama_host = os.getenv("OLLAMA_BASE_URL", "http://192.168.1.100:11434")
            self.current_model = os.getenv("OLLAMA_MODEL", "gemma3")
            self.llm = ollama.Client(host=ollama_host)
            import langchain_community.chat_models
            self.langchain_llm = langchain_community.chat_models.ChatOllama(
                model=self.current_model,
                base_url=ollama_host
            )

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
        logger.debug("Initializing chat prompt template.")
        return langchain_core.prompts.ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful AI assistant. Keep your answers concise."),
                langchain_core.prompts.MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

    def _get_session_history(self, session_id: str):
        """Returns the appropriate ChatMessageHistory for the session."""
        logger.debug(f"Getting session history for session_id: {session_id}")
        if config.config.LLM_PROVIDER == "ollama":
            import os
            os.makedirs(os.path.dirname(config.config.MEMORY_DB_PATH), exist_ok=True)
            from langchain_community.chat_message_histories import ChatMessageHistory
            return ChatMessageHistory()
        elif config.config.LLM_PROVIDER == "gemini":
            if not config.config.MEMORY_DB_URL:
                logger.warning("MEMORY_DB_URL not set for production. Using in-memory history.")
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
        logger.debug("Initializing message history wrapper.")
        return langchain_core.runnables.history.RunnableWithMessageHistory(
            self.chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def invoke(self, user_input: str, session_id: str):
        """Invokes the agent with user input and session ID."""
        logger.info(f"Invoking agent for session {session_id} with input: {user_input[:80]}{'...' if len(user_input) > 80 else ''}")
        response = self.with_message_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        logger.debug(f"Agent response: {getattr(response, 'content', str(response))[:100]}")
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
