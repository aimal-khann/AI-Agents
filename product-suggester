from dotenv import load_dotenv
import os 
from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner
from agents import AsyncOpenAI 

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Setup external model client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent = Agent(
    name="HealthProductSuggester",
    instructions=(
        "You are a professional doctor. "
        "Only suggest health related products or medicines based on the user's symptoms. "
        "Also explain briefly why the product is suitable. "
        "Do not recommend non-health items. "
    )
)

if __name__ == "__main__":
    print("👨‍⚕️ Welcome to HealthCare Assistant")
    user_input = input("🩺 Please describe your health issue: ")
    response = Runner.run_sync(agent, input=user_input, run_config=config)
    print("💊 Recommendation:", response.final_output)
