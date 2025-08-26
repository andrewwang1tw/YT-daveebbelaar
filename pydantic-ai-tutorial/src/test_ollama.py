from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent
 
provider = OpenAIProvider(
    base_url="http://localhost:11434/v1/" 
)
 
model_ollama = OpenAIModel(
    model_name="gemma3:4b",
    provider=provider
)

agent = Agent(
    model=model_ollama, 
    system_prompt=['Reply in one sentence']
)

response = agent.run_sync('The capital of Taiwan is ?')
print(response.output)