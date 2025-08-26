"""
Intelligence: The "brain" that processes information and makes decisions using LLMs.
This component handles context understanding, instruction following, and response generation.

More info: https://platform.openai.com/docs/guides/text?api-mode=responses
"""

#from openai import OpenAI

import ollama
import asyncio

async def basic_intelligence(prompt: str) -> str:
    """
    client = OpenAI()
    response = client.responses.create(model="gpt-4o", input=prompt)
    return response.output_text
    """
    
    OLLAMA_MODEL = 'gemma3:4b'
    ollama_async_client = ollama.AsyncClient()   
    await ollama_async_client.pull(OLLAMA_MODEL)    
    response = await ollama_async_client.chat(
                    model=OLLAMA_MODEL, 
                    messages=[{'role': 'user', 'content': prompt}]
                )
    
    #return response
    return response['message']['content']


async def main():
    """Main function to run the asynchronous code."""
    result = await basic_intelligence(prompt="What is artificial intelligence?")
    print(f"\n Basic Intelligence Output:\n\n {result}")
    

if __name__ == "__main__":
    # This is the correct way to run an asynchronous function
    # It creates an event loop to execute the code.
    asyncio.run(main())