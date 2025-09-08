import requests
import json
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

#Only for Jupyter Notebooks
import nest_asyncio
nest_asyncio.apply()

#--------------------------------------------------------
# Initialize the OpenAI client for Ollama   
# And add instructor wrapper for Pydantic validation
#--------------------------------------------------------
client = OpenAI(base_url="http://localhost:11434/v1/", api_key="")
client = instructor.patch(client)


# --------------------------------------------------------------
# Define the knowledge base retrieval tool
# --------------------------------------------------------------
def search_kb(question: str):
    """
    Load the whole knowledge base from the JSON file.
    (This is a mock function for demonstration purposes, we don't search)
    """
    import os
    kb_path = os.path.join(os.path.dirname(__file__), "kb.json")
    
    with open(kb_path, "r") as f:
        return json.load(f)
    
    """
    for record in kb:
        if record["question"].strip().lower() == question.strip().lower():
            return {
                "answer": record["answer"],
                "source": record["id"]
            }
    """

# --------------------------------------------------------------
# Step 1: Call model with search_kb tool defined
# --------------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Get the answer to the user's question from the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

system_prompt = "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store. "

system_prompt = """
    You are a helpful assistant. You will be given a list of records from a knowledge base. 
    Each record has an 'id', 'question', and 'answer'.

    Your job is to:
    1. Find the record whose question best matches the user's question.
    2. Return the answer and the correct 'id' of that record as 'source'.

    If you cannot find a relevant record, say 'I don't know' and use source: 0.
    """

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the return policy ?"},
]

model="llama3.1:8b"

completion = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,
    temperature=0, 
)

# --------------------------------------------------------------
# Step 2: Model decides to call function(s)
# --------------------------------------------------------------
completion.model_dump()


# --------------------------------------------------------------
# Step 3: Execute search_kb function
# --------------------------------------------------------------
def call_function(name, args):
    if name == "search_kb":
        return search_kb(**args)


for tool_call in completion.choices[0].message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    messages.append(completion.choices[0].message)

    result = call_function(name, args)
    #print ("result: ", result)
    
    messages.append(
        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
    )



# --------------------------------------------------------------
# Step 4: Supply result and call model again
# --------------------------------------------------------------

class KBResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question.")
    source: int = Field(description="The record id of the answer.")
    


completion_2 = client.chat.completions.parse(
    model=model,
    messages=messages,
    tools=tools,
    temperature=0, 
    response_format=KBResponse,
)

# --------------------------------------------------------------
# Step 5: Check model response
# --------------------------------------------------------------

final_response = completion_2.choices[0].message.parsed
print("\ncompletion_2 answer: ", final_response.answer)
print("completion_2 source: ", final_response.source)


# --------------------------------------------------------------
# Question that doesn't trigger the tool
# --------------------------------------------------------------

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "台南現在氣溫如何? 如果你無法回答, 就回答說你不知道. 而不要什都不回應"},
]

completion_3 = client.chat.completions.parse(
    model=model,
    messages=messages,
    tools=tools,
    temperature=0.3,
    response_format=KBResponse 
)

final_response = completion_3.choices[0].message.parsed
print("\ncompletion_3: ", final_response)
