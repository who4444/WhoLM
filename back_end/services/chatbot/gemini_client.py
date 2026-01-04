from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
from FlagEmbedding import BGEM3FlagModel 
import os
from llama_index.core import Settings

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

# loads models
llm = Gemini(model_name=GEMINI_MODEL, api_key=GOOGLE_API_KEY)
Settings.llm = llm


def generate_response(sys_prompt, context, query):
    formatted_prompt = f"""
    {sys_prompt}

    Using only the following context information, answer the user's query. 
    If the answer is not in the context, say "I don't know".

    ### Context:
    {context}

    ### User Query:
    {query}

    ### Answer:
    """
    response = llm.complete(formatted_prompt)
    return response.text






