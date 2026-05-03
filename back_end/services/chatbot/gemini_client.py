import os
import logging
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini

load_dotenv()

logger = logging.getLogger(__name__)

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
        model = os.getenv("GEMINI_MODEL")
        if not api_key:
            raise RuntimeError("GOOGLE_GEMINI_API_KEY not set")
        if not model:
            raise RuntimeError("GEMINI_MODEL not set")
        logger.info(f"Initializing Gemini LLM: {model}")
        _llm = Gemini(model_name=model, api_key=api_key)
    return _llm


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
    llm = _get_llm()
    response = llm.complete(formatted_prompt)
    return response.text
