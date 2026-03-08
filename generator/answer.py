import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_api_key = os.getenv("GROQ_API_KEY")
if not _api_key:
    raise RuntimeError("GROQ_API_KEY is not set. Check your .env file.")

client = Groq(api_key=_api_key)

SYSTEM_PROMPT = """You are an expert code documentation assistant.
When explaining code:
- Start with what the function does in one sentence
- Explain key logic steps
- Mention edge cases or error handling if present
- Keep it concise and developer-friendly
- If the user references a previous question or answer, use that context."""


def generate_answer(query, retrieved_code, repo_name=None, history=None):
    """Non-streaming version (kept for CLI use)."""
    context = "\n\n---\n\n".join(retrieved_code)
    repo_info = f"Repo: {repo_name}\n" if repo_name else ""

    user_prompt = f"""{repo_info}User Question:
{query}

Relevant Code:
{context}

Explain clearly how the code works."""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return response.choices[0].message.content


def stream_answer(query, retrieved_code, repo_name=None, history=None):
    """Streaming version — yields text chunks as they arrive."""
    context = "\n\n---\n\n".join(retrieved_code)
    repo_info = f"Repo: {repo_name}\n" if repo_name else ""

    user_prompt = f"""{repo_info}User Question:
{query}

Relevant Code:
{context}

Explain clearly how the code works."""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        stream=True
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta