import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from services.prompt_builder import build_prompt
from config import LLM_MODEL, TEMPERATURE

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    timeout=200
)

MODEL = os.getenv("OPENROUTER_MODEL", LLM_MODEL)

SYSTEM_PROMPT = """
You are Lumixa AI.
You are an intelligent Knowledge Assistant.

Rules:
- Answer only using the provided context.
- If the answer is not present, clearly say that you couldn't find it.
- Never hallucinate facts.
- Be concise and professional.
- When possible, organize answers using bullet points.
"""

def stream_llm_response(messages, temperature=TEMPERATURE):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=temperature,
            messages=messages,
            stream=True
        )
        inside_think = False
        buffer = ""
        for chunk in response:
            if not chunk.choices or not chunk.choices[0].delta.content:
                continue
            token = chunk.choices[0].delta.content
            buffer += token
            while buffer:
                if inside_think:
                    end_idx = buffer.find("</think>")
                    if end_idx != -1:
                        buffer = buffer[end_idx + len("</think>"):]
                        inside_think = False
                    else:
                        if len(buffer) > 8:
                            buffer = buffer[-8:]
                        break
                else:
                    start_idx = buffer.find("<think>")
                    if start_idx != -1:
                        before = buffer[:start_idx]
                        if before:
                            yield before
                        buffer = buffer[start_idx + len("<think>"):]
                        inside_think = True
                    elif "<" in buffer:
                        safe_idx = buffer.rfind("<")
                        if safe_idx > 0:
                            yield buffer[:safe_idx]
                            buffer = buffer[safe_idx:]
                        break
                    else:
                        yield buffer
                        buffer = ""
        if buffer and not inside_think:
            yield buffer
    except Exception as e:
        yield f"Sorry, I encountered an error during generation: {e}"

def summarize_stream(text):
    words = text.split()
    if len(words) > 12000:
        text = " ".join(words[:12000]) + "\n\n[Content truncated for summary due to length limits]"
    prompt = f"""
You are Lumixa AI.

Create a Summary of the following content.
Summarize the context in detail
Try to give more summary as much as possible
Generate a comprehensive summary while preserving all important concepts and details.
Instructions:
- Cover ALL major concepts.
- Do NOT skip important sections.
- Preserve the chronological flow whenever possible.
- Explain technical concepts clearly.
- Include important examples mentioned.
- Mention definitions when introduced.
- Keep important numbers, statistics and facts.
- Organize the summary into meaningful headings.
- Use bullet points where appropriate.
- Write in professional documentation style.
- The summary should be detailed rather than short.
- If the content is long, the summary should also be long.
- Never intentionally shorten the content.

Content:
{text}
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    return stream_llm_response(messages, temperature=0.3)

def summarize(text):
    return "".join(list(summarize_stream(text)))

def generate_insights(summary):
    prompt = f"""
Return ONLY valid JSON.

Schema:
{{
    "takeaways":[
        "...",
        "...",
        "...",
        "...",
        "..."
    ],
    "topics":[
        "...",
        "...",
        "...",
        "...",
        "..."
    ]
}}

Do not include markdown.

Summary:
{summary}
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    try:
        content = "".join(list(stream_llm_response(messages, temperature=0.2))).strip()
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception:
        return {"takeaways": [], "topics": []}

def ask_question_stream(question, context, messages=None):
    if messages is None:
        messages = []
    prompt = build_prompt(
        messages=messages,
        context=context,
        question=question
    )
    api_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    return stream_llm_response(api_messages, temperature=0.2)

def ask_question(question, context, messages=None):
    return "".join(list(ask_question_stream(question, context, messages)))

def generate_alternative_queries(question: str) -> list[str]:
    """
    Given a user's question, uses the LLM to generate 2 alternative formulations
    of the query to improve search coverage (Query Expansion).
    Returns a list of alternative queries (including the original one).
    """
    prompt = f"""
Given the search query below, generate exactly 2 alternative variations of this query that mean the same thing but use different vocabulary or phrasing.
These will be used to retrieve relevant documents from a database.

Return ONLY a valid JSON list of strings, for example:
[
  "first alternative query",
  "second alternative query"
]

Do not include any other text, reasoning, markdown blocks (like ```json), or explanations.

Query: {question}
"""
    messages = [
        {"role": "system", "content": "You are a helpful query translation assistant. You speak only in JSON arrays of strings."},
        {"role": "user", "content": prompt}
    ]

    alternative_queries = []
    try:
        content = "".join(list(stream_llm_response(messages, temperature=0.2))).strip()

        # Clean potential markdown output formatting
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(content)
        if isinstance(parsed, list):
            for q in parsed:
                if isinstance(q, str) and q.strip():
                    alternative_queries.append(q.strip())
    except Exception as e:
        print(f"Query expansion error: {e}")

    return alternative_queries
