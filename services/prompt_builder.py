def build_prompt(messages, context, question):
    conversation = ""
    for message in messages:
        role = message["role"].capitalize()
        conversation += f"{role}: {message['content']}\n"
    prompt = f"""
========================
SYSTEM
========================

You are Lumixa AI.
You are an intelligent Knowledge Assistant.

Rules:
- Answer ONLY from the provided context.
- Never hallucinate.
- If the answer is unavailable, reply:
  "I couldn't find this information in the uploaded knowledge."
- If multiple sources discuss the answer, combine them naturally.
- Use bullet points whenever appropriate.
- Keep answers concise and professional.

========================
CONVERSATION
========================
{conversation}
========================
CONTEXT
========================
{context}
========================
QUESTION
========================
{question}
========================
ANSWER
========================
"""
    return prompt
