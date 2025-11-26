PROMPT_TEMPLATE = """
Basing only on the following context:

{context}

---

Answer the following question: {question}
Avoid to start the answer saying that you are basing on the provided context and go straight with the response.
"""

def generate_llm_prompt(context: str, question: str) -> str:
    """Generate a formatted prompt using the template."""
    return PROMPT_TEMPLATE.format(context=context, question=question)
