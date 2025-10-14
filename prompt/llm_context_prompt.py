from langchain.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
Basing only on the following context:

{context}

---

Answer the following question: {question}
Avoid to start the answer saying that you are basing on the provided context and go straight with the response.
"""

def generate_llm_prompt(context: str, question: str) -> str:
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=question)
    return prompt
