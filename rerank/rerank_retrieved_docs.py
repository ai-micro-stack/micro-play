import os
from dotenv import load_dotenv, set_key

load_dotenv()
RELEVANT_DOCS = int(os.getenv("RELEVANT_DOCS"))


def get_top_relevant_sources(results, k=3):
    sources, scores, num = [], [], 0
    for doc, score in results:
        metadata = doc.metadata
        file = metadata.get("file", "unknown")
        page = metadata.get("page", "unknown")
        filename = os.path.basename(file)
        sources.append(f"{filename} | page {page}")
        scores.append(score)
        num += 1
        if num >= k:
            break
    sources = list(set(sources))
    references = []
    for i in range(len(sources)):
        references.append(f"[{i+1}] {sources[i]}")
    return references


def get_context_from_documents(results, k=3):
    sources = get_top_relevant_sources(results)
    enhanced_context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    return enhanced_context_text, list(sources)
