from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings:
    """
    class that implements two methods to be called from Chroma
    """

    def __init__(self, model: str):
        self.model = model
        self.embedder = SentenceTransformer(model, device="cpu")

    def embed_documents(self, texts: list[str]):
        vectors = self.embedder.encode(texts, batch_size=2)
        return vectors

    def embed_query(self, text: str):
        vectors = self.embedder.encode([text])
        return vectors[0]
