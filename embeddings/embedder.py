from fastembed import TextEmbedding
import numpy as np

_model = None

def get_model():
    global _model
    if _model is None:
        _model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _model


def create_embedding_text(func):
    class_prefix = f"{func['class_name']}." if func.get("class_name") else ""
    return f"""Function: {class_prefix}{func['function_name']}
File: {func['file_path']}
Docstring: {func['docstring'] or 'None'}
Code:
{func['code']}"""


def embed_batch(texts):
    """Embed a list of texts, return numpy array."""
    embeddings = list(get_model().embed(texts))
    return np.array(embeddings)


def embed_functions(functions):
    texts = [create_embedding_text(f) for f in functions]
    return embed_batch(texts), texts