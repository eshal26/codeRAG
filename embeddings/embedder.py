from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embedding_text(func):
    class_prefix = f"{func['class_name']}." if func.get("class_name") else ""
    return f"""Function: {class_prefix}{func['function_name']}
File: {func['file_path']}
Docstring: {func['docstring'] or 'None'}
Code:
{func['code']}"""


def embed_functions(functions):
    texts = [create_embedding_text(f) for f in functions]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings, texts