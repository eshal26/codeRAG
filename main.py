from ingestion.ingest import ingest_repo
from retriever.vector_store import search_repo, search_all_repos, list_indexed_repos
from generator.answer import generate_answer
from embeddings.embedder import model


def main():
    print("\n=== Code Documentation Assistant ===\n")

    while True:
        print("\nOptions:")
        print("  1. Add a GitHub repo")
        print("  2. Ask a question")
        print("  3. List indexed repos")
        print("  4. Exit")

        choice = input("\nChoice: ").strip()

        if choice == "1":
            url = input("GitHub URL: ").strip()
            ingest_repo(url)

        elif choice == "2":
            repos = list_indexed_repos()
            if not repos:
                print("No repos indexed yet. Add one first.")
                continue

            print(f"Indexed repos: {repos}")
            scope = input("Search which repo? (name or 'all'): ").strip()
            query = input("Your question: ").strip()

            query_embedding = model.encode(query)

            if scope == "all":
                results = search_all_repos(query_embedding, k=3)
            else:
                results = search_repo(query_embedding, scope, k=3)

            if not results:
                print("No results found.")
                continue

            retrieved_texts = [r["text"] for r in results]
            answer = generate_answer(query, retrieved_texts, repo_name=scope)

            print("\n--- Answer ---")
            print(answer)

            print("\n--- Sources ---")
            for r in results:
                f = r["function"]
                class_prefix = f"{f['class_name']}." if f.get("class_name") else ""
                print(f"  [{r['repo']}] {class_prefix}{f['function_name']} | {f['file_path'].split('/')[-1]} line {f['start_line']} (score: {r['score']:.3f})")

        elif choice == "3":
            repos = list_indexed_repos()
            print("Indexed repos:", repos if repos else "None")

        elif choice == "4":
            break


if __name__ == "__main__":
    main()