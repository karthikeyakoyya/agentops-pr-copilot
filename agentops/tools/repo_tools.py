"""Tools the PR Copilot agent can call.

`retrieve_context` builds a local Chroma index over the target repo's
Python files so the agent grounds its fix in real code (RAG) instead of
guessing from the diff text alone. `run_tests` actually shells out to
pytest against the target repo — real signal, not a mocked pass/fail.
"""

import glob
import os
import subprocess

from langchain_core.tools import tool

_vectorstore_cache: dict[str, object] = {}


def _get_vectorstore(repo_path: str):
    """Lazily build (and cache per repo_path) a Chroma index of the
    repo's Python source. Rebuilding per-run is intentionally simple for
    a scaffold; a production version would incrementally index on file
    change instead of a full rebuild per query."""
    if repo_path in _vectorstore_cache:
        return _vectorstore_cache[repo_path]

    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    else:
        # No first-party Anthropic embeddings API yet — default to a
        # local sentence-transformers model so RAG works end-to-end
        # without an OpenAI key.
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    docs = []
    for path in glob.glob(os.path.join(repo_path, "**", "*.py"), recursive=True):
        try:
            with open(path, encoding="utf-8") as f:
                docs.append(Document(page_content=f.read(), metadata={"path": path}))
        except (OSError, UnicodeDecodeError):
            continue

    if not docs:
        docs = [Document(page_content="(no Python source found at this path)", metadata={"path": repo_path})]

    store = Chroma.from_documents(
        docs, embeddings, collection_name="pr_copilot_repo", persist_directory=".chroma_repo"
    )
    _vectorstore_cache[repo_path] = store
    return store


@tool
def retrieve_context(query: str, repo_path: str, k: int = 4) -> str:
    """Retrieve the k most relevant source snippets from the target repo
    for `query`. Always call this before proposing a fix — ground the
    change in the actual codebase instead of guessing."""
    store = _get_vectorstore(repo_path)
    results = store.similarity_search(query, k=k)
    if not results:
        return "No relevant context found."
    return "\n---\n".join(
        f"[{r.metadata.get('path')}]\n{r.page_content[:800]}" for r in results
    )


@tool
def run_tests(repo_path: str, test_path: str = "") -> str:
    """Run pytest against the target repo (or a subpath of it) and
    return a summary of the results. Call this before diagnosing a
    failure and again if you want to confirm a proposed fix would pass."""
    target = os.path.join(repo_path, test_path) if test_path else repo_path
    try:
        result = subprocess.run(
            ["pytest", target, "-q", "--no-header"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout or result.stderr
        return output[-2000:] if output else "(pytest produced no output)"
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        return f"Could not run tests: {exc}"


TOOLS = [retrieve_context, run_tests]
