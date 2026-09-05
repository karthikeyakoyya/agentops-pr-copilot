"""Provider-agnostic LLM factory.

Every other module imports `get_llm()` — never a vendor SDK directly.
Switching LLM providers is a one-line env var change, not a refactor,
which is the "not tightly coupled to a single LLM vendor" requirement
in concrete form rather than a slide.
"""

import os


def get_llm(temperature: float = 0.0):
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            temperature=temperature,
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-3.5-flash"),
            temperature=temperature,
        )    
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
            temperature=temperature,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{provider}'. Set it to 'anthropic', 'openai', or "
        "'openai', or add a new branch to get_llm()."
    )
