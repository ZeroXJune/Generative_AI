"""Framework interop for the Personal Assistant AI."""

from .langchain_adapters import (
    PersonalAssistantEmbeddings,
    PersonalAssistantRetriever,
    to_documents,
)

__all__ = [
    "PersonalAssistantEmbeddings",
    "PersonalAssistantRetriever",
    "to_documents",
]
