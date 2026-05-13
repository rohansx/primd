"""livekit-primd — predictive turn-cache for LiveKit Agents.

See https://github.com/rohansx/primd for the underlying primd runtime.
"""

from livekit_primd.client import Hit, PrimdClient, QueryResult
from livekit_primd.retriever import (
    DEFAULT_SYSTEM_PROMPT,
    PrimdRetriever,
    attach_to_voice_assistant,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "Hit",
    "PrimdClient",
    "PrimdRetriever",
    "QueryResult",
    "attach_to_voice_assistant",
]

__version__ = "0.1.0"
