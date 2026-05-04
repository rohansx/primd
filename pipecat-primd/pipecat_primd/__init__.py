"""pipecat-primd: open-source VoiceAgentRAG plugin for Pipecat.

Drop the `PrimdRetriever` FrameProcessor into a Pipecat pipeline and primd
will speculate on partial transcripts during STT, return cached results at
end-of-utterance in microseconds, and pre-warm the next likely answer
during TTS playback.
"""

from pipecat_primd.client import Hit, PrimdClient, QueryResult

__version__ = "0.1.0"
__all__ = ["Hit", "PrimdClient", "QueryResult", "PrimdRetriever"]


def __getattr__(name: str):
    """Lazy-load PrimdRetriever so the package imports even without pipecat-ai.

    The retriever depends on pipecat-ai (declared as an optional extra). If a
    user installs only ``pipecat-primd`` for the standalone client, importing
    the retriever should fail loudly at access time, not at package import.
    """
    if name == "PrimdRetriever":
        from pipecat_primd.retriever import PrimdRetriever as _PrimdRetriever

        return _PrimdRetriever
    raise AttributeError(f"module 'pipecat_primd' has no attribute {name!r}")
