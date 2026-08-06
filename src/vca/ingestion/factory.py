"""Selects a transcription source based on configuration.

Resolution order for mode="auto":
  1. If VCA_FORCE_SIMULATED is set -> simulated.
  2. If Azure key + region are configured -> Azure Speech.
  3. Otherwise -> simulated.
"""
from __future__ import annotations

from pathlib import Path

from ..config import Config, load_config
from .azure_speech import AzureSpeechSource
from .base import TranscriptionSource
from .simulated import SimulatedSource


def azure_available(cfg: Config) -> bool:
    return bool(cfg.azure_speech_key and cfg.azure_speech_region) and not cfg.force_simulated


def make_source(
    conversation_id: str,
    transcript_path: str | Path | None = None,
    cfg: Config | None = None,
) -> TranscriptionSource:
    """Build the appropriate transcription source.

    If a ``transcript_path`` is given, a simulated replay source is used
    regardless of Azure availability (useful for the demo and tests).
    """
    cfg = cfg or load_config()
    delay = float(cfg.ingestion.get("simulated_delay_seconds", 0.0))

    if transcript_path is not None:
        return SimulatedSource.from_file(transcript_path, delay_seconds=delay)

    if azure_available(cfg):
        return AzureSpeechSource(
            conversation_id=conversation_id,
            speech_key=cfg.azure_speech_key,  # type: ignore[arg-type]
            speech_region=cfg.azure_speech_region,  # type: ignore[arg-type]
        )

    raise ValueError(
        "No transcript_path supplied and Azure Speech is not configured. "
        "Provide a transcript file or set AZURE_SPEECH_KEY/REGION."
    )
