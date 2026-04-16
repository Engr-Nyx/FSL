"""Optional LLM-based sentence rewriting using the Anthropic API.

When ``ENABLE_LLM_REWRITE=true`` and ``ANTHROPIC_API_KEY`` is set, raw
gloss-to-sentence output can be polished into fluent Tagalog or English.

If the API key is missing or the call fails, the mapper output is returned
unchanged — the LLM rewrite is always optional / non-blocking.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_FIL = """Ikaw ay isang eksperto sa wikang Filipino at Filipino Sign Language (FSL).
Bibigyan ka ng isang mala-raw na pangungusap na nakabatay sa FSL gloss sequence.
I-rewrite ito para maging natural na Filipino/Tagalog na pangungusap.
Huwag magdagdag ng impormasyon na wala sa orihinal.
Ibalik lamang ang rewritten na pangungusap — walang paliwanag."""

_SYSTEM_PROMPT_EN = """You are an expert in Filipino Sign Language (FSL) and natural English.
You will receive a raw sentence derived from an FSL gloss sequence.
Rewrite it into a natural, fluent English sentence.
Do not add information not present in the original.
Return only the rewritten sentence — no explanations."""


class LLMRewriter:
    """Polishes raw translated sentences using Claude."""

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001") -> None:
        self._api_key = api_key
        self._model = model
        self._client: Optional[object] = None
        self._available = False

        if not api_key:
            logger.info("LLMRewriter: no API key provided — rewriting disabled.")
            return

        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
            self._available = True
            logger.info("LLMRewriter: Anthropic client ready (model=%s)", model)
        except ImportError:
            logger.warning("LLMRewriter: anthropic package not installed — pip install anthropic")

    @property
    def available(self) -> bool:
        return self._available

    def rewrite(self, sentence: str, lang: str = "fil") -> str:
        """Rewrite *sentence* into fluent Tagalog or English.

        Args:
            sentence: Raw translated sentence from SentenceMapper.
            lang: 'fil' for Tagalog, 'en' for English.

        Returns:
            Polished sentence, or the original if rewriting fails / is disabled.
        """
        if not self._available or not sentence.strip():
            return sentence

        system = _SYSTEM_PROMPT_FIL if lang == "fil" else _SYSTEM_PROMPT_EN

        try:
            response = self._client.messages.create(  # type: ignore[union-attr]
                model=self._model,
                max_tokens=256,
                system=system,
                messages=[{"role": "user", "content": sentence}],
            )
            return response.content[0].text.strip()
        except Exception as exc:
            logger.warning("LLMRewriter: rewrite failed (%s) — returning original", exc)
            return sentence
