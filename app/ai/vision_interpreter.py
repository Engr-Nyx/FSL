"""Claude Vision-based FSL interpreter.

Sends a sampled sequence of video frames to Claude Vision and returns
Filipino/English translations of the FSL signs being performed.
No training data or custom model weights required — Claude interprets
signs directly from images.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Ikaw ay isang dalubhasa sa Filipino Sign Language (FSL). Ang iyong pangunahing trabaho ay makilala ang mga FSL sign mula sa mga larawan ng kamay at katawan, at isalin ang mga ito sa natural na Filipino/Tagalog.

=== GABAY SA MGA KARANIWANG FSL SIGNS ===

PAMILYA (Family):
- NANAY / MOTHER: Bukas na kamay, mga daliri nakakalat, malapit sa BABA — tapikin nang dalawang beses
- TATAY / FATHER: Bukas na kamay, mga daliri nakakalat, malapit sa NOO — tapikin nang dalawang beses
- LOLO / GRANDFATHER: Patag na kamay malapit sa NOO, gumalaw pababa at palabas
- LOLA / GRANDMOTHER: Patag na kamay malapit sa BABA, gumalaw pababa at palabas
- KAPATID / SIBLING: Ituro ang balikat, gumalaw pasulong
- ANAK / CHILD: Yumakap at uguin parang batang dala
- ASAWA / SPOUSE: Ikabit ang mga daliri ng dalawang kamay
- PAMILYA / FAMILY: Dalawang F-kamay, gumuhit ng bilog na nagtatagpo

PAGBATI (Greetings):
- KUMUSTA / HOW ARE YOU: Ihampas ang bukas na kamay, thumbs up
- MAGANDANG UMAGA / GOOD MORNING: Buksan ang mga kamay (maganda), itaas ang isang kamay (sunrise)
- PAALAM / GOODBYE: Hipain ang kamay, o ulit-ulitin ang pagsasara ng mga daliri
- SALAMAT / THANK YOU: Patag na kamay mula sa labi pasulong at pababa
- PAUMANHIN / SORRY: Saradong kamao gumalaw paligid ng dibdib
- PAKIUSAP / PLEASE: Patag na kamay sa dibdib gumalaw paligid

PANGKARANIWANG SALITA (Common):
- OO / YES: Saradong kamao umiling-iling pataas-pababa
- HINDI / NO: Dalawang daliri umiling-iling pakaliwa-pakanan
- KUMAIN / EAT: Mga daliri nakakulob, paulit-ulit na tutungo sa bibig
- UMINOM / DRINK: C-hugis na kamay tatagilid sa bibig
- TUBIG / WATER: W-kamay, tapikin ang labi nang dalawang beses
- PAGKAIN / FOOD: Mga daliri nakakulob, tapikin ang bibig
- GUSTO / WANT: Dalawang kamay may kurbadong daliri, hilahin palapit sa katawan
- TULONG / HELP: Isang kamao sa bukas na palad, parehong itaas
- BAHAY / HOME: Ikonekta ang mga daliri (bubong), pababa (pader)
- MABUTI / GOOD: Thumbs up, o patag na kamay mula sa baba pasulong

DAMDAMIN (Emotions):
- MAHAL / LOVE: Dalawang braso krossado sa dibdib
- MAGANDA / BEAUTIFUL: Bukas na kamay gumalaw paligid ng mukha, isara
- MASAYA / HAPPY: Bukas na kamay sa dibdib, paulit-ulit na paakyat
- MALUNGKOT / SAD: Bukas na kamay, pababa mula sa antas ng mukha
- GALIT / ANGRY: Kurbadong kamay sa mukha, hilahin pababa nang may puwersa

KALUSUGAN (Health):
- OSPITAL / HOSPITAL: H-kamay, gumuhit ng + sa itaas ng braso
- DOKTOR / DOCTOR: Dalawang daliri, tapikin ang loob ng pulso
- SAKIT / SICK: Gitnang daliri sa noo at tiyan, pareho pahilig

NUMERO: 1=isang daliri, 2=dalawa, 3=tatlo (kasama thumb), atbp.

=== GRAMÁTICA NG FSL ===
- Ayos ng salita: Paksa-Komento (hindi SVO)
- Ang mga facial expression ay gramatiko: Kilay pataas = tanong (OO/HINDI), Kilay bagsak = WH-tanong
- HINDI para sa negasyon

=== PANUTO ===
Suriin ang LAHAT ng larawan bilang isang pagkakasunod-sunod — hanapin ang galaw at posisyon ng kamay kaugnay ng mukha/katawan.

Para sa NANAY: ang kamay dapat MALAPIT SA BABA
Para sa TATAY: ang kamay dapat MALAPIT SA NOO

MAHAHALAGANG PANUTO:
1. PALAGI kang mag-interpret ng anumang makikita mong kamay o kilos — kahit hindi sigurado, ibigay ang iyong pinakamabuting hulaan.
2. Kapag nakakita ng kamay MALAPIT SA BABA → NANAY; MALAPIT SA NOO → TATAY.
3. Gamitin ang gabay sa itaas para matukoy ang pinaka-angkop na sign.
4. HUWAG mag-iwan ng empty — lagi kang mag-guess.

Sumagot LAMANG ng valid JSON (walang markdown, walang paliwanag):
{
  "glosses": ["SIGN1", "SIGN2"],
  "sentence_fil": "Natural na pangungusap sa Filipino",
  "sentence_en": "Natural English sentence",
  "confidence": "high|medium|low"
}

Tanging kung WALANG KAMAY sa lahat ng larawan: {"glosses":[],"sentence_fil":"","sentence_en":"","confidence":"low"}
HUWAG maglagay ng kahit anong teksto sa labas ng JSON."""

_FALLBACK: dict = {"glosses": [], "sentence_fil": "", "sentence_en": "", "confidence": "low"}

_interpreter_instance: Optional["VisionInterpreter"] = None


class VisionInterpreter:
    """Interprets FSL signs from video frames using Claude Vision.

    Call ``VisionInterpreter.get()`` to obtain the shared singleton.
    """

    def __init__(self, api_key: str, model: str = "claude-opus-4-6") -> None:
        self._api_key = api_key
        self._model = model
        self._client: Optional[object] = None
        self._available = False

        if not api_key:
            logger.warning("VisionInterpreter: no ANTHROPIC_API_KEY — AI interpretation disabled.")
            return

        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
            self._available = True
            logger.info("VisionInterpreter ready (model=%s)", model)
        except ImportError:
            logger.error(
                "VisionInterpreter: 'anthropic' package not installed. "
                "Run: pip install anthropic"
            )

    @property
    def available(self) -> bool:
        return self._available

    def interpret(
        self,
        frames: list[bytes],
        max_frames: int = 10,
        lang: str = "fil",
        lm_snapshots: list | None = None,
    ) -> dict:
        """Interpret FSL from a sequence of JPEG frame bytes.

        Args:
            frames: List of JPEG image bytes (from video or webcam).
            max_frames: Maximum frames to send (sampled evenly to stay within limits).
            lang: Primary language hint — 'fil' for Tagalog, 'en' for English.

        Returns:
            {glosses, sentence_fil, sentence_en, confidence}
        """
        if not self._available or not frames:
            return dict(_FALLBACK)

        sampled = _sample_frames(frames, max_frames)
        if not sampled:
            return dict(_FALLBACK)

        # Build content blocks: one image per sampled frame + instruction text
        content: list[dict] = []
        for jpeg_bytes in sampled:
            b64 = base64.standard_b64encode(jpeg_bytes).decode()
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                },
            })

        # Build landmark-based position description (very helpful for NANAY vs TATAY etc.)
        lm_description = _describe_landmarks(lm_snapshots) if lm_snapshots else ""

        user_text = (
            f"Ang {len(sampled)} na larawan sa itaas ay mga frame mula sa FSL signing clip "
            f"({len(frames)} kabuuang frames na nakolekta).\n"
        )
        if lm_description:
            user_text += f"\nMEDIAPIPE HAND TRACKING DATA:\n{lm_description}\n"
            user_text += "\nGamitin ang positional data sa itaas para matukoy ang tamang FSL sign."
        user_text += "\n\nI-interpret ang mga FSL signs at ibalik sa Filipino."

        content.append({"type": "text", "text": user_text})

        try:
            logger.info("Calling Claude with %d sampled frames (original=%d)", len(sampled), len(frames))
            response = self._client.messages.create(  # type: ignore[union-attr]
                model=self._model,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            raw = response.content[0].text.strip()
            logger.info("Claude raw response: %s", raw[:300])
            result = _parse_json(raw)
            out = {
                "glosses": result.get("glosses", []),
                "sentence_fil": result.get("sentence_fil", ""),
                "sentence_en": result.get("sentence_en", ""),
                "confidence": result.get("confidence", "low"),
            }
            logger.info("Parsed result: glosses=%s fil='%s' conf=%s", out["glosses"], out["sentence_fil"], out["confidence"])
            return out
        except Exception as exc:
            logger.warning("VisionInterpreter.interpret failed: %s", exc)
            return dict(_FALLBACK)

    # ── Singleton ─────────────────────────────────────────────────────────────

    @classmethod
    def get(cls) -> "VisionInterpreter":
        global _interpreter_instance
        if _interpreter_instance is None:
            from app.config import settings
            _interpreter_instance = cls(
                api_key=settings.anthropic_api_key,
                model=settings.ai_model,
            )
        return _interpreter_instance


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sample_frames(frames: list[bytes], n: int) -> list[bytes]:
    """Return up to n frames sampled evenly across the list."""
    if not frames:
        return []
    if len(frames) <= n:
        return list(frames)
    step = len(frames) / n
    return [frames[int(i * step)] for i in range(n)]


# Absolute Y position map (fallback when no face landmarks)
_POSITION_MAP_ABS = [
    (0.00, 0.20, "itaas ng ulo (above head)"),
    (0.20, 0.32, "noo/forehead (TATAY area)"),
    (0.32, 0.45, "mata/ilong (eyes/nose area)"),
    (0.45, 0.58, "bibig/baba (mouth/CHIN — NANAY area)"),
    (0.58, 0.72, "leeg/dibdib (neck/chest — SALAMAT/MAHAL area)"),
    (0.72, 1.00, "tiyan/ibaba (stomach area)"),
]

# Face-relative Y position map (0 = forehead, 1 = chin, 2+ = chest/below)
_POSITION_MAP_REL = [
    (-1.50, -0.30, "itaas ng ulo (well above head)"),
    (-0.30,  0.22, "noo/forehead — TATAY zone"),
    ( 0.22,  0.55, "mata/ilong (between forehead and chin)"),
    ( 0.55,  1.30, "baba/chin — NANAY zone"),
    ( 1.30,  1.80, "leeg/dibdib (neck/chest — SALAMAT zone)"),
    ( 1.80,  2.50, "dibdib (chest — MAHAL/MASAYA zone)"),
    ( 2.50,  4.00, "tiyan/ibaba (stomach/below)"),
]


def _describe_landmarks(lm_snapshots: list) -> str:
    """Convert MediaPipe Holistic landmark snapshots into a text description for Claude.

    Uses face-relative Y (relative_y) when available — this gives Claude the precise
    anatomical position of each hand relative to the signer's face, which is the key
    distinguishing feature for FSL family/greeting signs.

    lm_snapshots: list of per-frame lists.  Each hand dict contains:
        wrist_y, wrist_x, label, and optionally relative_y + face_ref.
    """
    if not lm_snapshots:
        return ""

    lines = []

    # Check whether we have face-relative data available
    all_hands = [h for snap in lm_snapshots for h in (snap or [])]
    has_face_rel = any("relative_y" in h for h in all_hands)

    if has_face_rel:
        lines.append("NOTE: Coordinate system = FACE-RELATIVE (0=forehead, 1=chin, 2+=chest). "
                     "Positive values are BELOW the forehead.")

    # Sample evenly from snapshots
    step = max(1, len(lm_snapshots) // 8)
    for i, snapshot in enumerate(lm_snapshots[::step][:8]):
        if not snapshot:
            continue
        hand_descs = []
        for hand in snapshot:
            label = hand.get("label", "")
            side = "Kanan" if label == "Right" else "Kaliwa" if label == "Left" else ""
            wx = hand.get("wrist_x", 0.5)
            lr = "kanan" if wx > 0.6 else "kaliwa" if wx < 0.4 else "gitna"

            if has_face_rel and "relative_y" in hand:
                ry = hand["relative_y"]
                pos = "hindi matukoy"
                for y_min, y_max, desc in _POSITION_MAP_REL:
                    if y_min <= ry < y_max:
                        pos = desc
                        break
                hand_descs.append(
                    f"{side} kamay: rel_y={ry:+.2f} ({pos}), x={wx:.2f} ({lr})"
                )
            else:
                wy = hand.get("wrist_y", 0.5)
                pos = "hindi matukoy"
                for y_min, y_max, desc in _POSITION_MAP_ABS:
                    if y_min <= wy < y_max:
                        pos = desc
                        break
                hand_descs.append(
                    f"{side} kamay: y={wy:.2f} ({pos}), x={wx:.2f} ({lr})"
                )

        if hand_descs:
            lines.append(f"Frame {(i*step)+1}: {'; '.join(hand_descs)}")

    return "\n".join(lines)


def _parse_json(text: str) -> dict:
    """Extract and parse JSON from Claude's response (handles markdown blocks)."""
    # Strip markdown code fences if present
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            stripped = part.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                continue
    return json.loads(text)
