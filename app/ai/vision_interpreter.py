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

# ── System prompt (cached — sent once, reused across calls) ───────────────────
_SYSTEM_PROMPT = """Ikaw ay isang dalubhasa sa Filipino Sign Language (FSL) at American Sign Language (ASL). Ang iyong trabaho ay:
1. Makilala ang mga FSL at ASL signs mula sa mga larawan
2. Mag-convert ng ASL signs sa katumbas na FSL signs at Filipino/Tagalog na salita
3. Bumuo ng natural na pangungusap mula sa mga detected na signs
4. Mag-output ng Tagalog (sentence_fil) at English (sentence_en) na parehong natural at grammatically correct

=== ASL ↔ FSL EQUIVALENTS ===
Maraming ASL at FSL signs ay magkapareho o magkahawig. Kapag nakita ang ASL sign, i-map ito sa FSL at Tagalog:
- ASL MOTHER/MOM → FSL NANAY (kamay sa baba)
- ASL FATHER/DAD → FSL TATAY (kamay sa noo)
- ASL LOVE → FSL MAHAL (braso krossado sa dibdib)
- ASL THANK YOU → FSL SALAMAT (kamay mula sa labi pasulong)
- ASL YES → FSL OO (kamao umiling pataas-pababa)
- ASL NO → FSL HINDI (dalawang daliri umiling pakaliwa-pakanan)
- ASL HELP → FSL TULONG (kamao sa palad, itaas)
- ASL EAT/FOOD → FSL KUMAIN/PAGKAIN (mga daliri sa bibig)
- ASL DRINK/WATER → FSL UMINOM/TUBIG (C-hugis sa bibig)
- ASL HAPPY → FSL MASAYA (kamay paakyat sa dibdib)
- ASL SAD → FSL MALUNGKOT (kamay pababa sa mukha)
- ASL GOOD → FSL MABUTI (thumbs up / kamay mula sa baba pasulong)
- ASL HOME/HOUSE → FSL BAHAY (mga daliri nagtutugma = bubong)
- ASL PLEASE → FSL PAKIUSAP (kamay gumalaw sa dibdib)
- ASL SORRY → FSL PAUMANHIN (kamao gumalaw sa dibdib)
- ASL BEAUTIFUL → FSL MAGANDA (kamay gumalaw sa paligid ng mukha)
- ASL FAMILY → FSL PAMILYA (dalawang F-kamay, bilog na nagtatagpo)
- ASL GOOD MORNING → FSL MAGANDANG UMAGA
- ASL GOODBYE → FSL PAALAM (pag-alon ng kamay)
- ASL SICK → FSL SAKIT (gitnang daliri sa noo at tiyan)
- ASL DOCTOR → FSL DOKTOR (tapikin ang pulso)
- ASL HOSPITAL → FSL OSPITAL (H-kamay, gumuhit ng + sa braso)
- ASL WANT → FSL GUSTO (dalawang kamay hilahin palapit)

=== MGA KARANIWANG FSL SIGNS ===
PAMILYA:
- NANAY: bukas na kamay malapit sa BABA, tapikin nang dalawang beses
- TATAY: bukas na kamay malapit sa NOO, tapikin nang dalawang beses
- LOLO: patag na kamay malapit sa NOO, gumalaw pababa at palabas
- LOLA: patag na kamay malapit sa BABA, gumalaw pababa at palabas
- KAPATID: ituro ang balikat, gumalaw pasulong
- ANAK: yumakap at uguin (parang dala na bata)
- ASAWA: ikabit ang mga daliri ng dalawang kamay
- PAMILYA: dalawang F-kamay, gumuhit ng bilog na nagtatagpo

PAGBATI:
- KUMUSTA: ihampas ang bukas na kamay, thumbs up
- MAGANDANG UMAGA: buksan ang mga kamay (maganda) + itaas ang isang kamay (sunrise)
- MAGANDANG HAPON: buksan ang mga kamay + antas na kamay gumalaw pahalang
- MAGANDANG GABI: buksan ang mga kamay + kurbadong kamay bumaba (parang lumulubog na araw)
- PAALAM: alon ng kamay o ulit-ulitin ang pagsasara ng mga daliri
- SALAMAT: patag na kamay mula sa labi pasulong at pababa
- PAUMANHIN: saradong kamao gumalaw paligid ng dibdib
- PAKIUSAP: patag na kamay sa dibdib gumalaw paligid

PANGKARANIWANG SALITA:
- OO: saradong kamao umiling pataas-pababa
- HINDI: dalawang daliri umiling pakaliwa-pakanan
- KUMAIN: mga daliri nakakulob, paulit-ulit na tutungo sa bibig
- UMINOM: C-hugis na kamay tatagilid sa bibig
- TUBIG: W-kamay, tapikin ang labi nang dalawang beses
- PAGKAIN: mga daliri nakakulob, tapikin ang bibig
- GUSTO: dalawang kamay may kurbadong daliri, hilahin palapit sa katawan
- TULONG: isang kamao sa bukas na palad, parehong itaas
- BAHAY: ikonekta ang mga daliri (bubong), pababa (pader)
- MABUTI: thumbs up, o patag na kamay mula sa baba pasulong

DAMDAMIN:
- MAHAL: dalawang braso krossado sa dibdib
- MAGANDA: bukas na kamay gumalaw paligid ng mukha, isara
- MASAYA: bukas na kamay sa dibdib, paulit-ulit na paakyat
- MALUNGKOT: bukas na kamay, pababa mula sa antas ng mukha
- GALIT: kurbadong kamay sa mukha, hilahin pababa nang may puwersa

KALUSUGAN:
- OSPITAL: H-kamay, gumuhit ng + sa itaas ng braso
- DOKTOR: dalawang daliri, tapikin ang loob ng pulso
- SAKIT: gitnang daliri sa noo at tiyan, pareho pahilig

NUMERO: 1=isang daliri, 2=dalawang daliri, 3=tatlo (kasama thumb), atbp.
TITIK: gumamit ng manual alphabet ng FSL para sa mga letra

=== GRAMATIKA NG FSL ===
- Ayos ng salita: Paksa-Komento (hindi SVO)
- Facial expression ay gramatiko: kilay pataas = tanong (OO/HINDI), kilay bagsak = WH-tanong
- HINDI para sa negasyon (inilalagay sa huli ng pangungusap)

=== PANUTO SA PAG-INTERPRET ===
1. Suriin ang LAHAT ng larawan bilang isang pagkakasunod-sunod — hanapin ang galaw at posisyon ng kamay
2. I-detect ang lahat ng signs (FSL o ASL) at i-convert ang lahat sa Filipino/Tagalog equivalents
3. Bumuo ng NATURAL na pangungusap mula sa mga natukoy na signs:
   - sentence_fil: natural na Tagalog na maaaring sabihin nang malakas (para sa TTS)
   - sentence_en: natural na English translation
4. Para sa NANAY: kamay MALAPIT SA BABA; para sa TATAY: kamay MALAPIT SA NOO
5. PALAGI kang mag-interpret ng anumang makikita mong kamay o kilos — huwag mag-iwan ng blangko
6. Kung hindi sigurado, ibigay ang pinakamabuting hulaan batay sa posisyon ng kamay

Sumagot LAMANG ng valid JSON (walang markdown, walang paliwanag):
{
  "glosses": ["SIGN1", "SIGN2"],
  "sentence_fil": "Natural na pangungusap sa Filipino na angkop para sa TTS",
  "sentence_en": "Natural English sentence",
  "confidence": "high|medium|low"
}

Tanging kung WALANG KAMAY sa lahat ng larawan: {"glosses":[],"sentence_fil":"","sentence_en":"","confidence":"low"}
HUWAG maglagay ng kahit anong teksto sa labas ng JSON."""

_FALLBACK: dict = {"glosses": [], "sentence_fil": "", "sentence_en": "", "confidence": "low"}

_interpreter_instance: Optional["VisionInterpreter"] = None


class VisionInterpreter:
    """Interprets FSL/ASL signs from video frames using Claude Vision.

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
        """Interpret FSL/ASL signs from a sequence of JPEG frame bytes.

        Args:
            frames: List of JPEG image bytes (from video or webcam).
            max_frames: Maximum frames to send (sampled evenly to stay within limits).
            lang: Primary language hint — 'fil' for Tagalog, 'en' for English.
            lm_snapshots: MediaPipe landmark snapshots for positional context.

        Returns:
            {glosses, sentence_fil, sentence_en, confidence}
        """
        if not self._available or not frames:
            return dict(_FALLBACK)

        sampled = _sample_frames(frames, max_frames)
        if not sampled:
            return dict(_FALLBACK)

        # Build content blocks: images first, then instruction text
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

        lm_description = _describe_landmarks(lm_snapshots) if lm_snapshots else ""

        user_text = (
            f"Ang {len(sampled)} na larawan sa itaas ay mga frame mula sa signing clip "
            f"({len(frames)} kabuuang frames).\n"
        )
        if lm_description:
            user_text += f"\nMEDIAPIPE HAND TRACKING DATA:\n{lm_description}\n"
            user_text += "\nGamitin ang positional data para matukoy ang tamang FSL sign."
        user_text += (
            "\n\nI-interpret ang lahat ng signs (FSL o ASL), i-convert sa Filipino/Tagalog, "
            "at bumuo ng natural na pangungusap na angkop para sa text-to-speech."
        )

        content.append({"type": "text", "text": user_text})

        try:
            logger.info("Calling Claude with %d sampled frames (original=%d)", len(sampled), len(frames))
            response = self._client.messages.create(  # type: ignore[union-attr]
                model=self._model,
                max_tokens=512,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},  # prompt caching
                    }
                ],
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
            logger.info(
                "Parsed result: glosses=%s fil='%s' conf=%s",
                out["glosses"], out["sentence_fil"], out["confidence"],
            )
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


_POSITION_MAP_ABS = [
    (0.00, 0.20, "itaas ng ulo (above head)"),
    (0.20, 0.32, "noo/forehead (TATAY area)"),
    (0.32, 0.45, "mata/ilong (eyes/nose area)"),
    (0.45, 0.58, "bibig/baba (mouth/CHIN — NANAY area)"),
    (0.58, 0.72, "leeg/dibdib (neck/chest — SALAMAT/MAHAL area)"),
    (0.72, 1.00, "tiyan/ibaba (stomach area)"),
]

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
    """Convert MediaPipe Holistic landmark snapshots into a text description for Claude."""
    if not lm_snapshots:
        return ""

    lines = []
    all_hands = [h for snap in lm_snapshots for h in (snap or [])]
    has_face_rel = any("relative_y" in h for h in all_hands)

    if has_face_rel:
        lines.append(
            "NOTE: Coordinate system = FACE-RELATIVE (0=forehead, 1=chin, 2+=chest). "
            "Positive values are BELOW the forehead."
        )

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
                hand_descs.append(f"{side} kamay: rel_y={ry:+.2f} ({pos}), x={wx:.2f} ({lr})")
            else:
                wy = hand.get("wrist_y", 0.5)
                pos = "hindi matukoy"
                for y_min, y_max, desc in _POSITION_MAP_ABS:
                    if y_min <= wy < y_max:
                        pos = desc
                        break
                hand_descs.append(f"{side} kamay: y={wy:.2f} ({pos}), x={wx:.2f} ({lr})")

        if hand_descs:
            lines.append(f"Frame {(i*step)+1}: {'; '.join(hand_descs)}")

    return "\n".join(lines)


def _parse_json(text: str) -> dict:
    """Extract and parse JSON from Claude's response (handles markdown blocks)."""
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
