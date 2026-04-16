"""FSL gloss sequence → Tagalog / English sentence.

FSL Grammar Rules
-----------------
1. Topic-Comment word order (not SVO).
2. WH-question fronting.
3. HINDI negation before verb.
4. Finger-spelling sequences collapsed into a single token.
5. Consecutive duplicate glosses removed (smoothing artefact).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

# ── Translation tables ────────────────────────────────────────────────────────

GLOSS_TO_TAGALOG: dict[str, str] = {
    # Numbers
    "1": "isa", "2": "dalawa", "3": "tatlo", "4": "apat", "5": "lima",
    "6": "anim", "7": "pito", "8": "walo", "9": "siyam", "10": "sampu",
    # Alphabet – keep as-is
    **{chr(c): chr(c) for c in range(ord("A"), ord("Z") + 1)},
    # Family
    "NANAY": "nanay", "TATAY": "tatay", "KAPATID": "kapatid",
    "LOLO": "lolo", "LOLA": "lola", "ANAK": "anak",
    "ASAWA": "asawa", "PAMILYA": "pamilya",
    # Greetings
    "KUMUSTA": "kumusta", "MAGANDANG_UMAGA": "magandang umaga",
    "MAGANDANG_HAPON": "magandang hapon", "MAGANDANG_GABI": "magandang gabi",
    "PAALAM": "paalam",
    # Polite
    "SALAMAT": "salamat", "PAUMANHIN": "paumanhin", "PAKIUSAP": "pakiusap",
    # Yes/No
    "OO": "oo", "HINDI": "hindi",
    # Actions
    "KUMAIN": "kumain", "UMINOM": "uminom", "GUSTO": "gusto",
    "MAHAL": "mahal", "TULONG": "tulong",
    # Food/drink
    "TUBIG": "tubig", "PAGKAIN": "pagkain", "MASARAP": "masarap",
    # Emotions
    "MAGANDA": "maganda", "MASAYA": "masaya", "MALUNGKOT": "malungkot",
    "GALIT": "galit", "TAKOT": "takot", "MABUTI": "mabuti",
    "HINDI_MABUTI": "hindi mabuti",
    # Places / time
    "OSPITAL": "ospital", "DOKTOR": "doktor", "SAKIT": "sakit",
    "BAHAY": "bahay", "TRABAHO": "trabaho", "PAARALAN": "paaralan",
    "ARAW": "araw", "GABI": "gabi",
    "NGAYON": "ngayon", "BUKAS": "bukas", "KAHAPON": "kahapon",
}

GLOSS_TO_ENGLISH: dict[str, str] = {
    # Numbers
    "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    # Alphabet
    **{chr(c): chr(c) for c in range(ord("A"), ord("Z") + 1)},
    # Family
    "NANAY": "mother", "TATAY": "father", "KAPATID": "sibling",
    "LOLO": "grandfather", "LOLA": "grandmother", "ANAK": "child",
    "ASAWA": "spouse", "PAMILYA": "family",
    # Greetings
    "KUMUSTA": "how are you", "MAGANDANG_UMAGA": "good morning",
    "MAGANDANG_HAPON": "good afternoon", "MAGANDANG_GABI": "good evening",
    "PAALAM": "goodbye",
    # Polite
    "SALAMAT": "thank you", "PAUMANHIN": "sorry", "PAKIUSAP": "please",
    # Yes/No
    "OO": "yes", "HINDI": "no",
    # Actions
    "KUMAIN": "eat", "UMINOM": "drink", "GUSTO": "want",
    "MAHAL": "love", "TULONG": "help",
    # Food/drink
    "TUBIG": "water", "PAGKAIN": "food", "MASARAP": "delicious",
    # Emotions
    "MAGANDA": "beautiful", "MASAYA": "happy", "MALUNGKOT": "sad",
    "GALIT": "angry", "TAKOT": "afraid", "MABUTI": "good",
    "HINDI_MABUTI": "not good",
    # Places / time
    "OSPITAL": "hospital", "DOKTOR": "doctor", "SAKIT": "sick",
    "BAHAY": "home", "TRABAHO": "work", "PAARALAN": "school",
    "ARAW": "day", "GABI": "night",
    "NGAYON": "now", "BUKAS": "tomorrow", "KAHAPON": "yesterday",
}

WH_GLOSSES = {"ANO", "SINO", "SAAN", "KAILAN", "BAKIT", "PAANO", "ILAN"}


@dataclass
class GlossBuffer:
    _glosses: list[str] = field(default_factory=list)
    _last_gloss: Optional[str] = None
    _change_counter: int = field(default=0, init=False)
    min_change_count: int = 3

    def push(self, gloss: str) -> bool:
        if gloss in ("<BLANK>", ""):
            return False
        if gloss == self._last_gloss and self._change_counter < self.min_change_count:
            return False
        if gloss != self._last_gloss:
            self._change_counter = 0
        self._last_gloss = gloss
        self._glosses.append(gloss)
        self._change_counter += 1
        return True

    def flush(self) -> list[str]:
        out = list(self._glosses)
        self._glosses.clear()
        self._last_gloss = None
        self._change_counter = 0
        return out

    @property
    def current(self) -> list[str]:
        return list(self._glosses)

    def __len__(self) -> int:
        return len(self._glosses)


class SentenceMapper:
    def map(self, glosses: list[str], lang: str = "fil") -> str:
        if not glosses:
            return ""
        collapsed = self._collapse_fingerspelling(glosses)
        ordered = self._apply_grammar(collapsed)
        table = GLOSS_TO_TAGALOG if lang == "fil" else GLOSS_TO_ENGLISH
        words = [table.get(g, g.lower()) for g in ordered]
        return self._render(words, ordered)

    def _collapse_fingerspelling(self, glosses: list[str]) -> list[str]:
        result, buf = [], []
        for g in glosses:
            if len(g) == 1 and g.isalpha():
                buf.append(g)
            else:
                if buf:
                    result.append("".join(buf))
                    buf.clear()
                result.append(g)
        if buf:
            result.append("".join(buf))
        return result

    def _apply_grammar(self, glosses: list[str]) -> list[str]:
        wh = [i for i, g in enumerate(glosses) if g in WH_GLOSSES]
        if wh:
            idx = wh[-1]
            return [glosses[idx]] + [g for i, g in enumerate(glosses) if i != idx]
        return glosses

    def _render(self, words: list[str], glosses: list[str]) -> str:
        is_question = any(g in WH_GLOSSES for g in glosses)
        text = " ".join(words)
        text = text[0].upper() + text[1:] if text else text
        return text + ("?" if is_question else ".")
