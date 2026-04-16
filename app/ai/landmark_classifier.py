"""Rule-based FSL sign classifier using MediaPipe hand + face landmarks.

Coordinate system
-----------------
When a face is detected the classifier uses FACE-RELATIVE Y coordinates:
  relative_y = (wrist_y - forehead_y) / face_height

  < 0     = above forehead
  0       = at forehead     → TATAY / LOLO zone
  0 – 0.5 = eye/nose area
  0.5–1.2 = mouth/chin      → NANAY / SALAMAT / KUMAIN zone
  1.2–2.0 = neck/shoulder
  2.0–3.5 = chest           → MAHAL / MASAYA zone
  > 3.5   = stomach

When no face is detected it falls back to absolute wrist_y (0=top, 1=bottom).

Two recognition paths
---------------------
1. Position path  — wrist zone + hand-open/closed + motion → FSL vocabulary signs
2. Handshape path — finger extension pattern → letters & numbers
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Position-based FSL sign rules ────────────────────────────────────────────
# (gloss, fil, en, y_min, y_max, needs_open, motion, note)
# y values are in face-relative units (0=forehead, 1=chin).
# Fallback absolute ranges are used when face is absent; those are wider.

_SIGN_RULES = [
    # ── Family signs (position is the primary distinguisher) ──────────────
    # TATAY: open hand at/above forehead
    ("TATAY",     "Tatay",     "Father",         -0.6,  0.22, True,  "any",        "open palm at forehead"),
    # NANAY: open hand at chin/mouth
    ("NANAY",     "Nanay",     "Mother",          0.60,  1.30, True,  "any",        "open palm at chin"),
    # LOLO: same forehead zone but with outward movement
    ("LOLO",      "Lolo",      "Grandfather",    -0.6,  0.22, True,  "moving_out", "palm forehead outward"),
    # LOLA: same chin zone but with outward movement
    ("LOLA",      "Lola",      "Grandmother",     0.60,  1.30, True,  "moving_out", "palm chin outward"),
    ("ANAK",      "Anak",      "Child",           3.00,  5.50, False, "rocking",    "cradling at waist"),
    ("KAPATID",   "Kapatid",   "Sibling",         0.10,  1.50, False, "any",        "shoulder point"),
    ("PAMILYA",   "Pamilya",   "Family",          0.20,  1.80, True,  "circular",   "F-hands circle"),
    ("ASAWA",     "Asawa",     "Spouse",          0.30,  1.60, False, "any",        "link fingers"),

    # ── Greetings ─────────────────────────────────────────────────────────
    ("SALAMAT",   "Salamat",   "Thank you",       0.50,  1.20, True,  "moving_out", "palm from lips forward"),
    ("PAUMANHIN", "Paumanhin", "Sorry",           1.80,  3.20, False, "circular",   "fist circles chest"),
    ("KUMUSTA",   "Kumusta",   "How are you",     0.10,  1.60, True,  "waving",     "flat hand wave"),
    ("PAALAM",    "Paalam",    "Goodbye",        -0.30,  1.20, True,  "waving",     "waving goodbye"),
    ("KAMUSTA",   "Kamusta",   "Hello",           0.10,  1.60, True,  "waving",     "wave"),

    # ── Actions ───────────────────────────────────────────────────────────
    ("KUMAIN",    "Kumain",    "Eat",             0.45,  1.00, False, "tapping",    "fingers tap mouth"),
    ("UMINOM",    "Uminom",    "Drink",           0.45,  1.00, False, "tilting",    "C-shape to mouth"),
    ("TUBIG",     "Tubig",     "Water",           0.45,  1.00, False, "tapping",    "W-hand taps lip"),
    ("MATULOG",   "Matulog",   "Sleep",          -0.30,  0.40, True,  "any",        "hands by cheek/temple"),

    # ── Common ────────────────────────────────────────────────────────────
    ("MABUTI",    "Mabuti",    "Good",           -0.50,  1.20, False, "any",        "thumbs up (any face level)"),
    ("OO",        "Oo",        "Yes",             0.20,  2.00, False, "nodding",    "fist nods"),
    ("HINDI",     "Hindi",     "No",              0.10,  1.80, False, "waving",     "fingers shake"),
    ("TULONG",    "Tulong",    "Help",            1.60,  3.50, False, "moving_up",  "fist on palm rises"),
    ("BAHAY",     "Bahay",     "Home",            0.20,  2.00, True,  "roof",       "fingertips form roof"),
    ("GUSTO",     "Gusto",     "Want",            1.60,  3.50, False, "pulling",    "curved fingers pull in"),
    ("WALA",      "Wala",      "None",            0.30,  2.20, True,  "waving",     "open hands shake"),
    ("MAYROON",   "Mayroon",   "There is",        0.30,  2.20, True,  "moving_out", "open hands forward"),
    ("SINO",      "Sino",      "Who",             0.20,  1.20, False, "any",        "L-hand near lips"),
    ("ANO",       "Ano",       "What",            0.10,  1.80, False, "waving",     "bent fingers wave"),
    ("SAAN",      "Saan",      "Where",           0.10,  1.80, True,  "waving",     "index finger waves"),

    # ── Emotions (chest area) ─────────────────────────────────────────────
    ("MAHAL",     "Mahal",     "Love",            1.80,  3.50, False, "crossed",    "arms crossed on chest"),
    ("MASAYA",    "Masaya",    "Happy",           1.80,  3.50, True,  "moving_up",  "palms brush up chest"),
    ("MALUNGKOT", "Malungkot", "Sad",             0.25,  1.40, True,  "moving_down","hands drop from face"),
    ("GALIT",     "Galit",     "Angry",           0.20,  1.30, False, "pulling",    "claw from face"),
    ("TAKOT",     "Takot",     "Afraid",          1.50,  3.00, False, "shaking",    "hands shake at chest"),
    ("MAGANDA",   "Maganda",   "Beautiful",      -0.30,  0.80, True,  "circular",   "open hand around face"),

    # ── Health ────────────────────────────────────────────────────────────
    ("SAKIT",     "Sakit",     "Sick/Pain",      -0.50,  4.00, False, "any",        "middle fingers forehead+stomach"),
    ("DOKTOR",    "Doktor",    "Doctor",          1.50,  3.00, False, "tapping",    "two fingers tap wrist"),
    ("OSPITAL",   "Ospital",   "Hospital",        0.20,  2.00, False, "any",        "H-hand draws plus"),
]

# ── Absolute-y fallback ranges (used when no face detected) ──────────────────
# These are wider and less accurate but better than nothing.
_SIGN_RULES_ABS = [
    ("TATAY",     "Tatay",     "Father",          0.10,  0.42, True,  "any",        "forehead zone"),
    ("NANAY",     "Nanay",     "Mother",          0.38,  0.66, True,  "any",        "chin zone"),
    ("LOLO",      "Lolo",      "Grandfather",     0.10,  0.42, True,  "moving_out", "forehead+out"),
    ("LOLA",      "Lola",      "Grandmother",     0.38,  0.66, True,  "moving_out", "chin+out"),
    ("ANAK",      "Anak",      "Child",           0.62,  0.92, False, "rocking",    "waist"),
    ("SALAMAT",   "Salamat",   "Thank you",       0.40,  0.72, True,  "moving_out", "chin forward"),
    ("PAUMANHIN", "Paumanhin", "Sorry",           0.52,  0.82, False, "circular",   "chest"),
    ("KUMUSTA",   "Kumusta",   "How are you",     0.22,  0.65, True,  "waving",     "wave"),
    ("PAALAM",    "Paalam",    "Goodbye",         0.15,  0.60, True,  "waving",     "wave"),
    ("KUMAIN",    "Kumain",    "Eat",             0.35,  0.60, False, "tapping",    "mouth"),
    ("UMINOM",    "Uminom",    "Drink",           0.33,  0.60, False, "tilting",    "mouth"),
    ("TUBIG",     "Tubig",     "Water",           0.33,  0.60, False, "tapping",    "lip"),
    ("MABUTI",    "Mabuti",    "Good",            0.15,  0.58, False, "any",        "thumbs up"),
    ("OO",        "Oo",        "Yes",             0.28,  0.70, False, "nodding",    "fist nod"),
    ("HINDI",     "Hindi",     "No",              0.22,  0.65, False, "waving",     "shake"),
    ("TULONG",    "Tulong",    "Help",            0.42,  0.80, False, "moving_up",  "rise"),
    ("BAHAY",     "Bahay",     "Home",            0.22,  0.65, True,  "roof",       "roof"),
    ("GUSTO",     "Gusto",     "Want",            0.42,  0.80, False, "pulling",    "pull"),
    ("MAHAL",     "Mahal",     "Love",            0.52,  0.85, False, "crossed",    "chest"),
    ("MASAYA",    "Masaya",    "Happy",           0.50,  0.85, True,  "moving_up",  "chest up"),
    ("MALUNGKOT", "Malungkot", "Sad",             0.28,  0.68, True,  "moving_down","drop"),
    ("GALIT",     "Galit",     "Angry",           0.26,  0.68, False, "pulling",    "face"),
    ("SAKIT",     "Sakit",     "Sick",            0.18,  0.80, False, "any",        "middle fingers"),
    ("DOKTOR",    "Doktor",    "Doctor",          0.42,  0.76, False, "tapping",    "wrist"),
]


# ── Handshape rules (finger extension patterns → letters & numbers) ───────────
# Pattern: (thumb, index, middle, ring, pinky) — True=extended, False=curled
# base_score: confidence weight (higher = more distinctive pattern)

_HANDSHAPE_RULES = [
    # Numbers (all thumb-specified for precise matching)
    ((False, True,  False, False, False), "1",  "Isa",    "One",    0.92),
    ((False, True,  True,  False, False), "2",  "Dalawa", "Two",    0.90),
    ((False, True,  True,  True,  False), "3",  "Tatlo",  "Three",  0.88),
    ((False, True,  True,  True,  True),  "4",  "Apat",   "Four",   0.88),
    ((True,  True,  True,  True,  True),  "5",  "Lima",   "Five",   0.87),
    ((True,  False, False, False, True),  "6",  "Anim",   "Six",    0.84),
    ((True,  False, False, False, False), "10", "Sampu",  "Ten",    0.78),

    # Distinctive FSL letters
    ((False, False, False, False, True),  "I",  "I",  "Letter I",  0.90),
    ((True,  True,  False, False, False), "L",  "L",  "Letter L",  0.88),
    ((True,  False, False, False, True),  "Y",  "Y",  "Letter Y",  0.87),
    ((False, True,  True,  False, False), "V",  "V",  "Letter V",  0.85),
    ((False, True,  True,  True,  False), "W",  "W",  "Letter W",  0.84),
    ((False, True,  True,  True,  True),  "B",  "B",  "Letter B",  0.82),
    ((True,  True,  True,  False, False), "K",  "K",  "Letter K",  0.80),
    ((False, False, False, False, False), "A",  "A",  "Letter A",  0.62),
    ((False, True,  False, False, True),  "H",  "H",  "Letter H",  0.76),
    ((True,  True,  True,  True,  False), "F",  "F",  "Letter F",  0.72),
]


# ── Public entry point ────────────────────────────────────────────────────────

def classify_from_landmarks(lm_snapshots: list) -> Optional[dict]:
    """Classify FSL sign from a sequence of landmark snapshots.

    Each snapshot is a list of hand dicts:
      {wrist_y, wrist_x, relative_y (optional), label,
       lms:[21×{x,y}], tips, knuckles, face_ref (optional)}

    relative_y is face-normalised (0=forehead, 1=chin) and is used when
    present for much higher position accuracy.

    Returns {glosses, sentence_fil, sentence_en, confidence} or None.
    Always returns something if hands were observed.
    """
    if not lm_snapshots:
        return None

    hand_obs = [h for snap in lm_snapshots for h in (snap or [])]
    if not hand_obs:
        return None

    # ── Decide whether we have reliable face-relative data ────────────────
    rel_ys = [h["relative_y"] for h in hand_obs if "relative_y" in h]
    use_face_relative = len(rel_ys) >= len(hand_obs) * 0.5  # at least half have it

    if use_face_relative:
        avg_y = sum(rel_ys) / len(rel_ys)
        rules = _SIGN_RULES
        coord_mode = "face-relative"
    else:
        avg_y = sum(h.get("wrist_y", 0.5) for h in hand_obs) / len(hand_obs)
        rules = _SIGN_RULES_ABS
        coord_mode = "absolute"

    avg_x      = sum(h.get("wrist_x", 0.5) for h in hand_obs) / len(hand_obs)
    open_ratio = sum(1 for h in hand_obs if _is_open(h)) / len(hand_obs)
    is_open    = open_ratio >= 0.50

    # Finger extensions (averaged, majority vote)
    ext_lists = [_get_finger_extensions(h) for h in hand_obs]
    n = len(ext_lists)
    avg_ext = [sum(e[i] for e in ext_lists) / n >= 0.50 for i in range(5)]

    # Mouth/brow context from the most recent face_ref available
    face_ctx = None
    for h in reversed(hand_obs):
        if h.get("face_ref"):
            face_ctx = h["face_ref"]
            break

    motion = _detect_motion(hand_obs)

    logger.info(
        "Classifier [%s]: y=%.2f open=%.0f%% ext=%s motion=%s mouth_open=%s n=%d",
        coord_mode, avg_y, open_ratio * 100,
        "".join("1" if e else "0" for e in avg_ext),
        motion,
        face_ctx.get("mouth_open") if face_ctx else "?",
        len(hand_obs),
    )

    # ── Run both recognition paths ────────────────────────────────────────
    shape_match = _match_handshape(avg_ext)
    pos_match   = _match_position(avg_y, is_open, motion, rules)

    # ── Winner selection ──────────────────────────────────────────────────
    _UNAMBIGUOUS = {"1", "2", "I", "L", "Y", "H", "6", "10"}
    _LOCATION    = {"TATAY", "NANAY", "LOLO", "LOLA"}

    if shape_match and pos_match:
        sg, sf, se, ss = shape_match
        pg, pf, pe, ps = pos_match

        if sg in _UNAMBIGUOUS:
            # Very distinctive handshape wins regardless of position
            winner = shape_match
        elif pg in _LOCATION and use_face_relative:
            # When face-relative coords are available, trust position for
            # TATAY/NANAY — their whole identity is about face position.
            winner = pos_match
        elif pg in _LOCATION and avg_y < (0.25 if not use_face_relative else 0.45):
            winner = pos_match   # still trust extreme positions even without face
        else:
            winner = shape_match if ss >= ps else pos_match
    elif shape_match:
        winner = shape_match
    elif pos_match:
        winner = pos_match
    else:
        # Hands present but nothing matched — mouth context may help
        if face_ctx and face_ctx.get("mouth_open"):
            return {
                "glosses":      ["KUMAIN"],
                "sentence_fil": "Kumain",
                "sentence_en":  "Eat",
                "confidence":   "low",
            }
        logger.info("Classifier: no match (y=%.2f mode=%s)", avg_y, coord_mode)
        return {
            "glosses":      ["?"],
            "sentence_fil": "Hindi makilala ang sign",
            "sentence_en":  "Sign not recognized",
            "confidence":   "low",
        }

    gloss, fil, en, score = winner
    conf = "high" if score >= 0.88 else "medium" if score >= 0.72 else "low"

    logger.info("Classifier matched: %s (score=%.2f conf=%s mode=%s)",
                gloss, score, conf, coord_mode)
    return {
        "glosses":      [gloss],
        "sentence_fil": fil,
        "sentence_en":  en,
        "confidence":   conf,
    }


# ── Private helpers ───────────────────────────────────────────────────────────

def _get_finger_extensions(hand: dict) -> list[bool]:
    """[thumb, index, middle, ring, pinky] — True = extended."""
    lms = hand.get("lms", [])

    if len(lms) >= 21:
        thumb_ext  = lms[4]["y"] < lms[3]["y"] - 0.005
        index_ext  = lms[8]["y"] < lms[5]["y"] - 0.015
        middle_ext = lms[12]["y"] < lms[9]["y"] - 0.015
        ring_ext   = lms[16]["y"] < lms[13]["y"] - 0.015
        pinky_ext  = lms[20]["y"] < lms[17]["y"] - 0.015
        return [thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext]

    tips     = hand.get("tips", [])
    knuckles = hand.get("knuckles", [])
    wrist_y  = hand.get("wrist_y", 0.5)

    if not tips:
        return [False, False, False, False, False]

    thumb_ext = tips[0].get("y", wrist_y) < wrist_y - 0.06 if len(tips) > 0 else False

    if len(tips) >= 5 and len(knuckles) >= 4:
        index_ext  = tips[1].get("y", 0.5) < knuckles[0].get("y", 0.5) - 0.01
        middle_ext = tips[2].get("y", 0.5) < knuckles[1].get("y", 0.5) - 0.01
        ring_ext   = tips[3].get("y", 0.5) < knuckles[2].get("y", 0.5) - 0.01
        pinky_ext  = tips[4].get("y", 0.5) < knuckles[3].get("y", 0.5) - 0.01
    else:
        thr = wrist_y - 0.04
        index_ext  = len(tips) > 1 and tips[1].get("y", wrist_y) < thr
        middle_ext = len(tips) > 2 and tips[2].get("y", wrist_y) < thr
        ring_ext   = len(tips) > 3 and tips[3].get("y", wrist_y) < thr
        pinky_ext  = len(tips) > 4 and tips[4].get("y", wrist_y) < thr

    return [thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext]


def _is_open(hand: dict) -> bool:
    ext = _get_finger_extensions(hand)
    return sum(ext[1:]) >= 3


def _match_handshape(avg_ext: list[bool]) -> Optional[tuple]:
    best_score = -1.0
    best_match = None

    for pattern, gloss, fil, en, base_score in _HANDSHAPE_RULES:
        defined = [(p, a) for p, a in zip(pattern, avg_ext) if p is not None]
        if not defined:
            continue
        matched = sum(1 for p, a in defined if p == a)
        wrong   = len(defined) - matched
        ratio   = matched / len(defined)
        if ratio < 0.70:
            continue
        score = base_score * ratio - wrong * 0.18
        if score > best_score:
            best_score = score
            best_match = (gloss, fil, en, score)

    return best_match


def _match_position(avg_y: float, is_open: bool, motion: str,
                    rules: list) -> Optional[tuple]:
    best_score = -999.0
    best_match = None

    for gloss, fil, en, y_min, y_max, needs_open, req_motion, _ in rules:
        if not (y_min <= avg_y <= y_max):
            continue

        score = 1.0
        score += 0.35 if (needs_open and is_open) else (-0.25 if needs_open else 0.20 if not is_open else 0.0)

        if req_motion != "any":
            score += 0.45 if motion == req_motion else -0.10

        score -= (y_max - y_min) * 0.30   # wider range → less specific → lower score

        if score > best_score:
            best_score = score
            best_match = (gloss, fil, en, best_score)

    if best_match:
        gloss, fil, en, raw = best_match
        normalised = min(1.0, max(0.0, (raw + 0.5) / 2.2))
        return (gloss, fil, en, normalised)
    return None


def _detect_motion(hand_obs: list) -> str:
    if len(hand_obs) < 4:
        return "any"

    ys = [h.get("wrist_y", 0.5) for h in hand_obs]
    xs = [h.get("wrist_x", 0.5) for h in hand_obs]

    y_range = max(ys) - min(ys)
    x_range = max(xs) - min(xs)

    def reversals(seq):
        return sum(
            1 for i in range(1, len(seq) - 1)
            if (seq[i] - seq[i-1]) * (seq[i+1] - seq[i]) < 0
        )

    y_rev = reversals(ys)
    x_rev = reversals(xs)

    if y_range > 0.05 and y_rev >= 2:   return "tapping"
    if x_range > 0.06 and x_rev >= 2:   return "waving"
    if y_range > 0.06 and ys[-1] > ys[0] + 0.03: return "moving_down"
    if y_range > 0.06 and ys[0] > ys[-1] + 0.03: return "moving_up"
    if x_range > 0.05:                   return "moving_out"
    return "static"
