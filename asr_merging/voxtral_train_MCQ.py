#!/usr/bin/env python3
"""Train/evaluate Voxtral on JSONL multiple-choice audio understanding tasks.

This script is separate from ASR-oriented training in `voxtral_train_router.py` and
is focused on acoustic/semantic MCQ supervision.

Expected JSONL schema per line:
{
  "path": "relative/or/absolute/audio.wav",
  "questions": [
    {
      "question_id": "Q001",
      "question_stem": "...",
      "options": [{"label": "A", "text": "..."}, ...],
      "correct_answer": "A"
    }
  ]
}
"""

from __future__ import annotations

import argparse
import datetime as dt
from collections import Counter, OrderedDict, defaultdict
import hashlib
import inspect
import json
import math
import os
import random
import re
import shlex
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple,\
     Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from datasets import Dataset, load_from_disk
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments, VoxtralForConditionalGeneration, VoxtralProcessor
from asr_merging.voxtral_train_router import (
    _offline_aware_from_pretrained_kwargs,
    _resolve_pretrained_source,
)

# Match voxtral_train_router.py runtime behavior:
# on some CUDA/PyTorch stacks, cuBLAS StridedBatched can fail while cuBLASLt works.
try:
    torch.backends.cuda.preferred_blas_library("cublaslt")
except Exception:
    pass


def _normalize_label(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _normalize_choice_label(s: str) -> str:
    return (s or "").strip().upper()


def _build_dynamic_choice_map(options: Sequence[Dict]) -> Dict[str, str]:
    choice_map: Dict[str, str] = {}
    for opt in options:
        if not isinstance(opt, dict):
            continue
        label = _normalize_choice_label(str(opt.get("label", "")))
        if not label:
            continue
        text = "" if opt.get("text") is None else str(opt.get("text"))
        choice_map[label] = text
    return choice_map


def _resolve_correct_choice_dynamic(choice_map: Dict[str, str], answer_gt: str) -> Optional[str]:
    answer_raw = (answer_gt or "").strip()
    answer_label = _normalize_choice_label(answer_raw)
    if answer_label in choice_map:
        return answer_label

    answer_norm = _normalize_label(answer_raw)
    for key, value in choice_map.items():
        if _normalize_label(value) == answer_norm:
            return key
    return None


def _select_multiple_choice_option(text: str, options: Sequence[str]) -> Optional[str]:
    raw = (text or "").strip()
    if not raw:
        return None

    allowed = [_normalize_choice_label(str(x)) for x in options]
    allowed_set = set(allowed)

    raw_up = _normalize_choice_label(raw)
    if raw_up in allowed_set:
        return raw_up

    raw_norm = _normalize_label(raw)
    for label in allowed:
        if _normalize_label(label) == raw_norm:
            return label

    for token in re.findall(r"[A-Za-z0-9_]+", raw_up):
        if token in allowed_set:
            return token

    for label in sorted(allowed, key=len):
        if len(label) == 1 and label in raw_up:
            return label

    return None


def _resolve_jsonl_audio_path(raw_path: str, audio_root: Optional[str]) -> str:
    p = Path(raw_path)

    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        if audio_root:
            root = Path(audio_root)
            candidates.append(root / p)

            parts = list(p.parts)
            if "mlc-slm-2nd-dev" in parts:
                idx = parts.index("mlc-slm-2nd-dev")
                if idx + 1 < len(parts) and parts[idx + 1] != "data":
                    with_data = parts[: idx + 1] + ["data"] + parts[idx + 1 :]
                    candidates.append(root / Path(*with_data))

        candidates.append(p)

    for cand in candidates:
        if cand.exists():
            return str(cand.resolve())

    if p.is_absolute():
        return str(p)
    if audio_root:
        return str((Path(audio_root) / p).resolve())
    return str(p.resolve())


@dataclass
class MCQSample:
    sample_id: str
    audio_path: str
    question: str
    prompt_text: str
    choice_map: Dict[str, str]
    gold_choice: str
    metadata: Dict


@dataclass
class MCQTaskData:
    name: str
    samples: List[MCQSample]


def _load_transcript(audio_path: str, transcript_dir: Optional[str],
                     max_words: int = 0) -> str:
    """Load the plain-text transcript for *audio_path* from *transcript_dir*.

    Returns an empty string only when the transcript file is not found or
    *transcript_dir* is None/empty.

    Hallucination-loop detection: if any 5-gram appears more than 5 times the
    transcript is treated as corrupted.  The clean tokens before the loop onset
    are kept and an error notice is appended so the model knows the transcript is
    unreliable::

        {pre-loop text}
        [Note: ASR transcription may contain errors]

    When *max_words* > 0 the (possibly truncated) transcript is further limited
    to that many words (applied before the error notice).
    """
    if not transcript_dir:
        return ""
    stem = Path(audio_path).stem
    txt_path = Path(transcript_dir) / f"{stem}.txt"
    if not txt_path.exists():
        return ""
    text = txt_path.read_text(encoding="utf-8").strip()
    words = text.split()
    has_errors = False
    # Hallucination-loop filter: sliding-window density check.
    # A real loop fills a 150-word window with many repetitions of one 5-gram.
    # A naturally repeated phrase (e.g. "how about you i am") appears at most
    # 1-2 times per window and is never flagged.
    if len(words) >= 6:
        from collections import Counter
        _NGRAM, _WIN, _MIN = 5, 150, 5
        _step = max(1, _WIN // 3)
        _loop_phrase, _loop_start = None, len(words)
        for _ws in range(0, max(1, len(words) - _WIN + 1), _step):
            _ww = words[_ws: _ws + _WIN]
            _ng = [" ".join(_ww[_i:_i+_NGRAM]) for _i in range(len(_ww)-_NGRAM)]
            _cnt = Counter(_ng)
            if not _cnt:
                continue
            _top, _c = _cnt.most_common(1)[0]
            if _c > _MIN:
                _tp = _top.split()
                _n = len(_tp)
                _seen = False
                for _i in range(len(words) - _n + 1):
                    if words[_i:_i+_n] == _tp:
                        if _seen:
                            _loop_start = _i
                            break
                        _seen = True
                else:
                    _loop_start = _ws
                _loop_phrase = _top
                break
        if _loop_phrase is not None:
            has_errors = True
            words = words[:_loop_start]
    if max_words > 0 and len(words) > max_words:
        words = words[:max_words]
    clean_text = " ".join(words)
    if has_errors:
        return (clean_text + "\n[Note: ASR transcription may contain errors]").strip()
    return clean_text if max_words > 0 else text


_MCQ_AUDIO_FOCUS_PREFIX: str = (
    "Listen carefully to the audio \u2014 the answer depends on how the "
    "speaker sounds (tone, intonation, pace, voice quality), not just "
    "the words spoken. "
)


def _build_mcq_prompt(question: str, choice_map: Dict[str, str]) -> str:
    choice_lines = [f"{lbl}. {txt}" for lbl, txt in choice_map.items()]
    return (
        _MCQ_AUDIO_FOCUS_PREFIX
        + "Choose the most suitable answer from the options below. "
        f"You must respond with only one label from: {', '.join(choice_map.keys())}.\n\n"
        f"Question: {question}\n\n"
        + "\n".join(choice_lines)
    )


# ---------------------------------------------------------------------------
# ICLShot: one in-context learning shot for training augmentation.
# Fields: (audio_filename, prompt_text, gold_answer)
# ---------------------------------------------------------------------------
class _ICLShot(NamedTuple):
    audio_fname: str   # filename relative to the ICL audio directory
    prompt_text: str   # full MCQ prompt (with audio-focus prefix)
    answer: str        # gold answer label (A, B, C or D)


# ---------------------------------------------------------------------------
# 20 pre-built ICL shot sets for training.
# Each set contains 6 shots: (4opt-A, 4opt-B, 4opt-C, 4opt-D, 2opt-A, 2opt-B)
# covering 21 distinct languages across sets.
# A random set is selected once per batch in VoxtralMCQCollator.
# ---------------------------------------------------------------------------
_ICL_TRAINING_SETS: List[List[_ICLShot]] = [
    # --- Set 0 ---
    [
        # 4opt-A | English_Filipino
        (
            "tr_s00_4optA_20004_001_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: What is the female speaker's tone in 1437.522 - 1444.132 when she speculates, \"I think, they will not have us pay for anything?\"\n\n"
                "A. Hopeful and questioning\nB. Demanding and certain\nC. Angry and sarcastic\nD. Sad and regretful"
            ),
            "A",
        ),
        # 4opt-B | English_American
        (
            "tr_s00_4optB_0707_001.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: In the segment 60.45-67.96, the speaker quotes someone saying, \"I'm not gonna leave this is public transport...\". How does the speaker deliver this quote?\n\n"
                "A. In a quiet, shy voice.\nB. In a matter-of-fact, slightly defiant tone.\nC. In a loud, shouting voice.\nD. In a tearful, pleading tone."
            ),
            "B",
        ),
        # 4opt-C | French_Canada
        (
            "tr_s00_4optC_1033_003_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: At 113.883, when the speaker says, « Robe, je pense à porter, Je ne sais pas vraiment quoi porter comme robe... », what do her pauses and hesitant tone reveal?\n\n"
                "A. She is annoyed by her friend's question.\nB. She has a secret plan she doesn't want to reveal.\nC. She is truly undecided about her choice of dress.\nD. She tries to remember the name of a creator."
            ),
            "C",
        ),
        # 4opt-D | Vietnamese
        (
            "tr_s00_4optD_0257_009_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: Based on the voice, what are the genders of the speaker during the time period 25.041-29.614 and the speaker during the time period 29.614-33.867?\n\n"
                "A. Both are male.\nB. Both are female.\nC. The first speaker is male, the second is female.\nD. The first speaker is female, the second is male."
            ),
            "D",
        ),
        # 2opt-A | French
        (
            "tr_s00_2optA_0311_005_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: Compare the volume of the same speaker's voice (O2) at [358.478-360.557] (\"J'ai bossé et je vais donner de mon mieux.\") and at [300.893-306.868] (\"Pour l'instant je suis à la maison je fais rien\"). Which statement is correct?\n\n"
                "A. The volume is higher in the first segment.\nB. The volume is lower in the first segment."
            ),
            "A",
        ),
        # 2opt-B | Tagalog
        (
            "tr_s00_2optB_60080_003.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: Compare the speech rate of the first speaker (O1) between 98.890 - 109.218 and 262.081 - 267.036. In which segment is it slower and seemingly more emotional?\n\n"
                "A. At 98.890 - 109.218, where he said that his view of his friend is almost like a sibling's.\nB. At 262.081 - 267.036, where he said to his friend that he is sharing all the problems with him."
            ),
            "B",
        ),
    ],
    # --- Set 1 ---
    [
        # 4opt-A | French_Canada
        (
            "tr_s01_4optA_0058_004_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: Comparing the volume of the speaker's voice at 476.25 s (\"Titanic!\") and at 606.80 s (\"J'pense\"), which difference is most notable?\n\n"
                "A. The volume is louder at 476.25 s.\nB. The volume is louder at 606.80 s.\nC. The volume is the same in both cases.\nD. The volume is whispered in both cases."
            ),
            "A",
        ),
        # 4opt-B | English_Australian
        (
            "tr_s01_4optB_1617_014_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: At 379.39-383.96, the speaker says \"You know, as... Outgrown, what a traditional, kids party.\" The pause after \"as\" and the restart with a new word indicates that the speaker:\n\n"
                "A. was changing their mind about their point.\nB. was searching for a better word to complete their thought.\nC. was interrupted by the other speaker.\nD. decided their original point was completely wrong."
            ),
            "B",
        ),
        # 4opt-C | German
        (
            "tr_s01_4optC_0180_005_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: At the end of the story about the hamster and the screwdriver, the narrator says: \"So one doesn't know whether she killed it, but.\" (1219.328 - 1222.271). What does the combination of the uncertain formulation and the tone of the unfinished \"but\" imply?\n\n"
                "A. She firmly believes in her friend's innocence.\nB. She is really uncertain and does not want to pass a judgment.\nC. It strongly suggests that she believes her friend killed the hamster.\nD. She finds the story funny and doesn't think any further about it."
            ),
            "C",
        ),
        # 4opt-D | Japanese
        (
            "tr_s01_4optD_0474_001_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: When comparing the segments 「私は建築家になりたいですって」 starting at 118.4 seconds and 「でもなってないけど」 starting at 126.3 seconds from the female speaker's speech, how does the volume of the voice change?\n\n"
                "A. The volume of the voice hardly changes.\nB. Growing larger and larger\nC. The 「でもなってないけど」 part suddenly becomes larger\nD. The voice becomes quieter at the 「でもなってないけど」 part."
            ),
            "D",
        ),
        # 2opt-A | Turkish
        (
            "tr_s01_2optA_0441_007_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: What tonality does the speaker's voice have when stating that they studied international relations at Marmara University between 76.67 and 79.57?\n\n"
                "A. Deep and male voice\nB. Thin and female voice"
            ),
            "A",
        ),
        # 2opt-B | French
        (
            "tr_s01_2optB_1166_005_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: In the sentence « Bah tu sais euh super ! Super euh.. » (between 15.36 and 17.94), what does the speaker's final hesitation (« euh.. ») combined with her upbeat tone suggest?\n\n"
                "A. She is about to announce bad news.\nB. She is simply cheerful and looks for words to continue the conversation."
            ),
            "B",
        ),
    ],
    # --- Set 2 ---
    [
        # 4opt-A | German
        (
            "tr_s02_4optA_0180_005_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: Which emotion does the speaker primarily convey through her tone in the statement \"Oh no, I hope nothing bad happens?\" (25.868 - 27.613)?\n\n"
                "A. Concern\nB. Curiosity\nC. Indifference\nD. Joy"
            ),
            "A",
        ),
        # 4opt-B | Portuguese
        (
            "tr_s02_4optB_2282_003_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: In the phrase \"que também tá a eh não, não me recordo o nome\" (415.021 - 418.326), what does the combination of the sound \"eh\" and the pause indicate?\n\n"
                "A. Strong disagreement with the interlocutor.\nB. A genuine difficulty in remembering information.\nC. An attempt to interrupt someone else.\nD. A joke."
            ),
            "B",
        ),
        # 4opt-C | English_Indian
        (
            "tr_s02_4optC_00700_002_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: In the sentence \"but I I do not play, cricket, that is why you say that I am not physically good\" [708.89, 712.80], which word does the speaker stress to create a contrast?\n\n"
                "A. not\nB. play\nC. cricket\nD. physically"
            ),
            "C",
        ),
        # 4opt-D | Portuguese
        (
            "tr_s02_4optD_2273_008_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: At 138.788, the speaker says \"...de útil para a sociedade, ah, foi,\". What do the pause and the sound \"ah\" at this point in the sentence—elements not fully captured by the comma in the transcription—indicate?\n\n"
                "A. He is making a dramatic pause to create suspense.\nB. He is pausing to remember what he wanted to say next.\nC. He is taking a pause to breathe deeply.\nD. He is pausing to organize his thoughts before expressing the central idea of his motivation."
            ),
            "D",
        ),
        # 2opt-A | French
        (
            "tr_s02_2optA_0311_005_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: Comparez le volume de la voix de la même locutrice (O2) à [358.478-360.557] (\"J'ai bossé et je vais donner de mon mieux.\") et à [300.893-306.868] (\"Pour l'instant je suis à la maison je fais rien\"). Quelle affirmation est correcte ?\n\n"
                "A. Le volume est plus élevé dans le premier segment.\nB. Le volume est plus bas dans le premier segment."
            ),
            "A",
        ),
        # 2opt-B | French
        (
            "tr_s02_2optB_0345_002_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: At the beginning of his statement at 24.536, what sound does the first speaker emit to express his fatigue towards school?\n\n"
                "A. A laugh\nB. A sigh (pouf)"
            ),
            "B",
        ),
    ],
    # --- Set 3 ---
    [
        # 4opt-A | Portuguese
        (
            "tr_s03_4optA_2239_036_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: What is the interlocutor's general mood at 527.117-532.432 when describing their arrival at a new place (\"Quando cheguei lá era frio, era novembro...\")?\n\n"
                "A. It sounds melancholic and solitary\nB. He sounds happy and enthusiastic\nC. He sounds angry and frustrated\nD. He sounds indifferent"
            ),
            "A",
        ),
        # 4opt-B | Japanese
        (
            "tr_s03_4optB_0315_004_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: From 688.74 seconds to 689.36 seconds, the speaker responds with 「嘘」. Based on the tone of voice at this moment, which emotion is conveyed most strongly?\n\n"
                "A. Suspecting the other person\nB. Surprise accompanied by sympathy\nC. The joy of having an interesting conversation\nD. Strong anger"
            ),
            "B",
        ),
        # 4opt-C | Korean
        (
            "tr_s03_4optC_1138_004.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: While the male narrator is explaining the plot of the movie, he briefly pauses just before uttering the protagonist's line \"나는 죽지 않았어\" at 1122.09 seconds. What effect does this pause have on the storytelling?\n\n"
                "A. The story becomes boring, so I pause for a moment.\nB. I stop because I can't think of what to say next.\nC. Create dramatic tension before delivering shocking content.\nD. Pauses to observe the other person's reaction."
            ),
            "C",
        ),
        # 4opt-D | English_American
        (
            "tr_s03_4optD_0598_001.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: At 133.10s, a speaker agrees that some art is lazy and says it \"just feels like s-\". Based on the sharp cut-off sound in the audio and the context, what word were they likely about to say?\n\n"
                "A. something\nB. stuff\nC. silly\nD. shit"
            ),
            "D",
        ),
        # 2opt-A | Turkish
        (
            "tr_s03_2optA_0441_016_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: During the time interval of 81.84-85.59, what does the speaker's tone and expression imply when saying \"Ama benziyor Dabbe\" about the argumentative woman at Gratis?\n\n"
                "A. Don't mock the situation and make jokes by comparing it to a horror film.\nB. Don't believe that the woman is really a horror movie character."
            ),
            "A",
        ),
        # 2opt-B | Spanish_Mexico
        (
            "tr_s03_2optB_1085_006_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: When the speaker from 268.32-273.92 says that people associate the beach and heat with \"fiesta, descansar\", does their tone suggest personal approval or merely an objective description?\n\n"
                "A. His enthusiastic tone shows that he completely agrees and loves it.\nB. Its tone is descriptive and neutral, simply reporting a common association."
            ),
            "B",
        ),
    ],
    # --- Set 4 ---
    [
        # 4opt-A | Spanish_Mexico
        (
            "tr_s04_4optA_1176_001_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: In fragment 102.705-109.615, what emotion does the speaker convey when describing the experience of eating warm tamales prepared by their mother on a rainy day?\n\n"
                "A. Nostalgia and pleasure\nB. Anxiety and haste\nC. Anger and frustration\nD. Boredom and monotony"
            ),
            "A",
        ),
        # 4opt-B | English_Australian
        (
            "tr_s04_4optB_1127_015_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: What is the speaker's tone when he says \"And he said, Chard. Oh, come on\" around [346.39, 352.27]?\n\n"
                "A. He is angry at his friend.\nB. He is imitating his friend's persuasive tone.\nC. He is making fun of his friend.\nD. He is genuinely pleading with the listener."
            ),
            "B",
        ),
        # 4opt-C | Spanish
        (
            "tr_s04_4optC_2085_003_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: At [2903.994, 2910.103], the speaker says \"pero hubiera estado bien, ¿no? [...] pero bueno, como hemos dicho, pues,\". What attitude does the expression \"pero bueno\" convey in the audio?\n\n"
                "A. Joy for the new movies coming out.\nB. Anger over the end of the trilogy.\nC. Resignation and acceptance that that phase has ended.\nD. Hope that the director makes more movies."
            ),
            "C",
        ),
        # 4opt-D | English_Australian
        (
            "tr_s04_4optD_1141_006_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: In the sentence at 161.38s, \"It's their world, and you're living in it, but with dogs, it's all about, you\", which word receives the most stress at the very end to emphasize the contrast between cats and dogs?\n\n"
                "A. world\nB. dogs\nC. about\nD. you"
            ),
            "D",
        ),
        # 2opt-A | Korean
        (
            "tr_s04_2optA_0220_003.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: What emotion does the speaker's exclamation \"Ah, goodness\" convey at segment 226.542?\n\n"
                "A. A deep sigh and sense of regret as recalling the past\nB. Dislike and dissatisfaction with the other person's words"
            ),
            "A",
        ),
        # 2opt-B | Turkish
        (
            "tr_s04_2optB_0441_013_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: While conveying the speaker's father's words between 666.553 and 670.561 (\"Şifa işte, içelim. Şifa, şifa.\"), what kind of change does the speaker make in volume?\n\n"
                "A. Lowering his voice, he whispers.\nB. Speaks by raising his voice and imitating."
            ),
            "B",
        ),
    ],
    # --- Set 5 ---
    [
        # 4opt-A | French_Canada
        (
            "tr_s05_4optA_0117_006_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: At 1190.996-1193.078, the interlocutor says \"Très bien ça, c'est votre avis, c'est votre avis.\". Based on the tone of his voice, what is his attitude when saying this?\n\n"
                "A. He politely acknowledges the other's opinion while indicating the end of the discussion.\nB. He angrily rejects the other's opinion.\nC. He agrees enthusiastically with the other's opinion.\nD. He is confused by the other's opinion."
            ),
            "A",
        ),
        # 4opt-B | German
        (
            "tr_s05_4optB_0195_002_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: What tone does the speaker use at 499.427 when he says \"Oh, man\"?\n\n"
                "A. Happy and excited\nB. Frustrated and sighing\nC. Anxious and worried\nD. Neutral and emotionless"
            ),
            "B",
        ),
        # 4opt-C | English_Australian
        (
            "tr_s05_4optC_1118_010_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: At [228.62, 230.06], in response to the question \"What do you think the best part about Internet is?\", the speaker says, \"There really isn't.\" How does his tone add to the meaning of his words?\n\n"
                "A. His cheerful tone suggests he is joking.\nB. His hesitant tone suggests he is unsure of his answer.\nC. His flat, immediate delivery conveys a strong, settled pessimism.\nD. His questioning tone suggests he wants the other person to answer."
            ),
            "C",
        ),
        # 4opt-D | Spanish_Mexico
        (
            "tr_s05_4optD_1176_001_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: What ingredient gives a bitter-sweet or sweet touch to German potato salad, according to the description at 893.443-901.348?\n\n"
                "A. The celery\nB. Smoked sausage\nC. The mayonnaise\nD. The carrot"
            ),
            "D",
        ),
        # 2opt-A | Thai
        (
            "tr_s05_2optA_0318_002_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: The speaker's tone from 203.871 - 209.842 reflects what emotion the mother felt when the second dog died.\n\n"
                "A. Great sorrow and sadness\nB. Anger that cannot keep a dog"
            ),
            "A",
        ),
        # 2opt-B | French
        (
            "tr_s05_2optB_0345_002_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: At 772.630, the second speaker repeats the word \"L'isolement\". Given the tone and the immediate repetition by the first speaker, what is the meaning of this word in context?\n\n"
                "A. The thermal insulation of a house.\nB. The fact that a person remains alone and cut off from the world."
            ),
            "B",
        ),
    ],
    # --- Set 6 ---
    [
        # 4opt-A | Japanese
        (
            "tr_s06_4optA_0130_004_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: What state might the speaker be in, based on the unclear statement 「まぁ、東京の方にあるなんか、ねはなんか？」 around the 150-second mark and the tone used at that time?\n\n"
                "A. I can't recall the store's name accurately and seems uncertain as I try to remember it.\nB. Asking the other person for the location of the store.\nC. Criticism of a store in Tokyo.\nD. Intentionally hiding the name of the store."
            ),
            "A",
        ),
        # 4opt-B | German
        (
            "tr_s06_4optB_0213_001_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: In the time range 60.28 - 67.54, the speaker responds to the idea that food is healthy fast food with: \"Really? The question is, is it actually fast food?\" What does his tone reveal about his actual stance?\n\n"
                "A. He is genuinely shocked and can't believe it.\nB. He is amused and playfully questions the term \"Fast Food\" because the preparation took a long time.\nC. He is angry because he considers fast food unhealthy.\nD. He agrees completely with the term \"Fast Food\"."
            ),
            "B",
        ),
        # 4opt-C | English_Australian
        (
            "tr_s06_4optC_1025_011_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: In response to the idea of fame fading, one speaker says, \"it'd be very humbling\" (1260.90s). Judging by the dry and understated tone in the audio, what is the implied meaning?\n\n"
                "A. It would be a spiritually enlightening experience\nB. It would be a positive lesson in humility\nC. It would be an extremely difficult and ego-crushing experience\nD. It would be a relief to no longer be famous"
            ),
            "C",
        ),
        # 4opt-D | English_American
        (
            "tr_s06_4optD_0707_001.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: In the segment 453.00-455.12, which word does the speaker emphasize to contrast her feeling with fear?\n\n"
                "A. even\nB. scared\nC. just\nD. stressed"
            ),
            "D",
        ),
        # 2opt-A | Spanish_Mexico
        (
            "tr_s06_2optA_1169_004_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: In the segment 519.013 - 523.502, when the speaker says \"a mí personalmente se me hace algo ridículo\", what emotion does their tone convey?\n\n"
                "A. Strong disdain and criticism.\nB. Joy and fun."
            ),
            "A",
        ),
        # 2opt-B | French
        (
            "tr_s06_2optB_1166_005_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: In the sentence « Chao Chao ! Bisou ! Bye bye ! » (at 1256.35), what is the speaker's tone?\n\n"
                "A. Cold and distant\nB. Warm and friendly"
            ),
            "B",
        ),
    ],
    # --- Set 7 ---
    [
        # 4opt-A | Vietnamese
        (
            "tr_s07_4optA_0537_002_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: Why does the person speaking from 197.76 to 205.33 feel that being an adult is really \"chán\"?\n\n"
                "A. Because he no longer receives care from his parents and has to worry about work.\nB. Because his dreams never became reality.\nC. Because he has too much free time.\nD. Because he remembers his childhood friends."
            ),
            "A",
        ),
        # 4opt-B | Portuguese
        (
            "tr_s07_4optB_2127_008_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: When speaker O2 asks \"Ma patinhas é um esporto?\" (72.273-74.010), what does their tone of voice reveal about their understanding?\n\n"
                "A. He's making a joke about the word \"patinhas\".\nB. He is genuinely perplexed and questioning whether \"patinhas\" is really a sport.\nC. He agrees that skating is a sport, but uses the wrong word.\nD. He is angry because the other speaker mentioned a sport that he doesn't know."
            ),
            "B",
        ),
        # 4opt-C | French_Canada
        (
            "tr_s07_4optC_1013_003_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: Between 554.278 and 559.958, the speaker says \"Je sais pas comment j'ai même vécu jusqu'à maintenant\". Given her slightly amused tone, how should this phrase be interpreted?\n\n"
                "A. She suffers from amnesia and does not remember her life without a phone.\nB. She is seriously anxious about living without a phone.\nC. She uses hyperbole to humorously emphasize how essential the phone has become in her life.\nD. She raises a real question because she doesn't understand how people used to do things."
            ),
            "C",
        ),
        # 4opt-D | English_Australian
        (
            "tr_s07_4optD_1025_011_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: At 500.92s, the speaker begins with \"Yeah, for sure, but...\". Although \"Yeah, for sure\" literally means agreement, what does the speaker's tone in the audio signal he is about to do?\n\n"
                "A. End the conversation\nB. Tell a personal story\nC. Emphatically agree with the previous point\nD. Introduce a contrasting opinion or downside"
            ),
            "D",
        ),
        # 2opt-A | German
        (
            "tr_s07_2optA_0285_004_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: When the speaker says at 369.44-372.16, \"Ah yes, but Digga, Matthew would never do that,\" what does this imply about her tone regarding her belief in the story?\n\n"
                "A. She doesn't believe in the accusation at all and finds it absurd.\nB. She defends Matthew, but has private doubts."
            ),
            "A",
        ),
        # 2opt-B | Turkish
        (
            "tr_s07_2optB_0441_013_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: During the time interval from 520.453 to 524.956 seconds, on which word segment is the stress placed in \"Yani sebze yemeklerini, et yemeklerine tercih etmem\", and what does this mean?\n\n"
                "A. The word \"tercih etmem\" is emphasized to convey a definite expression of rejection.\nB. \"et yemeklerine\" The word cluster is emphasized to establish a contrast with vegetable-based meals."
            ),
            "B",
        ),
    ],
    # --- Set 8 ---
    [
        # 4opt-A | Portuguese
        (
            "tr_s08_4optA_2314_008_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: At timestamp 786.231, speaker O1 says \"É sério?\" with a high pitch and almost laughing, after hearing that the athlete removed his hat to perform the qualification jump. What does this combination of words and tone mean?\n\n"
                "A. He is expressing amazement and amusement at the athlete's confidence and style.\nB. He is genuinely questioning the truth of the story.\nC. He is criticizing the athlete for being arrogant.\nD. He did not understand the meaning of \"tirou o boné\"."
            ),
            "A",
        ),
        # 4opt-B | Turkish
        (
            "tr_s08_4optB_0441_013_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: Between 541.034 and 543.930 seconds, why does the speaker experience a pause in the form of \"sö-\" when saying \"Bazen sö- sipariş ediyorum eve gelsin\"?\n\n"
                "A. Because he forgot what to say.\nB. Starting with the word \"Söylüyorum\" and correcting itself with the more appropriate word \"sipariş ediyorum\".\nC. Because his attention was distracted by the phone ringing.\nD. The other speaker cut him off."
            ),
            "B",
        ),
        # 4opt-C | Turkish
        (
            "tr_s08_4optC_0441_033_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: The speaker says in a cheerful tone between 1796.85 and 1799.48, \"Gayet was happy, we shook all the constellations.\" Considering the word \"shook\" in conjunction with the overall content and tone of the speech, what does it mean?\n\n"
                "A. We praised and exalted all the zodiac signs.\nB. We conducted a deep and scientific analysis of all zodiac signs.\nC. We discussed rumors about all zodiac signs and criticized their negative traits.\nD. We counted all the zodiac signs in sequence."
            ),
            "C",
        ),
        # 4opt-D | German
        (
            "tr_s08_4optD_0215_021_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: Which emotion is conveyed by the speaker who, at 357.342, says: \"Ah! Now everything makes sense.\"?\n\n"
                "A. True confusion\nB. Real trouble\nC. Sincere agreement\nD. Played, funny realization"
            ),
            "D",
        ),
        # 2opt-A | English_Filipino
        (
            "tr_s08_2optA_00370_006_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: Based on the pitch and vocal characteristics of the two speakers throughout the dialogue, what is most likely true?\n\n"
                "A. Both speakers are female.\nB. One speaker is male and one is female."
            ),
            "A",
        ),
        # 2opt-B | Spanish_Mexico
        (
            "tr_s08_2optB_1169_004_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: In the segment 265.502-268.227, a speaker says \"Y yo, guao\" after hearing that the text is art. What does their tone reveal about the meaning of \"guao\" (wow)?\n\n"
                "A. Expresses genuine admiration for the idea.\nB. Express disbelief and surprise, not admiration."
            ),
            "B",
        ),
    ],
    # --- Set 9 ---
    [
        # 4opt-A | English_American
        (
            "tr_s09_4optA_0660_003.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: What is the tone of the speaker at 634.41-637.88 when they say, \"that sounds kinda sketchy\"?\n\n"
                "A. Suspicious and doubtful\nB. Certain and confident\nC. Joyful and amused\nD. Angry and accusatory"
            ),
            "A",
        ),
        # 4opt-B | English_British
        (
            "tr_s09_4optB_0497_004_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: At 1260.892, a speaker says, 'it's crazy to think that I am a Gen Z'. What does their tone of voice, combined with this statement, imply about their self-perception?\n\n"
                "A. They are proud to be a Gen Z and embrace all its stereotypes.\nB. They feel disconnected from the typical image of their generation and relate more to an older one.\nC. They are angry about being labeled as a Gen Z.\nD. They are genuinely confused about which generation they belong to."
            ),
            "B",
        ),
        # 4opt-C | Tagalog
        (
            "tr_s09_4optC_00276_001_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: In the statement, \"Hindi na nakakain ng karne. Hotdog na lang.\" (693.16-696.11), what is indicated by the speaker's slightly rising tone?\n\n"
                "A. He is serious about complaining and sad.\nB. He really prefers hotdogs over meat.\nC. This is a cheap way of expressing that they cannot afford to buy expensive meat.\nD. He suggested that they cook hotdogs for breakfast."
            ),
            "C",
        ),
        # 4opt-D | English_American
        (
            "tr_s09_4optD_0588_002.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: In the utterance at 1403.74-1407.78 (\"but compared to post secondary exams those exams were simple.\"), which word receives the most stress to highlight the main point of comparison?\n\n"
                "A. compared\nB. post secondary\nC. exams\nD. simple"
            ),
            "D",
        ),
        # 2opt-A | French
        (
            "tr_s09_2optA_1166_005_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: How does the volume of the speaker's voice change from the moment she says « Ah ! C'est vrai. » (at 617.76) to the moment she says « Bah oui j'exerce là-bas maintenant je fais mes stages là-bas. » (at 621.93)?\n\n"
                "A. The volume increases.\nB. The volume decreases."
            ),
            "A",
        ),
        # 2opt-B | Russian
        (
            "tr_s09_2optB_4112_001_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: In response to the request to name thirty favorite films, one of the interviewees at 51.836s says: «Тридцать?». What emotion does his tone convey?\n\n"
                "A. Wonder\nB. Amazement"
            ),
            "B",
        ),
    ],
    # --- Set 10 ---
    [
        # 4opt-A | English_Indian
        (
            "tr_s10_4optA_10694_001_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: In the segment from 559.262s to 567.020s, a speaker says \"I think I feel, I think I feel... I feel not good. I don't feel good.\" What does the hesitation and self-correction in his speech reveal?\n\n"
                "A. He is struggling to find the right words to articulate his negative feelings.\nB. He is trying to remember a specific event that made him feel bad.\nC. He is pretending to be sick to end the conversation.\nD. He is becoming angry and is trying to control his temper."
            ),
            "A",
        ),
        # 4opt-B | Italian
        (
            "tr_s10_4optB_0190_003_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: What is the predominant emotion in O1's tone when speaking about the weather in Finland, saying \"always gray\" and \"it hasn't been very beautiful\" (415.211-458.122)?\n\n"
                "A. Joy\nB. Disappointment\nC. Anger\nD. Fear"
            ),
            "B",
        ),
        # 4opt-C | Urdu
        (
            "tr_s10_4optC_0268_001_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: What is the tone when the speaker mentions during the period 91.806-98.397 that \"General Nalj\" has been renamed to \"General Awareness\"?\n\n"
                "A. Happiness and enthusiasm\nB. Disapproval and complaints\nC. Non-partisan and informative\nD. Jokes and criticism"
            ),
            "C",
        ),
        # 4opt-D | Italian
        (
            "tr_s10_4optD_0085_002_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: What attitude does the tone of voice of the person who says \"Uscire dalla comfort zone, è difficile alla fine\" (1111.77-1116.17) reveal?\n\n"
                "A. Judgment and criticism\nB. Surprise and disbelief\nC. Amusement and irony\nD. Empathy and understanding"
            ),
            "D",
        ),
        # 2opt-A | Portuguese
        (
            "tr_s10_2optA_2128_015_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: At the beginning of the dialogue (4.181 - 8.000), which speaker has the deeper voice?\n\n"
                "A. The speaker who says \"I am recorder number one.\"\nB. The speaker who says \"I am recorder number two.\""
            ),
            "A",
        ),
        # 2opt-B | French_Canada
        (
            "tr_s10_2optB_1033_003_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: Based on the voices of the two speakers in the first 30 seconds, which one has a generally higher-pitched tone?\n\n"
                "A. The first speaker (I'm the recorder, one...)\nB. The second speaker (I am Recorder Two...)"
            ),
            "B",
        ),
    ],
    # --- Set 11 ---
    [
        # 4opt-A | Russian
        (
            "tr_s11_4optA_4098_001_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: How should the phrase «Ну, заработали, значит, чё сделаешь» (142.713) be interpreted, considering its literal meaning and the dismissive, fatalistic tone in the audio?\n\n"
                "A. As an expression of cynical fatalism, not as a serious moral condemnation.\nB. As full and sincere agreement with Stalin's decision on exile.\nC. As an expression of deep sympathy for the suffering people.\nD. As an attempt to joke about a very serious topic."
            ),
            "A",
        ),
        # 4opt-B | French_Canada
        (
            "tr_s11_4optB_1044_004_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: Between 282.23 and 286.94, there is a long pause. What does this pause, followed by the statement « Pendant... un an ou deux ans, j'ai juste complètement arrêté ça », indicate about the interlocutor's (O1) reflection?\n\n"
                "A. He waits for the other speaker to speak.\nB. He tries to remember the exact duration of his musical phase.\nC. He was interrupted by an external noise.\nD. He suddenly finds himself at odds with himself."
            ),
            "B",
        ),
        # 4opt-C | Tagalog
        (
            "tr_s11_4optC_00276_001_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: Upon learning about the samyang-eating contest (870.84-883.29), the speaker said, \"Ay grabe, ang anghang... Di nga namin naubos eh.\" What is indicated by his tone when referring to this experience?\n\n"
                "A. Deep regret for having joined the contest.\nB. Pride because he tried it.\nC. A funny memory of suffering due to the storm.\nD. Ang anger of his companions who did not help him to finish this."
            ),
            "C",
        ),
        # 4opt-D | Vietnamese
        (
            "tr_s11_4optD_0318_001_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: Which emotion is expressed through the word \"Hả?\" spoken at [383.544, 383.921]?\n\n"
                "A. Joy and excitement\nB. Understanding and empathy\nC. Confusion and inability to hear clearly\nD. Shock and disbelief"
            ),
            "D",
        ),
        # 2opt-A | Thai
        (
            "tr_s11_2optA_0318_002_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: During the period 670.574 - 677.049, the speaker's statement of \"เราคงเลือกยี่ห้อโกลเด้น ไม่ใช่เราคงเลือกสายพันธุ์\" indicates something about both the content and the tone.\n\n"
                "A. The speaker accidentally uses the wrong word and quickly corrects their own statement.\nB. The speaker is teasing another listener."
            ),
            "A",
        ),
        # 2opt-B | English_American
        (
            "tr_s11_2optB_0571_002.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: At 905.256, one speaker says \"Of course\" in response to a question. Based on the distinct voice, which speaker is it?\n\n"
                "A. The speaker who has been explaining their gym routine (O2).\nB. The speaker who has been asking questions about fitness (O1)."
            ),
            "B",
        ),
    ],
    # --- Set 12 ---
    [
        # 4opt-A | English_Filipino
        (
            "tr_s12_4optA_00412_005_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: Throughout the entire dialogue, both speakers express a desire for car upgrades or changes. What is a common underlying motivation for both of them?\n\n"
                "A. To personalize their vehicles and enhance their driving experience.\nB. To sell their current cars for a higher price.\nC. To compete with each other for the best car.\nD. To reduce their insurance costs."
            ),
            "A",
        ),
        # 4opt-B | Spanish_Mexico
        (
            "tr_s12_4optB_1169_004_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: In segment 1308.483 - 1314.493, the speaker describes returning to old styles as a form \"bastante irónica de ver cómo estaba funcionando el mundo\", followed by a pause and the conclusion \"porque siento que así es como pasa con todo\". What does this discourse structure reveal?\n\n"
                "A. The speaker is unsure of their argument and is seeking the other's approval.\nB. The speaker is using the pause to generalize their observation about art into a universal principle regarding culture and fashion.\nC. The speaker stops because he realizes that his argument is contradictory.\nD. The speaker takes a pause to give the other person time to process a very complex idea."
            ),
            "B",
        ),
        # 4opt-C | French_Canada
        (
            "tr_s12_4optC_1044_004_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: At 646.52, interlocutor O2 says « Ouais mais moyennement ». Given his tone and the context, what does he really mean about his experience with the violin?\n\n"
                "A. He loved it but doesn't want to admit it.\nB. He disliked it but remained polite.\nC. Her experience has been mixed, with both positive and negative aspects.\nD. He has no clear opinion on the matter."
            ),
            "C",
        ),
        # 4opt-D | German
        (
            "tr_s12_4optD_0215_021_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: Welche Emotion wird durch den Sprecher vermittelt, der bei 357.342 ausruft: „Ah! Jetzt macht das alles Sinn.“?\n\n"
                "A. Echte Verwirrung\nB. Echter Ärger\nC. Aufrichtige Zustimmung\nD. Gespielte, komische Erkenntnis"
            ),
            "D",
        ),
        # 2opt-A | English_British
        (
            "tr_s12_2optA_1137_019_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: Which speaker, based on their voice, discusses the pros and cons of online versus in-store shopping for a smartwatch between 139.329 and 154.865?\n\n"
                "A. The male speaker\nB. The female speaker"
            ),
            "A",
        ),
        # 2opt-B | Korean
        (
            "tr_s12_2optB_1139_001.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: From the utterance between 419.370 seconds and 436.561 seconds, the speaker discusses the cold of winter in Shanghai. Based on the intonation stress, what does the speaker emphasize as the real issue?\n\n"
                "A. The inconvenience of having to wear a lot of clothes outside\nB. The cold felt inside the house, especially when sleeping at night"
            ),
            "B",
        ),
    ],
    # --- Set 13 ---
    [
        # 4opt-A | Japanese
        (
            "tr_s13_4optA_0147_002_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: At 35.41 seconds, the speaker says 「ああ、そうですか？」. Given the neutral and polite tone, what do you think is the primary function of this statement?\n\n"
                "A. Recognize the other person's information and smooth out the flow of conversation.\nB. To express strong suspicion\nC. Request more detailed information\nD. Demonstrating boredom with a topic"
            ),
            "A",
        ),
        # 4opt-B | Portuguese_Brazil
        (
            "tr_s13_4optB_0084_005_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: When the speaker from 575.880 to 580.046 says \"...it happens that you just leave it, right?\", her tone is one of resignation. What experience is she describing?\n\n"
                "A. The joy of finally being able to go to an expensive concert.\nB. The frustration and the decision to give up going to a concert because it is too expensive.\nC. The anger over the shows being so expensive.\nD. Indifference towards the price of concerts."
            ),
            "B",
        ),
        # 4opt-C | English_British
        (
            "tr_s13_4optC_1226_008_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: The first speaker says, \"Yeah, it makes for good stories. That's for sure\" (410.138-412.559), reflecting on his bizarre job. What does his tone add to the literal meaning of his words?\n\n"
                "A. A sense of deep regret and that the stories are not worth the trouble.\nB. A boastful tone, suggesting he is proud of his professional resilience.\nC. A weary but amused tone, implying that the entertainment value is the main positive takeaway from an otherwise negative experience.\nD. A neutral, factual tone, as if he were reading from a report."
            ),
            "C",
        ),
        # 4opt-D | Turkish
        (
            "tr_s13_4optD_0441_033_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: The tone and style of delivery used by the speaker between 144.26 and 147.96 when saying, \"I mean, I can sit and cry for hours over nothing,\" indicates what about their perception of this trait?\n\n"
                "A. Complained about this feature and wanted to change it.\nB. He saw this feature as a weakness and felt ashamed.\nC. He exaggerated about this feature and actually it's not that emotional.\nD. I accept this feature as part of my personality and describe it as a normal situation without complaining about it."
            ),
            "D",
        ),
        # 2opt-A | French_Canada
        (
            "tr_s13_2optA_1013_003_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: Between 167.134 and 184.476, two people are speaking. Which of the two has a voice that is systematically deeper?\n\n"
                "A. The first person who speaks in this segment (167.134-173.850).\nB. The second person speaking in this segment (173.972-184.476)."
            ),
            "A",
        ),
        # 2opt-B | Spanish_Mexico
        (
            "tr_s13_2optB_1169_004_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: Around the 10:52 mark (652.594), a speaker says \"Siento que\", followed by a pause, and then shifts to compare with the music. What does this pause-transition indicate?\n\n"
                "A. What happened to what he was going to say about painting?\nB. That is presenting a new argument by drawing an analogy with another field that it knows well."
            ),
            "B",
        ),
    ],
    # --- Set 14 ---
    [
        # 4opt-A | Turkish
        (
            "tr_s14_4optA_0441_007_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: The speaker \"...pardon bulgar göçmeni mi dedim?\" says between 160.41 and 168.37: This pause and self-correction expression, as clearly understood from both the text and the audio, reveals what?\n\n"
                "A. Realized and corrected the confusion between the two different friends' stories he had told.\nB. That he lied about his friend's origins and was caught doing so\nC. Forgot that part of the conversation and is trying to save time\nD. The other speaker corrected himself and agreed with him."
            ),
            "A",
        ),
        # 4opt-B | Tagalog
        (
            "tr_s14_4optB_60080_003.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: At 1012.8 - 1015.375, the first speaker said \"tol mali yan, wag yan.\" What does his tone indicate about his prangka, yet without a hostile tone?\n\n"
                "A. He is a bad influence.\nB. For him, true friendship involves honest correction of wrongs.\nC. He always wanted to fight with his friends.\nD. He was dominant in their friendship."
            ),
            "B",
        ),
        # 4opt-C | Spanish_Mexico
        (
            "tr_s14_4optC_1080_006_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: In 1200.660 - 1202.194, the first speaker responds \"Sí me ha pasado, pero no\". What does the tone of voice and the way in which he ends the phrase imply?\n\n"
                "A. That in reality never happened to him.\nB. What's scary about it and why he doesn't want to talk about it.\nC. That although it has happened to him, it diminishes the significance of the experience.\nD. That is contradicting the other speaker."
            ),
            "C",
        ),
        # 4opt-D | French_Canada
        (
            "tr_s14_4optD_1033_003_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: At 576.448, a female speaker says « J'ai pas, j'ai pas trop fan » about a coffee. How does her tone alter the literal meaning of her words?\n\n"
                "A. Her curious tone suggests she'd like to taste it again.\nB. Her hesitant tone shows that she is unsure of her opinion.\nC. Her neutral tone shows that she has no opinion.\nD. Her direct and slightly flat tone shows that she really didn't like him."
            ),
            "D",
        ),
        # 2opt-A | Thai
        (
            "tr_s14_2optA_0318_002_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: From 821.304 to 829.118, the speaker says \"ทุกวันนี้ก็เลี้ยงแบบก็ทิ้งทิ้งขว้างขว้างนะ\", but based on the tone and the following sentence (\"แต่มันก็อยู่ดีกินดีแบบอ้วนอะไรอย่างเงี้ยทุกตัวเลย\"), what is the true meaning of this statement?\n\n"
                "A. It's humble, but in reality, it's well taken care of.\nB. It's a complaint about having to care for a cat unwillingly and abandoning it."
            ),
            "A",
        ),
        # 2opt-B | English_British
        (
            "tr_s14_2optB_1284_013_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: Which speaker, identifiable by their voice, talks about having an allergic reaction to piri piri?\n\n"
                "A. The speaker with the lower-pitched voice\nB. The speaker with the higher-pitched voice"
            ),
            "B",
        ),
    ],
    # --- Set 15 ---
    [
        # 4opt-A | Italian
        (
            "tr_s15_4optA_0100_004_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: In the segment 279.364 - 284.798, the female speaker says: \"È vero, è vero, ma diciamo che adoriamo fare shopping, un po' tutto l'anno\". What attitude does her tone of voice suggest?\n\n"
                "A. A guilty but amused admission of her love for shopping.\nB. A deep frustration with his habit of spending.\nC. A sarcastic tone, as if he completely disliked shopping.\nD. A flat and indifferent tone."
            ),
            "A",
        ),
        # 4opt-B | Japanese
        (
            "tr_s15_4optB_0130_004_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: Based on the unnatural ending and tone of the utterance 「後のあんまり合ってはいないですけども、その服ねなんか？」 starting at 522 seconds, what are we likely to infer the speaker (O1) was trying to convey?\n\n"
                "A. I regret not having met my friends anymore.\nB. I'm trying to recall a specific story about the clothes my friend sold me, but I'm having trouble putting it into words.\nC. Trying to subtly convey that the clothes given by a friend were not to their liking\nD. Asking the other person if they know about that friend"
            ),
            "B",
        ),
        # 4opt-C | Italian
        (
            "tr_s15_4optC_0204_001_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: In segment 712.226-716.462, the speaker says: \"Cioè a me a sorridere e non piangere...\". Considering the context in which he is criticized for his lack of experience, what is his true emotional state, suggested by his tone of voice?\n\n"
                "A. True joy and optimism\nB. Deep sadness and despair\nC. Cynical and bitter ironic resignation\nD. Confusion and uncertainty"
            ),
            "C",
        ),
        # 4opt-D | Portuguese_Brazil
        (
            "tr_s15_4optD_0084_001_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: In the utterance \"E dessa vez realmente foi né?\" (199.38-201.03), what does the speaker emphasize with their tone of voice?\n\n"
                "A. That she always thought she had covid, but never did.\nB. That she doesn't believe it was covid.\nC. When was the first time she felt sick?\nD. That, unlike the other times when he suspected, this time the diagnosis was positive."
            ),
            "D",
        ),
        # 2opt-A | English_Indian
        (
            "tr_s15_2optA_00126_002_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: Based on their voice, which speaker expresses a preference for villains more than heroes?\n\n"
                "A. The speaker with the higher-pitched voice\nB. The speaker with the lower-pitched voice"
            ),
            "A",
        ),
        # 2opt-B | Portuguese_Brazil
        (
            "tr_s15_2optB_0084_005_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: In the segment 44.659 - 52.095, the speaker says \"I saw that you went, yes, to the festival...\". What is the function of the word 'is', which is followed by a brief pause in the audio?\n\n"
                "A. It's a question to confirm the information.\nB. It's a filler word (filler) while it organizes the sentence."
            ),
            "B",
        ),
    ],
    # --- Set 16 ---
    [
        # 4opt-A | French_Canada
        (
            "tr_s16_4optA_0058_004_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: À 418.82 s, l'intervenant dit de Brad Pitt : \"j'peux pas me souvenir de son dernier film\". Comment sa pause et son ton juste avant de mentionner \"Ocean's Thirteen\" (444.74) renforcent-ils cette affirmation ?\n\n"
                "A. Ils montrent qu'il a dû réfléchir longuement, suggérant que l'acteur est moins présent dans son esprit.\nB. Ils prouvent qu'il mentait et qu'il connaissait le film depuis le début.\nC. Ils indiquent qu'il n'aime pas le film et hésite à le nommer.\nD. Ils montrent qu'il est distrait et pense à autre chose."
            ),
            "A",
        ),
        # 4opt-B | Thai
        (
            "tr_s16_4optB_0288_003_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: Based on O1's speech throughout the conversation about past technology (such as old YouTube, the pre-cellphone era), what is his overall attitude toward that period?\n\n"
                "A. Feeling sad and wanting to forget that difficult period.\nB. Feeling nostalgic and viewing it as a good time\nC. Feeling indifferent, no special emotional attachment.\nD. Feeling amused by the technological lag of the past."
            ),
            "B",
        ),
        # 4opt-C | Portuguese_Brazil
        (
            "tr_s16_4optC_0140_003_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: Why does the speaker (O2) feel tired about grilling meat at social gatherings?\n\n"
                "A. Because he doesn't like to socialize.\nB. Because he thinks grilled meat is too expensive.\nC. Because the task of cooking always remains for him.\nD. Because he thinks that electric barbecue is not good."
            ),
            "C",
        ),
        # 4opt-D | English_Australian
        (
            "tr_s16_4optD_1118_010_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: When recounting how he helped an old lady with her email for two hours [882.81, 898.12], what is the speaker's prevailing emotion?\n\n"
                "A. Annoyance at how long it took.\nB. Pride in his technical skills.\nC. Frustration with the old lady's incompetence.\nD. Satisfaction and warmth from helping someone grateful."
            ),
            "D",
        ),
        # 2opt-A | Turkish
        (
            "tr_s16_2optA_0441_016_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: When the speaker \"Nesini beğenmedin diyor ya, beğenmedim işte\" says this between 1442.33 and 1444.83, which emotion is the tone strongly emphasizing, contrary to the simple statement in the text?\n\n"
                "A. Dizziness and anxiety\nB. Grief and regret"
            ),
            "A",
        ),
        # 2opt-B | English_British
        (
            "tr_s16_2optB_1137_019_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: Based on the voice, what is the gender of the speaker who says they are looking for a new pair of running shoes between 37.304 and 44.836?\n\n"
                "A. Male\nB. Female"
            ),
            "B",
        ),
    ],
    # --- Set 17 ---
    [
        # 4opt-A | Vietnamese
        (
            "tr_s17_4optA_0593_001_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: At 133.180, the woman says to the man: \"Ừ, anh cũng biết nhiều phết nhờ.\". What does her tone suggest about the true meaning of this statement?\n\n"
                "A. A sincere and slightly surprised compliment.\nB. A bitter remark that he only knows the basics.\nC. A question to check if he knows anything else.\nD. A complaint that he talks too much."
            ),
            "A",
        ),
        # 4opt-B | Vietnamese
        (
            "tr_s17_4optB_0055_006_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: Based on the surprised tone in the question \"phải tiêm hả bạn?\" (897.506 - 900.588), what can be inferred about speaker O1's knowledge?\n\n"
                "A. She's pretending not to know to tease the other person.\nB. She really didn't know that cats needed vaccinations.\nC. She knows but disagrees with the vaccination.\nD. She is asking again to confirm information she already knows."
            ),
            "B",
        ),
        # 4opt-C | English_American
        (
            "tr_s17_4optC_0598_001.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: At the end of the conversation (397.26s), a speaker says, \"...and that's the end. Okay we're done.\" How does the speaker's tone in the audio clarify the meaning of \"Okay we're done\"?\n\n"
                "A. It's a serious and abrupt command to stop the conversation immediately.\nB. It's a sad statement, expressing disappointment that the conversation is over.\nC. It's a playful, joking remark to signal the end of the recording.\nD. It's a formal announcement marking the official conclusion."
            ),
            "C",
        ),
        # 4opt-D | Portuguese_Brazil
        (
            "tr_s17_4optD_0140_003_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: At 7:31, when describing the food they ate at Renan's house using the word 'Bad,' what is the speaker's tone?\n\n"
                "A. Playful and exaggerated.\nB. Thoughtful and uncertain.\nC. Sarcastic.\nD. Direct, short and negative."
            ),
            "D",
        ),
        # 2opt-A | Italian
        (
            "tr_s17_2optA_0103_003_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: During the conversation, which of the two speakers has a voice that is generally deeper?\n\n"
                "A. The interlocutor who found the kitten (O1)\nB. The interlocutor listening to the story of the kitten (O2)"
            ),
            "A",
        ),
        # 2opt-B | English_British
        (
            "tr_s17_2optB_1284_013_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: Which speaker, identifiable by their higher-pitched voice, mentions being a \"good girl\"?\n\n"
                "A. The speaker who ordered orange and cinnamon hot chocolate\nB. The speaker who ordered pistachio hot chocolate"
            ),
            "B",
        ),
    ],
    # --- Set 18 ---
    [
        # 4opt-A | English_Indian
        (
            "tr_s18_4optA_00205_002_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: When the speaker says 'Hmm good absolutely...' at 592.58, what does the combination of the initial sound 'Hmm' and the emphatic tone on 'absolutely' convey?\n\n"
                "A. A brief moment of thought followed by strong, enthusiastic agreement.\nB. Hesitation and doubt, followed by forced agreement.\nC. Boredom with the conversation, trying to end it quickly.\nD. Confusion about the previous statement, followed by a guess."
            ),
            "A",
        ),
        # 4opt-B | Portuguese_Brazil
        (
            "tr_s18_4optB_0155_005_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: Compare the speech rate of the same speaker in 40.818-46.261 (complaining about the internet) and 517.913-525.524 (describing a positive use of the internet). What can be observed?\n\n"
                "A. The pace is faster when describing positive uses and slower when complaining.\nB. The pace is notably faster when criticizing and more deliberate when describing positive usage.\nC. The speech rate remains constant and unchanged in both moments.\nD. The speaking pace is slow and hesitant in both segments."
            ),
            "B",
        ),
        # 4opt-C | Vietnamese
        (
            "tr_s18_4optC_0318_001_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: At the segment [437.398, 441.864], speaker O1 says \"Mẹ lông nó vàng chẳng vịt mạ vàng\". What does his tone convey about the intent of this statement?\n\n"
                "A. He agreed that the duck was a gold-feathered duck.\nB. He was curious about the difference between yellow feathers and yellow rice.\nC. He dismissed the lie, claiming it was just a common yellow-feathered duck, and referred to the other person as \"đừng có nói phét\".\nD. He is praising the duck with golden-looking feathers."
            ),
            "C",
        ),
        # 4opt-D | French_Canada
        (
            "tr_s18_4optD_1033_003_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: At 314.858, the interlocutor's discourse is fragmented as she describes a friend (« Ouais, comme, elle est comment, mais, comme... »). What does this hesitant way of expressing her feelings toward the situation suggest?\n\n"
                "A. She admires her friend's courage.\nB. She is indifferent to this friend's behavior.\nC. She searches for the right words to express her joy.\nD. She expresses a certain discomfort or disapproval."
            ),
            "D",
        ),
        # 2opt-A | Portuguese
        (
            "tr_s18_2optA_2186_036_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: At the beginning of the dialogue (17.70-27.51), which voice is heard first?\n\n"
                "A. A female voice asking about music platforms.\nB. A female voice that responds liking to listen to radio."
            ),
            "A",
        ),
        # 2opt-B | Japanese
        (
            "tr_s18_2optB_0130_004_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: Comparing the speaking pace of speaker (O1) in the 32-45 second interval and the 145-149 second interval, which interval has a faster speaking speed?\n\n"
                "A. 32-45 second interval\nB. Section from 145 to 149 seconds"
            ),
            "B",
        ),
    ],
    # --- Set 19 ---
    [
        # 4opt-A | Portuguese
        (
            "tr_s19_4optA_2127_008_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: How does the volume of speaker O2's voice change when saying \"É vinte e seis mano!\" (244.820-246.551) compared to their previous speech?\n\n"
                "A. The volume increases significantly.\nB. The volume decreases significantly.\nC. The volume remains exactly the same.\nD. The speaker whispers the phrase."
            ),
            "A",
        ),
        # 4opt-B | Tagalog
        (
            "tr_s19_4optB_60012_002.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: At 713.006 - 714.460, what emotion is being conveyed through the speaker's tone as he says \"Alangan kainin mo ng buo iyan.\"\n\n"
                "A. Surprise\nB. Frenzied fight\nC. Hurt\nD. Serious explanation"
            ),
            "B",
        ),
        # 4opt-C | Urdu
        (
            "tr_s19_4optC_1319_001_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: From 524.27 to 531.59, when the woman wants to know more about coal smoke, what does her tone indicate?\n\n"
                "A. Boredom and lack of interest\nB. Doubt and uncertainty\nC. Intense suspicion and passion\nD. Fear and anxiety"
            ),
            "C",
        ),
        # 4opt-D | Vietnamese
        (
            "tr_s19_4optD_0257_009_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B, C, D.\n\n"
                "Question: What emotion is conveyed by the speaker in the statement \"đấy nên là mình làm việc mình có vẻ mình không hứng thú lắm\" (595.165-598.549)?\n\n"
                "A. Happy and humorous.\nB. Angry and frustrated.\nC. Indifferent and uninterested.\nD. A bit disappointed and low on energy."
            ),
            "D",
        ),
        # 2opt-A | Japanese
        (
            "tr_s19_2optA_0263_002_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: When comparing the speaking pace of speaker O1, which is faster between 481.05-488.84 (the Manchurian speech) and 25.69-31.14 (the discussion on the definition of pets)?\n\n"
                "A. While talking about Manchurian issues\nB. When talking about the definition of pets"
            ),
            "A",
        ),
        # 2opt-B | English_Australian
        (
            "tr_s19_2optB_1025_011_phone.wav",
            (
                "Listen carefully to the audio — the answer depends on how the speaker sounds (tone, intonation, pace, voice quality), not just the words spoken. Choose the most suitable answer from the options below. You must respond with only one label from: A, B.\n\n"
                "Question: Which speaker, based on their voice, brings up the example of Cristiano Ronaldo having the most followers on Instagram?\n\n"
                "A. The female speaker\nB. The male speaker"
            ),
            "B",
        ),
    ],
]


def load_jsonl_audio_mcq(
    jsonl_path: str,
    audio_root: Optional[str],
    max_questions_per_audio: int,
    max_samples: int,
    seed: int,
) -> MCQTaskData:
    src = Path(jsonl_path)
    if not src.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    samples: List[MCQSample] = []

    with src.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw_audio_path = str(row.get("path") or row.get("audio_path") or "").strip()
            if not raw_audio_path:
                continue
            audio_path = _resolve_jsonl_audio_path(raw_audio_path, audio_root)

            questions = row.get("questions") or []
            if not isinstance(questions, list):
                continue
            if max_questions_per_audio > 0:
                questions = questions[:max_questions_per_audio]

            for qi, q in enumerate(questions):
                if not isinstance(q, dict):
                    continue

                question = "" if q.get("question_stem") is None else str(q.get("question_stem"))
                choice_map = _build_dynamic_choice_map(q.get("options") or [])
                if len(choice_map) < 2:
                    continue

                gold_raw = "" if q.get("correct_answer") is None else str(q.get("correct_answer"))
                gold_choice = _resolve_correct_choice_dynamic(choice_map, gold_raw)
                if gold_choice is None:
                    continue

                qid = q.get("question_id") or str(qi)
                sample_id = f"jsonl:{line_no}:{qid}"
                samples.append(
                    MCQSample(
                        sample_id=sample_id,
                        audio_path=audio_path,
                        question=question,
                        prompt_text=_build_mcq_prompt(question, choice_map),
                        choice_map=choice_map,
                        gold_choice=gold_choice,
                        metadata={
                            "jsonl_line": line_no,
                            "source_audio_path": raw_audio_path,
                            "question_id": qid,
                            "question": question,
                            "sample_weight": float(row.get("sample_weight", 1.0)),
                            "task_name": q.get("task_name") if q.get("task_name") is not None else row.get("task_name"),
                            "language": q.get("language") if q.get("language") is not None else row.get("language"),
                            "difficulty": q.get("difficulty") if q.get("difficulty") is not None else row.get("difficulty"),
                            "category": q.get("category") if q.get("category") is not None else row.get("category"),
                            "subtype": q.get("subtype") if q.get("subtype") is not None else row.get("subtype"),
                            "sub-category": q.get("sub-category") if q.get("sub-category") is not None else row.get("sub-category"),
                            "sub-sub-category": q.get("sub-sub-category") if q.get("sub-sub-category") is not None else row.get("sub-sub-category"),
                            "linguistics_sub_discipline": q.get("linguistics_sub_discipline")
                            if q.get("linguistics_sub_discipline") is not None
                            else row.get("linguistics_sub_discipline"),
                        },
                    )
                )

    rng = random.Random(seed)
    rng.shuffle(samples)
    if max_samples > 0:
        samples = samples[: min(max_samples, len(samples))]

    if not samples:
        raise RuntimeError("No usable MCQ samples found in JSONL.")

    return MCQTaskData(name="jsonl_audio_mc", samples=samples)


def _extract_language_from_path(audio_path: str) -> str:
    """Extract root language from the audio file's parent directory name.

    Audio paths follow the convention:
        mlc-slm-2nd-dev/{Language[_Dialect]}/filename.wav

    Examples:
        mlc-slm-2nd-dev/English_Australian/1127_015_phone.wav  → 'English'
        mlc-slm-2nd-dev/French_Canada/file.wav                 → 'French'
        mlc-slm-2nd-dev/Japanese/file.wav                      → 'Japanese'

    The language folder is always the direct parent of the audio file
    (``Path.parts[-2]``). The root language is the part before the first
    underscore, so dialect suffixes (``_Australian``, ``_Brazil``, etc.) are
    stripped — all English variants are grouped as ``'English'``, etc.
    """
    parts = Path(audio_path.replace("\\", "/")).parts
    if len(parts) >= 2:
        lang_dir = parts[-2]  # parent directory = language folder
        return lang_dir.split("_")[0]
    return "unknown"


def _samples_to_hf_dataset(samples: List[MCQSample]) -> Dataset:
    rows = []
    for s in samples:
        rows.append(
            {
                "sample_id": s.sample_id,
                "audio_path": s.audio_path,
                "question": s.question,
                "prompt_text": s.prompt_text,
                "gold_choice": s.gold_choice,
                "choice_labels": list(s.choice_map.keys()),
                "choices": s.choice_map,
                "metadata": s.metadata,
                "sample_weight": float(s.metadata.get("sample_weight", 1.0)),
                "language": _extract_language_from_path(s.audio_path),
            }
        )
    return Dataset.from_list(rows)


def _dataset_to_samples(ds: Dataset) -> List[MCQSample]:
    out: List[MCQSample] = []
    for row in ds:
        metadata = row.get("metadata") or {}
        choice_map = row.get("choices") or {}
        out.append(
            MCQSample(
                sample_id=str(row.get("sample_id") or ""),
                audio_path=str(row.get("audio_path") or ""),
                question=str(row.get("question") or metadata.get("question") or ""),
                prompt_text=str(row.get("prompt_text") or ""),
                choice_map={str(k): str(v) for k, v in choice_map.items()},
                gold_choice=str(row.get("gold_choice") or ""),
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        )
    return out


def _resolve_processed_split_dir(cache_dir: Path, split_name: str) -> Path:
    return cache_dir / split_name / "processed"


def _resolve_sharded_split_dir(cache_dir: Path, split_name: str, num_shards: int, shard_index: int) -> Path:
    shard_tag = f"shard_{int(shard_index):05d}_of_{int(num_shards):05d}"
    return cache_dir / "shards" / split_name / shard_tag / "processed"


def _load_cache_split_dataset(
    cache_dir: Path,
    split_name: str,
    num_shards: int,
    shard_index: int,
) -> Dataset:
    shard_dir = _resolve_sharded_split_dir(cache_dir, split_name, num_shards, shard_index)
    if num_shards > 1 and shard_dir.exists():
        return load_from_disk(str(shard_dir))

    split_dir = _resolve_processed_split_dir(cache_dir, split_name)
    if not split_dir.exists():
        raise FileNotFoundError(f"Cached split not found: {split_dir}")

    ds = load_from_disk(str(split_dir))
    if num_shards > 1:
        indices = [i for i in range(len(ds)) if (i % num_shards) == shard_index]
        ds = ds.select(indices)
    return ds


def _split_train_eval(samples: List[MCQSample], eval_fraction: float, seed: int) -> Tuple[List[MCQSample], List[MCQSample]]:
    if eval_fraction <= 0.0:
        return samples, []
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_eval = max(1, int(round(len(samples) * eval_fraction)))
    n_eval = min(n_eval, max(0, len(samples) - 1))
    eval_idx = set(idx[:n_eval])
    train_samples = [s for i, s in enumerate(samples) if i not in eval_idx]
    eval_samples = [s for i, s in enumerate(samples) if i in eval_idx]
    return train_samples, eval_samples


def _apply_shard(samples: List[MCQSample], num_shards: int, shard_index: int) -> List[MCQSample]:
    if num_shards <= 1:
        return samples
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards).")
    return [s for i, s in enumerate(samples) if (i % num_shards) == shard_index]


def _load_audio_ids_for_shard(cache_dir: Path, split_name: str, num_shards: int, shard_index: int) -> Set[str]:
    shard_tag = f"shard_{int(shard_index):05d}_of_{int(num_shards):05d}"
    p = cache_dir / "audio_shards" / split_name / f"{shard_tag}_audio_ids.json"
    if not p.exists():
        raise FileNotFoundError(f"Audio shard mapping not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    ids = data.get("audio_ids") or []
    return {str(x) for x in ids}


def _audio_id_from_path(audio_path: str) -> str:
    # Keep this aligned with build_mcq_cache_from_jsonl.py.
    import hashlib

    return hashlib.sha1(str(audio_path).encode("utf-8")).hexdigest()


def _apply_audio_shard(
    samples: List[MCQSample],
    audio_shard_cache_dir: str,
    split_name: str,
    num_shards: int,
    shard_index: int,
) -> List[MCQSample]:
    if num_shards <= 1:
        return samples
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards).")

    cache_dir = Path(audio_shard_cache_dir)
    shard_audio_ids = _load_audio_ids_for_shard(cache_dir, split_name, num_shards, shard_index)
    return [s for s in samples if _audio_id_from_path(s.audio_path) in shard_audio_ids]


def _resolve_adapter_path(path_str: str) -> Optional[Path]:
    cp = Path(path_str)
    if not cp.exists():
        return None

    if (cp / "adapter_config.json").exists():
        return cp

    final_model = cp / "final_model"
    if (final_model / "adapter_config.json").exists():
        return final_model

    checkpoints = sorted(
        [p for p in cp.glob("checkpoint-*") if p.is_dir() and (p / "adapter_config.json").exists()],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    return checkpoints[-1] if checkpoints else None


def _choose_dtype(use_bf16: bool, use_fp16: bool) -> torch.dtype:
    if use_bf16:
        return torch.bfloat16
    if use_fp16:
        return torch.float16
    return torch.float32


def _load_model(
    model_id: str,
    model_mode: str,
    adapter_path: Optional[str],
    dtype: torch.dtype,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    base_adapter_path: Optional[str] = None,
):
    """Load the Voxtral model and apply LoRA.

    When *base_adapter_path* is provided the workflow is:
      1. Load the frozen pre-trained base model.
      2. Load the ASR (or other pre-trained) LoRA adapter from *base_adapter_path*.
      3. Merge the adapter weights into the base model with ``merge_and_unload``
         so the resulting weights act as a better-initialised starting point.
      4. Apply a fresh MCQ LoRA on top of the merged model (same as the
         ``baseline`` path but starting from the merged weights instead of
         the raw pre-trained model).
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank} if torch.cuda.is_available() else "auto"
    load_kwargs = _offline_aware_from_pretrained_kwargs({"torch_dtype": dtype, "device_map": device_map})
    model = VoxtralForConditionalGeneration.from_pretrained(
        _resolve_pretrained_source(model_id),
        **load_kwargs,
    )

    # Optional: merge a pre-trained ASR adapter into the base weights before
    # attaching the new MCQ LoRA, giving it a better-initialised starting point.
    if base_adapter_path:
        resolved_base = _resolve_adapter_path(base_adapter_path)
        if resolved_base is None:
            raise FileNotFoundError(
                f"Could not resolve --base-adapter-path from: {base_adapter_path}"
            )
        print(f"[base_adapter] Loading and merging pre-trained adapter: {resolved_base}")
        model = PeftModel.from_pretrained(model, str(resolved_base), is_trainable=False)
        model = model.merge_and_unload()
        print("[base_adapter] Adapter merged into base weights.")

    if model_mode == "adapter":
        if not adapter_path:
            raise ValueError("--adapter-path is required when --model-mode adapter")
        resolved = _resolve_adapter_path(adapter_path)
        if resolved is None:
            raise FileNotFoundError(f"Could not resolve adapter path from: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(resolved), is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            bias="none",
            target_modules="all-linear",
        )
        model = get_peft_model(model, lora_cfg)

    return model


def _print_trainable_params(model) -> None:
    try:
        if hasattr(model, "get_nb_trainable_parameters"):
            trainable, total = model.get_nb_trainable_parameters()
        else:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        pct = (100.0 * float(trainable) / float(total)) if total else 0.0
        print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] Could not compute trainable parameter summary: {type(e).__name__}: {e}")


def _is_encoder_param_name(name: str) -> bool:
    """True for audio encoder (tower/feature-extractor) parameters only."""
    n = (name or "").lower()
    encoder_tokens = [
        "audio_tower",
        "audio_encoder",
        "speech_encoder",
        "feature_extractor",
    ]
    return any(tok in n for tok in encoder_tokens)


def _is_connector_param_name(name: str) -> bool:
    """True for multi-modal connector/projector parameters only."""
    n = (name or "").lower()
    connector_tokens = [
        "multi_modal_projector",
        "modality_projector",
        "audio_projector",
        "projector",
        "connector",
        "bridge",
    ]
    return any(tok in n for tok in connector_tokens)


def _is_encoder_connector_param_name(name: str) -> bool:
    """True for encoder OR connector parameters (kept for backward compat)."""
    return _is_encoder_param_name(name) or _is_connector_param_name(name)


def _split_dual_lr_named_groups(model) -> Dict[str, List[Tuple[str, torch.nn.Parameter]]]:
    """Two groups: encoder+connector vs LLM."""
    no_decay_terms = ["bias", "layernorm.weight", "layer_norm.weight", "norm.weight"]
    grouped: Dict[str, List[Tuple[str, torch.nn.Parameter]]] = {
        "enc_decay": [],
        "enc_nodecay": [],
        "llm_decay": [],
        "llm_nodecay": [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_enc = _is_encoder_connector_param_name(name)
        is_nodecay = any(t in name.lower() for t in no_decay_terms)
        if is_enc and is_nodecay:
            grouped["enc_nodecay"].append((name, param))
        elif is_enc:
            grouped["enc_decay"].append((name, param))
        elif is_nodecay:
            grouped["llm_nodecay"].append((name, param))
        else:
            grouped["llm_decay"].append((name, param))
    return grouped


def _split_triple_lr_named_groups(model) -> Dict[str, List[Tuple[str, torch.nn.Parameter]]]:
    """Three groups: encoder / connector / LLM (each split into decay/nodecay)."""
    no_decay_terms = ["bias", "layernorm.weight", "layer_norm.weight", "norm.weight"]
    grouped: Dict[str, List[Tuple[str, torch.nn.Parameter]]] = {
        "encoder_decay": [], "encoder_nodecay": [],
        "connector_decay": [], "connector_nodecay": [],
        "llm_decay": [], "llm_nodecay": [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_enc  = _is_encoder_param_name(name)
        is_conn = _is_connector_param_name(name)
        is_nodecay = any(t in name.lower() for t in no_decay_terms)
        if is_enc:
            grouped["encoder_nodecay" if is_nodecay else "encoder_decay"].append((name, param))
        elif is_conn:
            grouped["connector_nodecay" if is_nodecay else "connector_decay"].append((name, param))
        else:
            grouped["llm_nodecay" if is_nodecay else "llm_decay"].append((name, param))
    return grouped


def _print_dual_lr_group_preview(model, topk: int) -> None:
    groups = _split_dual_lr_named_groups(model)
    order = ["enc_decay", "enc_nodecay", "llm_decay", "llm_nodecay"]
    topk = max(1, int(topk))
    print("Dual-LR dry-run: parameter group preview")
    for key in order:
        items = groups[key]
        n_tensors = len(items)
        n_params = sum(int(p.numel()) for _, p in items)
        print(f"  - {key}: tensors={n_tensors:,}, params={n_params:,}")
        top_items = sorted(items, key=lambda x: int(x[1].numel()), reverse=True)[:topk]
        for rank, (name, p) in enumerate(top_items, start=1):
            print(f"      {rank:02d}. {name} [{int(p.numel()):,}]")


def _print_triple_lr_group_preview(model, topk: int) -> None:
    groups = _split_triple_lr_named_groups(model)
    order = ["encoder_decay", "encoder_nodecay", "connector_decay", "connector_nodecay", "llm_decay", "llm_nodecay"]
    topk = max(1, int(topk))
    print("Triple-LR dry-run: parameter group preview")
    for key in order:
        items = groups[key]
        n_params = sum(int(p.numel()) for _, p in items)
        print(f"  - {key}: tensors={len(items):,}, params={n_params:,}")
        top_items = sorted(items, key=lambda x: int(x[1].numel()), reverse=True)[:topk]
        for rank, (name, p) in enumerate(top_items, start=1):
            print(f"      {rank:02d}. {name} [{int(p.numel()):,}]")


class TripleLRTrainer(Trainer):
    """Trainer with three independent LR groups: encoder / connector / LLM."""

    def __init__(
        self,
        *args,
        encoder_learning_rate: float,
        connector_learning_rate: float,
        llm_learning_rate: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.encoder_learning_rate = float(encoder_learning_rate)
        self.connector_learning_rate = float(connector_learning_rate)
        self.llm_learning_rate = float(llm_learning_rate)

    def create_optimizer(self):  # type: ignore[override]
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        named_groups = _split_triple_lr_named_groups(opt_model)
        grouped = {k: [p for _, p in v] for k, v in named_groups.items()}

        total_trainable = sum(int(p.numel()) for p in opt_model.parameters() if p.requires_grad)

        lr_map = {
            "encoder_decay":    (self.encoder_learning_rate,   float(self.args.weight_decay)),
            "encoder_nodecay":  (self.encoder_learning_rate,   0.0),
            "connector_decay":  (self.connector_learning_rate, float(self.args.weight_decay)),
            "connector_nodecay":(self.connector_learning_rate, 0.0),
            "llm_decay":        (self.llm_learning_rate,       float(self.args.weight_decay)),
            "llm_nodecay":      (self.llm_learning_rate,       0.0),
        }
        optimizer_grouped_parameters = [
            {"params": grouped[k], "lr": lr, "weight_decay": wd}
            for k, (lr, wd) in lr_map.items() if grouped[k]
        ]

        enc_params  = sum(int(p.numel()) for p in grouped["encoder_decay"]    + grouped["encoder_nodecay"])
        conn_params = sum(int(p.numel()) for p in grouped["connector_decay"]  + grouped["connector_nodecay"])
        llm_params  = sum(int(p.numel()) for p in grouped["llm_decay"]        + grouped["llm_nodecay"])
        print(
            "Triple-LR optimizer enabled: "
            f"encoder lr={self.encoder_learning_rate:g} ({enc_params:,} params), "
            f"connector lr={self.connector_learning_rate:g} ({conn_params:,} params), "
            f"llm lr={self.llm_learning_rate:g} ({llm_params:,} params), "
            f"total trainable={total_trainable:,}."
        )
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            betas=(float(self.args.adam_beta1), float(self.args.adam_beta2)),
            eps=float(self.args.adam_epsilon),
        )
        return self.optimizer


class DualLRTrainer(Trainer):
    """Trainer with two LR groups: encoder+multi_modal_projector vs LLM."""

    def __init__(
        self,
        *args,
        encoder_connector_learning_rate: float,
        llm_learning_rate: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.encoder_connector_learning_rate = float(encoder_connector_learning_rate)
        self.llm_learning_rate = float(llm_learning_rate)

    def create_optimizer(self):  # type: ignore[override]
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        named_groups = _split_dual_lr_named_groups(opt_model)
        grouped = {k: [p for _, p in v] for k, v in named_groups.items()}

        total_trainable = 0
        for _, param in opt_model.named_parameters():
            if param.requires_grad:
                total_trainable += int(param.numel())

        optimizer_grouped_parameters = []
        if grouped["enc_decay"]:
            optimizer_grouped_parameters.append(
                {
                    "params": grouped["enc_decay"],
                    "lr": float(self.encoder_connector_learning_rate),
                    "weight_decay": float(self.args.weight_decay),
                }
            )
        if grouped["enc_nodecay"]:
            optimizer_grouped_parameters.append(
                {
                    "params": grouped["enc_nodecay"],
                    "lr": float(self.encoder_connector_learning_rate),
                    "weight_decay": 0.0,
                }
            )
        if grouped["llm_decay"]:
            optimizer_grouped_parameters.append(
                {
                    "params": grouped["llm_decay"],
                    "lr": float(self.llm_learning_rate),
                    "weight_decay": float(self.args.weight_decay),
                }
            )
        if grouped["llm_nodecay"]:
            optimizer_grouped_parameters.append(
                {
                    "params": grouped["llm_nodecay"],
                    "lr": float(self.llm_learning_rate),
                    "weight_decay": 0.0,
                }
            )

        enc_params = sum(int(p.numel()) for p in grouped["enc_decay"] + grouped["enc_nodecay"])
        llm_params = sum(int(p.numel()) for p in grouped["llm_decay"] + grouped["llm_nodecay"])
        print(
            "Dual-LR optimizer enabled: "
            f"encoder+multi_modal_projector lr={self.encoder_connector_learning_rate:g} ({enc_params:,} params), "
            f"llm lr={self.llm_learning_rate:g} ({llm_params:,} params), "
            f"total trainable={total_trainable:,}."
        )

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            betas=(float(self.args.adam_beta1), float(self.args.adam_beta2)),
            eps=float(self.args.adam_epsilon),
        )
        return self.optimizer


class LanguageBalancedSampler(DistributedSampler):
    """DDP-aware sampler that interleaves samples by language each epoch.

    Samples are grouped by their ``language`` column.  At each epoch, within
    each language group the indices are shuffled (seeded by seed+epoch), then
    all groups are interleaved:  L0[0], L1[0], ..., LN[0], L0[1], L1[1], …

    Because every contiguous window of *num_languages* positions covers each
    language exactly once, each GPU's strided slice (rank::num_replicas) also
    cycles evenly across languages, giving every gradient step diverse language
    coverage.

    Inherits from ``torch.utils.data.distributed.DistributedSampler`` so that
    accelerate / HF Trainer recognise it as already-distributed and do not
    replace it with their own sampler.
    """

    def __init__(
        self,
        dataset,
        language_key: str = "language",
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
        drop_last: bool = True,
    ):
        # Initialise DistributedSampler for isinstance compatibility; we
        # override __iter__ / __len__ completely.
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, seed=seed, drop_last=drop_last)

        # Fast column access on HF Dataset
        langs: list = dataset[language_key]
        lang_to_idx: dict = defaultdict(list)
        for i, lang in enumerate(langs):
            lang_to_idx[str(lang) if lang else "unknown"].append(i)
        self._lang_to_idx: dict = dict(lang_to_idx)
        self._languages: list = sorted(self._lang_to_idx.keys())

        n_langs = len(self._languages)
        min_per_lang = min(len(v) for v in self._lang_to_idx.values())
        # Round total to a multiple of num_replicas so slices are equal-length
        total = (min_per_lang * n_langs // num_replicas) * num_replicas
        self._num_samples_per_replica: int = total // num_replicas
        self._total_size: int = total

        langs_str = ", ".join(
            f"{l}({len(self._lang_to_idx[l])})" for l in self._languages
        )
        print(
            f"[LanguageBalancedSampler] {n_langs} languages | "
            f"min_per_lang={min_per_lang} | "
            f"total={total} | per_replica={self._num_samples_per_replica}\n"
            f"  {langs_str}"
        )

    # set_epoch(epoch) is inherited from DistributedSampler and sets self.epoch ✓

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        # Shuffle within each language group
        shuffled = {
            lang: rng.permutation(indices).tolist()
            for lang, indices in self._lang_to_idx.items()
        }
        # Interleave across languages
        n = min(len(shuffled[l]) for l in self._languages)
        interleaved: list = []
        for i in range(n):
            for lang in self._languages:
                interleaved.append(shuffled[lang][i])

        interleaved = interleaved[: self._total_size]
        # Strided split: rank 0 → positions 0, R, 2R, …; rank 1 → 1, R+1, …
        indices = interleaved[self.rank : self._total_size : self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self._num_samples_per_replica


class WeightedLossTrainer(Trainer):
    """Trainer that scales the per-step CE loss by a ``sample_weight`` tensor
    passed through the batch (produced by VoxtralMCQCollator).

    Works correctly with batch_size=1 per GPU: the weight is a scalar that
    multiplies the model's already mean-reduced loss, effectively up-weighting
    or down-weighting each sample's gradient contribution.

    Weight conventions (set per JSONL row):
      1.0  – English audio, English questions (en/en original)
      2.0  – non-English audio, NLLB-translated English questions
      3.0  – non-English audio, original cross-lingual questions (non-en/non-en)
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("sample_weight", None)
        outputs = model(**inputs)
        loss = outputs.loss
        if loss is not None and weights is not None:
            w = weights.to(loss.device).mean()
            loss = loss * w
        return (loss, outputs) if return_outputs else loss


class CheckpointLossLoggerCallback(TrainerCallback):
    """Save latest train/eval losses each time a checkpoint is written.

    Also accumulates all training losses logged between consecutive eval/save
    points so that *avg_train_loss_since_last_eval* is available for
    generalisation-gap checkpoint selection.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.log_path = self.output_dir / "training_ckpt_log.jsonl"
        self._latest_train_loss: Optional[float] = None
        self._latest_eval_loss: Optional[float] = None
        # Accumulated train losses since the last checkpoint/eval flush
        self._train_losses_since_last_eval: List[float] = []
        # Track the best gen-gap checkpoint seen so far (copied to a fixed dir)
        self._best_gap: Optional[float] = None
        self._best_gap_dir: Path = Path(output_dir) / "best_gen_gap_checkpoint"

    @staticmethod
    def _as_float_or_none(v) -> Optional[float]:
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    @staticmethod
    def _latest_from_history(state, key: str) -> Optional[float]:
        hist = getattr(state, "log_history", None) or []
        for row in reversed(hist):
            if isinstance(row, dict) and key in row:
                try:
                    return float(row[key])
                except Exception:
                    continue
        return None

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs:
            return
        if "loss" in logs and "eval_loss" not in logs:
            v = self._as_float_or_none(logs.get("loss"))
            if v is not None:
                self._latest_train_loss = v
                self._train_losses_since_last_eval.append(v)
        if "eval_loss" in logs:
            self._latest_eval_loss = self._as_float_or_none(logs.get("eval_loss"))

    def on_save(self, args, state, control, **kwargs):  # type: ignore[override]
        if not getattr(state, "is_world_process_zero", True):
            return

        step = int(getattr(state, "global_step", 0) or 0)
        epoch = self._as_float_or_none(getattr(state, "epoch", None))
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{step}"

        train_loss = self._latest_train_loss
        eval_loss = self._latest_eval_loss
        if train_loss is None:
            train_loss = self._latest_from_history(state, "loss")
        if eval_loss is None:
            eval_loss = self._latest_from_history(state, "eval_loss")

        # Average of all training losses logged since the previous save/eval
        if self._train_losses_since_last_eval:
            avg_train = sum(self._train_losses_since_last_eval) / len(self._train_losses_since_last_eval)
        else:
            avg_train = train_loss  # fallback to latest if nothing accumulated

        gen_gap = abs(eval_loss - avg_train) if (eval_loss is not None and avg_train is not None) else None

        payload = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "global_step": step,
            "epoch": epoch,
            "checkpoint_dir": str(ckpt_dir),
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "avg_train_loss_since_last_eval": avg_train,
            "generalization_gap": gen_gap,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        # Reset accumulator after each checkpoint so next interval is fresh
        self._train_losses_since_last_eval = []

        # Keep a copy of the best gen-gap checkpoint seen so far (adapter files only)
        if gen_gap is not None and ckpt_dir.exists():
            if self._best_gap is None or gen_gap < self._best_gap:
                self._best_gap = gen_gap
                import shutil as _shutil
                # Only copy adapter files — skip optimizer/rng states to save disk
                _ADAPTER_SUFFIXES = {".safetensors", ".json", ".bin", ".txt", ".md"}
                _SKIP_NAMES = {"optimizer.pt", "scheduler.pt"}
                tmp = self._best_gap_dir.with_name(self._best_gap_dir.name + "_tmp")
                if tmp.exists():
                    _shutil.rmtree(tmp)
                tmp.mkdir(parents=True)
                for item in ckpt_dir.iterdir():
                    if item.name in _SKIP_NAMES or item.name.startswith("rng_state"):
                        continue
                    if item.suffix in _ADAPTER_SUFFIXES or item.is_dir():
                        dest = tmp / item.name
                        if item.is_dir():
                            _shutil.copytree(item, dest)
                        else:
                            _shutil.copy2(item, dest)
                if self._best_gap_dir.exists():
                    _shutil.rmtree(self._best_gap_dir)
                tmp.rename(self._best_gap_dir)
                print(f"[gen_gap] New best checkpoint saved: gap={gen_gap:.4f} step={step}")


def _pick_best_ckpt_by_gen_gap(log_path: Path) -> Optional[Path]:
    """Return the checkpoint dir that minimises |eval_loss - avg_train_loss|.

    Using the absolute difference avoids biasing toward very early checkpoints
    where train loss is still high (making the signed gap spuriously negative).
    Falls back to min eval_loss if gap information is unavailable.
    """
    if not log_path.exists():
        return None
    entries = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Filter to entries that have an eval_loss and a valid checkpoint dir
    valid = [
        e for e in entries
        if e.get("eval_loss") is not None and e.get("checkpoint_dir")
        and Path(e["checkpoint_dir"]).exists()
    ]
    if not valid:
        return None

    # Prefer entries with gap; fall back to min eval_loss
    with_gap = [e for e in valid if e.get("generalization_gap") is not None]
    if with_gap:
        best = min(with_gap, key=lambda e: abs(e["generalization_gap"]))
        print(
            f"[gen_gap] Best checkpoint: {best['checkpoint_dir']}  "
            f"gap={best['generalization_gap']:.4f}  "
            f"eval_loss={best['eval_loss']:.4f}  "
            f"avg_train_loss={best.get('avg_train_loss_since_last_eval')}"
        )
    else:
        best = min(valid, key=lambda e: e["eval_loss"])
        print(
            f"[gen_gap] Falling back to min eval_loss: {best['checkpoint_dir']}  "
            f"eval_loss={best['eval_loss']:.4f}"
        )
    return Path(best["checkpoint_dir"])


def _build_voxtral_chat_input_features(processor, chat_audio: List[np.ndarray]) -> torch.Tensor:
    target_frames = 3000
    pad_to_samples = 480000
    feature_tensors = []
    for audio_array in chat_audio:
        audio_inputs = processor.feature_extractor(
            audio_array,
            sampling_rate=16000,
            padding=True,
            truncation=False,
            pad_to_multiple_of=pad_to_samples,
        )
        raw_features = None
        if isinstance(audio_inputs, dict):
            raw_features = audio_inputs.get("input_features")
        elif hasattr(audio_inputs, "input_features"):
            raw_features = audio_inputs.input_features
        elif isinstance(audio_inputs, (list, tuple)) and audio_inputs:
            raw_features = audio_inputs[0]

        if raw_features is None:
            raise RuntimeError("Could not extract input_features for Voxtral chat audio.")

        feats = np.asarray(raw_features, dtype=np.float32)
        if feats.ndim == 3 and feats.shape[0] == 1:
            feats = feats[0]
        if feats.ndim == 2:
            if feats.shape[1] % target_frames != 0:
                pad_size = target_frames - (feats.shape[1] % target_frames)
                feats = np.pad(feats, ((0, 0), (0, pad_size)), mode="constant")
            feats = feats.reshape(feats.shape[0], -1, target_frames).transpose(1, 0, 2)
        elif feats.ndim == 3:
            if feats.shape[2] != target_frames:
                raise RuntimeError(f"Unexpected chunked feature shape: {feats.shape}")
        else:
            raise RuntimeError(f"Unexpected feature shape: {feats.shape}")

        feature_tensors.append(torch.as_tensor(feats))

    return torch.cat(feature_tensors, dim=0)


def _to_1d_long(x) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.tensor(x)
    t = t.to(dtype=torch.long)
    if t.ndim == 0:
        t = t.unsqueeze(0)
    elif t.ndim >= 2:
        t = t[0]
    return t.reshape(-1)


def _parse_time_value(token: str) -> Optional[float]:
    s = (token or "").strip().lower()
    s = s.replace(",", ".")
    if not s:
        return None

    if ":" in s:
        parts = s.split(":")
        try:
            vals = [float(x) for x in parts]
        except Exception:
            return None
        if len(vals) == 2:
            mm, ss = vals
            if ss < 0:
                return None
            return max(0.0, mm * 60.0 + ss)
        if len(vals) == 3:
            hh, mm, ss = vals
            if mm < 0 or ss < 0:
                return None
            return max(0.0, hh * 3600.0 + mm * 60.0 + ss)
        return None

    m = re.match(
        r"^([0-9]+(?:[\.,][0-9]+)?)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds)?$",
        s,
    )
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2) or "s"
    if unit.startswith("h"):
        return max(0.0, val * 3600.0)
    if unit.startswith("m"):
        return max(0.0, val * 60.0)
    return max(0.0, val)


def _extract_time_ranges_from_text(text: str) -> List[Tuple[float, float]]:
    q = str(text or "")
    if not q:
        return []

    ranges: List[Tuple[float, float]] = []
    used_spans: List[Tuple[int, int]] = []

    time_atom = (
        r"(?:"
        r"\d+(?::\d{1,2}){1,2}"
        r"|"
        r"\d+(?:[\.,]\d+)?(?:\s*(?:h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds))?"
        r")"
    )
    range_re = re.compile(rf"(?P<a>{time_atom})\s*(?:-|–|—|to)\s*(?P<b>{time_atom})", flags=re.IGNORECASE)

    for m in range_re.finditer(q):
        a = _parse_time_value(m.group("a"))
        b = _parse_time_value(m.group("b"))
        if a is None or b is None:
            continue
        lo, hi = (a, b) if a <= b else (b, a)
        ranges.append((lo, hi))
        used_spans.append((m.start(), m.end()))

    # Bracket-range format used by MLC26 challenge: [22.12, 24.83] (float seconds,
    # comma-separated in square brackets).  Must come after range_re so that spans
    # are registered in used_spans before the colon-time fallback runs.
    bracket_range_re = re.compile(
        r"\[\s*(?P<a>\d+(?:[.,]\d+)?)\s*,\s*(?P<b>\d+(?:[.,]\d+)?)\s*\]"
    )
    for m in bracket_range_re.finditer(q):
        a = _parse_time_value(m.group("a"))
        b = _parse_time_value(m.group("b"))
        if a is None or b is None:
            continue
        lo, hi = (a, b) if a <= b else (b, a)
        ranges.append((lo, hi))
        used_spans.append((m.start(), m.end()))

    point_kw_re = re.compile(
        rf"(?:at|around|near|timestamp|time)\s*(?P<t>{time_atom})",
        flags=re.IGNORECASE,
    )
    for m in point_kw_re.finditer(q):
        t = _parse_time_value(m.group("t"))
        if t is None:
            continue
        ranges.append((t, t))
        used_spans.append((m.start(), m.end()))

    def _overlaps_used(start: int, end: int) -> bool:
        for s0, e0 in used_spans:
            if not (end <= s0 or start >= e0):
                return True
        return False

    colon_time_re = re.compile(r"(?<![\d:])\d+(?::\d{1,2}){1,2}(?![\d:])")
    for m in colon_time_re.finditer(q):
        if _overlaps_used(m.start(), m.end()):
            continue
        t = _parse_time_value(m.group(0))
        if t is None:
            continue
        ranges.append((t, t))

    # Keep deterministic order and remove near-duplicates.
    ranges = sorted(ranges, key=lambda x: (x[0], x[1]))
    deduped: List[Tuple[float, float]] = []
    for lo, hi in ranges:
        if deduped and abs(deduped[-1][0] - lo) < 1e-3 and abs(deduped[-1][1] - hi) < 1e-3:
            continue
        deduped.append((lo, hi))
    return deduped


def _format_seconds_like_token(seconds: float, token: str) -> str:
    t = str(token or "").strip().lower().replace(",", ".")
    v = max(0.0, float(seconds))

    if ":" in t:
        parts = t.split(":")
        sec_rounded = int(round(v))
        if len(parts) == 3 or v >= 3600:
            hh = sec_rounded // 3600
            mm = (sec_rounded % 3600) // 60
            ss = sec_rounded % 60
            return f"{hh:02d}:{mm:02d}:{ss:02d}"
        mm = sec_rounded // 60
        ss = sec_rounded % 60
        return f"{mm:02d}:{ss:02d}"

    m = re.match(
        r"^([0-9]+(?:[\.,][0-9]+)?)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds)?$",
        t,
    )
    if not m:
        return f"{int(round(v))}s"

    unit = m.group(2) or "s"
    has_decimal = "." in (m.group(1) or "")

    if unit.startswith("h"):
        val = v / 3600.0
    elif unit.startswith("m"):
        val = v / 60.0
    else:
        val = v

    if has_decimal:
        return f"{val:.2f}{unit}"
    return f"{int(round(val))}{unit}"


def _map_time_to_concat_local(time_s: float, windows: Sequence[Tuple[float, float]]) -> Optional[float]:
    t = float(time_s)
    offset = 0.0
    for start_s, end_s in windows:
        a = float(start_s)
        b = float(end_s)
        if b <= a:
            continue
        if a <= t <= b:
            return offset + (t - a)
        offset += (b - a)
    return None


def _rewrite_timestamps_to_cropped_local_time(
    text: str,
    windows: Sequence[Tuple[float, float]],
) -> str:
    if not text:
        return text

    valid_windows = sorted(
        [
            (float(a), float(b))
            for a, b in windows
            if b is not None and a is not None and float(b) > float(a)
        ],
        key=lambda x: (x[0], x[1]),
    )
    if not valid_windows:
        return text

    q = str(text)
    time_atom = (
        r"(?:"
        r"\d+(?::\d{1,2}){1,2}"
        r"|"
        r"\d+(?:[\.,]\d+)?(?:\s*(?:h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds))?"
        r")"
    )
    range_re = re.compile(
        rf"(?P<a>{time_atom})(?P<lws>\s*)(?P<sep>-|–|—|to)(?P<rws>\s*)(?P<b>{time_atom})",
        flags=re.IGNORECASE,
    )
    point_kw_re = re.compile(
        rf"(?P<prefix>(?:at|around|near|timestamp|time)\s*)(?P<t>{time_atom})",
        flags=re.IGNORECASE,
    )
    colon_time_re = re.compile(r"(?<![\d:])\d+(?::\d{1,2}){1,2}(?![\d:])")

    def _range_repl(m: re.Match) -> str:
        a_tok = m.group("a")
        b_tok = m.group("b")
        a = _parse_time_value(a_tok)
        b = _parse_time_value(b_tok)
        if a is None or b is None:
            return m.group(0)
        a_new = _map_time_to_concat_local(a, valid_windows)
        b_new = _map_time_to_concat_local(b, valid_windows)
        if a_new is None or b_new is None:
            return m.group(0)
        return (
            f"{_format_seconds_like_token(a_new, a_tok)}"
            f"{m.group('lws')}{m.group('sep')}{m.group('rws')}"
            f"{_format_seconds_like_token(b_new, b_tok)}"
        )

    q = range_re.sub(_range_repl, q)

    def _point_repl(m: re.Match) -> str:
        tok = m.group("t")
        t = _parse_time_value(tok)
        if t is None:
            return m.group(0)
        t_new = _map_time_to_concat_local(t, valid_windows)
        if t_new is None:
            return m.group(0)
        return f"{m.group('prefix')}{_format_seconds_like_token(t_new, tok)}"

    q = point_kw_re.sub(_point_repl, q)

    used_spans: List[Tuple[int, int]] = []
    for m in range_re.finditer(q):
        used_spans.append((m.start(), m.end()))
    for m in point_kw_re.finditer(q):
        used_spans.append((m.start(), m.end()))

    def _overlaps_used(start: int, end: int) -> bool:
        for s0, e0 in used_spans:
            if not (end <= s0 or start >= e0):
                return True
        return False

    pieces: List[str] = []
    pos = 0
    for m in colon_time_re.finditer(q):
        if _overlaps_used(m.start(), m.end()):
            continue
        tok = m.group(0)
        t = _parse_time_value(tok)
        if t is None:
            continue
        t_new = _map_time_to_concat_local(t, valid_windows)
        if t_new is None:
            continue
        pieces.append(q[pos : m.start()])
        pieces.append(_format_seconds_like_token(t_new, tok))
        pos = m.end()
    if pos == 0:
        return q
    pieces.append(q[pos:])
    return "".join(pieces)


def _load_audio_for_cropping(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    arr, sr = sf.read(str(audio_path), dtype="float32")
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    arr = np.asarray(arr, dtype=np.float32)

    if int(sr) != int(target_sr):
        try:
            import librosa

            arr = librosa.resample(arr, orig_sr=int(sr), target_sr=int(target_sr))
            sr = int(target_sr)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Audio resample failed for {audio_path}: {e}")

    return arr, int(sr)


def _build_audio_source_from_question(
    audio_path: str,
    question_text: str,
    sample_id: str,
    crop_from_question_refs: bool,
    crop_collar_seconds: float,
    random_crop_seconds: float,
) -> Tuple[object, str, bool, Optional[List[Tuple[float, float]]]]:
    # Returns (audio_source, audio_cache_key, allow_cache, crop_windows).
    # audio_source is either original path (str) or cropped waveform (np.ndarray).
    if not bool(crop_from_question_refs):
        return str(audio_path), str(audio_path), True, None

    waveform, sr = _load_audio_for_cropping(str(audio_path), target_sr=16000)
    total_samples = int(waveform.shape[0])
    if total_samples <= 0:
        return str(audio_path), str(audio_path), True, None

    duration_s = float(total_samples) / float(sr)
    refs = _extract_time_ranges_from_text(str(question_text))
    collar = max(0.0, float(crop_collar_seconds))

    if refs:
        segments: List[np.ndarray] = []
        windows: List[Tuple[float, float]] = []
        for lo, hi in refs:
            start_s = max(0.0, float(lo) - collar)
            end_s = min(duration_s, float(hi) + collar)
            if end_s <= start_s:
                continue
            s0 = int(start_s * sr)
            s1 = int(end_s * sr)
            if s1 <= s0:
                continue
            segments.append(waveform[s0:s1])
            windows.append((start_s, end_s))

        if segments:
            merged = np.concatenate(segments, axis=0).astype(np.float32, copy=False)
            # Cap merged crop to random_crop_seconds to prevent OOM on GPUs:
            # questions comparing distant segments can produce a crop spanning the
            # full audio (e.g. 1200s → 40 audio chunks → OOM on 64 GB H100).
            # Use 0 (no cap) when random_crop_seconds=0 (eval mode) to preserve
            # full-audio behaviour; only cap during training (random_crop_seconds>0).
            max_crop_s = max(0.0, float(random_crop_seconds)) if float(random_crop_seconds) > 0 else 0.0
            max_crop_n = int(max_crop_s * sr)
            if max_crop_n > 0 and len(merged) > max_crop_n:
                merged = merged[:max_crop_n]
            key_payload = {
                "audio_path": str(audio_path),
                "mode": "refs",
                "windows": [[round(a, 3), round(b, 3)] for a, b in windows],
            }
            key = "crop:" + hashlib.sha1(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()
            return merged, key, True, windows

    # No explicit references: random crop on-the-fly.
    crop_s = max(0.0, float(random_crop_seconds))
    crop_n = int(crop_s * sr)
    if crop_n > 0 and crop_n < total_samples:
        max_start = total_samples - crop_n
        start = random.randint(0, max_start)
        end = start + crop_n
        chunk = waveform[start:end].astype(np.float32, copy=False)
        key_payload = {
            "audio_path": str(audio_path),
            "mode": "random",
            "sample_id": str(sample_id),
            "start": int(start),
            "end": int(end),
        }
        key = "crop:" + hashlib.sha1(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()
        return chunk, key, False, None

    # Fallback: return full waveform.  Only cap during training (random_crop_seconds>0)
    # to prevent OOM; when random_crop_seconds=0 (eval mode) return full audio so that
    # eval results match the original baseline.
    max_crop_s = max(0.0, float(random_crop_seconds)) if float(random_crop_seconds) > 0 else 0.0
    max_crop_n = int(max_crop_s * sr)
    waveform_out = waveform[:max_crop_n] if max_crop_n > 0 and len(waveform) > max_crop_n else waveform
    return waveform_out.astype(np.float32, copy=False), str(audio_path), True, None


def _encode_chat_sample(
    processor: VoxtralProcessor,
    model_id: str,
    prompt_language: str,
    audio_source: object,
    prompt_text: str,
    assistant_text: str,
    max_prompt_tokens: int,
    audio_prompt_cache: Optional[OrderedDict] = None,
    audio_prompt_cache_size: int = 0,
    audio_cache_key: Optional[str] = None,
    allow_audio_cache: bool = True,
    use_chat_template: bool = False,
    use_transcription_hint: bool = False,
    icl_shots: Optional[List[Tuple[str, str, str]]] = None,
) -> Dict[str, torch.Tensor]:
    # ---------------------------------------------------------------------------
    # Multi-turn ICL path: when icl_shots is provided, prepend N (audio, prompt,
    # answer) turns to the conversation.  Labels for ICL turns are masked (-100);
    # only the final answer is supervised.  Always uses apply_chat_template.
    # Each icl_shots element is (prompt_text, audio_file_path, gold_answer).
    # apply_chat_template requires file paths for audio, not numpy arrays.
    # ---------------------------------------------------------------------------
    if icl_shots:
        conversation = []
        for shot_prompt, shot_audio_path, shot_answer in icl_shots:
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "audio", "path": str(shot_audio_path)},
                    {"type": "text", "text": shot_prompt},
                ],
            })
            conversation.append({
                "role": "assistant",
                "content": shot_answer,
            })
        # Main question (user only — assistant answer appended manually for label masking)
        # apply_chat_template only accepts file path/url/base64 — NOT raw numpy arrays.
        # CRITICAL: always cap audio to 30 s (one Whisper chunk = 1 500 tokens).
        #   • crops ≥ 30 s  → collar=30 on each side of a timestamp window can easily
        #     produce 60-120 s;  multi-reference questions can span the full session.
        #   • no-timestamp fallback → full session waveform (hundreds of seconds).
        #   • string path (crop_from_question_refs=False) → full file, any duration.
        # Without this cap the Whisper feature extractor splits audio >30 s into
        # multiple 30-s chunks, multiplying audio-token count and causing OOM during
        # the backward pass even with just 1 ICL shot.
        _WHISPER_MAX_SAMPLES = 480_000  # 30 s × 16 000 Hz = 1 chunk = 1 500 tokens
        _tmp_audio_path: Optional[str] = None
        import soundfile as _sf_icl
        if isinstance(audio_source, str):
            # String path: use soundfile.info (header-only, fast) to check duration
            # before loading anything expensive.
            _info = _sf_icl.info(audio_source)
            if _info.frames > int(_info.samplerate * 30):
                # File is longer than 30 s — read only the first 30 s at original SR.
                _arr, _sr = _sf_icl.read(
                    audio_source, frames=int(_info.samplerate * 30), dtype="float32"
                )
                if _arr.ndim > 1:
                    _arr = _arr.mean(axis=1)  # to mono
                _tmp_fd, _tmp_audio_path = tempfile.mkstemp(suffix=".wav")
                os.close(_tmp_fd)
                _sf_icl.write(_tmp_audio_path, _arr, _sr)
                main_audio_path = _tmp_audio_path
            else:
                main_audio_path = str(audio_source)
        else:
            # Numpy array (cropped waveform at 16 kHz): cap at 480 000 samples.
            _arr = np.asarray(audio_source, dtype=np.float32)
            if len(_arr) > _WHISPER_MAX_SAMPLES:
                _arr = _arr[:_WHISPER_MAX_SAMPLES]
            _tmp_fd, _tmp_audio_path = tempfile.mkstemp(suffix=".wav")
            os.close(_tmp_fd)
            _sf_icl.write(_tmp_audio_path, _arr, 16000)
            main_audio_path = _tmp_audio_path
        conversation.append({
            "role": "user",
            "content": [{"type": "audio", "path": main_audio_path}, {"type": "text", "text": prompt_text}],
        })
        try:
            prompt_enc = processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
        finally:
            if _tmp_audio_path is not None:
                try:
                    os.unlink(_tmp_audio_path)
                except OSError:
                    pass
        prompt_ids = _to_1d_long(prompt_enc["input_ids"])
        prompt_mask = _to_1d_long(prompt_enc["attention_mask"])
        input_features = prompt_enc["input_features"]

        target_ids = processor.tokenizer(
            str(assistant_text),
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0].to(dtype=torch.long)
        eos_id = processor.tokenizer.eos_token_id
        if eos_id is not None:
            target_ids = torch.cat([target_ids, torch.tensor([eos_id], dtype=torch.long)], dim=0)

        input_ids = torch.cat([prompt_ids, target_ids], dim=0)
        attention_mask = torch.cat(
            [prompt_mask, torch.ones((target_ids.shape[0],), dtype=torch.long)], dim=0
        )
        labels = input_ids.clone()
        labels[: prompt_ids.shape[0]] = -100  # mask prompt+ICL turns

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_features": input_features,
        }

    # ---------------------------------------------------------------------------
    # Standard single-turn path (original logic below, unchanged).
    # ---------------------------------------------------------------------------
    # Transcription-hint mode: embed "lang:{language}\n[TRANSCRIBE]\n" inside [INST] as a
    # prefix to the question text.  This keeps the question inside the instruction block
    # (unlike the raw apply_transcription_request path) while exposing the same token cue
    # that the base model was pre-trained on.  Implies use_chat_template=True.
    if use_transcription_hint:
        use_chat_template = True
        hint = f"lang:{prompt_language}\n[TRANSCRIBE]\n"
        prompt_text = hint + prompt_text
    conversation = [
        {
            "role": "user",
            "content": [
                ({"type": "audio", "path": str(audio_source)} if isinstance(audio_source, str) else {"type": "audio", "audio": audio_source}),
                {"type": "text", "text": prompt_text},
            ],
        },
        {
            "role": "assistant",
            "content": assistant_text,
        },
    ]

    try:
        cache_key = str(audio_cache_key) if audio_cache_key else (str(audio_source) if isinstance(audio_source, str) else None)
        cache_entry = None
        use_audio_prompt_cache = (
            bool(allow_audio_cache)
            and cache_key is not None
            and audio_prompt_cache is not None
            and int(audio_prompt_cache_size) > 0
        )
        if use_audio_prompt_cache:
            cache_entry = audio_prompt_cache.get(cache_key)
            if cache_entry is not None:
                audio_prompt_cache.move_to_end(cache_key)

        if cache_entry is None:
            if use_chat_template:
                # Use apply_chat_template matching the eval encoding path exactly.
                # Pass user-only conversation so we can append assistant tokens
                # manually and set assistant_mask correctly.
                user_only_conversation = [
                    {
                        "role": "user",
                        "content": [
                            ({"type": "audio", "path": str(audio_source)} if isinstance(audio_source, str) else {"type": "audio", "audio": audio_source}),
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                prompt_enc = processor.apply_chat_template(
                    user_only_conversation,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                prompt_ids = _to_1d_long(prompt_enc["input_ids"])
                prompt_mask = _to_1d_long(prompt_enc["attention_mask"])
                input_features = prompt_enc["input_features"]
            else:
                # Prefer the robust transcription-request API used in voxtral_train_router.py.
                if isinstance(audio_source, str):
                    prompt = processor.apply_transcription_request(
                        language=str(prompt_language),
                        model_id=str(model_id),
                        audio=[str(audio_source)],
                        format=["WAV"],
                        sampling_rate=16000,
                        return_tensors="pt",
                    )
                else:
                    prompt = processor.apply_transcription_request(
                        language=str(prompt_language),
                        model_id=str(model_id),
                        audio=[audio_source],
                        sampling_rate=16000,
                        return_tensors="pt",
                    )
                prompt_ids = _to_1d_long(prompt["input_ids"])
                prompt_mask = _to_1d_long(prompt["attention_mask"])
                input_features = prompt["input_features"]

            if use_audio_prompt_cache:
                input_features_cached = input_features
                if isinstance(input_features_cached, torch.Tensor):
                    input_features_cached = input_features_cached.detach().cpu()
                audio_prompt_cache[cache_key] = {
                    "prompt_ids": prompt_ids.detach().cpu(),
                    "prompt_mask": prompt_mask.detach().cpu(),
                    "input_features": input_features_cached,
                }
                audio_prompt_cache.move_to_end(cache_key)
                while len(audio_prompt_cache) > int(audio_prompt_cache_size):
                    audio_prompt_cache.popitem(last=False)
        else:
            prompt_ids = cache_entry["prompt_ids"].clone()
            prompt_mask = cache_entry["prompt_mask"].clone()
            input_features = cache_entry["input_features"]
            if isinstance(input_features, torch.Tensor):
                input_features = input_features.clone()

        # apply_chat_template already embeds prompt_text inside [INST]...[/INST],
        # so skip the manual append to avoid double-counting the question text.
        if not use_chat_template:
            prompt_text_ids = processor.tokenizer(
                str(prompt_text),
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0].to(dtype=torch.long)
            if prompt_text_ids.ndim == 0:
                prompt_text_ids = prompt_text_ids.unsqueeze(0)
            prompt_text_ids = prompt_text_ids.reshape(-1)

            if prompt_text_ids.shape[0] > 0:
                prompt_ids = torch.cat([prompt_ids, prompt_text_ids], dim=0)
                prompt_mask = torch.cat(
                    [prompt_mask, torch.ones((prompt_text_ids.shape[0],), dtype=torch.long)],
                    dim=0,
                )

        if int(max_prompt_tokens) > 0 and prompt_ids.shape[0] > int(max_prompt_tokens):
            prompt_ids = prompt_ids[-int(max_prompt_tokens) :]
            prompt_mask = prompt_mask[-int(max_prompt_tokens) :]

        target_ids = processor.tokenizer(
            str(assistant_text),
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0].to(dtype=torch.long)
        eos_id = processor.tokenizer.eos_token_id
        if eos_id is not None:
            target_ids = torch.cat([target_ids, torch.tensor([eos_id], dtype=torch.long)], dim=0)

        input_ids = torch.cat([prompt_ids, target_ids], dim=0)
        attention_mask = torch.cat(
            [prompt_mask, torch.ones((target_ids.shape[0],), dtype=torch.long)],
            dim=0,
        )
        assistant_mask = torch.zeros((input_ids.shape[0],), dtype=torch.bool)
        assistant_mask[prompt_ids.shape[0] :] = True
    except Exception:
        try:
            # Older API: same processor path but no assistant mask support.
            encoded = processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
            input_ids = _to_1d_long(encoded["input_ids"])
            attention_mask = _to_1d_long(encoded["attention_mask"])
            input_features = encoded["input_features"]

            prompt_text_ids = processor.tokenizer(
                str(prompt_text),
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0].to(dtype=torch.long)
            if prompt_text_ids.ndim == 0:
                prompt_text_ids = prompt_text_ids.unsqueeze(0)
            prompt_text_ids = prompt_text_ids.reshape(-1)
            if prompt_text_ids.shape[0] > 0:
                input_ids = torch.cat([input_ids, prompt_text_ids], dim=0)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((prompt_text_ids.shape[0],), dtype=torch.long)],
                    dim=0,
                )

            assistant_mask = None
        except Exception:
            # Tokenizer fallback for versions where processor template path is unavailable.
            tmp_audio_path = None
            if isinstance(audio_source, str):
                fallback_audio_path = str(audio_source)
            else:
                fallback_audio_arr = np.asarray(audio_source, dtype=np.float32)
                try:
                    import soundfile as sf

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                        tmp_audio_path = tf.name
                    sf.write(tmp_audio_path, fallback_audio_arr, 16000)
                    fallback_audio_path = tmp_audio_path
                except Exception as e:  # noqa: BLE001
                    raise RuntimeError(f"Failed to materialize cropped audio for tokenizer fallback: {e}")

            user_message = {
                "role": "user",
                "content": [
                    {"type": "audio", "path": fallback_audio_path},
                    {"type": "text", "text": str(prompt_text)},
                ],
            }

            try:
                try:
                    encoded = processor.tokenizer.apply_chat_template(
                        [user_message],
                        return_tensors=None,
                        return_assistant_tokens_mask=True,
                    )
                except Exception:
                    encoded = processor.tokenizer.apply_chat_template(
                        [user_message],
                        return_tensors=None,
                    )
            finally:
                if tmp_audio_path is not None:
                    try:
                        Path(tmp_audio_path).unlink(missing_ok=True)
                    except Exception:
                        pass

            chat_audio = encoded.pop("audio", None)
            if chat_audio is None:
                raise RuntimeError("Tokenizer chat template did not return audio for fallback.")

            # Build teacher-forcing targets manually from assistant text to avoid
            # strict validator errors for chats ending with assistant in older APIs.
            prompt_ids = _to_1d_long(encoded["input_ids"])
            prompt_mask = _to_1d_long(encoded["attention_mask"])

            prompt_text_ids = processor.tokenizer(
                str(prompt_text),
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0].to(dtype=torch.long)
            if prompt_text_ids.ndim == 0:
                prompt_text_ids = prompt_text_ids.unsqueeze(0)
            prompt_text_ids = prompt_text_ids.reshape(-1)
            if prompt_text_ids.shape[0] > 0:
                prompt_ids = torch.cat([prompt_ids, prompt_text_ids], dim=0)
                prompt_mask = torch.cat(
                    [prompt_mask, torch.ones((prompt_text_ids.shape[0],), dtype=torch.long)],
                    dim=0,
                )

            if int(max_prompt_tokens) > 0 and prompt_ids.shape[0] > int(max_prompt_tokens):
                prompt_ids = prompt_ids[-int(max_prompt_tokens) :]
                prompt_mask = prompt_mask[-int(max_prompt_tokens) :]

            target_ids = processor.tokenizer(
                str(assistant_text),
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0].to(dtype=torch.long)
            eos_id = processor.tokenizer.eos_token_id
            if eos_id is not None:
                target_ids = torch.cat([target_ids, torch.tensor([eos_id], dtype=torch.long)], dim=0)

            input_ids = torch.cat([prompt_ids, target_ids], dim=0)
            attention_mask = torch.cat(
                [prompt_mask, torch.ones((target_ids.shape[0],), dtype=torch.long)],
                dim=0,
            )
            input_features = _build_voxtral_chat_input_features(processor, chat_audio)

            assistant_mask = torch.zeros((input_ids.shape[0],), dtype=torch.bool)
            assistant_mask[prompt_ids.shape[0] :] = True

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    if assistant_mask is not None and assistant_mask.shape[0] == labels.shape[0]:
        labels[~assistant_mask] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "input_features": input_features,
    }


class VoxtralMCQCollator:
    def __init__(
        self,
        processor: VoxtralProcessor,
        model_id: str,
        prompt_language: str,
        max_prompt_tokens: int,
        audio_prompt_cache_size: int,
        crop_from_question_refs: bool,
        remap_timestamps_after_crop: bool,
        crop_collar_seconds: float,
        random_crop_seconds: float,
        use_chat_template: bool = False,
        use_transcription_hint: bool = False,
        transcript_dir: Optional[str] = None,
        transcript_max_words: int = 0,
        icl_audio_dir: Optional[str] = None,
        n_icl_shots: int = 1,
        seed: int = 42,
    ):
        self.processor = processor
        self.model_id = model_id
        self.prompt_language = prompt_language
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.audio_prompt_cache_size = max(0, int(audio_prompt_cache_size))
        self.audio_prompt_cache: OrderedDict = OrderedDict()
        self.crop_from_question_refs = bool(crop_from_question_refs)
        self.remap_timestamps_after_crop = bool(remap_timestamps_after_crop)
        self.crop_collar_seconds = float(crop_collar_seconds)
        self.random_crop_seconds = float(random_crop_seconds)
        self.use_chat_template = bool(use_chat_template)
        self.use_transcription_hint = bool(use_transcription_hint)
        self.transcript_dir = transcript_dir
        self.transcript_max_words = int(transcript_max_words)
        tok = self.processor.tokenizer
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        # ICL shot rotation: store (prompt_text, audio_path, answer) tuples.
        # File paths are passed directly to apply_chat_template (requires paths, not arrays).
        self._icl_shot_sets: Optional[List[List[Tuple[str, str, str]]]] = None
        if icl_audio_dir:
            icl_dir = Path(icl_audio_dir)
            loaded_sets: List[List[Tuple[str, str, str]]] = []
            for shot_set in _ICL_TRAINING_SETS:
                shots_loaded = []
                for shot in shot_set:
                    fname, prompt, answer = shot[0], shot[1], shot[2]
                    fpath = icl_dir / fname
                    if not fpath.exists():
                        raise FileNotFoundError(
                            f"ICL audio clip not found: {fpath}. "
                            "Re-run the ICL clip extraction script."
                        )
                    shots_loaded.append((prompt, str(fpath), answer))
                loaded_sets.append(shots_loaded)
            self._icl_shot_sets = loaded_sets
            self._icl_rng = __import__("random").Random(seed)
        self._n_icl_shots = max(1, int(n_icl_shots))

    def __call__(self, features: List[dict]) -> Dict[str, torch.Tensor]:
        # Pick one ICL set for the whole batch (rotation per batch).
        batch_icl_shots: Optional[List[Tuple[str, str, str]]] = None
        if self._icl_shot_sets:
            full_set = self._icl_rng.choice(self._icl_shot_sets)
            n = min(self._n_icl_shots, len(full_set))
            batch_icl_shots = self._icl_rng.sample(full_set, n)

        encoded = []
        for f in features:
            audio_source, audio_cache_key, allow_audio_cache, crop_windows = _build_audio_source_from_question(
                audio_path=str(f["audio_path"]),
                question_text=str(f.get("question") or f.get("prompt_text") or ""),
                sample_id=str(f.get("sample_id") or ""),
                crop_from_question_refs=bool(self.crop_from_question_refs),
                crop_collar_seconds=float(self.crop_collar_seconds),
                random_crop_seconds=float(self.random_crop_seconds),
            )
            prompt_text = str(f["prompt_text"])
            # Optionally prepend ASR transcript so the model is trained to leverage text.
            transcript = _load_transcript(
                str(f["audio_path"]), self.transcript_dir, self.transcript_max_words
            )
            if transcript:
                prompt_text = f"Transcript:\n{transcript}\n\n{prompt_text}"
            if bool(self.remap_timestamps_after_crop) and crop_windows:
                prompt_text = _rewrite_timestamps_to_cropped_local_time(
                    text=prompt_text,
                    windows=crop_windows,
                )
            encoded.append(
                _encode_chat_sample(
                    processor=self.processor,
                    model_id=self.model_id,
                    prompt_language=self.prompt_language,
                    audio_source=audio_source,
                    prompt_text=prompt_text,
                    assistant_text=str(f["gold_choice"]),
                    max_prompt_tokens=self.max_prompt_tokens,
                    audio_prompt_cache=self.audio_prompt_cache,
                    audio_prompt_cache_size=self.audio_prompt_cache_size,
                    audio_cache_key=audio_cache_key,
                    allow_audio_cache=allow_audio_cache,
                    use_chat_template=self.use_chat_template,
                    use_transcription_hint=self.use_transcription_hint,
                    icl_shots=batch_icl_shots,
                )
            )

        max_len = max(x["input_ids"].shape[0] for x in encoded)

        def pad_1d(x: torch.Tensor, fill: int) -> torch.Tensor:
            if x.shape[0] == max_len:
                return x
            pad = torch.full((max_len - x.shape[0],), fill, dtype=x.dtype)
            return torch.cat([x, pad], dim=0)

        input_ids = torch.stack([pad_1d(x["input_ids"], self.pad_id) for x in encoded], dim=0)
        attention_mask = torch.stack([pad_1d(x["attention_mask"], 0) for x in encoded], dim=0)
        labels = torch.stack([pad_1d(x["labels"], -100) for x in encoded], dim=0)
        input_features = torch.cat([x["input_features"] for x in encoded], dim=0)
        sample_weights = torch.tensor(
            [float(f.get("sample_weight", 1.0)) for f in features], dtype=torch.float32
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_features": input_features,
            "sample_weight": sample_weights,
        }


def _run_audio_instruction(
    model,
    processor,
    audio_path: str,
    prompt: str,
    max_new_tokens: int,
    audio_cache: Optional[OrderedDict] = None,
    audio_cache_size: int = 64,
    prompt_prefix: str = "",
    crop_from_question_refs: bool = False,
    crop_collar_seconds: float = 30.0,
    random_crop_seconds: float = 0.0,
    question_text: str = "",
    transcript_dir: Optional[str] = None,
    transcript_max_words: int = 0,
) -> str:
    """Run a single audio MCQ instruction, optionally caching encoded audio features by path.

    Args:
        prompt_prefix: Optional string prepended to ``prompt`` before encoding.
            Used by ``--use-transcription-hint-format`` to embed the
            ``lang:{lang}\n[TRANSCRIBE]\n`` cue inside the [INST] block.
        crop_from_question_refs: When True, apply timestamp-based audio cropping
            before inference.  Questions with detected time references are cropped
            to the relevant window (± ``crop_collar_seconds``); questions without
            time references receive full audio when ``random_crop_seconds=0``
            (default) or a random crop otherwise.  When a crop is applied the
            audio feature cache is bypassed for that sample.
        crop_collar_seconds: Padding in seconds around detected timestamps.
        random_crop_seconds: Fallback crop duration (seconds) when no timestamps
            are detected.  0 (default) passes full audio for those questions.
        question_text: Raw question text used for timestamp detection.  Falls
            back to ``prompt`` if empty.
    """
    effective_prompt = (prompt_prefix + prompt) if prompt_prefix else prompt

    # Optionally prepend the ASR transcript so the LLM can cross-reference text
    # with audio when answering the MCQ.
    transcript = _load_transcript(audio_path, transcript_dir, transcript_max_words)
    if transcript:
        effective_prompt = f"Transcript:\n{transcript}\n\n{effective_prompt}"

    # ------------------------------------------------------------------
    # Eval-time audio cropping (opt-in via crop_from_question_refs=True)
    # When a crop results in a numpy array we bypass the path-keyed cache
    # because each question has a unique crop window.
    # ------------------------------------------------------------------
    use_path_cache = True
    if crop_from_question_refs:
        crop_audio_source, _crop_key, _allow_cache, _crop_windows = _build_audio_source_from_question(
            audio_path=audio_path,
            question_text=question_text or prompt,
            sample_id="",
            crop_from_question_refs=True,
            crop_collar_seconds=crop_collar_seconds,
            random_crop_seconds=random_crop_seconds,
        )
        if isinstance(crop_audio_source, np.ndarray):
            # processor.apply_chat_template requires a file path for audio;
            # save the cropped ndarray to a temp WAV (same pattern as training
            # data preparation at the sf.write fallback path).
            import soundfile as sf
            # Use system temp dir — do NOT write to the audio file's parent dir
            # or it will pollute the dataset with tmp*_crop.wav files.
            with tempfile.NamedTemporaryFile(suffix="_crop.wav", delete=False,
                                             dir=tempfile.gettempdir()) as _tf:
                _crop_tmp_path = _tf.name
            sf.write(_crop_tmp_path, np.asarray(crop_audio_source, dtype=np.float32), 16000)
            audio_content: Dict = {"type": "audio", "path": _crop_tmp_path}
            use_path_cache = False  # cropped ndarray — no path-based caching
        else:
            audio_content = {"type": "audio", "path": str(crop_audio_source)}
    else:
        audio_content = {"type": "audio", "path": audio_path}

    conversation = [
        {
            "role": "user",
            "content": [
                audio_content,
                {"type": "text", "text": effective_prompt},
            ],
        }
    ]

    # Audio-feature cache: if the same audio_path appears in multiple questions,
    # reuse the encoded features rather than re-loading and re-encoding from disk.
    # Disabled when crop_from_question_refs produces a per-question ndarray crop.
    cached_features = None
    if use_path_cache and audio_cache is not None and audio_path in audio_cache:
        cached_features = audio_cache[audio_path]
        audio_cache.move_to_end(audio_path)

    if cached_features is not None:
        # Rebuild conversation without audio file reference so processor only tokenizes text,
        # then inject the cached input_features manually.
        text_conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": audio_path},
                    {"type": "text", "text": effective_prompt},
                ],
            }
        ]
        try:
            inputs = processor.apply_chat_template(
                text_conversation,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs["input_features"] = cached_features.to(model.device)
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        except Exception:
            cached_features = None  # fall through to normal path

    if cached_features is None:
        try:
            inputs = processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
            # Store audio features in cache before moving to device
            # Cache key is the original audio_path (not dependent on prompt_prefix).
            # Skip when use_path_cache is False (per-question crop was applied).
            if use_path_cache and audio_cache is not None and "input_features" in inputs:
                audio_cache[audio_path] = inputs["input_features"].cpu().clone()
                audio_cache.move_to_end(audio_path)
                while len(audio_cache) > max(1, int(audio_cache_size)):
                    audio_cache.popitem(last=False)
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        except Exception as chat_err:  # noqa: BLE001
            if not getattr(_run_audio_instruction, "_warned_template_fallback", False):
                print(
                    "Warning: processor.apply_chat_template failed; using tokenizer chat-template fallback. "
                    f"error={type(chat_err).__name__}: {chat_err}"
                )
                _run_audio_instruction._warned_template_fallback = True

            encoded = processor.tokenizer.apply_chat_template(
                [conversation],
                return_tensors=None,
            )
            if isinstance(encoded, dict):
                chat_audio = encoded.pop("audio", None)
            else:
                chat_audio = encoded["audio"] if "audio" in encoded else None
                if "audio" in encoded:
                    del encoded["audio"]

            if chat_audio is None:
                raise RuntimeError("Tokenizer chat template did not return audio for fallback.")

            inputs = {
                "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
                "input_features": _build_voxtral_chat_input_features(processor, chat_audio),
            }
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    prompt_len = inputs["input_ids"].shape[1]
    decoded = processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
    return decoded[0] if decoded else ""


def _compute_accuracy_breakdown(rows: List[Dict], field_name: str) -> Dict[str, Dict[str, float]]:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in rows:
        if row.get("error"):
            continue
        metadata = row.get("metadata") or {}
        key = metadata.get(field_name)
        if key in (None, ""):
            continue
        key_str = str(key)
        counts[key_str]["total"] += 1
        counts[key_str]["correct"] += int(bool(row.get("is_correct")))

    out: Dict[str, Dict[str, float]] = {}
    for key, v in sorted(counts.items()):
        total = int(v["total"])
        correct = int(v["correct"])
        out[key] = {
            "n_total": total,
            "n_correct": correct,
            "accuracy": (correct / total) if total > 0 else 0.0,
        }
    return out


def _add_mcq_breakdowns(summary: Dict, rows: List[Dict]) -> Dict:
    summary["language_breakdown"] = _compute_accuracy_breakdown(rows, "language")
    summary["difficulty_breakdown"] = _compute_accuracy_breakdown(rows, "difficulty")
    summary["category_breakdown"] = _compute_accuracy_breakdown(rows, "category")
    summary["subtype_breakdown"] = _compute_accuracy_breakdown(rows, "subtype")
    summary["task_name_breakdown"] = _compute_accuracy_breakdown(rows, "task_name")
    summary["sub_category_breakdown"] = _compute_accuracy_breakdown(rows, "sub-category")
    summary["sub_sub_category_breakdown"] = _compute_accuracy_breakdown(rows, "sub-sub-category")
    summary["linguistics_sub_discipline_breakdown"] = _compute_accuracy_breakdown(
        rows,
        "linguistics_sub_discipline",
    )
    return summary


def evaluate_mcq(
    model,
    processor,
    samples: List[MCQSample],
    max_new_tokens: int,
    output_dir: Path,
    audio_cache_size: int = 64,
    prompt_prefix: str = "",
    crop_from_question_refs: bool = False,
    crop_collar_seconds: float = 30.0,
    random_crop_seconds: float = 0.0,
    transcript_dir: Optional[str] = None,
    transcript_max_words: int = 0,
) -> Dict:
    rows = []
    correct = 0
    error_counts: Counter[str] = Counter()
    audio_cache: OrderedDict = OrderedDict()

    for i, s in enumerate(samples, start=1):
        try:
            out = _run_audio_instruction(
                model=model,
                processor=processor,
                audio_path=s.audio_path,
                prompt=s.prompt_text,
                max_new_tokens=max_new_tokens,
                audio_cache=audio_cache,
                audio_cache_size=audio_cache_size,
                prompt_prefix=prompt_prefix,
                crop_from_question_refs=crop_from_question_refs,
                crop_collar_seconds=crop_collar_seconds,
                random_crop_seconds=random_crop_seconds,
                question_text=s.question,
                transcript_dir=transcript_dir,
                transcript_max_words=transcript_max_words,
            )
            pred_choice = _select_multiple_choice_option(out, list(s.choice_map.keys()))
            ok = pred_choice == s.gold_choice
            correct += int(ok)
            rows.append(
                {
                    "sample_id": s.sample_id,
                    "audio_path": s.audio_path,
                    "question": s.question,
                    "gold_choice": s.gold_choice,
                    "pred_choice": pred_choice,
                    "is_correct": bool(ok),
                    "response": out,
                    "choices": s.choice_map,
                    "metadata": s.metadata,
                }
            )
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            error_counts[err] += 1
            rows.append(
                {
                    "sample_id": s.sample_id,
                    "audio_path": s.audio_path,
                    "question": s.question,
                    "gold_choice": s.gold_choice,
                    "pred_choice": None,
                    "is_correct": False,
                    "response": "",
                    "error": err,
                    "choices": s.choice_map,
                    "metadata": s.metadata,
                }
            )

        if i % 20 == 0 or i == len(samples):
            done = max(1, i)
            print(f"[eval] {i}/{len(samples)} processed | acc_so_far={correct/done:.4f}")

    n = len(samples)
    acc = (correct / n) if n > 0 else 0.0
    summary = {
        "n_total": n,
        "n_correct": correct,
        "accuracy": acc,
        "n_error": int(sum(error_counts.values())),
        "error_counts": dict(sorted(error_counts.items(), key=lambda kv: kv[0])),
    }
    summary = _add_mcq_breakdowns(summary, rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mcq_eval_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "mcq_eval_predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate Voxtral on JSONL audio MCQ")

    p.add_argument("--model-id", default="mistralai/Voxtral-Mini-3B-2507")
    p.add_argument("--model-mode", choices=["baseline", "adapter"], default="baseline")
    p.add_argument("--adapter-path", default=None)
    p.add_argument(
        "--base-adapter-path",
        default=None,
        help="Path to a pre-trained LoRA adapter (e.g. an ASR-fine-tuned adapter) to load "
             "and merge into the base model weights *before* attaching the new MCQ LoRA. "
             "The merge uses PeftModel.merge_and_unload() so the original adapter rank / "
             "target-module constraints are discarded and the new MCQ LoRA starts from the "
             "merged weights.  Only used with --model-mode baseline.",
    )

    p.add_argument("--train-jsonl", required=False)
    p.add_argument("--eval-jsonl", default=None)
    p.add_argument("--test-jsonl", default=None,
                   help="Additional held-out JSONL evaluated after training (e.g. original unbalanced organisers). "
                        "Results saved to mcq_test_metrics.json in the experiment folder.")
    p.add_argument("--audio-root", default=None)
    p.add_argument("--prompt-language", default="en")
    p.add_argument("--use-chat-template-for-training", action="store_true",
                   help="Use apply_chat_template instead of apply_transcription_request for training "
                        "samples, matching the eval encoding path exactly.")
    p.add_argument("--use-transcription-hint-format", action="store_true",
                   help="Embed the transcription request cue (lang:{lang}\\n[TRANSCRIBE]\\n) inside "
                        "the [INST] block as a prefix to the question text, then close [INST] and "
                        "let the model generate the MCQ answer.  Applies the same prefix at eval "
                        "time so train/eval formats match.  Implies apply_chat_template internally "
                        "and is mutually exclusive with --use-chat-template-for-training.")
    p.add_argument("--mcq-cache-dir", default=None)
    p.add_argument("--cache-train-split", default="train")
    p.add_argument("--cache-eval-split", default="dev")
    p.add_argument("--audio-shard-cache-dir", default=None)
    p.add_argument("--audio-shard-train-split", default="train")

    p.add_argument("--do-train", action="store_true")
    p.add_argument("--do-eval", action="store_true")
    p.add_argument("--eval-fraction", type=float, default=0.0)

    p.add_argument("--max-questions-per-audio", type=int, default=0)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-eval-samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)

    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--train-batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument(
        "--encoder-learning-rate",
        type=float,
        default=None,
        help="LR for audio encoder parameters only. When set together with --connector-learning-rate and --llm-learning-rate, enables triple-LR mode.",
    )
    p.add_argument(
        "--connector-learning-rate",
        type=float,
        default=None,
        help="LR for connector/projector parameters only. Requires --encoder-learning-rate and --llm-learning-rate.",
    )
    p.add_argument(
        "--encoder-connector-learning-rate",
        type=float,
        default=None,
        help="LR for encoder+connector as one group (dual-LR mode). Requires --llm-learning-rate. Ignored if --encoder-learning-rate is set.",
    )
    p.add_argument(
        "--llm-learning-rate",
        type=float,
        default=None,
        help="LR for LLM/LoRA parameters. Required for dual-LR and triple-LR modes.",
    )
    p.add_argument(
        "--dual-lr-dry-run",
        action="store_true",
        help="Print dual-LR parameter groups and exit before training.",
    )
    p.add_argument(
        "--dual-lr-dry-run-topk",
        type=int,
        default=20,
        help="Number of top parameter names to print per dual-LR group in dry-run mode.",
    )
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--early-stopping-patience", type=int, default=0,
                   help="Stop training if eval_loss does not improve for this many evals. 0 = disabled.")
    p.add_argument(
        "--best-ckpt-strategy",
        choices=["min_eval_loss", "min_generalization_gap"],
        default="min_eval_loss",
        help=(
            "How to pick the checkpoint saved as final_model after training. "
            "'min_eval_loss' (default): standard HuggingFace load_best_model_at_end on eval_loss. "
            "'min_generalization_gap': pick the checkpoint that minimises "
            "(eval_loss - avg_train_loss_between_evals), which prefers the point "
            "where the model generalises best relative to its training fit. "
            "NOTE: keeps ALL checkpoints on disk when active (save_total_limit=None)."
        ),
    )
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=0,
        help="Hard token limit applied to the prompt during training (truncates from the left). "
             "0 (default) disables truncation — the audio crop flags are the preferred memory guard.",
    )
    p.add_argument(
        "--eval-crop-from-question-refs",
        action="store_true",
        help="Apply timestamp-based audio cropping during evaluation/inference. "
             "Questions with detected time references are cropped to the relevant window "
             "(± --eval-crop-collar-seconds); questions without time references receive "
             "full audio when --eval-random-crop-seconds=0 (default).",
    )
    p.add_argument(
        "--eval-crop-collar-seconds",
        type=float,
        default=30.0,
        help="Collar in seconds around detected timestamps for eval crop. Default 30.0.",
    )
    p.add_argument(
        "--eval-random-crop-seconds",
        type=float,
        default=0.0,
        help="Fallback crop duration (seconds) for questions without timestamps during eval. "
             "0 (default) passes full audio for those questions.",
    )
    p.add_argument(
        "--transcript-dir",
        default=None,
        metavar="DIR",
        help="Directory containing ASR transcript .txt files (one per session, named {stem}.txt). "
             "When set, the full transcript is prepended to every MCQ prompt as "
             "'Transcript:\\n{text}\\n\\n'. Off by default.",
    )
    p.add_argument(
        "--transcript-max-words",
        type=int,
        default=0,
        help="Truncate transcript to at most this many words before inserting into the prompt. "
             "0 (default) = no truncation (use full transcript).",
    )
    p.add_argument(
        "--audio-prompt-cache-size",
        type=int,
        default=256,
        help="Collator-side LRU cache size (audio-path keyed) for reusable audio prompt encoding; set 0 to disable.",
    )
    p.add_argument(
        "--crop-from-question-refs",
        action="store_true",
        help="Enable collator-side audio cropping based on timestamp/range mentions in the question text.",
    )
    p.add_argument(
        "--remap-timestamps-after-crop",
        action="store_true",
        help=(
            "When using --crop-from-question-refs and explicit timestamp ranges, rewrite question/prompt "
            "timestamps to concatenated-local crop time."
        ),
    )
    p.add_argument(
        "--crop-collar-seconds",
        type=float,
        default=30.0,
        help="Extra seconds added to both sides of each detected timestamp/range.",
    )
    p.add_argument(
        "--random-crop-seconds",
        type=float,
        default=300.0,
        help="When no timestamp/range is detected, use an on-the-fly random crop of this duration (seconds).",
    )

    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--freeze-encoder-connector",
        action="store_true",
        help="Set requires_grad=False on all encoder+multi_modal_projector params. LLM/LoRA still trains.",
    )
    p.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Set requires_grad=False on audio encoder params only. Connector+LLM still train.",
    )
    p.add_argument(
        "--freeze-llm",
        action="store_true",
        help="Set requires_grad=False on all LLM/LoRA params. Encoder+connector still trains.",
    )

    p.add_argument("--output-root", default="experiments")
    p.add_argument("--experiment-name", default="voxtral_mcq")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--timestamped-exp-dir", action="store_true")
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="Path to a checkpoint dir (or 'last') to resume training from.")

    p.add_argument("--use-bf16", action="store_true")
    p.add_argument("--no-use-bf16", dest="use_bf16", action="store_false")
    p.set_defaults(use_bf16=True)
    p.add_argument("--use-fp16", action="store_true")
    p.add_argument(
        "--weighted-loss", action="store_true", default=False,
        help=(
            "Enable per-sample loss weighting using the 'sample_weight' field "
            "stored in each JSONL row. Weights are read by the collator and "
            "applied in WeightedLossTrainer.compute_loss(). "
            "Convention: en/en=1.0, nllb non-en/en=2.0, orig CL non-en/non-en=3.0."
        ),
    )
    p.add_argument(
        "--balance-by-language", action="store_true", default=False,
        help=(
            "Use LanguageBalancedSampler: interleaves training samples by "
            "language at each epoch so every batch covers all languages. "
            "Works with all trainer types and is DDP/accelerate-compatible. "
            "Recommended for multilingual datasets with --grad-accum-steps >= 8."
        ),
    )
    p.add_argument(
        "--train-icl-audio-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory containing pre-extracted ICL audio clips (tr_s*.wav). "
            "When set, enables per-batch ICL rotation: a random set from the "
            "20 built-in _ICL_TRAINING_SETS is prepended to each training batch "
            "as masked (non-supervised) in-context demonstrations. "
            "ICL audio clips are pre-loaded once at collator init time. "
            "Example: data/mlc26_task2/icl_examples"
        ),
    )
    p.add_argument(
        "--train-icl-n-shots",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of ICL shots sampled per training batch from the selected "
            "6-shot set. Fewer shots = lower memory. Default 1 (recommended for "
            "training to avoid OOM with 30-second audio ICL clips)."
        ),
    )

    return p.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    base = Path(args.output_root)
    exp = str(args.experiment_name)
    if args.timestamped_exp_dir:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        return base / f"{exp}_{stamp}"
    return base / exp


def _save_run_metadata(output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    args_payload = {
        "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
        "argv": list(sys.argv),
        "args": vars(args),
    }
    (output_dir / "run_args.json").write_text(
        json.dumps(args_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    cmd = "python " + " ".join(shlex.quote(str(x)) for x in sys.argv)
    cmd_sh_path = output_dir / "run_command.sh"
    cmd_sh_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + cmd + "\n",
        encoding="utf-8",
    )
    try:
        cmd_sh_path.chmod(0o755)
    except Exception:
        pass

    (output_dir / "run_command.txt").write_text(cmd + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.do_train and not args.do_eval:
        args.do_eval = True

    # Distributed rank — used to guard filesystem operations and single-GPU eval
    # in multi-process torchrun launches.  Falls back to 0 for single-process runs.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Runtime safety: many GPUs (e.g., pre-Ampere) cannot run BF16 matmuls.
    # In that case fallback to FP16 automatically to avoid cublasGemmEx invalid-value crashes.
    if bool(args.use_bf16) and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        print(
            "[warn] CUDA BF16 is not supported on this GPU/runtime. "
            "Falling back to FP16 to avoid CUDA cublas errors."
        )
        args.use_bf16 = False
        if not bool(args.use_fp16):
            args.use_fp16 = True

    if bool(args.use_bf16) and bool(args.use_fp16):
        # Keep settings deterministic; BF16 takes precedence when supported.
        args.use_fp16 = False

    if bool(args.remap_timestamps_after_crop) and not bool(args.crop_from_question_refs):
        print(
            "[warn] --remap-timestamps-after-crop has no effect unless --crop-from-question-refs is enabled."
        )

    if args.mcq_cache_dir:
        cache_dir = Path(args.mcq_cache_dir)
        train_ds = _load_cache_split_dataset(
            cache_dir=cache_dir,
            split_name=str(args.cache_train_split),
            num_shards=int(args.num_shards),
            shard_index=int(args.shard_index),
        )
        if args.max_train_samples > 0:
            train_ds = train_ds.select(range(min(int(args.max_train_samples), len(train_ds))))

        eval_samples: List[MCQSample] = []
        eval_ds: Optional[Dataset] = None
        if args.do_eval or args.do_train:
            eval_split = str(args.cache_eval_split).strip()
            if eval_split:
                eval_ds = _load_cache_split_dataset(
                    cache_dir=cache_dir,
                    split_name=eval_split,
                    num_shards=1,
                    shard_index=0,
                )
                if args.max_eval_samples > 0:
                    eval_ds = eval_ds.select(range(min(int(args.max_eval_samples), len(eval_ds))))
                eval_samples = _dataset_to_samples(eval_ds)

        train_samples = _dataset_to_samples(train_ds)
    else:
        if not args.train_jsonl:
            raise ValueError("Either --train-jsonl or --mcq-cache-dir is required.")

        train_task = load_jsonl_audio_mcq(
            jsonl_path=args.train_jsonl,
            audio_root=args.audio_root,
            max_questions_per_audio=max(0, int(args.max_questions_per_audio)),
            max_samples=max(0, int(args.max_train_samples)),
            seed=int(args.seed),
        )

        if args.eval_jsonl:
            eval_task = load_jsonl_audio_mcq(
                jsonl_path=args.eval_jsonl,
                audio_root=args.audio_root,
                max_questions_per_audio=max(0, int(args.max_questions_per_audio)),
                max_samples=max(0, int(args.max_eval_samples)),
                seed=int(args.seed),
            )
            train_samples = train_task.samples
            eval_samples = eval_task.samples
        else:
            train_samples, eval_samples = _split_train_eval(
                train_task.samples,
                eval_fraction=float(args.eval_fraction),
                seed=int(args.seed),
            )
            if args.max_eval_samples > 0 and eval_samples:
                eval_samples = eval_samples[: min(int(args.max_eval_samples), len(eval_samples))]

        if args.audio_shard_cache_dir and int(args.num_shards) > 1:
            train_samples = _apply_audio_shard(
                samples=train_samples,
                audio_shard_cache_dir=str(args.audio_shard_cache_dir),
                split_name=str(args.audio_shard_train_split),
                num_shards=int(args.num_shards),
                shard_index=int(args.shard_index),
            )
        else:
            train_samples = _apply_shard(train_samples, int(args.num_shards), int(args.shard_index))
        train_ds = _samples_to_hf_dataset(train_samples)
        eval_ds = _samples_to_hf_dataset(eval_samples) if eval_samples else None

    print(f"Train samples: {len(train_samples)}")
    print(f"Eval samples: {len(eval_samples)}")

    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_run_metadata(output_dir=output_dir, args=args)

    processor = VoxtralProcessor.from_pretrained(_resolve_pretrained_source(args.model_id))
    dtype = _choose_dtype(use_bf16=bool(args.use_bf16), use_fp16=bool(args.use_fp16))
    model = _load_model(
        model_id=args.model_id,
        model_mode=args.model_mode,
        adapter_path=args.adapter_path,
        dtype=dtype,
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        base_adapter_path=args.base_adapter_path or None,
    )
    _print_trainable_params(model)

    if getattr(args, "freeze_encoder_connector", False):
        for _n, _p in model.named_parameters():
            if _is_encoder_connector_param_name(_n):
                _p.requires_grad_(False)
        print("[freeze] encoder+connector frozen; re-printing trainable params:")
        _print_trainable_params(model)

    if getattr(args, "freeze_encoder", False):
        for _n, _p in model.named_parameters():
            if _is_encoder_param_name(_n):
                _p.requires_grad_(False)
        print("[freeze] encoder frozen; re-printing trainable params:")
        _print_trainable_params(model)

    if getattr(args, "freeze_llm", False):
        for _n, _p in model.named_parameters():
            if not _is_encoder_connector_param_name(_n):
                _p.requires_grad_(False)
        print("[freeze] LLM/LoRA frozen; re-printing trainable params:")
        _print_trainable_params(model)

    if args.do_train:
        # ── LR mode detection ───────────────────────────────────────────────────
        triple_lr_enabled = (
            args.encoder_learning_rate is not None
            or args.connector_learning_rate is not None
        )
        if triple_lr_enabled:
            if args.encoder_learning_rate is None or args.connector_learning_rate is None or args.llm_learning_rate is None:
                raise ValueError(
                    "Triple-LR mode requires all three: --encoder-learning-rate, "
                    "--connector-learning-rate, and --llm-learning-rate."
                )
            dual_lr_enabled = False
        else:
            dual_lr_enabled = args.encoder_connector_learning_rate is not None or args.llm_learning_rate is not None
            if dual_lr_enabled and (args.encoder_connector_learning_rate is None or args.llm_learning_rate is None):
                raise ValueError(
                    "Both --encoder-connector-learning-rate and --llm-learning-rate must be set together."
                )
        if bool(args.dual_lr_dry_run) and not (dual_lr_enabled or triple_lr_enabled):
            raise ValueError(
                "--dual-lr-dry-run requires LR flags (dual or triple mode)."
            )
        if bool(args.dual_lr_dry_run):
            if triple_lr_enabled:
                _print_triple_lr_group_preview(model, topk=int(args.dual_lr_dry_run_topk))
                print("Triple-LR dry-run complete. Exiting before trainer initialization.")
            else:
                _print_dual_lr_group_preview(model, topk=int(args.dual_lr_dry_run_topk))
                print("Dual-LR dry-run complete. Exiting before trainer initialization.")
            return

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        collator = VoxtralMCQCollator(
            processor=processor,
            model_id=str(args.model_id),
            prompt_language=str(args.prompt_language),
            max_prompt_tokens=int(args.max_prompt_tokens),
            audio_prompt_cache_size=int(args.audio_prompt_cache_size),
            crop_from_question_refs=bool(args.crop_from_question_refs),
            remap_timestamps_after_crop=bool(args.remap_timestamps_after_crop),
            crop_collar_seconds=float(args.crop_collar_seconds),
            random_crop_seconds=float(args.random_crop_seconds),
            use_chat_template=bool(args.use_chat_template_for_training),
            use_transcription_hint=bool(args.use_transcription_hint_format),
            transcript_dir=args.transcript_dir,
            transcript_max_words=int(args.transcript_max_words),
            icl_audio_dir=args.train_icl_audio_dir,
            n_icl_shots=int(args.train_icl_n_shots),
            seed=int(args.seed),
        )

        train_args_kwargs = {
            "output_dir": str(output_dir),
            "per_device_train_batch_size": int(args.train_batch_size),
            "per_device_eval_batch_size": int(args.eval_batch_size),
            "gradient_accumulation_steps": int(args.grad_accum_steps),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "warmup_ratio": float(args.warmup_ratio),
            "logging_steps": int(args.logging_steps),
            "save_steps": int(args.save_steps),
            "eval_steps": int(args.eval_steps),
            "num_train_epochs": int(args.num_epochs),
            "bf16": bool(args.use_bf16),
            "fp16": bool(args.use_fp16),
            "remove_unused_columns": False,
            "dataloader_num_workers": 0,
            "save_strategy": "steps",
            "load_best_model_at_end": (
                int(args.early_stopping_patience) > 0
                and args.best_ckpt_strategy == "min_eval_loss"
            ),
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "save_total_limit": (
                2 if int(args.early_stopping_patience) > 0 else None
            ),
            "report_to": [],
        }

        eval_mode = "steps" if eval_ds is not None else "no"
        ta_params = inspect.signature(TrainingArguments.__init__).parameters
        if "evaluation_strategy" in ta_params:
            train_args_kwargs["evaluation_strategy"] = eval_mode
        elif "eval_strategy" in ta_params:
            train_args_kwargs["eval_strategy"] = eval_mode

        train_args = TrainingArguments(**train_args_kwargs)

        weighted_loss = bool(args.weighted_loss)
        trainer_cls = TripleLRTrainer if triple_lr_enabled else (DualLRTrainer if dual_lr_enabled else (WeightedLossTrainer if weighted_loss else Trainer))

        # Inject language-balanced sampler into whichever trainer class was selected
        if bool(getattr(args, "balance_by_language", False)):
            _base_cls = trainer_cls
            def _lang_balanced_get_train_sampler(self, dataset=None):
                return LanguageBalancedSampler(
                    dataset=self.train_dataset,
                    language_key="language",
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )
            trainer_cls = type(
                _base_cls.__name__ + "LangBal",
                (_base_cls,),
                {"_get_train_sampler": _lang_balanced_get_train_sampler},
            )
        trainer_kwargs = {
            "model": model,
            "args": train_args,
            "data_collator": collator,
            "train_dataset": train_ds,
            "eval_dataset": eval_ds,
            "callbacks": [
                CheckpointLossLoggerCallback(output_dir=output_dir),
                # EarlyStoppingCallback requires load_best_model_at_end=True.
                # With min_generalization_gap strategy that flag is False, so we
                # skip the callback and rely on running the full training duration
                # then picking the best checkpoint post-hoc.
                *([EarlyStoppingCallback(early_stopping_patience=int(args.early_stopping_patience))]
                  if int(args.early_stopping_patience) > 0
                  and args.best_ckpt_strategy == "min_eval_loss"
                  else []),
            ],
        }
        if triple_lr_enabled:
            trainer_kwargs["encoder_learning_rate"]   = float(args.encoder_learning_rate)
            trainer_kwargs["connector_learning_rate"] = float(args.connector_learning_rate)
            trainer_kwargs["llm_learning_rate"]       = float(args.llm_learning_rate)
        elif dual_lr_enabled:
            trainer_kwargs["encoder_connector_learning_rate"] = float(args.encoder_connector_learning_rate)
            trainer_kwargs["llm_learning_rate"] = float(args.llm_learning_rate)

        trainer = trainer_cls(**trainer_kwargs)

        resume_ckpt = args.resume_from_checkpoint
        if resume_ckpt and resume_ckpt != "last":
            # Resolve relative paths against the output_dir
            p_ckpt = Path(resume_ckpt)
            if not p_ckpt.is_absolute():
                p_ckpt = output_dir / p_ckpt
            resume_ckpt = str(p_ckpt)
        if resume_ckpt:
            # Transformers ≥ 4.51 / torch < 2.6 compatibility fixes when
            # loading optimizer state from our own trusted checkpoint.
            # Bypass the torch.load CVE-2025-32434 version guard — must patch
            # the name as imported inside transformers.trainer (not just the
            # source module) because the trainer has already bound it.
            try:
                import transformers.trainer as _tr_module
                import transformers.utils.import_utils as _tu
                _noop = lambda: None
                _tu.check_torch_load_is_safe = _noop
                _tr_module.check_torch_load_is_safe = _noop
            except Exception:
                pass
        trainer.train(resume_from_checkpoint=resume_ckpt or None)

        # --- Checkpoint selection & final_model saving ---
        final_dir = output_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)

        if args.best_ckpt_strategy == "min_generalization_gap":
            import shutil
            # The callback saved the best-gap checkpoint to best_gen_gap_checkpoint/
            best_ckpt = output_dir / "best_gen_gap_checkpoint"
            if not best_ckpt.exists():
                # Fall back to scanning the log (handles resume case)
                log_path = output_dir / "training_ckpt_log.jsonl"
                best_ckpt = _pick_best_ckpt_by_gen_gap(log_path)
            # Only rank 0 performs filesystem operations to avoid race conditions
            if local_rank == 0:
                if best_ckpt is not None and best_ckpt.exists():
                    print(f"[gen_gap] Copying best gen-gap checkpoint → final_model: {best_ckpt}")
                    if final_dir.exists():
                        shutil.rmtree(final_dir)
                    shutil.copytree(best_ckpt, final_dir)
                    processor.save_pretrained(str(final_dir))
                    print(f"[gen_gap] final_model written from {best_ckpt.name}")
                else:
                    print("[gen_gap] WARNING: no best gap checkpoint found; falling back to last model.")
                    trainer.save_model(str(final_dir))
                    processor.save_pretrained(str(final_dir))
        else:
            # min_eval_loss: load_best_model_at_end=True already reloaded the best ckpt
            trainer.save_model(str(final_dir))
            processor.save_pretrained(str(final_dir))

        print(f"Training complete. Saved model/artifacts to: {final_dir}")

    # Post-training MCQ accuracy evaluation — only rank 0 in multi-GPU runs to
    # avoid 4× redundant inference and file-write races.
    if local_rank == 0 and args.do_eval:
        eval_model = model
        eval_prompt_prefix = (
            f"lang:{args.prompt_language}\n[TRANSCRIBE]\n"
            if args.use_transcription_hint_format
            else ""
        )
        summary = evaluate_mcq(
            model=eval_model,
            processor=processor,
            samples=eval_samples,
            max_new_tokens=int(args.max_new_tokens),
            output_dir=output_dir,
            prompt_prefix=eval_prompt_prefix,
            crop_from_question_refs=bool(args.eval_crop_from_question_refs),
            crop_collar_seconds=float(args.eval_crop_collar_seconds),
            random_crop_seconds=float(args.eval_random_crop_seconds),
            transcript_dir=args.transcript_dir,
            transcript_max_words=int(args.transcript_max_words),
        )
        print("\nEvaluation complete (eval set):")
        print(f"  Accuracy: {summary['accuracy']:.4f}")
        print(f"  n_total: {summary['n_total']}")
        print(f"  n_correct: {summary['n_correct']}")
        print(f"  Metrics saved to: {output_dir / 'mcq_eval_metrics.json'}")

    if local_rank == 0 and args.test_jsonl:
        test_task = load_jsonl_audio_mcq(
            jsonl_path=args.test_jsonl,
            audio_root=args.audio_root,
            max_questions_per_audio=int(args.max_questions_per_audio),
            max_samples=0,
            seed=int(args.seed),
        )
        test_samples = test_task.samples
        print(f"\nTest set (original): {len(test_samples)} samples from {args.test_jsonl}")
        test_summary = evaluate_mcq(
            model=model,
            processor=processor,
            samples=test_samples,
            max_new_tokens=int(args.max_new_tokens),
            output_dir=output_dir,
            prompt_prefix=eval_prompt_prefix if args.do_eval else (
                f"lang:{args.prompt_language}\n[TRANSCRIBE]\n"
                if args.use_transcription_hint_format
                else ""
            ),
            crop_from_question_refs=bool(args.eval_crop_from_question_refs),
            crop_collar_seconds=float(args.eval_crop_collar_seconds),
            random_crop_seconds=float(args.eval_random_crop_seconds),
            transcript_dir=args.transcript_dir,
            transcript_max_words=int(args.transcript_max_words),
        )
        # Rename outputs so they don't overwrite the eval set results
        for stem in ("mcq_eval_metrics.json", "mcq_eval_predictions.jsonl"):
            src = output_dir / stem
            if src.exists():
                src.rename(output_dir / stem.replace("eval", "test"))
        print("\nTest evaluation complete (original organisers):")
        print(f"  Accuracy: {test_summary['accuracy']:.4f}")
        print(f"  n_total: {test_summary['n_total']}")
        print(f"  n_correct: {test_summary['n_correct']}")
        print(f"  Metrics saved to: {output_dir / 'mcq_test_metrics.json'}")


if __name__ == "__main__":
    main()
