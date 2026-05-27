#!/usr/bin/env python3
"""Create Kaldi-style files from MLC-style raw WAV + segment TXT folders.

Expected raw layout examples:
- train: <root>/<Language>/<Dialect>/<Speaker>/<file>.wav + <file>.txt
- dev:   <root>/<Language>/<file>.wav + <file>.txt
- dev:   <root>/<Language>/<Dialect>/<file>.wav + <file>.txt

For each split this script writes:
  <output_root>/<split>/
    wav.scp
    segments
    text
    utt2lang
    utt2spk
    reco2lang

TXT format expected per line (MLC26 style):
  <start_sec> <end_sec> <speaker> <transcript>
"""

from __future__ import annotations

import argparse
import json
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


SEGMENT_LINE_RE = re.compile(
    r"^\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+(\S+)\s+(.+?)\s*$"
)


@dataclass
class SegmentRow:
    utt_id: str
    rec_id: str
    start_sec: float
    end_sec: float
    speaker: str
    text: str
    language: str


def _normalize_token(s: str) -> str:
    return " ".join(str(s).replace("_", " ").split()).strip().lower()


def _sanitize_id(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-.:]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _build_language_label(language_dir: str, dialect_dir: str, mode: str) -> str:
    base = _normalize_token(language_dir)
    dialect = _normalize_token(dialect_dir)
    if mode == "base":
        return base
    if base == dialect:
        return base
    return f"{base} ({dialect})"


def _wav_duration_seconds(wav_path: Path) -> Optional[float]:
    try:
        with wave.open(str(wav_path), "rb") as wf:
            n_frames = wf.getnframes()
            sr = wf.getframerate()
            if sr <= 0:
                return None
            return float(n_frames) / float(sr)
    except Exception:
        return None


def _parse_segment_lines(txt_path: Path) -> Tuple[List[Tuple[float, float, str, str]], int]:
    rows: List[Tuple[float, float, str, str]] = []
    non_matching_nonempty = 0
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = SEGMENT_LINE_RE.match(line)
        if m:
            start = float(m.group(1))
            end = float(m.group(2))
            speaker = m.group(3).strip()
            text = m.group(4).strip()
            if end > start and text:
                rows.append((start, end, speaker, text))
        elif line.strip():
            non_matching_nonempty += 1
    return rows, non_matching_nonempty


def _centiseconds(x: float) -> int:
    return int(round(float(x) * 100.0))


def _load_recording_allowlist(path: Optional[Path]) -> Optional[Set[str]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Recording allowlist file not found: {path}")
    out = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip().lower()
        if s:
            out.add(s)
    return out


def _load_segment_key_allowlist(path: Optional[Path]) -> Optional[Set[Tuple[str, str, int, int]]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Segment-key allowlist file not found: {path}")
    out: Set[Tuple[str, str, int, int]] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split("|")
        if len(parts) != 4:
            continue
        rec = parts[0].strip().lower()
        spk = parts[1].strip().upper()
        try:
            start_cs = int(parts[2].strip())
            end_cs = int(parts[3].strip())
        except Exception:
            continue
        out.add((rec, spk, start_cs, end_cs))
    return out


def _collect_split_rows(
    split_root: Path,
    label_mode: str,
    fallback_to_whole_file: bool,
    recording_allowlist: Optional[Set[str]] = None,
    segment_key_allowlist: Optional[Set[Tuple[str, str, int, int]]] = None,
) -> Tuple[List[Tuple[str, Path, str]], List[SegmentRow], Dict[str, int]]:
    if not split_root.exists():
        raise FileNotFoundError(f"Split root does not exist: {split_root}")

    rec_rows: List[Tuple[str, Path, str]] = []
    seg_rows: List[SegmentRow] = []

    stats = {
        "wav_total": 0,
        "missing_txt": 0,
        "invalid_layout": 0,
        "segment_rows": 0,
        "segment_txt_nonmatching_lines": 0,
        "fallback_whole_file_rows": 0,
        "filtered_recording_not_allowlisted": 0,
        "filtered_segment_not_allowlisted": 0,
    }

    for wav_path in sorted(split_root.rglob("*.wav")):
        stats["wav_total"] += 1

        rel = wav_path.relative_to(split_root)
        if len(rel.parts) < 2:
            stats["invalid_layout"] += 1
            continue

        language_dir = rel.parts[0]
        dialect_dir = rel.parts[1] if len(rel.parts) >= 3 else language_dir
        language = _build_language_label(language_dir, dialect_dir, label_mode)

        rec_id = _sanitize_id("__".join(rel.with_suffix("").parts))
        rec_key = rec_id.lower()
        rec_stem_key = wav_path.stem.lower()
        if recording_allowlist is not None and rec_key not in recording_allowlist and rec_stem_key not in recording_allowlist:
            stats["filtered_recording_not_allowlisted"] += 1
            continue

        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            stats["missing_txt"] += 1
            continue

        segs, nonmatching = _parse_segment_lines(txt_path)
        stats["segment_txt_nonmatching_lines"] += nonmatching

        if segs:
            kept_for_rec = 0
            for idx, (start, end, speaker, text) in enumerate(segs, start=1):
                spk_clean = (_sanitize_id(speaker) or "spk_unk")
                seg_key = (rec_key, spk_clean.upper(), _centiseconds(start), _centiseconds(end))
                seg_key_stem = (rec_stem_key, spk_clean.upper(), _centiseconds(start), _centiseconds(end))
                if (
                    segment_key_allowlist is not None
                    and seg_key not in segment_key_allowlist
                    and seg_key_stem not in segment_key_allowlist
                ):
                    stats["filtered_segment_not_allowlisted"] += 1
                    continue
                utt_id = f"{rec_id}__seg{idx:05d}"
                seg_rows.append(
                    SegmentRow(
                        utt_id=utt_id,
                        rec_id=rec_id,
                        start_sec=start,
                        end_sec=end,
                        speaker=spk_clean,
                        text=text,
                        language=language,
                    )
                )
                kept_for_rec += 1
            stats["segment_rows"] += kept_for_rec
            if kept_for_rec > 0:
                rec_rows.append((rec_id, wav_path.resolve(), language))
            continue

        if fallback_to_whole_file:
            text_raw = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            if text_raw:
                dur = _wav_duration_seconds(wav_path)
                if dur and dur > 0.0:
                    seg_key = (rec_key, "SPK_UNK", 0, _centiseconds(float(dur)))
                    seg_key_stem = (rec_stem_key, "SPK_UNK", 0, _centiseconds(float(dur)))
                    if (
                        segment_key_allowlist is not None
                        and seg_key not in segment_key_allowlist
                        and seg_key_stem not in segment_key_allowlist
                    ):
                        stats["filtered_segment_not_allowlisted"] += 1
                        continue
                    utt_id = f"{rec_id}__seg00001"
                    seg_rows.append(
                        SegmentRow(
                            utt_id=utt_id,
                            rec_id=rec_id,
                            start_sec=0.0,
                            end_sec=float(dur),
                            speaker="spk_unk",
                            text=text_raw,
                            language=language,
                        )
                    )
                    rec_rows.append((rec_id, wav_path.resolve(), language))
                    stats["fallback_whole_file_rows"] += 1

    # Deduplicate recordings if needed.
    seen = set()
    rec_rows_unique: List[Tuple[str, Path, str]] = []
    for rec_id, wav_path, lang in rec_rows:
        if rec_id in seen:
            continue
        seen.add(rec_id)
        rec_rows_unique.append((rec_id, wav_path, lang))

    return rec_rows_unique, seg_rows, stats


def _write_kaldi_split(
    split_dir: Path,
    rec_rows: Iterable[Tuple[str, Path, str]],
    seg_rows: Iterable[SegmentRow],
) -> Dict[str, int]:
    split_dir.mkdir(parents=True, exist_ok=True)

    rec_rows = list(rec_rows)
    seg_rows = list(seg_rows)

    wav_scp = split_dir / "wav.scp"
    segments = split_dir / "segments"
    text = split_dir / "text"
    utt2lang = split_dir / "utt2lang"
    utt2spk = split_dir / "utt2spk"
    reco2lang = split_dir / "reco2lang"

    with wav_scp.open("w", encoding="utf-8") as f:
        for rec_id, wav_path, _ in rec_rows:
            f.write(f"{rec_id} {wav_path}\n")

    with reco2lang.open("w", encoding="utf-8") as f:
        for rec_id, _, lang in rec_rows:
            f.write(f"{rec_id} {lang}\n")

    with segments.open("w", encoding="utf-8") as fseg, text.open("w", encoding="utf-8") as ftxt, utt2lang.open(
        "w", encoding="utf-8"
    ) as flang, utt2spk.open("w", encoding="utf-8") as fspk:
        for row in seg_rows:
            fseg.write(f"{row.utt_id} {row.rec_id} {row.start_sec:.6f} {row.end_sec:.6f}\n")
            ftxt.write(f"{row.utt_id} {row.text}\n")
            flang.write(f"{row.utt_id} {row.language}\n")
            fspk.write(f"{row.utt_id} {row.speaker}\n")

    summary = {
        "recordings": len(rec_rows),
        "segments": len(seg_rows),
        "languages": len({r.language for r in seg_rows}),
    }
    (split_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Kaldi files from MLC-style raw WAV/TXT metadata")
    p.add_argument("--train-root", required=True)
    p.add_argument("--dev-root", required=True)
    p.add_argument("--test-root", default=None, help="Defaults to --dev-root if omitted")
    p.add_argument("--output-root", default="data/cache/kaldi/mlc26")
    p.add_argument("--label-mode", choices=["base", "dialect"], default="base")
    p.add_argument(
        "--fallback-to-whole-file",
        action="store_true",
        help="If TXT is not in segment format, use whole-file transcript with [0, wav_duration]",
    )
    p.add_argument(
        "--recording-allowlist",
        default=None,
        help="Optional file with one recording id per line (e.g., new_recording_ids.txt from diff_mlc26_vs_mlc25).",
    )
    p.add_argument(
        "--segment-key-allowlist",
        default=None,
        help=(
            "Optional file with segment keys in format rec|spk|start_cs|end_cs "
            "(e.g., segment_keys__mlc26_new_only.txt from diff_mlc26_vs_mlc25 --write-segment-key-files)."
        ),
    )
    p.add_argument("--overwrite", action="store_true", help="Remove output split dirs before writing")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_root)
    recording_allowlist = _load_recording_allowlist(Path(args.recording_allowlist) if args.recording_allowlist else None)
    segment_key_allowlist = _load_segment_key_allowlist(Path(args.segment_key_allowlist) if args.segment_key_allowlist else None)

    split_roots = {
        "train": Path(args.train_root),
        "dev": Path(args.dev_root),
        "test": Path(args.test_root) if args.test_root else Path(args.dev_root),
    }

    all_stats = {}
    for split_name, split_root in split_roots.items():
        rec_rows, seg_rows, stats = _collect_split_rows(
            split_root=split_root,
            label_mode=args.label_mode,
            fallback_to_whole_file=bool(args.fallback_to_whole_file),
            recording_allowlist=recording_allowlist,
            segment_key_allowlist=segment_key_allowlist,
        )

        split_dir = output_root / split_name
        if split_dir.exists() and args.overwrite:
            for p in sorted(split_dir.rglob("*"), reverse=True):
                if p.is_file() or p.is_symlink():
                    p.unlink()
                elif p.is_dir():
                    p.rmdir()
            split_dir.rmdir()

        summary = _write_kaldi_split(split_dir=split_dir, rec_rows=rec_rows, seg_rows=seg_rows)
        all_stats[split_name] = {
            "input_root": str(split_root),
            "recording_allowlist": args.recording_allowlist,
            "segment_key_allowlist": args.segment_key_allowlist,
            "collection": stats,
            "written": summary,
        }
        print(
            f"[{split_name}] recordings={summary['recordings']:,} segments={summary['segments']:,} "
            f"languages={summary['languages']}"
        )

    (output_root / "manifest.json").write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    print(f"Wrote Kaldi files under: {output_root}")


if __name__ == "__main__":
    main()
