#!/usr/bin/env python3
"""
Split a per-question challenge JSONL into N contiguous session-based shards.

Sessions are assigned to shards in contiguous blocks preserving their order of
first appearance in the input file.  Within each shard the original line order
is preserved.  Merging shard hyp.txt files in ascending shard index order
therefore reproduces the original JSONL ordering required by the challenge scorer.

Usage
-----
python -m asr_merging.scripts.split_challenge_jsonl \
    --jsonl-path data/mlc26_task2/task2_phase1_questions_options.jsonl \
    --output-dir  data/mlc26_task2/challenge_eval_shards \
    --num-shards 4
"""
import argparse
import json
from collections import OrderedDict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl-path", required=True, help="Input JSONL file (one question per line).")
    p.add_argument("--output-dir", required=True, help="Directory to write shard files into.")
    p.add_argument("--num-shards", type=int, default=4, help="Number of shards (default: 4).")
    args = p.parse_args()

    src = Path(args.jsonl_path)
    out_dir = Path(args.output_dir)
    n_shards = max(1, int(args.num_shards))

    if not src.exists():
        raise FileNotFoundError(f"Input JSONL not found: {src}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Pass 1: collect all lines, track session order of first appearance ---
    raw_lines: list[str] = []
    session_order: "OrderedDict[str, None]" = OrderedDict()

    with src.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            sid = str(row.get("session_id") or "").strip()
            if not sid:
                sid = "__unknown__"
            session_order.setdefault(sid, None)
            raw_lines.append(stripped)

    unique_sessions = list(session_order.keys())
    n_sessions = len(unique_sessions)
    print(f"Input : {src.name}  —  {len(raw_lines)} questions, {n_sessions} unique sessions")

    # --- Assign sessions to contiguous shard blocks ---
    session_to_shard: dict[str, int] = {}
    for i, sid in enumerate(unique_sessions):
        session_to_shard[sid] = i * n_shards // n_sessions

    shard_session_counts: list[int] = [0] * n_shards
    for sid, sh in session_to_shard.items():
        shard_session_counts[sh] += 1

    # --- Pass 2: write shard files ---
    shard_paths = [out_dir / f"shard_{i:02d}_of_{n_shards:02d}.jsonl" for i in range(n_shards)]
    handles = [p.open("w", encoding="utf-8") for p in shard_paths]

    shard_line_counts = [0] * n_shards
    for stripped in raw_lines:
        row = json.loads(stripped)
        sid = str(row.get("session_id") or "").strip() or "__unknown__"
        sh = session_to_shard[sid]
        handles[sh].write(stripped + "\n")
        shard_line_counts[sh] += 1

    for h in handles:
        h.close()

    # --- Report ---
    total_q = sum(shard_line_counts)
    print(f"\n{'Shard':<8} {'File':<45} {'Sessions':>8} {'Questions':>10}")
    print("-" * 75)
    for i, sp in enumerate(shard_paths):
        print(f"{i:<8} {sp.name:<45} {shard_session_counts[i]:>8} {shard_line_counts[i]:>10}")
    print("-" * 75)
    print(f"{'Total':<8} {'':<45} {n_sessions:>8} {total_q:>10}")
    print(f"\nMerge command (preserves original order):")
    cats = " ".join(str(sp) for sp in shard_paths)
    print(f"  cat {cats} > merged_hyp.txt  # for JSONL merging")
    print(f"  (for hyp.txt after eval: cat shard_*_hyp.txt in shard index order)")


if __name__ == "__main__":
    main()
