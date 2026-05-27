"""Command-line interface for ASR-merging helpers."""

from __future__ import annotations

import argparse
import sys


def main() -> None:  # pragma: no cover
    """Entry point for `python -m asr_merging` and `asr_merging`."""

    parser = argparse.ArgumentParser(
        prog="asr_merging",
        description="ASR-merging command-line tools",
        epilog=(
            "Quick start:\n"
            "  python -m asr_merging whisper-turbo-router -- --config-json configuration/whisper_turbo_mlc_train_eval_baseline.json\n"
            "  python -m asr_merging seamless-router -- --config-json configuration/seamless_mlc_train_eval_baseline.json\n"
            "  python -m asr_merging voxtral-eval -- --source mlc --splits dev test --checkpoint-path experiments/mlc_train_eval_29k_20260406_224234"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    seamless = subparsers.add_parser(
        "seamless-router",
        help="Run the Seamless train/eval router.",
        description="Pass-through wrapper for asr_merging.seamless_train_router",
    )
    seamless.add_argument(
        "router_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to seamless_train_router (prefix with --).",
    )

    whisper = subparsers.add_parser(
        "whisper-turbo-router",
        help="Run the Whisper-Turbo train/eval router.",
        description="Pass-through wrapper for asr_merging.whisper_turbo_train_router",
    )
    whisper.add_argument(
        "router_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to whisper_turbo_train_router (prefix with --).",
    )

    voxtral_eval = subparsers.add_parser(
        "voxtral-eval",
        help="Run the Voxtral evaluation router.",
        description="Pass-through wrapper for asr_merging.voxtral_eval_router",
    )
    voxtral_eval.add_argument(
        "router_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to voxtral_eval_router (prefix with --).",
    )

    args = parser.parse_args()

    if args.command == "seamless-router":
        from . import seamless_train_router

        forwarded = list(args.router_args or [])
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]

        original_argv = sys.argv
        try:
            sys.argv = ["seamless_train_router"] + forwarded
            seamless_train_router.main()
        finally:
            sys.argv = original_argv
        return

    if args.command == "whisper-turbo-router":
        from . import whisper_turbo_train_router

        forwarded = list(args.router_args or [])
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]

        original_argv = sys.argv
        try:
            sys.argv = ["whisper_turbo_train_router"] + forwarded
            whisper_turbo_train_router.main()
        finally:
            sys.argv = original_argv
        return

    if args.command == "voxtral-eval":
        from . import voxtral_eval_router

        forwarded = list(args.router_args or [])
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]

        original_argv = sys.argv
        try:
            sys.argv = ["voxtral_eval_router"] + forwarded
            voxtral_eval_router.main()
        finally:
            sys.argv = original_argv
        return

    parser.print_help()
