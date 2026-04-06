"""Entry point for asr_merging."""

import sys

from asr_merging.cli import main  # pragma: no cover

if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) == 1:
        print("Tip: start with Whisper-Turbo router:")
        print("  python -m asr_merging whisper-turbo-router -- --config-json configuration/whisper_turbo_mlc_train_eval_baseline.json\n")
    main()
