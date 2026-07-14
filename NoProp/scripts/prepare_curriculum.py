"""
prepare_curriculum.py — Convert curriculum JSONL to char-level token sequences.

Reads all curriculum phases, extracts text fields, maps each character
to an integer ID 4-255 (ASCII), and saves as JSONL with input_ids.

Usage:
    python prepare_curriculum.py [--phases 0,1,2,3,4,5]
                                 [--max-len 2048]
                                 [--output curriculum_tokens.jsonl]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def char_to_id(c: str) -> int:
    """Map a character to a token ID in range [4, 255]."""
    code = ord(c)
    if code < 4:
        return 4  # control chars → minimum
    if code > 255:
        return 255  # non-ASCII → max
    return code


def text_to_ids(text: str, max_len: int = 2048) -> list[int]:
    """Convert text to a list of character-level token IDs."""
    ids = [char_to_id(c) for c in text[:max_len]]
    return ids


def main():
    parser = argparse.ArgumentParser(description="Prepare curriculum data")
    parser.add_argument("--phases", type=str, default="0,1,2,3,4,5",
                        help="Comma-separated phase numbers")
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--output", type=str, default="curriculum_tokens.jsonl")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "curriculum_data"))
    args = parser.parse_args()

    phases = [int(p.strip()) for p in args.phases.split(",")]
    data_dir = args.data_dir
    output_path = args.output

    total_seqs = 0
    total_chars = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for phase in phases:
            phase_dir = os.path.join(data_dir, f"phase{phase:02d}_*")
            # Find the phase directory
            import glob
            matches = glob.glob(os.path.join(data_dir, f"phase{phase:02d}_*"))
            if not matches:
                print(f"  Phase {phase}: not found (skipping)")
                continue
            phase_path = matches[0]
            samples_path = os.path.join(phase_path, "samples.jsonl")

            if not os.path.exists(samples_path):
                print(f"  Phase {phase}: samples.jsonl not found (skipping)")
                continue

            count = 0
            chars = 0
            with open(samples_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)

                    # Extract text fields
                    texts = []
                    for key in ("input", "analysis", "final_answer", "verification"):
                        val = item.get(key, "")
                        if val and isinstance(val, str):
                            texts.append(val)

                    if not texts:
                        continue

                    combined = " | ".join(texts)
                    ids = text_to_ids(combined, args.max_len)

                    out.write(json.dumps({"input_ids": ids}, ensure_ascii=False) + "\n")
                    count += 1
                    chars += len(ids)

            print(f"  Phase {phase}: {count:>5d} sequences, {chars:>7d} chars")
            total_seqs += count
            total_chars += chars

    print(f"\n  Total: {total_seqs} sequences, {total_chars} chars")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
