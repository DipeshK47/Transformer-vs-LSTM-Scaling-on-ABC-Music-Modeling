import os
import subprocess
import traceback
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Path to the midi2abc binary that you built
MIDI2ABC_BIN = "/scratch/dk5288/code/my_project/midi2abc/abcmidi/midi2abc"

# Input and output roots
INPUT_DIR = "/scratch/dk5288/data/lmd_full"
OUTPUT_DIR = "/scratch/dk5288/data/abc_midi2abc"

# Error log
LOG_FILE = "midi2abc_conversion_errors.log"


def process_file(midi_path: str):
    """Convert a single MIDI file to ABC using midi2abc."""
    midi_path = os.path.abspath(midi_path)
    rel = os.path.relpath(midi_path, INPUT_DIR)

    out_dir = os.path.join(OUTPUT_DIR, os.path.dirname(rel))
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(rel))[0]
    out_path = os.path.join(out_dir, base + ".abc")

    try:
        # Skip if already converted and non empty
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return midi_path, True, "skipped_existing"

        # Call midi2abc
        result = subprocess.run(
            [MIDI2ABC_BIN, "-f", midi_path, "-o", out_path],
            stdout=subprocess.DEVNULL,   # ignore stdout
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"midi2abc failed:\n{result.stderr}")

        # Check output size
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError("Output ABC file is empty or missing")

        return midi_path, True, None

    except Exception:
        return midi_path, False, traceback.format_exc()


def find_midi_files(root_dir: str):
    """Return list of all .mid files under root_dir."""
    return [str(p) for p in Path(root_dir).rglob("*.mid") if p.is_file()]


if __name__ == "__main__":
    print(f"Scanning for MIDI files in {INPUT_DIR} ...")
    midi_files = find_midi_files(INPUT_DIR)
    print(f"Found {len(midi_files)} MIDI files.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    n_workers = cpu_count()
    print(f"Using {n_workers} worker processes")

    with Pool(n_workers) as pool:
        for res in tqdm(
            pool.imap_unordered(process_file, midi_files),
            total=len(midi_files),
            desc="midi2abc converting",
        ):
            results.append(res)

    success = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - success

    with open(LOG_FILE, "w") as f:
        for path, ok, err in results:
            if not ok:
                f.write(f"Failed: {path}\n{err}\n{'='*60}\n")

    print(f"Successfully converted: {success}")
    print(f"Failed conversions: {failed}")
    print(f"Error log saved to {LOG_FILE}")