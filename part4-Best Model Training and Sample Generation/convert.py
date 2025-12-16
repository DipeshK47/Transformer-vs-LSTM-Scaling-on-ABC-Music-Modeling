# convert_abc_to_midi.py

import os
from glob import glob

from music21 import converter

# Folder where your ABC samples are saved
SAMPLES_DIR = "/scratch/dk5288/code/my_project/part4/samples"
MIDI_DIR = os.path.join(SAMPLES_DIR, "midi")

def convert_abc_file(abc_path: str, midi_path: str):
    print(f"Converting {abc_path} -> {midi_path}")
    with open(abc_path, "r", encoding="utf8") as f:
        abc_text = f.read()
    try:
        score = converter.parseData(abc_text, format="abc")
        score.write("midi", fp=midi_path)
        print(f"Saved {midi_path}")
    except Exception as e:
        print(f"[WARN] Failed to convert {abc_path}: {e}")

def main():
    # Make sure both folders exist
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(MIDI_DIR, exist_ok=True)

    abc_files = sorted(glob(os.path.join(SAMPLES_DIR, "*.abc")))
    if not abc_files:
        print(f"No .abc files found in {SAMPLES_DIR}")
        return

    print(f"Found {len(abc_files)} ABC files")
    for abc_path in abc_files:
        base_name = os.path.splitext(os.path.basename(abc_path))[0]
        midi_path = os.path.join(MIDI_DIR, base_name + ".mid")
        convert_abc_file(abc_path, midi_path)

    print("Conversion complete")

if __name__ == "__main__":
    main()