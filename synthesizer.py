import os

from midi2audio import FluidSynth
from train import BUILD_PATH

MP3_PATH = os.path.join(BUILD_PATH, "song.mp3")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOUNDFONT_PATH = os.path.join(CURRENT_DIR, "soundfonts", "soundfonts.sf2")


def synthesize(midi_path, mp3_path=MP3_PATH, soundfont_path=SOUNDFONT_PATH):
    fs = FluidSynth(soundfont_path)
    fs.midi_to_audio(midi_path, mp3_path)


if __name__ == "__main__":
    synthesize("build/melody.mid")
