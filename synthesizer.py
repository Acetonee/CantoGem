import os
from pydub import AudioSegment

from midi2audio import FluidSynth
from paths import SOUNDFONT_MELODY_PATH, SOUNDFONT_CHORD_PATH, MELODY_MIDI_SAVE_PATH, CHORD_MIDI_SAVE_PATH, MELODY_WAV_PATH, CHORD_WAV_PATH, MP3_PATH



def synthesize():
    fs = FluidSynth(SOUNDFONT_MELODY_PATH)
    fs.midi_to_audio(MELODY_MIDI_SAVE_PATH, MELODY_WAV_PATH)
    fs = FluidSynth(SOUNDFONT_CHORD_PATH)
    fs.midi_to_audio(CHORD_MIDI_SAVE_PATH, CHORD_WAV_PATH)
    melody = AudioSegment.from_file(MELODY_WAV_PATH, format="wav")
    chords = AudioSegment.from_file(CHORD_WAV_PATH, format="wav")
    melody.overlay(chords).export(MP3_PATH)
    os.remove(MELODY_WAV_PATH)
    os.remove(CHORD_WAV_PATH)


if __name__ == "__main__":
    synthesize()
