import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

BUILD_PATH = "build"
SAVE_MODEL_PATH = os.path.join(BUILD_PATH, "model_weights.ckpt")
PLOT_PATH = os.path.join(BUILD_PATH, "training_plot.png")

MIDI_SAVE_PATH = os.path.join(BUILD_PATH, "melody.mid")
MELODY_MIDI_SAVE_PATH = os.path.join(BUILD_PATH, "melody_voice.mid")
CHORD_MIDI_SAVE_PATH = os.path.join(BUILD_PATH, "melody_chords.mid")
PROGRESS_PATH = os.path.join("serverside", "progressbar.txt")

MELODY_WAV_PATH = os.path.join(BUILD_PATH, "melody.wav")
CHORD_WAV_PATH = os.path.join(BUILD_PATH, "chords.wav")
MP3_PATH = os.path.join(BUILD_PATH, "song.mp3")
SOUNDFONT_MELODY_PATH = os.path.join(CURRENT_DIR, "soundfonts", "melody_e_guitar.sf2")
SOUNDFONT_CHORD_PATH = os.path.join(CURRENT_DIR, "soundfonts", "chords_cello.sf2")