import os
import json
from sys import stdout
from time import sleep

import music21 as m21
import numpy as np
import pycantonese as pc

RAW_DATA_PATH = "rawdata"
DATASET_PATH = "dataset"
MAPPING_PATH = "mappings"
SEQUENCE_LENGTH = 16


ACCEPTABLE_DURATIONS = [
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2,
    3,
    4
]

pitch_to_id = dict()
duration_to_id = dict()
with open(os.path.join(MAPPING_PATH, "pitch_mapping.json"), "r") as pitch_file:
    pitch_to_id = json.load(pitch_file)

with open(os.path.join(MAPPING_PATH, "duration_mapping.json"), "r") as duration_file:
    duration_to_id = json.load(duration_file)

# Determine the number of unique names and values
num_pitch = len(pitch_to_id)
num_duration = len(duration_to_id)
num_tone = 10
num_bar_internal = 16
num_bar_external = 4


def create_datasets_and_mapping(raw_data_path, save_dir):
    # load the songs
    print("Loading songs...")
    songs = load_songs(raw_data_path)
    print(f"Loaded {len(songs)} songs.")

    encoded_songs_combined = []

    for i, song in enumerate(songs):
        if not has_acceptable_duration(song, ACCEPTABLE_DURATIONS):
            continue

        song = transpose(song)
        encoded_song = encode_song(song)

        if encoded_songs_combined is None:
            encoded_songs_combined = encoded_song.copy()
        else:
            encoded_songs_combined += encoded_song

        with open(os.path.join(save_dir, str(i) + ".json"), "w") as fp:
            json.dump(encoded_song, fp, indent=4)

    create_mapping(encoded_songs_combined)


def load_songs(dataset_path):
    songs = []
    for path, subdir, files in os.walk(dataset_path):
        for sub in subdir:  # loop over subdirectories in path
            sub_path = os.path.join(path, sub)  # get full subdirectory path
            for file in os.listdir(sub_path):
                if file[-3:] == "mxl":
                    file_path = os.path.join(sub_path, file)  # get full file path
                    song = m21.converter.parse(file_path)
                    songs.append(song)
    return songs


def has_acceptable_duration(song, acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(song):
    # get key from song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # estimate key
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # get interval for transposition
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    transposed_song = song.transpose(interval)

    return transposed_song


def encode_song(song, time_step=0.25):

    elements = []
    current_note = {"pitch": None, "duration": None, "tone": None}

    for event in song.flat.notesAndRests:
        if isinstance(event, m21.note.Note):
            if current_note["pitch"] is None:
                current_note["pitch"] = event.pitch.midi
                current_note["duration"] = event.duration.quarterLength
                current_note["tone"] = get_tone(event.lyric)

            elif current_note["pitch"] == event.pitch.midi and event.tie is not None:
                current_note["duration"] += event.duration.quarterLength
                # No need add tone because the note before the tie must have the same tone

            else:
                elements.append({"pitch": current_note["pitch"], "duration": int(current_note["duration"] / time_step),
                                 "tone": current_note["tone"]})
                current_note["pitch"] = event.pitch.midi
                current_note["duration"] = event.duration.quarterLength
                current_note["tone"] = get_tone(event.lyric)

        elif isinstance(event, m21.note.Rest):
            if current_note["pitch"] is not None:
                elements.append({"pitch": current_note["pitch"], "duration": int(current_note["duration"] / time_step),
                                 "tone": current_note["tone"]})
                current_note = {"pitch": None, "duration": None, "tone": None}
            elements.append({"pitch": 0, "duration": int(event.duration.quarterLength / time_step), "tone": 0})

    if current_note["pitch"] is not None:
        elements.append({"pitch": current_note["pitch"], "duration": int(current_note["duration"] / time_step),
                         "tone": current_note["tone"]})

    return elements


def get_tone(word):
    jyutping = pc.characters_to_jyutping(word)
    return int(jyutping[0][1][-1])


def create_mapping(encoded_song):

    unique_pitches = set([element["pitch"] for element in encoded_song])
    unique_durations = set([element["duration"] for element in encoded_song])

    # Create dictionaries that map each name and value to an integer ID
    pitch_to_id = {name: i for i, name in enumerate(unique_pitches)}
    duration_to_id = {value: i for i, value in enumerate(unique_durations)}

    with open(os.path.join(MAPPING_PATH, "pitch_mapping.json"), "w") as pitch_file:
        json.dump(pitch_to_id, pitch_file, indent=4)

    with open(os.path.join(MAPPING_PATH, "duration_mapping.json"), "w") as duration_file:
        json.dump(duration_to_id, duration_file, indent=4)

def bulk_append(dictList, new_dict):
    for key, val in new_dict.items():
        dictList[key].append(val)

def generating_training_sequences(dataset_path=DATASET_PATH):
    # Give the network 4 bars of notes (64 time steps) and 4 bar of tones, with the tone that the target has

    inputs = {
        "pitch": [],
        "duration": [],
        "tone": [],
        "pos_internal": [],
        "pos_external": [],
    }
    outputs = {
        "pitch": [],
        "duration": [],
    }

    for path, _, files in os.walk(dataset_path):
        for fileId, filename in enumerate(files):
            if filename == '.DS_Store':
                continue

            stdout.write(f"\rProcessing {filename}   ({fileId + 1} / {len(files)})     ")
            stdout.flush()
            with open(os.path.join(path, filename)) as file:
                data = json.load(file)
                no_of_inputs = len(data) - SEQUENCE_LENGTH
                pos = 0
                onehot_vector_inputs = {
                    "pitch": [],
                    "duration": [],
                    "tone": [],
                    "pos_internal": [],
                    "pos_external": [],
                }
                onehot_vector_outputs = {
                    "pitch": [],
                    "duration": [],
                }
                for element in data:
                    pitch = int(element["pitch"])
                    duration = int(element["duration"])

                    pitch_id = pitch_to_id[str(pitch)]
                    duration_id = duration_to_id[str(duration)]
                    tone_id = int(element["tone"])  # Tone no need mapping

                    pos = (pos + duration) % 64
                    # Create a list of one-hot encoded vectors for each element
                    bulk_append(onehot_vector_outputs, {
                        "pitch": [int(pitch_id == i) for i in range(num_pitch)],
                        "duration": [int(duration_id == j) for j in range(num_duration)],
                    })
                    # Add position and tone data to input
                    bulk_append(onehot_vector_inputs, {
                        "pitch": [int(pitch_id == i) for i in range(num_pitch)],
                        "duration": [int(duration_id == j) for j in range(num_duration)],
                        "tone": [int(tone_id == k) for k in range(num_tone)],
                        # Note position within a single bar
                        "pos_internal": [int(pos == k) for k in range(16)],
                        # Note position within 4-bar phrase
                        "pos_external": [int((pos // 16) % 4 == k) for k in range(4)],
                    })

                for i in range(no_of_inputs):
                    bulk_append(inputs, { k: v[i:(i + SEQUENCE_LENGTH)] for k, v in onehot_vector_inputs.items() })
                    bulk_append(outputs, { k: v[i + SEQUENCE_LENGTH] for k, v in onehot_vector_outputs.items() })

    print("\nFinished file processing.")

    print(f"There are {len(inputs['pitch'])} sequences.")

    inputs = { k: np.array(v) for k, v in inputs.items() }
    outputs = { k: np.array(v) for k, v in outputs.items() }

    return inputs, outputs


def main():
    create_datasets_and_mapping(RAW_DATA_PATH, DATASET_PATH)


if __name__ == "__main__":
    main()
