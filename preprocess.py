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

REST_TONE = 0
LONG_REST_TONE = 7
END_TONE = 10

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
tones_to_intervals = dict()
with open(os.path.join(MAPPING_PATH, "pitch_mapping.json"), "r") as pitch_file:
    pitch_to_id = json.load(pitch_file)

with open(os.path.join(MAPPING_PATH, "duration_mapping.json"), "r") as duration_file:
    duration_to_id = json.load(duration_file)

with open("intelligible_intervals.json", "r") as interval_file:
    tones_to_intervals = json.load(interval_file)

id_to_pitch = { v: int(k) for k, v in pitch_to_id.items() }
id_to_duration = { v: int(k) for k, v in duration_to_id.items() }

# Determine the number of unique names and values
num_pitch = len(pitch_to_id)
num_duration = len(duration_to_id)
num_tone = 11
num_pos_internal = 16
num_pos_external = 4
num_when_rest = 8

input_params = ("pitch", "duration", "pos_internal", "pos_external", "valid_pitches") + tuple(["tone_" + str(i) for i in range(8)])
output_params = ("pitch", "duration")

param_shapes = {
    "pitch": num_pitch,
    "duration": num_duration,
    "pos_internal": num_pos_internal,
    "pos_external": num_pos_external,
    "valid_pitches": num_pitch,
}
for i in range(8):
    param_shapes["tone_" + str(i)] = num_tone


def create_datasets_and_mapping(raw_data_path, save_dir):
    # load the songs
    print("Loading songs...")
    encoded_songs = load_songs(raw_data_path)
    print(f"Loaded {len(encoded_songs)} songs.")

    encoded_songs_combined = []

    for i, encoded_song in enumerate(encoded_songs):
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
                    if not has_acceptable_duration(song, ACCEPTABLE_DURATIONS):
                        continue
                    print(file)
                    song = transpose(song)
                    song = encode_song(song)
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
                # print(event.lyric)

            elif current_note["pitch"] == event.pitch.midi and event.tie is not None:
                current_note["duration"] += event.duration.quarterLength
                # No need add tone because the note before the tie must have the same tone

            else:
                elements.append({"pitch": current_note["pitch"], "duration": int(current_note["duration"] / time_step),
                                 "tone": current_note["tone"]})

                # Save last note
                last_note = {key: value for key, value in current_note.items()}

                current_note["pitch"] = event.pitch.midi
                current_note["duration"] = event.duration.quarterLength
                current_note["tone"] = get_tone(event.lyric)

                # Check how many tones are not intelligible, fool the loss function
                key = str(last_note["tone"]) + "_" + str(current_note["tone"])
                if key in tones_to_intervals:
                    interval = current_note["pitch"] - last_note["pitch"]
                    if interval not in tones_to_intervals[key]:
                        elements.insert(-2, {"pitch": 0, "duration": 0, "tone": 0})

        elif isinstance(event, m21.note.Rest):
            if current_note["pitch"] is None:
                current_note["pitch"] = 0
                current_note["duration"] = event.duration.quarterLength
                current_note["tone"] = LONG_REST_TONE if current_note["duration"] >= 2 else REST_TONE

            elif current_note["pitch"] == 0:
                current_note["duration"] += event.duration.quarterLength
                current_note["tone"] = LONG_REST_TONE if current_note["duration"] >= 2 else REST_TONE

            else:
                elements.append({"pitch": current_note["pitch"], "duration": int(current_note["duration"] / time_step),
                                 "tone": current_note["tone"]})

                current_note["pitch"] = 0
                current_note["duration"] = event.duration.quarterLength
                current_note["tone"] = LONG_REST_TONE if current_note["duration"] >= 2 else REST_TONE

    if current_note["pitch"] is not None:
        elements.append({"pitch": current_note["pitch"], "duration": int(current_note["duration"] / time_step),
                         "tone": current_note["tone"]})

    return elements


def get_tone(word):
    jyutping = pc.characters_to_jyutping(word)
    return int(jyutping[0][1][-1])


def create_mapping(encoded_song):
    unique_pitches = list(set([element["pitch"] for element in encoded_song]))
    unique_durations = list(set([element["duration"] for element in encoded_song]))
    unique_pitches.sort()
    unique_durations.sort()

    # Create dictionaries that map each name and value to an integer ID
    pitch_to_id = {name: i for i, name in enumerate(unique_pitches)}
    duration_to_id = {value: i for i, value in enumerate(unique_durations)}

    with open(os.path.join(MAPPING_PATH, "pitch_mapping.json"), "w") as pitch_file:
        json.dump(pitch_to_id, pitch_file, indent=4)

    with open(os.path.join(MAPPING_PATH, "duration_mapping.json"), "w") as duration_file:
        json.dump(duration_to_id, duration_file, indent=4)


def bulk_append(dict_list, new_dict):
    for key, val in new_dict.items():
        dict_list[key].append(val)

def process_input_data(data, tones=None):
    pos = 0
    # treat as anticipation if long rest at start of song
    if data[0]["pitch"] == 0 and data[0]["duration"] >= 8:
        pos = -16

    onehot_vector_inputs = { k: [] for k in input_params }

    for index, element in enumerate(data):
        pitch = int(element["pitch"])
        duration = int(element["duration"])

        pitch_id = pitch_to_id[str(pitch)]
        duration_id = duration_to_id[str(duration)]

        pos += duration

        single_input = {
            "pitch": pitch_id,
            "duration": duration_id,
            # Note position within a single bar
            "pos_internal": pos % 16,
            # Note position within 4-bar phrase
            "pos_external": (pos // 16) % 4,
        }

        # Input tones, 0 = current, 1 = next, 2 = next next, etc
        for i in range(8):
            if tones == None:
                single_input["tone_" + str(i)] = END_TONE if index + i >= len(data) else data[index + i]["tone"]
            else:
                single_input["tone_" + str(i)] = END_TONE if index + i >= len(tones) else tones[index + i]
        
        valid_intervals = [0] * num_pitch
        interval_idx = str(single_input["tone_0"]) + "_" + str(single_input["tone_1"])
        # if the next tone is a rest, the only valid pitch is rest
        if single_input["tone_1"] in (REST_TONE, LONG_REST_TONE, END_TONE):
            valid_intervals[pitch_to_id["0"]] = 1
        # if starting from a rest, any note is valid
        elif interval_idx not in tones_to_intervals:
            valid_intervals = [1] * num_pitch
        # calculate a vector of valid pitches
        else:
            for interval in tones_to_intervals[interval_idx]:
                new_pitch = str(pitch + interval)
                if new_pitch in pitch_to_id:
                    valid_intervals[pitch_to_id[new_pitch]] = 1
        
        single_input = { k: [int(v == j) for j in range(param_shapes[k])] for k, v in single_input.items() }
        single_input["valid_pitches"] = valid_intervals

        bulk_append(onehot_vector_inputs, single_input)

    return onehot_vector_inputs

def generating_training_sequences(dataset_path=DATASET_PATH):
    # Give the network 4 bars of notes (64 time steps) and 4 bar of tones, with the tone that the target has

    inputs = { k: [] for k in input_params }
    outputs = { k: [] for k in output_params }

    for path, _, files in os.walk(dataset_path):
        for fileId, filename in enumerate(files):
            if filename == '.DS_Store':
                continue

            stdout.write(f"\rProcessing {filename}   ({fileId + 1} / {len(files)})")
            stdout.flush()
            with open(os.path.join(path, filename)) as file:
                data = json.load(file)
                no_of_inputs = len(data) - SEQUENCE_LENGTH

                onehot_vector_inputs = process_input_data(data)
                onehot_vector_outputs = { k: [] for k in output_params }

                for element in data:
                    pitch = int(element["pitch"])
                    duration = int(element["duration"])

                    pitch_id = pitch_to_id[str(pitch)]
                    duration_id = duration_to_id[str(duration)]

                    # Create a list of one-hot encoded vectors for each element
                    bulk_append(onehot_vector_outputs, {
                        "pitch": [int(pitch_id == i) for i in range(num_pitch)],
                        "duration": [int(duration_id == j) for j in range(num_duration)],
                    })

                for i in range(no_of_inputs):
                    bulk_append(inputs, {k: v[i:(i + SEQUENCE_LENGTH)] for k, v in onehot_vector_inputs.items()})
                    bulk_append(outputs, {k: v[i + SEQUENCE_LENGTH] for k, v in onehot_vector_outputs.items()})

    print("\nFinished file processing.")

    print(f"There are {len(inputs['pitch'])} sequences.")

    inputs = {k: np.array(v) for k, v in inputs.items()}
    outputs = {k: np.array(v) for k, v in outputs.items()}

    return inputs, outputs


def main():
    create_datasets_and_mapping(RAW_DATA_PATH, DATASET_PATH)


if __name__ == "__main__":
    main()
