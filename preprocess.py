import os
import json

import music21 as m21
import pycantonese as pc

RAW_DATA_PATH = "rawdata"
DATASET_PATH = "dataset"
MAPPING_PATH = "mappings"
SEQUENCE_LENGTH = 64


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

    print(encoded_songs_combined)
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

    for event in song.flat.notesAndRests:

        if isinstance(event, m21.note.Note):
            element = {"pitch": event.pitch.midi, "duration": int(event.duration.quarterLength / time_step)}
        elif isinstance(event, m21.note.Rest):
            element = {"pitch": 0, "duration": int(event.duration.quarterLength / time_step)}

        elements.append(element)

    return elements


def get_tone(word):
    jyutping = pc.characters_to_jyutping(word)
    return jyutping[0][1][-1]


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


def generating_training_sequences(dataset_path=DATASET_PATH, mapping_path=MAPPING_PATH):
    # Give the network 4 bars of notes (64 time steps) and 4 bar of tones, with the tone that the target has

    with open(os.path.join(mapping_path, "pitch_mapping.json"), "r") as pitch_file:
        pitch_to_id = json.load(pitch_file)

    with open(os.path.join(mapping_path, "duration_mapping.json"), "r") as duration_file:
        duration_to_id = json.load(duration_file)

    with open(os.path.join(dataset_path, "0.json")) as file:
        dataset = json.load(file)

    # Create a list of encoded elements, where each element is a tuple of (name_id, value_id)
    encoded_elements = [(pitch_to_id[str(element["pitch"])], duration_to_id[str(element["duration"])]) for element in
                        dataset]

    # Determine the number of unique names and values
    num_pitch = len(pitch_to_id)
    num_duration = len(duration_to_id)

    # Create a list of one-hot encoded vectors for each element
    encoded_vectors = [[int(pitch_id == i) for i in range(num_pitch)] + [int(duration_id == j) for j in range(num_duration)]
                       for pitch_id, duration_id in encoded_elements]
    print(encoded_vectors)


def main():
    create_datasets_and_mapping(RAW_DATA_PATH, DATASET_PATH)


if __name__ == "__main__":
    main()
