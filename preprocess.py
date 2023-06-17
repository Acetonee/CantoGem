import os
import json
import music21 as m21
import numpy as np
#import tensorflow as tf

RAW_DATA_PATH = "rawdata"
SAVE_DIR = "dataset"
SINGLE_SONGS_FILE_DATASET = "file_song_dataset"
SINGLE_LYRICS_FILE_DATASET = "file_lyrics_dataset"

MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64  # 64
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


def preprocess(dataset_path):
    pass

    # load the songs
    print("Loading songs...")
    songs = load_songs(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for i, song in enumerate(songs):

        # filter out songs
        if not has_acceptable_duration(song, ACCEPTABLE_DURATIONS):
            continue

        song = transpose(song)

        # encode songs with music time series representation
        encoded_song = encode_song(song)
        encoded_lyrics = encode_lyrics(song)

        # save songs to a text file
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)
            fp.write("\n")
            fp.write(encoded_lyrics)


def load_songs(dataset_path):
    songs = []

    for path, subdir, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "mxl":
                song = m21.converter.parse(os.path.join(path, file))
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
    encoded_song = []

    for event in song.flat.notesAndRests:
        # e.g. (C, 1), (D, 0.5) -> [60, "_", "_", "_", 62, "_", "_"]

        if isinstance(event, m21.note.Note):
            if event.tie is not None and event.tie.type == 'stop':
                symbol = "_"
            else:
                symbol = event.pitch.midi
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        steps = int(event.duration.quarterLength / time_step)

        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # cast list to str
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song


def encode_lyrics(song, time_step=0.25):
    encoded_lyrics = []

    for event in song.flat.notesAndRests:

        if isinstance(event, m21.note.Note):
            if event.tie is not None and event.tie.type == 'stop':
                symbol_lyric = "_"
            else:
                symbol_lyric = event.lyric

        elif isinstance(event, m21.note.Rest):
            symbol_lyric = "r"

        steps = int(event.duration.quarterLength / time_step)

        for step in range(steps):
            if step == 0:
                encoded_lyrics.append(symbol_lyric)
            else:
                encoded_lyrics.append("_")

    encoded_lyrics = " ".join(map(str, encoded_lyrics))

    return encoded_lyrics


def create_single_file_datasets(dataset_path, song_single_dataset_path, lyric_single_dataset_path, sequence_length):
    # Every input of LSTM must be of the same length
    new_song_delimiter = "/ " * sequence_length
    all_songs = ""
    all_lyrics = ""

    # load encoded songs and lyrics + add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            with open(os.path.join(path, file), "r") as fp:
                song = fp.readline().strip()
                lyrics = fp.readline().strip()
            all_songs = all_songs + song + " " + new_song_delimiter
            all_lyrics = all_lyrics + lyrics + " " + new_song_delimiter

    # remove last space
    all_songs = all_songs[:-1]
    all_lyrics = all_lyrics[:-1]

    with open(song_single_dataset_path, "w") as fp:
        fp.write(all_songs)

    with open(lyric_single_dataset_path, "w") as fp:
        fp.write(all_lyrics)

    return all_songs, all_lyrics


def create_mapping(songs, mapping_path):
    mappings = {}

    # identify the vocab

    songs = songs.split()

    # e.g. {1 5 3 2 4 1 5 3 5} set-> {1 2 3 4 5}
    vocabulary = list(set(songs))

    # create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # save vocab to json
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)


def convert_songs_to_int(songs):
    int_songs = []

    # load mappings
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    # cast songs string to a list
    songs = songs.split()

    # map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs


def generating_training_sequences(sequence_length):
    # Give the network 4 bars of notes (64 time steps), and ask it to predict the next one

    # TODO: load songs
    songs = ""  # load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    inputs = []
    targets = []

    # generate the training sequences (e.g. 100 notes -> 36 training samples)
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i + sequence_length])
        targets.append(int_songs[i + sequence_length])

    # one-hot encoding
    # input shape: (# of sequences, sequence length) -> (# of sequences, sequence length, vocabulary size)
    vocabulary_size = len(set(int_songs))
    # inputs = tf.keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    return inputs, targets


def main():
    preprocess(RAW_DATA_PATH)
    songs, lyrics = create_single_file_datasets(SAVE_DIR, SINGLE_SONGS_FILE_DATASET,
                                                SINGLE_LYRICS_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)

    # TODO: Adapt these lines of code
    # inputs, targets = generating_training_sequences(SEQUENCE_LENGTH)


if __name__ == "__main__":
    main()
