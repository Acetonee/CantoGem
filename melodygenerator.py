import math
import os
import random
import re
from enum import Enum

import numpy as np
import music21 as m21
import pycantonese as pc
from pycantonese.word_segmentation import Segmenter

from preprocess import id_to_pitch, id_to_duration, duration_to_id
from preprocess import input_params, output_params
from preprocess import process_input_data

from preprocess import SEQUENCE_LENGTH, PAD_TONE, REST_TONE, LONG_REST_TONE

from train import build_model

from harmoniser import harmonise, CHORD_DURATION
from synthesizer import synthesize

from paths import SAVE_MODEL_PATH, MELODY_MIDI_SAVE_PATH, CHORD_MIDI_SAVE_PATH, MIDI_SAVE_PATH, PROGRESS_PATH
CHORD_REFERENCE_DO = 36

RANGE = 19  # 1.5 octaves


class Voice(Enum):  # Lower limit of voices
    BASS = 53
    TENOR = 60
    ALTO = 55
    SOPRANO = 57


def _sample_with_temperature(probabilities, temperature):
    # temperature -> infinity -> Homogenous distribution
    # temperature -> 0 -> deterministic
    # temperature -> 1 -> keep probabilities
    # normalised probabilities again to guarantee floating point errors won't round stuff down to 0 in power step

    probabilities = probabilities / np.sum(probabilities)
    probabilities = np.power(probabilities, 1 / temperature)
    probabilities = probabilities / np.sum(probabilities)

    choices = range(len(probabilities))
    index = np.random.choice(choices, p=probabilities)

    return index


def onehot_input_from_seed(data, tones):
    onehot_input_dict = process_input_data(
        [{"pitch": 1, "duration": 0} for _ in range(SEQUENCE_LENGTH)] + data,
        [{"tone": PAD_TONE, "phrasing": 0} for _ in range(SEQUENCE_LENGTH)] + tones
    )
    onehot_input = [onehot_input_dict[k] for k in input_params]

    for i, onehot_vectors in enumerate(onehot_input):
        onehot_input[i] = np.array(onehot_vectors)[np.newaxis, ...]
    return onehot_input


def save_song(melody, voice, format="midi", midi_path=MIDI_SAVE_PATH):
    transposition_factor = get_transposition_factor(melody, voice)
    stream = save_melody(melody, transposition_factor)
    stream = save_chords(melody, stream, transposition_factor)
    stream = stream.transpose(transposition_factor)

    stream.write(format, midi_path)

    synthesize()


def save_melody(melody, transposition_factor):
    stream = m21.stream.Stream()
    for note in melody:
        if note["duration"] == 0:
            continue
        # 0 is shorthand for a rest
        if note["pitch"] == 0:
            m21_event = m21.note.Rest(quarterLength=note["duration"] / 4)
        else:
            m21_event = m21.note.Note(note["pitch"], quarterLength=note["duration"] / 4)
        stream.append(m21_event)
    stream.append(m21.note.Rest(quarterLength=2))
    stream.transpose(transposition_factor).write("midi", MELODY_MIDI_SAVE_PATH)
    return stream


def save_chords(melody, stream, transposition_factor):
    chord_progression = harmonise(melody)
    chords = [chord.construct_chord(CHORD_REFERENCE_DO) for chord in chord_progression]
    total_duration = sum([note["duration"] for note in melody])

    new_stream = m21.stream.Stream()
    for i in range(0, total_duration, CHORD_DURATION):
        new_chord = m21.chord.Chord(chords[i // CHORD_DURATION], quarterLength=CHORD_DURATION // 4)
        new_chord.volume = m21.volume.Volume(velocity=60)
        stream.insert(i // 4, new_chord)
        new_stream.append(new_chord)

    new_stream.transpose(transposition_factor).write("midi", CHORD_MIDI_SAVE_PATH)
    return stream


def get_transposition_factor(melody, voice):
    lowest_note = 1000
    for event in melody:
        # Reserve bottom 10 notes for other functions
        if event["pitch"] > 10:
            lowest_note = min(lowest_note, event["pitch"])

    return voice.value - lowest_note + random.choice([0, 1, -1])


class MelodyGenerator:

    def __init__(self, model_path=SAVE_MODEL_PATH):
        self.model_path = model_path
        self.model = build_model()
        self.model.load_weights(SAVE_MODEL_PATH).expect_partial()

    def generate_melody(self, all_tones, temperature):
        # create seed with start symbols
        current_melody = []

        with open(PROGRESS_PATH, "w") as progressbar_file:
            progressbar_file.write("0.00%")

        for _ in range(len(all_tones)):
            # create seed with start symbols
            onehot_seed = onehot_input_from_seed(current_melody, all_tones)

            valid_pitches = onehot_seed[input_params.index("valid_pitches")][0, -1]

            # make a prediction
            probabilities = self.model.predict(onehot_seed)
            probabilities[output_params.index("duration")][0][duration_to_id["0"]] = 0.0
            if np.sum(probabilities[output_params.index("pitch")][0] * (valid_pitches + 0.000005)) > 0:
                probabilities[output_params.index("pitch")][0] *= (valid_pitches + 0.000005)

            # choose semi-random note from probability distribution (pitch class, duration class)
            # temperature: temperature[key]((_ + 1) / len(all_tones)
            output_note = {
                key: _sample_with_temperature(probabilities[index][0], 0.1)
                for index, key in enumerate(output_params)
            }

            output_note["pitch"] = id_to_pitch[output_note["pitch"]]
            output_note["duration"] = id_to_duration[output_note["duration"]]
            print(output_note)

            current_melody.append(output_note)

            with open(PROGRESS_PATH, "w") as progressbar_file:
                progressbar_file.write("{:.2f}%".format((_ + 1) * 100 / len(all_tones)))

        print(current_melody)
        return current_melody


def get_bell_sigmoid(min_val, max_val, roughness):
    k = (max_val - min_val) * (1 + math.exp(-roughness / 6)) / (1 - math.exp(-roughness / 6))
    return lambda x: k / (1 + math.exp(roughness * (1 / 3 - x))) + k / (
            1 + math.exp(roughness * (x - 2 / 3))) - k + min_val


def parse_lyrics(lyrics):
    print("parsing lyrics:", lyrics)
    rest_tones_pos = []
    for char in lyrics:
        if char == ",":
            rest_tones_pos.append(REST_TONE)
        elif char == "|":
            rest_tones_pos.append(LONG_REST_TONE)
        else:
            rest_tones_pos.append(0)

    pure_words = lyrics.replace(",", "").replace("|", "")
    all_tones = []

    segmenter = Segmenter(disallow={"一個人"}, allow={"大江"})
    tokens = pc.parse_text(pure_words, segment_kwargs={"cls": segmenter}).tokens()

    for token in tokens:
        print(token)
        anglicised_words = re.split(r'(?<=[0-9])(?=[a-zA-Z])', token.jyutping)
        tones = [{"tone": int(char[-1]), "phrasing": len(anglicised_words) - idx} for idx, char in
                 enumerate(anglicised_words)]
        all_tones.extend(tones)

    for i in range(len(lyrics)):
        if rest_tones_pos[i] != 0:
            all_tones.insert(i, {"tone": rest_tones_pos[i], "phrasing": 5})

    print("Finished lyrics parsing")
    return all_tones


mg = MelodyGenerator()


def make_melody_response(lyrics, voice):
    print("Starting melody generation")

    tones = parse_lyrics(lyrics)
    print("Generating")

    melody = mg.generate_melody(tones, temperature={
        # temperature
        "pitch": get_bell_sigmoid(min_val=0.1, max_val=0.3, roughness=10),
        "duration": get_bell_sigmoid(min_val=0.05, max_val=0.2, roughness=10)
    })
    save_song(melody, voice)


if __name__ == "__main__":
    make_melody_response(",大江東去,浪淘盡,千古風流人物|"
                         "故壘西邊,人道是,三國周郎赤壁|"
                         "亂石崩雲,驚濤裂岸,捲起千堆雪|"
                         "江山如畫,一時多少豪傑|"
                         "遙想公瑾當年,小喬初嫁了,雄姿英發|"
                         "羽扇綸巾,談笑間,檣櫓灰飛煙滅|"
                         "故國神遊,多情應笑我,早生華髮|"
                         "人生如夢,一尊還酹江月", Voice.TENOR)