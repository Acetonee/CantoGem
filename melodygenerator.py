import keras.utils
import tensorflow as tf
import json
import numpy as np
import music21 as m21
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH, TONE_MAPPING, TONE_MAPPING_SIZE


class MelodyGenerator:

    def __init__(self, model_path="model.h5"):

        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._notes_mapping = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_melody(self, first_note, all_tones, max_sequence_length, temperature):

        # create seed with start symbols

        all_tones = all_tones.split()
        seed_notes = first_note.split()

        result_melody = seed_notes  # TODO: Adapt to tuple

        seed_tones = self._start_symbols
        seed_notes = self._start_symbols + seed_notes

        # map seeds to int
        seed_tones = [TONE_MAPPING[symbol] for symbol in seed_tones]
        seed_notes = [self._notes_mapping[symbol] for symbol in seed_notes]

        while True:

            # limit the seed to max_sequence_length
            seed_notes = seed_notes[-(max_sequence_length - 1):]  # E.g. Sequence length of 64, leave 1 for prediction
            seed_tones = seed_tones[-(max_sequence_length - 1):]

            print(seed_notes)
            print(seed_tones)

            input_notes = seed_notes
            input_notes.append(0)
            input_tones = seed_tones
            input_tones.append(TONE_MAPPING[all_tones[0]])

            # print(input_notes)
            # print(input_tones)

            # one-hot encode the seed
            onehot_seed_notes = tf.keras.utils.to_categorical(input_notes, num_classes=len(self._notes_mapping))
            onehot_seed_tones = tf.keras.utils.to_categorical(input_tones, num_classes=TONE_MAPPING_SIZE)

            # 2D->3D
            onehot_seed_notes = onehot_seed_notes[np.newaxis, ...]
            onehot_seed_tones = onehot_seed_tones[np.newaxis, ...]

            # Concat
            onehot_seed = tf.concat([onehot_seed_notes, onehot_seed_tones], 2)

            # make a prediction
            probabilities = self.model.predict(onehot_seed)[0]
            notes_output_int = self._sample_with_temperature(probabilities, temperature)

            seed_notes.append(notes_output_int)
            # map
            notes_output_symbol = [k for k, v in self._notes_mapping.items() if v == notes_output_int][0]

            tones_output_int = TONE_MAPPING["_"] if notes_output_symbol == "_" or "r" else TONE_MAPPING[all_tones[0]]

            seed_tones.append(tones_output_int)

            if notes_output_symbol != "_" or "r":
                all_tones.pop(0)

            if notes_output_symbol == "/" or len(all_tones) == 0:
                break

            result_melody.append(notes_output_symbol)

        return result_melody

    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="melody.mid"):

        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):

            if symbol != "_" or i + 1 == len(melody):

                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter  # 0.25 * 4 = 1

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1

                start_symbol = symbol

                # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1

        stream.write(format, file_name)

    def _sample_with_temperature(self, probabilities, temperature):
        # temperature -> infinity -> Homogenous distribution
        # temperature -> 0 -> deterministic
        # temperature -> 1 -> keep probabilities

        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)

        return index


if __name__ == "__main__":
    mg = MelodyGenerator()
    tones = "4 6 6 1 2 1 2 2 9 3"  # After 9 3
    initial_note = "60"

    melody = mg.generate_melody(initial_note, tones, SEQUENCE_LENGTH, 0.1)
    print(melody)
    mg.save_melody(melody)
