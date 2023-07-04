from tensorflow import keras, concat as onehot_concat

import numpy as np
import music21 as m21

from preprocess import END_TONE
from preprocess import pitch_to_id, duration_to_id
from preprocess import id_to_pitch, id_to_duration
from preprocess import input_params, output_params, param_shapes


class MelodyGenerator:

    def __init__(self, model_path="model.h5"):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

    def onehot_input_from_seed(self, data, tones):
        onehot_input = list(map(lambda _: [], input_params))
        pos = 0
        for index, element in enumerate(data):
            pitch = int(element["pitch"])
            duration = int(element["duration"])

            pitch_id = pitch_to_id[str(pitch)]
            duration_id = duration_to_id[str(duration)]
            current_tone_id = tones[index]  # Tone no need mapping
            next_tone_id = END_TONE if index + 1 >= len(tones) else tones[index + 1]

            pos = (pos + duration) % 64
            
            until_end = len(tones) - index - 1
            # Create a list of one-hot encoded vectors for each element
            # Add position and tone data to input
            single_input: dict[str, int] = {
                "pitch": pitch_id,
                "duration": duration_id,
                "current_tone": current_tone_id,
                "next_tone": next_tone_id,
                # Note position within a single bar
                "pos_internal": pos % 16,
                # Note position within 4-bar phrase
                "pos_external": (pos // 16) % 4,
                "when_end": until_end,
            }

            for i, key in enumerate(input_params):
                onehot_input[i].append([int(single_input[key] == k) for k in range(param_shapes[key])])
        for i, onehot_vectors in enumerate(onehot_input):
            onehot_input[i] = np.array(onehot_vectors)[np.newaxis, ...]
        return onehot_input

    def generate_melody(self, first_note, all_tones, temperature):

        all_tones = list(map(lambda x: int(x), all_tones.split()))
        
        # create seed with start symbols
        current_melody = [ first_note ]

        for _ in all_tones:
            # create seed with start symbols
            onehot_seed = self.onehot_input_from_seed(current_melody, all_tones)

            # make a prediction
            probabilities = self.model.predict(onehot_seed)

            # choose semi-random note from probability distribution (pitch class, duration class)
            output_note = {
                key: self._sample_with_temperature(probabilities[index][0], temperature[key])
                for index, key in enumerate(output_params)
            }

            output_note["pitch"] = id_to_pitch[output_note["pitch"]]
            output_note["duration"] = id_to_duration[output_note["duration"]]

            current_melody.append(output_note)

        return current_melody

    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="melody.mid"):
        stream = m21.stream.Stream()
        for note in melody:
            m21_event = m21.note.Rest(0)
            # 0 is shorthand for a rest
            if note["pitch"] == 0:
                m21_event = m21.note.Rest(quarterLength=note["duration"] * step_duration)
            else:
                m21_event = m21.note.Note(note["pitch"], quarterLength=note["duration"] * step_duration)
            stream.append(m21_event)

        stream.write(format, file_name)
        print(melody)
        print ("Melody saved")

    def _sample_with_temperature(self, probabilities, temperature):
        # temperature -> infinity -> Homogenous distribution
        # temperature -> 0 -> deterministic
        # temperature -> 1 -> keep probabilities
        probabilities = np.power(probabilities, 1 / temperature)
        probabilities = probabilities / np.sum(probabilities)

        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)

        return index


if __name__ == "__main__":
    mg = MelodyGenerator()
    # tones = "4 6 6 1 2 1 2 2 9 3"  # After 9 3
    tones = "1 3 4 4 4 1 2 5 6 1 1 5 4 3 4 6 1 1 1 4 6 3 1 1 2 6 1 1 1 3 4 4 6 2 2 5 1 4 1 2"
    initial_note = {
        "pitch": 67,
        "duration": 4
    }

    melody = mg.generate_melody(initial_note, tones, temperature={"pitch": 0.8, "duration": 1.3})
    mg.save_melody(melody)
