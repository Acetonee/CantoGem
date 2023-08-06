import os
import keras_tuner

from preprocess import generating_training_sequences
from preprocess import input_params, TESTING_DATASET_PATH, param_shapes
from train import SAVE_MODEL_PATH
from tensorflow import keras

TUNING_DIRECTORY = "tunings"
PROJECT_NAME = "CantoGem"


class PitchLoss(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, input_valid_pitches, output_pitch):
        # batch_size, last in sequence, whole range of pitches
        invalid_pitches = 1 - input_valid_pitches[:, -1, :]
        invalid_pitch_probabilities = keras.layers.Multiply()([invalid_pitches, output_pitch])
        # loss func: log(x + 1) where x is sum of probabilities over invalid pitches
        sum = keras.backend.sum(keras.backend.sum(invalid_pitch_probabilities)) * 5 + 1
        return keras.backend.log(sum)


def build_hyper_model(hp):
    inputs = dict()
    LSTM_processed_inputs = dict()
    outputs = dict()

    # create the model architecture
    for input_type in input_params:
        inputs[input_type] = keras.layers.Input(shape=(None, param_shapes[input_type]))
        tmp = keras.layers.LSTM(param_shapes[input_type], return_sequences=True)(inputs[input_type])
        LSTM_processed_inputs[input_type] = keras.layers.Dropout(hp.Float(f'input dropout: {input_type}', min_value=0.1,
                                                                          max_value=0.9, step=0.1))(tmp)

    combined_input = keras.layers.concatenate(list(LSTM_processed_inputs.values()))

    combined_dropout_1 = hp.Float('combined input dropout 1', min_value=0.1, max_value=0.9, step=0.1)
    combined_dropout_2 = hp.Float('combined input dropout 2', min_value=0.1, max_value=0.9, step=0.1)

    x = keras.layers.LSTM(512, return_sequences=True)(combined_input)
    x = keras.layers.Dropout(combined_dropout_1)(x)
    x = keras.layers.LSTM(512)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(combined_dropout_2)(x)
    x = keras.layers.Dense(256, activation="relu")(x)

    combined_dropout_3 = hp.Float('combined input dropout 3', min_value=0.1, max_value=0.9, step=0.1)

    pitch_output_dropout = hp.Float('pitch output dropout', min_value=0.1, max_value=0.9, step=0.1)
    duration_output_dropout = hp.Float('duration output dropout', min_value=0.1, max_value=0.9, step=0.1)

    tmp = keras.layers.Dense(128, activation="relu")(x)
    tmp = keras.layers.BatchNormalization()(tmp)
    tmp = keras.layers.Dropout(combined_dropout_3)(tmp)
    outputs["pitch"] = keras.layers.Dense(param_shapes["pitch"], activation="softmax", name="pitch")(tmp)
    tmp = keras.layers.Dropout(pitch_output_dropout)(outputs["pitch"])
    tmp = keras.layers.Dense(128, activation="relu")(keras.layers.concatenate([x, tmp]))
    tmp = keras.layers.BatchNormalization()(tmp)
    tmp = keras.layers.Dropout(duration_output_dropout)(tmp)
    outputs["duration"] = keras.layers.Dense(param_shapes["duration"], activation="softmax", name="duration")(tmp)

    model = keras.Model(list(inputs.values()), list(outputs.values()))
    model.add_loss(PitchLoss()(inputs["valid_pitches"], outputs["pitch"]))

    # compile model
    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(0.001),
                  metrics=["accuracy"])

    return model


def tune():

    # generate the training sequences
    inputs, targets = generating_training_sequences()
    testing_inputs, testing_targets = generating_training_sequences(dataset_path=TESTING_DATASET_PATH)

    tuner = keras_tuner.RandomSearch(
        hypermodel=build_hyper_model,
        directory=TUNING_DIRECTORY,
        project_name=PROJECT_NAME,
        objective="val_pitch_accuracy",
        max_trials=70,
    )

    input_mapped = {}
    for i, key in enumerate(inputs):
        new_key = f"input_{i + 1}"
        input_mapped[new_key] = inputs[key]

    testing_input_mapped = {}
    for i, key in enumerate(testing_inputs):
        new_key = f"input_{i + 1}"
        testing_input_mapped[new_key] = testing_inputs[key]

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    print("-" * 20)
    tuner.search_space_summary()
    tuner.search(x=input_mapped, y=targets, epochs=100, validation_data=(testing_input_mapped, testing_targets),
                 callbacks=[early_stopping])
    print("-" * 20)
    tuner.results_summary()

    models = tuner.get_best_models(num_models=1)
    best_model = models[0]

    # Evaluate the model
    print("--- Model evaluation --- ")
    best_model.evaluate(x=list(testing_inputs.values()), y=list(testing_targets.values()))
    best_model.save_weights(SAVE_MODEL_PATH)


if __name__ == "__main__":
    tune()
