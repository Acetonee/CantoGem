import os

from preprocess import generating_training_sequences
from preprocess import input_params, TESTING_DATASET_PATH, param_shapes
from tensorflow import keras

import matplotlib.pyplot as plt

LEARNING_RATE = 0.002
EPOCHS = 100
BATCH_SIZE = 50
BUILD_PATH = "build"
SAVE_MODEL_PATH = os.path.join(BUILD_PATH, "model_weights.ckpt")
PLOT_PATH = os.path.join(BUILD_PATH, "training_plot.png")


class PitchLoss(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, input_valid_pitches, output_pitch):
        # batch_size, last in sequence, whole range of pitches
        invalid_pitches = 1 - input_valid_pitches[:, -1, :]
        invalid_pitch_probabilities = keras.layers.Multiply()([invalid_pitches, output_pitch])
        # calculates the probability of invalid pitch averaged over batch size
        sum = keras.backend.sum(keras.backend.sum(invalid_pitch_probabilities)) * (1 / BATCH_SIZE)
        return keras.backend.log(sum * 5 + 1) * 2


# Optimised using keras tuner
INPUT_DROPOUTS = {
    "pitch": 0.4,
    "duration": 0.3,
    "pos_internal": 0.2,
    "pos_external": 0.3,
    "valid_pitches": 0.4,
    "tone_0": 0.8,
    "tone_1": 0.2,
    "tone_2": 0.3,
    "tone_3": 0.4,
    "tone_4": 0.5,
    "tone_5": 0.6,
    "tone_6": 0.7,
    "tone_7": 0.8,
    # phrasing still not accounted for, just put random here
    "phrasing": 0.6,
}


def build_model():
    inputs = dict()
    LSTM_processed_inputs = dict()

    outputs = dict()

    # create the model architecture
    for type in input_params:
        inputs[type] = keras.layers.Input(shape=(None, param_shapes[type]))
        tmp = keras.layers.LSTM(param_shapes[type], return_sequences=True)(inputs[type])
        # slowly increase dropout the farther a tone is
        LSTM_processed_inputs[type] = keras.layers.Dropout(INPUT_DROPOUTS[type])(tmp)

    combined_input = keras.layers.concatenate(list(LSTM_processed_inputs.values()))

    x = keras.layers.LSTM(128, return_sequences=True)(combined_input)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.LSTM(384)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(384, activation="relu")(x)

    tmp = keras.layers.Dense(384, activation="relu")(x)
    tmp = keras.layers.BatchNormalization()(tmp)
    tmp = keras.layers.Dropout(0.4)(tmp)
    outputs["pitch"] = keras.layers.Dense(param_shapes["pitch"], activation="softmax", name="pitch")(tmp)

    tmp = keras.layers.Dropout(0.8)(outputs["pitch"])
    tmp = keras.layers.Dense(128, activation="relu")(keras.layers.concatenate([x, tmp]))
    tmp = keras.layers.BatchNormalization()(tmp)
    tmp = keras.layers.Dropout(0.7)(tmp)
    outputs["duration"] = keras.layers.Dense(param_shapes["duration"], activation="softmax", name="dur.")(tmp)

    model = keras.Model(list(inputs.values()), list(outputs.values()))
    model.add_loss(PitchLoss()(inputs["valid_pitches"], outputs["pitch"]))

    # compile model
    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  metrics=["accuracy"])

    return model


def train():
    loadFromExist = input("Load model from existing? (Y/N) ").lower() == "y"
    print("Continuing training session." if loadFromExist else "Creating new model.")

    # generate the training sequences
    inputs, targets = generating_training_sequences()
    testing_inputs, testing_targets = generating_training_sequences(dataset_path=TESTING_DATASET_PATH)

    # build the network
    model = build_model()
    if loadFromExist:
        model.load_weights(SAVE_MODEL_PATH)

    # train the model
    # Create a callback that saves the model's weights
    cp_callback = [keras.callbacks.ModelCheckpoint(filepath=SAVE_MODEL_PATH, verbose=0, save_weights_only=True),
                   keras.callbacks.EarlyStopping(monitor="val_loss", patience=500, verbose=0)]

    history = model.fit(list(inputs.values()), list(targets.values()), epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=[cp_callback], validation_data=(testing_inputs.values(), testing_targets.values()))

    # Save the model
    model.save_weights(SAVE_MODEL_PATH)

    # Evaluate the model
    print("--- Model evaluation --- ")
    model.evaluate(x=list(testing_inputs.values()), y=list(testing_targets.values()))

    # Plot the model
    pitch_accuracies = history.history["pitch_accuracy"]
    duration_accuracies = history.history["dur._accuracy"]

    plt.plot(range(len(history.history['loss'])), pitch_accuracies, label="Pitch accuracy")
    plt.plot(range(len(history.history['loss'])), duration_accuracies, label="Duration accuracy")

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training accuracies")

    plt.show()
    plt.savefig(PLOT_PATH)


if __name__ == "__main__":
    train()
