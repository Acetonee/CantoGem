from preprocess import generating_training_sequences
from preprocess import input_params, output_params, param_shapes
from tensorflow import keras

import tensorflow as tf
import matplotlib.pyplot as plt

INPUT_UNITS = 50
OUTPUT_UNITS = 30
LEARNING_RATE = 0.001
EPOCHS = 90
BATCH_SIZE = 16
SAVE_MODEL_PATH = "model.h5"
PLOT_PATH = "./training_plot.png"


def build_model():
    input_layers = dict()
    inputs = dict()

    outputs = dict()

    # create the model architecture
    for type in input_params:
        input_layers[type] = keras.layers.Input(shape=(None, param_shapes[type]))
        tmp = keras.layers.LSTM(param_shapes[type], return_sequences=True)(input_layers[type])
        inputs[type] = keras.layers.Dropout(0.4)(tmp)

    combined_input = keras.layers.concatenate(list(inputs.values()))

    x = keras.layers.LSTM(512, return_sequences=True)(combined_input)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.LSTM(512)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(256, activation="relu")(x)

    tmp = keras.layers.Dense(128, activation="relu")(x)
    tmp = keras.layers.BatchNormalization()(tmp)
    tmp = keras.layers.Dropout(0.2)(tmp)
    outputs["pitch"] = keras.layers.Dense(param_shapes["pitch"], activation="softmax", name="pitch")(tmp)
    tmp = keras.layers.Dense(128, activation="relu")(keras.layers.concatenate([x, outputs["pitch"]]))
    tmp = keras.layers.BatchNormalization()(tmp)
    tmp = keras.layers.Dropout(0.2)(tmp)
    outputs["duration"] = keras.layers.Dense(param_shapes["duration"], activation="softmax", name="dur.")(tmp)

    model = keras.Model(list(inputs.values()), list(outputs.values()))

    # compile model
    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  metrics=["accuracy"])

    model.summary()

    return model


def train():
    loadFromExist = input("Load model from existing? (Y/N) ").lower() == "y"
    print("Continuing training session." if loadFromExist else "Creating new model.")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # generate the training sequences
    inputs, targets = generating_training_sequences()

    # build the network
    model = keras.models.load_model("./model.h5") if loadFromExist else build_model()

    # train the model
    # Create a callback that saves the model's weights
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=SAVE_MODEL_PATH, verbose=0)
    model.fit(list(inputs.values()), list(targets.values()), epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=[cp_callback])

    # save the model
    model.save(SAVE_MODEL_PATH)


if __name__ == "__main__":
    train()
