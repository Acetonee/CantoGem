from preprocess import generating_training_sequences, num_pitch, num_duration, num_tone, num_bar_internal, num_bar_external

from tensorflow import keras

INPUT_UNITS = 50
OUTPUT_UNITS = 30
LEARNING_RATE = 0.001
EPOCHS = 90
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"


def build_model():
    input_params = ["pitch", "duration", "tone", "pos_internal", "pos_external"]
    output_params = ["pitch", "duration"]

    shapes = {
        "pitch": num_pitch,
        "duration": num_duration,
        "tone": num_tone,
        "pos_internal": num_bar_internal,
        "pos_external": num_bar_external,
    }

    input_layers = dict()
    inputs = dict()

    outputs = dict()

    # create the model architecture
    for type in input_params:
        input_layers[type] = keras.layers.Input(shape=(None, shapes[type]))
        tmp = keras.layers.LSTM(shapes[type], return_sequences=True)(input_layers[type])
        inputs[type] = keras.layers.Dropout(0.2)(tmp)

    combined_input = keras.layers.concatenate(list(inputs.values()))
    
    x = keras.layers.LSTM(512, return_sequences=True)(combined_input)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LSTM(512)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(256, activation="relu")(x)

    for type in output_params:
        tmp = keras.layers.Dense(128, activation="relu")(x)
        tmp = keras.layers.BatchNormalization()(tmp)
        tmp = keras.layers.Dropout(0.2)(tmp)
        outputs[type] = keras.layers.Dense(shapes[type], activation="softmax", name=type)(tmp)

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
    # generate the training sequences
    inputs, targets = generating_training_sequences()

    # build the network
    model = keras.models.load_model("./model.h5") if loadFromExist else build_model()

    # train the model
    # Create a callback that saves the model's weights
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=SAVE_MODEL_PATH, verbose=0)
    model.fit(list(inputs.values()), list(targets.values()), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[cp_callback])

    # save the model
    model.save(SAVE_MODEL_PATH)

if __name__ == "__main__":
    train()
