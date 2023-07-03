from preprocess import generating_training_sequences

from tensorflow import keras

INPUT_UNITS = 50
OUTPUT_UNITS = 30
NUM_UNITS = [256]
LOSS = "categorical_crossentropy"
LEARNING_RATE = 0.0003
EPOCHS = 90
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"


def build_model():
    # create the model architecture
    input = keras.layers.Input(shape=(None, INPUT_UNITS))
    x = keras.layers.LSTM(NUM_UNITS[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(OUTPUT_UNITS, activation="softmax")(x)
    model = keras.Model(input, output)

    # compile model
    model.compile(loss=LOSS,
                  optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  metrics=["accuracy"])

    model.summary()

    return model


def train():
    loadFromExist = input("Load model from existing? (Y/N) ").lower() == "y"
    print("Continuing training session." if loadFromExist else "Creating new model.")
    # generate the training sequences
    inputs, targets = generating_training_sequences()

    print("Input Shape:", inputs.shape)
    print("Output shape:", targets.shape)

    # build the network
    model = keras.models.load_model('./model.h5') if loadFromExist else build_model()

    # train the model
    # Create a callback that saves the model's weights
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=SAVE_MODEL_PATH, verbose=0)
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[cp_callback])

    # save the model
    model.save(SAVE_MODEL_PATH)

if __name__ == "__main__":
    train()
