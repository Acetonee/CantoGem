import keras.layers
from preprocess import generating_training_sequences, SEQUENCE_LENGTH
import tensorflow as tf

# vocab size
OUTPUT_UNITS = 38
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"


def build_model(output_units, num_units, loss, learning_rate):

    input1 = tf.keras.layers.Input(shape=(None, output_units))
    input2 = tf.keras.layers.Input(shape=(None, output_units))
    combined_inputs = keras.layers.concatenate([input1, input2])

    x = keras.layers.LSTM(num_units[0])(combined_inputs)
    x = keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(output_units, activation="softmax")(x)
    model = tf.keras.Model(input, output)

    model.compile(loss=loss,
                  optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])
    model.summary()
    return model


def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):

    # generate the training sequences
    inputs_songs, inputs_lyrics, targets = generating_training_sequences(SEQUENCE_LENGTH)

    # build the network
    model = build_model(output_units, num_units, loss, learning_rate)

    # train the model
    model.fit([inputs_songs, inputs_lyrics], targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # save the model
    model.save(SAVE_MODEL_PATH)


if __name__ == "__main__":
    train()