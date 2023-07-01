import keras.layers
from preprocess import generating_training_sequences, SEQUENCE_LENGTH, TONE_MAPPING_SIZE, \
    SINGLE_SONGS_FILE_DATASET, SINGLE_LYRICS_FILE_DATASET
import tensorflow as tf

# vocab size
OUTPUT_UNITS = 25 + TONE_MAPPING_SIZE  # Change to actual mapping size in mapping.json
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"
TEST_INPUT_PATH = "testing_dataset/input_testing_dataset"
TEST_OUTPUT_PATH = "testing_dataset/output_testing_dataset"


def build_model(output_units, num_units, loss, learning_rate):
    input = tf.keras.layers.Input(shape=(BATCH_SIZE, OUTPUT_UNITS))  # Dk why None not working
    x = keras.layers.LSTM(num_units[0])(input)
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
    inputs_songs, targets = generating_training_sequences(SEQUENCE_LENGTH, SINGLE_SONGS_FILE_DATASET,
                                                          SINGLE_LYRICS_FILE_DATASET)

    # build the network
    model = build_model(output_units, num_units, loss, learning_rate)

    # train the model
    model.fit(inputs_songs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # save the model
    model.save(SAVE_MODEL_PATH)


def evaluate(testing_input_path=TEST_INPUT_PATH, testing_output_path=TEST_OUTPUT_PATH, model_path=SAVE_MODEL_PATH):
    model = tf.keras.models.load_model(model_path)
    inputs_songs, targets = generating_training_sequences(SEQUENCE_LENGTH, testing_output_path, testing_input_path)
    loss, acc = model.evaluate(inputs_songs, targets)
    print("loss:")
    print(loss)
    print("acc:")
    print(acc)


if __name__ == "__main__":
    # train()
    evaluate()
