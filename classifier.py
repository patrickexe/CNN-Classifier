from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np


def build_model(input_shape):
    model = keras.Sequential()

    model.add(keras.layers.InputLayer(input_shape=input_shape))
    model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

def load_data():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Convert values to float and normalise
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Reshape as CNN needs channel
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # One-hot encode with 10 classes (1 for each digit)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

def main():
    # Fetch and process MNIST dataset
    (x_train, y_train), (x_test, y_test) = load_data()

    # Create a CNN network
    model = build_model(x_train.shape[1:])

    # Train / fit the CNN network
    batch_size = 128
    epochs = 15
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    # Evaluate the trained network
    score = model.evaluate(x_test, y_test, verbose=0)
    print("\nTest loss:", score[0])
    print("Test accuracy:", score[1])


if __name__ == "__main__":
    main()