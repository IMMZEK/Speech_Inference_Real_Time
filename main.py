import numpy as np

import tensorflow as tf

from recording_helper import record_audio, terminate
from tf_helper import preprocess_audiobuffer


def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)

    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)

    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]

    return spectrogram


commands = ["down", "go", "left", "no", "right", "stop", "up", "yes"]
loaded_model = tf.saved_model.load("saved_old", tags=None, options=None)


def predict_mic():
    audio = record_audio()
    spec = preprocess_audiobuffer(audio)
    prediction = loaded_model(spec)
    label_pred = np.argmax(prediction, axis=1)
    logits = prediction['predictions']
    probabilities = tf.nn.softmax(logits).numpy()
    print(probabilities)
    command = commands[commands[0]]
    print("Predicted label:", command)
    return command


def main():
    from turtle_helper import move_turtle
    while True:
        command = predict_mic()
        move_turtle(command)
        if command == "stop":
            terminate()
            break

if __name__ == "__main__":
    main()
