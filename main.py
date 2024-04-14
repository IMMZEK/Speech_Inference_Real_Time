import sounddevice as sd
import numpy as np
import tensorflow as tf
from scipy.signal import spectrogram

# Parameters
sampling_rate = 16000  # Sampling rate of the microphone input
duration = 1  # Duration in seconds of each audio snippet
num_samples = sampling_rate * duration

# Load the TensorFlow model
model = tf.saved_model.load('saved')

def audio_callback(indata, frames, time, status):
    # Convert the audio input to a numpy array
    audio_data = np.squeeze(indata)
    # Generate a spectrogram
    _, _, Sxx = spectrogram(audio_data, fs=sampling_rate, nperseg=256, noverlap=128)
    Sxx = np.log(Sxx + 1e-10)  # Convert power to dB
    Sxx = tf.convert_to_tensor(Sxx, dtype=tf.float32)
    Sxx = tf.expand_dims(Sxx, 0)
    Sxx = tf.expand_dims(Sxx, -1)  # Add channel dimension
    
    # Perform inference
    prediction = model(Sxx, training=False)
    print("Predicted Class:", prediction)

# Set up the stream to capture audio
with sd.InputStream(callback=audio_callback, channels=1, samplerate=sampling_rate, blocksize=num_samples):
    print("Starting real-time audio inference. Speak into your microphone...")
    sd.sleep(duration * 1000)  # Keep the stream open
