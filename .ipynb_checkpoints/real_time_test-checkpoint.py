import pyaudio
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Load model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Set up audio input
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 4096  # Increase buffer size
audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True, frames_per_buffer=CHUNK)

def process_audio(data):
    np_audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    print("Max amplitude:", np.max(np.abs(np_audio)))  # Debug: check max amplitude in buffer
    np_audio /= 32768.0
    return processor(np_audio, sampling_rate=RATE, return_tensors="pt").input_values

def predict(input_values):
    input_values = input_values.float()  # Ensure input is float
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(predicted_ids)

while True:
    input_data = stream.read(CHUNK, exception_on_overflow=False)
    input_values = process_audio(input_data)
    transcription = predict(input_values)
    print(transcription)

stream.stop_stream()
stream.close()
audio.terminate()
