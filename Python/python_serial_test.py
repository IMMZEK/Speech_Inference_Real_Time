import pyaudio
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import serial

# Load model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 2048
BUFFER_SIZE = 25

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
audio_buffer = []

# UART settings
uart = serial.Serial("COM5", baudrate=9600, timeout=1)

# Command mapping
commands = {
    "no": 1,
    "yes": 2,
    "light on": 3
}

def process_audio(data):
    np_audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    np_audio /= 32768.0  # Normalize to [-1, 1]
    return processor(np_audio, sampling_rate=RATE, return_tensors="pt").input_values

def predict(input_values):
    input_values = input_values.float()  # Ensure input is float
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
    return transcription

def send_command(transcription):
    for phrase in commands:
        if phrase in transcription.lower():
            uart.write(str(commands[phrase]).encode())
            print(f"Sent '{phrase}' as {commands[phrase]}")
            break

while True:
    input_data = stream.read(CHUNK, exception_on_overflow=False)
    audio_buffer.append(input_data)
    if len(audio_buffer) >= BUFFER_SIZE:
        buffer_concat = b''.join(audio_buffer)
        input_values = process_audio(buffer_concat)
        transcriptions = predict(input_values)
        for transcription in transcriptions:
            send_command(transcription)
        audio_buffer = []

stream.stop_stream()
stream.close()
audio.terminate()
uart.close()