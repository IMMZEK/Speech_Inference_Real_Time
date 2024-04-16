import pyaudio
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

class AudioProcessor:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        # Audio settings
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 2048
        self.buffer_size = 25 # We can change this
        self.audio_interface = pyaudio.PyAudio()
        self.stream = self.audio_interface.open(format=self.format, channels=self.channels, 
                                                rate=self.rate, input=True, frames_per_buffer=self.chunk)
        self.audio_buffer = []

    def process_audio(self, data):
        np_audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        np_audio /= 32768.0  # Normalize to [-1, 1]
        return self.processor(np_audio, sampling_rate=self.rate, return_tensors="pt").input_values

    def predict(self, input_values):
        input_values = input_values.float()  # Ensure input is float
        with torch.no_grad():
            logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            return self.processor.batch_decode(predicted_ids)

    def get_transcription(self):
        input_data = self.stream.read(self.chunk, exception_on_overflow=False)
        self.audio_buffer.append(input_data)
        if len(self.audio_buffer) >= self.buffer_size:
            buffer_concat = b''.join(self.audio_buffer)  # Concatenate buffer chunks
            input_values = self.process_audio(buffer_concat)
            transcription = self.predict(input_values)
            self.audio_buffer = []  # Clear the buffer after processing
            return transcription
        return None

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio_interface.terminate()
