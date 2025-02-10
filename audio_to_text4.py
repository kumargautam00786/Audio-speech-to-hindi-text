import sys
import speech_recognition as sr
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QHBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
import os
from transformers import file_utils


############### Get the name of all the model installed
# Get the default cache directory (may be overridden by TRANSFORMERS_CACHE or HF_HOME)
cache_dir = file_utils.default_cache_path
print("Default Transformers cache directory:", cache_dir)

# List all items in the cache directory
print("\nCached files and directories:")
for item in os.listdir(cache_dir):
    print(item)



class TranscriptionThread(QThread):
    transcription_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.pipe = pipeline("automatic-speech-recognition", model="nalini2799/CDAC_hindispeechrecognition", device=0)
        self.running = False

    def run(self):
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            while self.running:
                print("Listening...")
                try:
                    audio = self.recognizer.listen(source, timeout=10)
                    print("Recognizing...")
                    text = self.pipe(audio.get_wav_data())["text"]
                    self.transcription_signal.emit(text)
                except sr.WaitTimeoutError:
                    print("Listening timed out, no speech detected.")
                except Exception as e:
                    self.transcription_signal.emit(f"Error: {e}")

    def start_transcription(self):
        self.running = True
        self.start()

    def stop_transcription(self):
        self.running = False
        self.quit()
        self.wait()

class TranscriptionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Live Transcription')
        self.setGeometry(100, 100, 900, 400)

        self.layout = QVBoxLayout()

        # Create a horizontal layout for the text displays
        self.text_layout = QHBoxLayout()

        # Text display for Model 1
        self.transcription_display1 = QTextEdit(self)
        self.transcription_display1.setReadOnly(True)
        self.text_layout.addWidget(self.transcription_display1)

        # Text display for Model 2
        self.transcription_display2 = QTextEdit(self)
        self.transcription_display2.setReadOnly(True)
        self.text_layout.addWidget(self.transcription_display2)

        # Text display for Model 3
        self.transcription_display3 = QTextEdit(self)
        self.transcription_display3.setReadOnly(True)
        self.text_layout.addWidget(self.transcription_display3)

        self.layout.addLayout(self.text_layout)

        self.record_button = QPushButton('Start Recording', self)
        self.record_button.clicked.connect(self.toggle_recording)
        self.layout.addWidget(self.record_button)

        self.setLayout(self.layout)

        self.transcription_thread = TranscriptionThread()
        self.transcription_thread.transcription_signal.connect(self.update_transcription)

        # Initialize different models for each text area with GPU support
        self.model1 = pipeline("automatic-speech-recognition", model="nalini2799/CDAC_hindispeechrecognition", device=0)

        # Load the processor and model for Oriserve/Whisper-Hindi2Hinglish-Prime
        processor = AutoProcessor.from_pretrained("Oriserve/Whisper-Hindi2Hinglish-Prime")
        model = AutoModelForSpeechSeq2Seq.from_pretrained("Oriserve/Whisper-Hindi2Hinglish-Prime")

        # Ensure that the tokenizer has a pad_token_id.
        # If not, assign the eos_token_id to pad_token_id.
        if not hasattr(processor.tokenizer, "pad_token_id") or processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

        # Initialize the pipeline using the separate tokenizer and feature_extractor from the processor.
        self.model2 = pipeline(
            "automatic-speech-recognition", 
            model=model, 
            tokenizer=processor.tokenizer, 
            feature_extractor=processor.feature_extractor, 
            device=0
        )

        self.model3 = pipeline("automatic-speech-recognition", model="nalini2799/CDAC_hindispeechrecognition", device=0)

    def toggle_recording(self):
        if self.transcription_thread.isRunning():
            self.transcription_thread.stop_transcription()
            self.record_button.setText('Start Recording')
        else:
            self.transcription_thread.start_transcription()
            self.record_button.setText('Stop Recording')

    def update_transcription(self, data):
        # If data isn't audio bytes, assume it's an error message.
        if not isinstance(data, bytes):
            message = str(data)
            self.transcription_display1.append(message)
            self.transcription_display2.append(message)
            self.transcription_display3.append(message)
            return

        # Process audio bytes with different models
        try:
            output1 = self.model1(data)["text"]
        except Exception as e:
            output1 = f"Model1 error: {e}"

        try:
            output2 = self.model2(data)["text"]
        except Exception as e:
            output2 = f"Model2 error: {e}"

        try:
            output3 = self.model3(data)["text"]
        except Exception as e:
            output3 = f"Model3 error: {e}"

        # Update text displays with the outputs
        self.transcription_display1.append(output1)
        self.transcription_display2.append(output2)
        self.transcription_display3.append(output3)

def main():
    app = QApplication(sys.argv)
    ex = TranscriptionApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()