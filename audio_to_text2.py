import speech_recognition as sr
import keyboard
import time
from transformers import pipeline, AutoProcessor, AutoModelForCTC
import tempfile
import wave
from docx import Document

recognizer = sr.Recognizer()
mic = sr.Microphone()

# Initialize the pipeline for automatic speech recognition
pipe = pipeline("automatic-speech-recognition", model="nalini2799/CDAC_hindispeechrecognition")

print("Press and HOLD 'b' to talk. Release 'b' to transcribe.")

while True:
    # 1) Wait until the user presses 'b'
    keyboard.wait('b')  # Blocks until 'b' is pressed
    print("Recording while 'b' is held down. Speak now!")
    
    # We'll accumulate multiple small chunks here
    frames = []
    
    with mic as source:
        # Optional: adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        # 2) Keep recording small chunks WHILE 'b' is still pressed
        while keyboard.is_pressed('b'):
            # You can record 0.5s, 1s, or any chunk you like
            chunk = recognizer.record(source, duration=1)
            frames.append(chunk)
            
    # 3) Once 'b' is released, we exit the loop above
    print("Stopped recording. Recognizing...")
    
    # Merge all recorded chunks into one AudioData
    if frames:
        combined_audio = sr.AudioData(
            b"".join([c.get_raw_data() for c in frames]),
            frames[0].sample_rate,
            frames[0].sample_width
        )
    else:
        # If the user tapped 'b' too quickly, no frames might exist
        print("No audio captured. Try holding 'b' longer next time.\n")
        time.sleep(0.5)
        continue
    
    # Now transcribe the merged audio
    try:
        # Convert the audio data to a WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            with wave.open(temp_wav, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(combined_audio.sample_width)
                wf.setframerate(combined_audio.sample_rate)
                wf.writeframes(combined_audio.get_raw_data())

            # Use the pipeline to transcribe the audio
            text = pipe(temp_wav.name)["text"]
            print("You said:", text)

            # Save the transcribed text to a Word document
            doc = Document()
            doc.add_paragraph(text)
            doc.save("transcription_output.docx")
            print("Transcription saved to 'transcription_output.docx'")
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
    
    print()  # Blank line for readability
    time.sleep(0.5)  # Small pause to avoid re-triggering immediately