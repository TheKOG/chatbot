#pip install openai-whisper
import zhconv
import os
import whisper
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("large").to(device)
print("Model loaded")

def traditional_to_simplified(traditional_text):
    simplified_text = zhconv.convert(traditional_text, 'zh-hans')
    return simplified_text

def Trans_from_file(audio_file_path):
    audio = whisper.load_audio(audio_file_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio,n_mels=128).to(device)
    options = whisper.DecodingOptions(language="zh", without_timestamps=True)
    with torch.no_grad():
        result = whisper.decode(model, mel, options)
    simplified_text = traditional_to_simplified(result.text)
    return simplified_text

def Trans(audio):
    sampling_rate, audio_data = audio
    audio = whisper.pad_or_trim(audio_data)
    mel = whisper.log_mel_spectrogram(audio, n_mels=128, sampling_rate=sampling_rate).to(device)
    options = whisper.DecodingOptions(language="zh", without_timestamps=True)
    with torch.no_grad():
        result = whisper.decode(model, mel, options)
    simplified_text = traditional_to_simplified(result.text)
    return simplified_text

if __name__ == "__main__":
    audio_dir_path = "./seg"
    for file_name in os.listdir(audio_dir_path):
        if file_name.endswith(".wav"):
            audio_file_path = os.path.join(audio_dir_path, file_name)
            print(f"Processing file: {audio_file_path}")
            result=Trans(audio_file_path)
            print("-" * 100)
            print(f"File: {file_name}")
            print("Prediction:", result)