import argparse
import librosa
from phonemizer import phonemize


def bangla_phonemize(text):
    phonemes = phonemize(
        text,
        backend="espeak",
        language="bn",
        strip=True,
        preserve_punctuation=False,
        with_stress=False
    )
    return phonemes


def get_audio_duration(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    duration = len(audio) / sr
    return duration


def main(audio_path, text):

    print("\n====== INPUT ======")
    print("Text:", text)

    words = text.split()
    print("\nWords:")
    for i, w in enumerate(words):
        print(f"{i+1}. {w}")

    print("\nLoading audio...")
    duration = get_audio_duration(audio_path)
    print(f"Audio duration: {duration:.3f} seconds")

    print("\nGenerating phonemes...")
    phonemes = bangla_phonemize(text)

    phoneme_string = phonemes.replace(" ", "")
    speaking_rate = len(phoneme_string) / duration if duration > 0 else 0

    print("\n====== PHONEMES ======")
    print(phonemes)

    print("\n====== STATS ======")
    print("Total phonemes:", len(phoneme_string))
    print("Speaking rate (phonemes/sec):", round(speaking_rate, 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--text", required=True, help="Bangla transcription")

    args = parser.parse_args()

    main(args.audio, args.text)