# from g2p import make_g2p

# transducer = make_g2p('eng', 'eng-ipa')

# def rate_apply(batch, rank=None, audio_column_name="audio", text_column_name="text"):
#     if isinstance(batch[text_column_name], list):  
#         speaking_rates = []
#         phonemes_list = []
#         if "speech_duration" in batch:
#             for text, audio_duration in zip(batch[text_column_name], batch["speech_duration"]):
#                 phonemes = transducer(text).output_string
#                 audio_duration = audio_duration if audio_duration != 0 else 0.01
#                 speaking_rate = len(phonemes) / audio_duration
#                 speaking_rates.append(speaking_rate)
#                 phonemes_list.append(phonemes)
#         else:
#             for text, audio in zip(batch[text_column_name], batch[audio_column_name]):
#                 phonemes = transducer(text).output_string
                
#                 sample_rate = audio["sampling_rate"]
#                 audio_length = len(audio["array"].squeeze()) / sample_rate
                
#                 speaking_rate = len(phonemes) / audio_length

                
#                 speaking_rates.append(speaking_rate)
#                 phonemes_list.append(phonemes)
        
#         batch["speaking_rate"] = speaking_rates
#         batch["phonemes"] = phonemes_list
#     else:
#         phonemes = transducer(batch[text_column_name]).output_string
#         if "speech_duration" in batch:
#             audio_length = batch["speech_duration"] if batch["speech_duration"] != 0 else 0.01
#         else:
#             sample_rate = batch[audio_column_name]["sampling_rate"]
#             audio_length = len(batch[audio_column_name]["array"].squeeze()) / sample_rate

#         speaking_rate = len(phonemes) / audio_length
        
#         batch["speaking_rate"] = speaking_rate
#         batch["phonemes"] = phonemes

#     return batch



from phonemizer.backend import EspeakBackend

# Module-level singleton — created once, reused for every call
_BN_BACKEND = None

def _get_backend():
    global _BN_BACKEND
    if _BN_BACKEND is None:
        _BN_BACKEND = EspeakBackend(
            language='bn',
            preserve_punctuation=True,
            with_stress=True,
            language_switch='remove-flags',
        )
    return _BN_BACKEND

def get_bn_phonemes(text):
    backend = _get_backend()
    # EspeakBackend.phonemize() expects a list, returns a list
    result = backend.phonemize([text], strip=True)
    return result[0] if result else ""


def rate_apply(batch, rank=None, audio_column_name="audio", text_column_name="text"):

    if isinstance(batch[text_column_name], list):
        speaking_rates = []
        phonemes_list = []

        texts = batch[text_column_name]

        if "speech_duration" in batch:
            durations = batch["speech_duration"]
            for text, audio_duration in zip(texts, durations):
                phonemes = get_bn_phonemes(text)
                duration = audio_duration if audio_duration > 0 else 0.01
                speaking_rates.append(len(phonemes) / duration)
                phonemes_list.append(phonemes)
        else:
            audios = batch[audio_column_name]
            for text, audio in zip(texts, audios):
                phonemes = get_bn_phonemes(text)
                sample_rate = audio["sampling_rate"]
                audio_length = len(audio["array"].squeeze()) / sample_rate
                audio_length = audio_length if audio_length > 0 else 0.01
                speaking_rates.append(len(phonemes) / audio_length)
                phonemes_list.append(phonemes)

        batch["speaking_rate"] = speaking_rates
        batch["phonemes"] = phonemes_list

    else:
        phonemes = get_bn_phonemes(batch[text_column_name])

        if "speech_duration" in batch:
            audio_length = batch["speech_duration"]
        else:
            sample_rate = batch[audio_column_name]["sampling_rate"]
            audio_length = len(batch[audio_column_name]["array"].squeeze()) / sample_rate

        audio_length = audio_length if audio_length > 0 else 0.01
        batch["speaking_rate"] = len(phonemes) / audio_length
        batch["phonemes"] = phonemes

    return batch