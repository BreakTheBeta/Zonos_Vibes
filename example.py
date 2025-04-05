def generator():
    # Can stream from your LLM or other source here, just partition the text into
    # sentences with nltk or rule based tokenizer. See example here:
    # https://stackoverflow.com/a/31505798
    for text, emotion, pitch_std in zip(texts, emotions, pitch_stds):
        yield {
            "text": text,
            "speaker": speaker,
            "emotion": emotion,
            "pitch_std": pitch_std,
            "language": "en-us",
        }

# Define chunk schedule: start with small chunks for faster initial output,
# then gradually increase to larger chunks for fewer cuts
stream_generator = model.stream(
    cond_dicts_generator=generator(),
    chunk_schedule=[17, *range(9, 100)],  # optimal schedule for RTX3090
    chunk_overlap=2,  # tokens to overlap between chunks (affects crossfade)
)

# Accumulate audio chunks as they are generated.
audio_chunks = []
for i, audio_chunk in enumerate(stream_generator):
    audio_chunks.append(audio_chunk)

audio = torch.cat(audio_chunks, dim=-1).cpu()
out_sr = model.autoencoder.sampling_rate
torchaudio.save("streaming.wav", audio, out_sr)
display(Audio(data=audio, rate=out_sr))