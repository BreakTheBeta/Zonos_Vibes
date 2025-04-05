#!/usr/bin/env python

import torch
import torchaudio
from flask import Flask, request, jsonify, send_file
import io
import scipy.io.wavfile
import os

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

# --- Configuration ---
MODEL_NAME = "Zyphra/Zonos-v0.1-transformer" # Or choose another supported model
DEFAULT_LANGUAGE = "en-us"
DEFAULT_CFG_SCALE = 2.0
DEFAULT_SEED = 420 # Or use None for random seed each time

# --- Global Variables ---
MODEL = None
SPEAKER_CACHE = {} # Simple cache for speaker embeddings {path: embedding}

# --- Flask App Initialization ---
app = Flask(__name__)

def load_model():
    """Loads the Zonos TTS model."""
    global MODEL
    if MODEL is None:
        print(f"Loading Zonos model: {MODEL_NAME}...")
        MODEL = Zonos.from_pretrained(MODEL_NAME, device=device)
        MODEL.requires_grad_(False).eval()
        print("Model loaded successfully.")
    return MODEL

def get_speaker_embedding(speaker_audio_path):
    """Loads audio and generates speaker embedding, using cache."""
    if not os.path.exists(speaker_audio_path):
        raise FileNotFoundError(f"Speaker audio file not found: {speaker_audio_path}")

    if speaker_audio_path in SPEAKER_CACHE:
        print(f"Using cached speaker embedding for: {speaker_audio_path}")
        return SPEAKER_CACHE[speaker_audio_path]
    else:
        print(f"Generating speaker embedding for: {speaker_audio_path}")
        wav, sr = torchaudio.load(speaker_audio_path)
        # Ensure mono
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)

        embedding = MODEL.make_speaker_embedding(wav, sr)
        embedding = embedding.to(device, dtype=torch.bfloat16) # Match Gradio interface dtype
        SPEAKER_CACHE[speaker_audio_path] = embedding
        print(f"Cached speaker embedding for: {speaker_audio_path}")
        return embedding

@app.route('/tts', methods=['POST'])
def text_to_speech():
    """
    Endpoint to generate speech from text using a specific speaker voice and speed.
    Expects JSON payload:
    {
        "text": "Text to synthesize.",
        "speaker_audio_path": "/path/to/speaker/voice.wav",
        "speaking_rate": 15.0,
        "language": "en-us" // Optional
    }
    Returns a WAV audio file.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    text = data.get('text')
    speaker_audio_path = data.get('speaker_audio_path')
    speaking_rate = data.get('speaking_rate')
    language = data.get('language', DEFAULT_LANGUAGE)

    if not text:
        return jsonify({"error": "Missing 'text' parameter"}), 400
    if not speaker_audio_path:
        return jsonify({"error": "Missing 'speaker_audio_path' parameter"}), 400
    if speaking_rate is None:
        return jsonify({"error": "Missing 'speaking_rate' parameter"}), 400

    try:
        speaking_rate = float(speaking_rate)
    except ValueError:
        return jsonify({"error": "'speaking_rate' must be a number"}), 400

    try:
        model = load_model() # Ensure model is loaded

        # Generate or retrieve speaker embedding
        speaker_embedding = get_speaker_embedding(speaker_audio_path)

        # Set seed
        seed = DEFAULT_SEED
        if seed is not None:
            torch.manual_seed(seed)
        else:
            # Use a random seed if default is None
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            torch.manual_seed(seed)
            print(f"Using random seed: {seed}")


        # Prepare conditioning dictionary (using defaults from Gradio for simplicity)
        cond_dict = make_cond_dict(
            text=text,
            language=language,
            speaker=speaker_embedding,
            speaking_rate=speaking_rate,
            # Add other defaults if needed, e.g., emotion=[0.5]*8
            device=device,
            unconditional_keys=["emotion", "vqscore_8", "fmax", "pitch_std", "dnsmos_ovrl", "speaker_noised"] # Make others unconditional like Gradio default
        )
        conditioning = model.prepare_conditioning(cond_dict)

        # Generate audio codes
        print("Generating audio...")
        codes = model.generate(
            prefix_conditioning=conditioning,
            audio_prefix_codes=None, # No prefix audio for now
            max_new_tokens=86 * 30, # Default max length
            cfg_scale=DEFAULT_CFG_SCALE,
            batch_size=1,
            sampling_params=dict(top_p=0, top_k=0, min_p=0, linear=0.5, conf=0.4, quad=0.0), # Default sampling
        )

        # Decode to waveform
        wav_out = model.autoencoder.decode(codes).cpu().detach()
        sr_out = model.autoencoder.sampling_rate

        # Ensure mono and correct shape
        if wav_out.dim() == 2 and wav_out.size(0) > 1:
            wav_out = wav_out[0:1, :]
        wav_numpy = wav_out.squeeze().numpy()

        # Convert numpy array to WAV in memory
        wav_buffer = io.BytesIO()
        scipy.io.wavfile.write(wav_buffer, sr_out, wav_numpy)
        wav_buffer.seek(0)

        print("Audio generation complete.")
        return send_file(
            wav_buffer,
            mimetype='audio/wav',
            as_attachment=False # Send inline
            # download_name='output.wav' # Use if as_attachment=True
        )

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    load_model() # Pre-load model on startup
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False) # Use debug=True for development only
