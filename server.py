#!/usr/bin/env python

import nltk.downloader
import torch
import torchaudio
from flask import Flask, request, jsonify, send_file
import io
import scipy.io.wavfile
import os
import numpy as np
import json # Added for pretty printing request
import zipfile # Added for returning multiple files
import tempfile # Added for temporary files during trimming
import os # Re-importing for clarity, used with tempfile
import nltk # Needed for sentence tokenization in the generator

# Ensure NLTK data is available (might need a one-time download)
try:
    nltk.data.find('tokenizers/punkt')
except Exception:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab')
    print("'punkt' downloaded.")


from TextCleaner import clean_text # Import the updated cleaning function
from AudioCleaner import trim_tts_audio # Import the audio trimming function
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict, supported_language_codes # Keep make_cond_dict for structure reference if needed
from zonos.utils import DEFAULT_DEVICE as device

# --- Configuration ---
MODEL_NAME = "Zyphra/Zonos-v0.1-transformer" # Or choose another supported model

# --- Defaults (aligned with test_server.py / gradio_interface.py) ---
DEFAULT_LANGUAGE = "en-us"
DEFAULT_EMOTIONS = [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2]
DEFAULT_VQ_SCORE = 0.78
DEFAULT_FMAX = 24000.0
DEFAULT_PITCH_STD = 45.0
DEFAULT_SPEAKING_RATE = 15.0
DEFAULT_DNSMOS = 4.0
DEFAULT_SPEAKER_NOISED = False
DEFAULT_CFG_SCALE = 2.0
DEFAULT_SEED = 420
DEFAULT_RANDOMIZE_SEED = True
DEFAULT_UNCONDITIONAL_KEYS = ["emotion"]
DEFAULT_LINEAR = 0.5
DEFAULT_CONFIDENCE = 0.40
DEFAULT_QUADRATIC = 0.00
DEFAULT_TOP_P = 0.0
DEFAULT_TOP_K = 0
DEFAULT_MIN_P = 0.0
# MAX_NEW_TOKENS removed, stream handles length

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
        model = load_model() # Ensure model is loaded for make_speaker_embedding
        wav, sr = torchaudio.load(speaker_audio_path)
        # Ensure mono
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)

        embedding = model.make_speaker_embedding(wav, sr)
        embedding = embedding.to(device, dtype=torch.bfloat16) # Match Gradio interface dtype
        SPEAKER_CACHE[speaker_audio_path] = embedding
        print(f"Cached speaker embedding for: {speaker_audio_path}")
        return embedding

@app.route('/tts', methods=['POST'])
def text_to_speech():
    """
    Endpoint to generate speech from text with advanced conditioning and sampling.
    Expects JSON payload with parameters matching test_server.py.
    Returns a WAV audio file.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    print("Received request payload:")
    print(json.dumps(data, indent=2)) # Log received data

    # --- Extract Parameters ---
    try:
        # Core
        text = data.get('text')
        language = data.get('language', DEFAULT_LANGUAGE)
        speaker_audio_path = data.get('speaker_audio_path')
        # prefix_audio_path removed

        # Conditioning
        emotion_list = data.get('emotion', DEFAULT_EMOTIONS)
        vq_score = float(data.get('vq_score', DEFAULT_VQ_SCORE))
        fmax = float(data.get('fmax', DEFAULT_FMAX))
        pitch_std = float(data.get('pitch_std', DEFAULT_PITCH_STD))
        speaking_rate = float(data.get('speaking_rate', DEFAULT_SPEAKING_RATE))
        dnsmos_ovrl = float(data.get('dnsmos_ovrl', DEFAULT_DNSMOS))
        speaker_noised = bool(data.get('speaker_noised', DEFAULT_SPEAKER_NOISED))

        # Generation
        cfg_scale = float(data.get('cfg_scale', DEFAULT_CFG_SCALE))
        seed = int(data.get('seed', DEFAULT_SEED))
        randomize_seed = bool(data.get('randomize_seed', DEFAULT_RANDOMIZE_SEED))
        unconditional_keys = data.get('unconditional_keys', DEFAULT_UNCONDITIONAL_KEYS)

        # Sampling
        linear = float(data.get('linear', DEFAULT_LINEAR))
        confidence = float(data.get('confidence', DEFAULT_CONFIDENCE))
        quadratic = float(data.get('quadratic', DEFAULT_QUADRATIC))
        top_p = float(data.get('top_p', DEFAULT_TOP_P))
        top_k = int(data.get('top_k', DEFAULT_TOP_K))
        min_p = float(data.get('min_p', DEFAULT_MIN_P))

        # combine_chunks removed

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid parameter type: {e}"}), 400

    # --- Validate Required Parameters ---
    if not text:
        return jsonify({"error": "Missing 'text' parameter"}), 400
    if not speaker_audio_path:
        return jsonify({"error": "Missing 'speaker_audio_path' parameter"}), 400
    if language not in supported_language_codes:
         return jsonify({"error": f"Unsupported language code: {language}. Supported: {supported_language_codes}"}), 400
    if not isinstance(emotion_list, list) or len(emotion_list) != 8:
        return jsonify({"error": "'emotion' must be a list of 8 numbers"}), 400
    if not isinstance(unconditional_keys, list):
        return jsonify({"error": "'unconditional_keys' must be a list of strings"}), 400


    try:
        model = load_model() # Ensure model is loaded

        # --- Clean Input Text ---
        print(f"Original text length: {len(text)}")
        cleaned_text = clean_text(text) # Use the new function
        if not cleaned_text:
             return jsonify({"error": "Text resulted in empty string after cleaning."}), 400
        print(f"Cleaned text length: {len(cleaned_text)}")
        # print(f"Cleaned text: {cleaned_text}") # Optional: print cleaned text

        # --- Prepare Inputs (Common) ---
        # Speaker Embedding
        speaker_embedding = get_speaker_embedding(speaker_audio_path)
        # Prefix audio logic removed

        # --- Seed Handling ---
        if randomize_seed:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            print(f"Using random seed: {seed}")
        else:
            print(f"Using fixed seed: {seed}")
        torch.manual_seed(seed) # Set the seed

        # --- Define Generator for model.stream ---
        def create_cond_generator():
            # Split cleaned text into sentences for streaming
            # This mimics the example.py approach for partitioning long text
            sentences = nltk.sent_tokenize(cleaned_text)
            print(f"Split cleaned text into {len(sentences)} sentences for streaming.")

            # Yield conditioning dictionary for each sentence
            for sentence_text in sentences:
                if not sentence_text.strip(): # Skip empty sentences
                    continue
                print(f"  Yielding sentence: {sentence_text[:80]}...") # Log start of sentence
                # Note: Reusing most parameters for each sentence.
                # If per-sentence variation (e.g., emotion) is needed, modify here.
                yield {
                    "text": sentence_text.strip(),
                    "speaker": speaker_embedding, # Use the loaded embedding
                    "language": language,
                    "emotion": emotion_list, # Pass the list directly
                    "pitch_std": pitch_std,
                    #"vq_score": vq_score, # Pass scalar values
                    "fmax": fmax,
                    "speaking_rate": speaking_rate,
                    "dnsmos_ovrl": dnsmos_ovrl,
                    "speaker_noised": speaker_noised,
                    # Add other relevant conditioning params if needed by model.stream's internal make_cond_dict
                }

        # --- Generate Audio using Streaming ---
        print("\nStarting audio generation stream...")
        stream_generator = model.stream(
            cond_dicts_generator=create_cond_generator(),
            # Define chunk schedule: start small, increase size (adjust based on hardware/latency needs)
            # Using the schedule from example.py as a starting point
            chunk_schedule=[17, *range(9, 100)],
            chunk_overlap=2, # From example.py
            cfg_scale=cfg_scale, # Pass generation params
            #unconditional_keys=unconditional_keys, # Pass uncond keys
            # Pass sampling params directly
            sampling_params=dict(
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                linear=linear,
                conf=confidence,
                quad=quadratic
            ),
            # Add other necessary params for model.stream if any
        )

        # --- Accumulate Audio Chunks ---
        audio_chunks = []
        print("Accumulating audio chunks from stream...")
        for i, audio_chunk in enumerate(stream_generator):
            print(f"  Received audio chunk {i+1} (shape: {audio_chunk.shape})")
            audio_chunks.append(audio_chunk.cpu()) # Move to CPU as received
        print(f"Finished accumulating {len(audio_chunks)} audio chunks.")

        if not audio_chunks:
            return jsonify({"error": "Streaming generated no audio chunks."}), 500

        # --- Concatenate Final Audio ---
        final_audio_tensor = torch.cat(audio_chunks, dim=-1)
        sr_out = model.autoencoder.sampling_rate
        print(f"Concatenated audio shape: {final_audio_tensor.shape}, Sample Rate: {sr_out}")
        print(f"Total combined audio length: {final_audio_tensor.shape[-1] / sr_out:.2f} seconds")

        # --- Trim Final Combined Audio ---
        # Convert tensor to numpy int16 for trimming function
        wav_numpy_full = final_audio_tensor.squeeze().cpu().numpy()
        wav_numpy_full = np.clip(wav_numpy_full, -1.0, 1.0)
        wav_int16_full = (wav_numpy_full * 32767).astype(np.int16)

        temp_in_file = None
        temp_out_file = None
        trimmed_wav_data = None
        try:
            # Write the full combined audio to a temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_in:
                temp_in_file = temp_in.name
                scipy.io.wavfile.write(temp_in_file, sr_out, wav_int16_full)

            # Create a temp file for the output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_out:
                temp_out_file = temp_out.name

            print(f"Trimming final combined audio...")
            trim_tts_audio(temp_in_file, temp_out_file) # Call the trimmer
            print(f"Trimming complete.")

            # Read the trimmed data back
            sr_trimmed, trimmed_wav_data = scipy.io.wavfile.read(temp_out_file)
            if sr_trimmed != sr_out:
                print(f"Warning: Sample rate mismatch after trimming. Expected {sr_out}, got {sr_trimmed}.")
                # Fallback to untrimmed if sample rate changed unexpectedly
                trimmed_wav_data = wav_int16_full
            else:
                print(f"Final trimmed audio data loaded (shape: {trimmed_wav_data.shape}).")
                print(f"Final trimmed audio length: {len(trimmed_wav_data) / sr_out:.2f} seconds")


        except Exception as trim_error:
            print(f"Error trimming final audio: {trim_error}. Returning untrimmed audio.")
            trimmed_wav_data = wav_int16_full # Fallback to untrimmed data
        finally:
            # Clean up temporary files
            if temp_in_file and os.path.exists(temp_in_file):
                os.remove(temp_in_file)
            if temp_out_file and os.path.exists(temp_out_file):
                os.remove(temp_out_file)

        if trimmed_wav_data is None:
             return jsonify({"error": "Failed to process audio after trimming."}), 500

        # --- Return Final Trimmed Audio ---
        # Convert final (potentially trimmed) numpy array to WAV in memory
        wav_buffer = io.BytesIO()
        scipy.io.wavfile.write(wav_buffer, sr_out, trimmed_wav_data.astype(np.int16)) # Use the final data
        wav_buffer.seek(0)

        print("Audio generation and processing complete.")
        return send_file(
            wav_buffer,
            mimetype='audio/wav',
            as_attachment=False # Send inline
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
    print("Starting Flask server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False) # Use debug=True for development only
