#!/usr/bin/env python

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

from TextCleaner import clean_and_split_text # Import the NEW cleaning and splitting function
from AudioCleaner import trim_tts_audio # Import the audio trimming function
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict, supported_language_codes
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
MAX_NEW_TOKENS = 86 * 30 # Default max length

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
        prefix_audio_path = data.get('prefix_audio_path') # Optional

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

        # New flag to control output format
        combine_chunks = data.get('combine_chunks', True) # Default to combining

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

        # --- Clean and Split Input Text ---
        print(f"Original text: {text}")
        text_chunks = clean_and_split_text(text) # Use the new function
        if not text_chunks:
             return jsonify({"error": "Text resulted in empty chunks after cleaning."}), 400
        print(f"Cleaned and split into {len(text_chunks)} chunks.")
        for i, chunk in enumerate(text_chunks):
            print(f"  Chunk {i+1}: {chunk}")

        # --- Prepare Inputs (Common) ---
        # Speaker Embedding
        speaker_embedding = get_speaker_embedding(speaker_audio_path)

        # Prefix Audio Codes (Optional)
        audio_prefix_codes = None
        if prefix_audio_path and os.path.exists(prefix_audio_path):
            print(f"Loading prefix audio: {prefix_audio_path}")
            wav_prefix, sr_prefix = torchaudio.load(prefix_audio_path)
            wav_prefix = wav_prefix.mean(0, keepdim=True) # Ensure mono
            wav_prefix = model.autoencoder.preprocess(wav_prefix, sr_prefix)
            wav_prefix = wav_prefix.to(device, dtype=torch.float32)
            audio_prefix_codes = model.autoencoder.encode(wav_prefix.unsqueeze(0))
            print("Prefix audio processed.")
        elif prefix_audio_path:
            print(f"Warning: Prefix audio path specified but not found: {prefix_audio_path}")

        # --- Generate Audio Chunk by Chunk ---
        all_wav_out_parts = []
        current_prefix_codes = audio_prefix_codes # Use provided prefix only for the first chunk

        # Seed Handling: Set seed once before the loop based on request params
        if randomize_seed:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            print(f"Using random seed: {seed}")
        else:
            print(f"Using fixed seed: {seed}")
        torch.manual_seed(seed) # Set the seed once

        for i, chunk_text in enumerate(text_chunks):
            print(f"\n--- Processing Chunk {i+1}/{len(text_chunks)} ---")
            print(f"Text: {chunk_text}")

            # Note: Seed is already set. If different seed per chunk is desired, modify here.

            # Tensors for conditioning (recreate for clarity, could optimize)
            emotion_tensor = torch.tensor(emotion_list, device=device).unsqueeze(0)
            vq_tensor = torch.tensor([vq_score] * 8, device=device).unsqueeze(0)

            # --- Prepare Conditioning for the current chunk ---
            cond_dict = make_cond_dict(
                text=chunk_text, # Use the current chunk's text
                language=language,
                speaker=speaker_embedding, # Re-use the same speaker embedding
                emotion=emotion_tensor,
                vqscore_8=vq_tensor,
                fmax=fmax,
                pitch_std=pitch_std,
                speaking_rate=speaking_rate,
                dnsmos_ovrl=dnsmos_ovrl,
                speaker_noised=speaker_noised,
                device=device,
                unconditional_keys=unconditional_keys,
            )
            conditioning = model.prepare_conditioning(cond_dict)

            # --- Generate Audio Codes for the current chunk ---
            print(f"Generating audio for chunk {i+1}...")
            codes = model.generate(
                prefix_conditioning=conditioning,
                # Use original prefix only for the first chunk.
                audio_prefix_codes=current_prefix_codes if i == 0 else None,
                max_new_tokens=MAX_NEW_TOKENS, # Apply max tokens per chunk
                cfg_scale=cfg_scale,
                batch_size=1, # Process one chunk at a time
                sampling_params=dict(
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    linear=linear,
                    conf=confidence,
                    quad=quadratic
                ),
            )

            # --- Decode to Waveform for the current chunk ---
            wav_out_chunk = model.autoencoder.decode(codes).cpu().detach()
            all_wav_out_parts.append(wav_out_chunk)
            print(f"Chunk {i+1} audio generated (shape: {wav_out_chunk.shape}).")

            # Optional: Prepare prefix for the *next* chunk (Advanced)
            # Optional: Prepare prefix for the *next* chunk (Advanced)
            # current_prefix_codes = model.autoencoder.encode(wav_out_chunk[:, :, -N:]) # Example

        # --- Process Output Based on combine_chunks Flag ---
        if not all_wav_out_parts:
             return jsonify({"error": "No audio parts were generated."}), 500

        sr_out = model.autoencoder.sampling_rate # Get sampling rate

        if combine_chunks:
            # --- Concatenate Audio Chunks (Default Behavior) ---
            print("\nCombining audio chunks...")
            wav_out = torch.cat(all_wav_out_parts, dim=2) # Concatenate along time dim
            print(f"Total combined audio length: {wav_out.shape[2] / sr_out:.2f} seconds (shape: {wav_out.shape})")

            # Ensure mono and correct shape
            if wav_out.shape[0] != 1 or wav_out.shape[1] != 1:
                 print(f"Warning: Unexpected concatenated shape {wav_out.shape}, attempting to reshape.")
                 wav_out = wav_out.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True) # Force to [1, 1, T]

            wav_numpy = wav_out.squeeze().numpy() # Shape (T,)

            # Convert float audio to 16-bit PCM
            wav_numpy = np.clip(wav_numpy, -1.0, 1.0)
            wav_int16 = (wav_numpy * 32767).astype(np.int16)

            # --- Trim the combined audio ---
            temp_in_file = None
            temp_out_file = None
            wav_buffer = io.BytesIO() # Initialize buffer
            try:
                # Create temporary files for input and output of trimming
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_in:
                    temp_in_file = temp_in.name
                    # Write the *untrimmed* combined audio to the temp input file
                    scipy.io.wavfile.write(temp_in_file, sr_out, wav_int16) 
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_out:
                    temp_out_file = temp_out.name

                print(f"Trimming combined audio (temp in: {temp_in_file}, temp out: {temp_out_file})...")
                # Call the trimmer
                trim_tts_audio(temp_in_file, temp_out_file) 
                print("Combined audio trimming complete.")

                # Read the *trimmed* data into the buffer for sending
                with open(temp_out_file, 'rb') as f_trimmed:
                    wav_buffer.write(f_trimmed.read())
                wav_buffer.seek(0)
                print("Trimmed audio loaded into response buffer.")

            except Exception as trim_error:
                print(f"Error trimming combined audio: {trim_error}")
                # Fallback: Send the untrimmed audio if trimming fails
                print("Trimming failed. Sending untrimmed audio as fallback.")
                wav_buffer = io.BytesIO() # Reset buffer
                scipy.io.wavfile.write(wav_buffer, sr_out, wav_int16) # Write original data
                wav_buffer.seek(0)
            finally:
                # Clean up temporary files
                if temp_in_file and os.path.exists(temp_in_file):
                    os.remove(temp_in_file)
                if temp_out_file and os.path.exists(temp_out_file):
                    os.remove(temp_out_file)
            # --- End Trimming ---

            # Send the (potentially trimmed) audio
            return send_file(
                wav_buffer, # This now contains trimmed data (or original if trimming failed)
                mimetype='audio/wav',
                as_attachment=False # Send inline
            )
        else:
            # --- Return Multiple Chunks as ZIP ---
            print("\nPackaging audio chunks into ZIP archive...")
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for i, wav_out_chunk in enumerate(all_wav_out_parts):
                    # Process each chunk individually
                    wav_numpy_chunk = wav_out_chunk.squeeze().cpu().numpy() # Shape (T,)
                    wav_numpy_chunk = np.clip(wav_numpy_chunk, -1.0, 1.0)
                    wav_int16_chunk = (wav_numpy_chunk * 32767).astype(np.int16)

                    # Write chunk to an in-memory WAV buffer
                    # --- Trim the chunk before adding to ZIP ---
                    temp_in_file = None
                    temp_out_file = None
                    try:
                        # Create temporary files for input and output of trimming
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_in:
                            temp_in_file = temp_in.name
                            scipy.io.wavfile.write(temp_in_file, sr_out, wav_int16_chunk)
                        
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_out:
                            temp_out_file = temp_out.name

                        print(f"Trimming chunk {i+1} (temp in: {temp_in_file}, temp out: {temp_out_file})...")
                        # Call the trimmer - ignore return values for now
                        trim_tts_audio(temp_in_file, temp_out_file) 
                        print(f"Trimming chunk {i+1} complete.")

                        # Read the *trimmed* data
                        with open(temp_out_file, 'rb') as f_trimmed:
                            trimmed_data = f_trimmed.read()

                        # Add the *trimmed* chunk WAV to the ZIP file
                        zip_filename = f"output_chunk_{i+1}_trimmed.wav" # Indicate trimmed
                        zipf.writestr(zip_filename, trimmed_data)
                        print(f"Added {zip_filename} to archive.")

                    except Exception as trim_error:
                        print(f"Error trimming chunk {i+1}: {trim_error}")
                        # Fallback: Add the untrimmed chunk if trimming fails? Or skip?
                        # For now, let's log and skip adding the failed chunk to avoid corrupting zip
                        print(f"Skipping chunk {i+1} due to trimming error.")
                    finally:
                        # Clean up temporary files
                        if temp_in_file and os.path.exists(temp_in_file):
                            os.remove(temp_in_file)
                        if temp_out_file and os.path.exists(temp_out_file):
                            os.remove(temp_out_file)
                    # --- End Trimming ---


            zip_buffer.seek(0)
            print("ZIP archive created.")
            return send_file(
                zip_buffer,
                mimetype='application/zip',
                as_attachment=True, # Force download
                download_name='audio_chunks.zip' # Suggest filename
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
