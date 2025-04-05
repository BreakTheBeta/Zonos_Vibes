import requests
import os
import wave
import io

# --- Configuration ---
SERVER_URL = "http://192.168.1.128:5000/tts"

TEST_TEXT = "This is an integration test for the TTS server. End."
TEST_TEXT = """
Fed only acts on hard data and predictions on hard data. They don't respond to politics. It's by design, if they did otherwise it would jeopardize fed independence. JPow made that point very clearly in the speech and I believe to address these sorts of calls to action.
The fed isn't acting until blood is in the streets, one way or the other; and stagflation implies hike. End '
'"""

# --- Speaker/Prefix ---
TEST_SPEAKER_PATH = "short.wav" # Use an existing audio file for speaker cloning
TEST_PREFIX_PATH = "assets/silence_100ms.wav" # Optional: Audio to continue from

# --- Conditioning Parameters ---
TEST_EMOTIONS = [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2] # Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral
TEST_VQ_SCORE = 0.78
TEST_FMAX = 24000.0
TEST_PITCH_STD = 45.0
TEST_SPEAKING_RATE = 15.0
TEST_DNSMOS = 4.0
TEST_SPEAKER_NOISED = False # Denoise speaker embedding?
TEST_LANGUAGE = "en-us" # Language code

# --- Generation Parameters ---
TEST_CFG_SCALE = 2.0
TEST_SEED = 420
TEST_RANDOMIZE_SEED = True
TEST_UNCONDITIONAL_KEYS = ["emotion"] # List of keys to make unconditional

# --- Sampling Parameters ---
# NovelAI Unified Sampler (set linear=0 to disable)
TEST_LINEAR = 0.5
TEST_CONFIDENCE = 0.40
TEST_QUADRATIC = 0.00
# Legacy Sampling (used if linear=0)
TEST_TOP_P = 0.0
TEST_TOP_K = 0 # Note: Gradio uses 'Min K', assuming it maps to top_k=0 if unused
TEST_MIN_P = 0.0

# --- Output ---
OUTPUT_FILENAME = "test_output.wav" # Optional: Save the received audio

def run_test():
    """Runs the integration test against the running TTS server."""
    print(f"--- Running TTS Server Integration Test ---")
    print(f"Target URL: {SERVER_URL}")

    # Verify the speaker audio file exists before sending the request
    if not os.path.exists(TEST_SPEAKER_PATH):
        print(f"Error: Test speaker audio file not found at '{TEST_SPEAKER_PATH}'.")
        print("Please ensure the file exists in the same directory as the test script or provide the correct path.")
        return False

    payload = {
        # Core
        "text": TEST_TEXT,
        "language": TEST_LANGUAGE,
        "speaker_audio_path": TEST_SPEAKER_PATH,
        "prefix_audio_path": TEST_PREFIX_PATH,
        # Conditioning
        "emotion": TEST_EMOTIONS,
        "vq_score": TEST_VQ_SCORE,
        "fmax": TEST_FMAX,
        "pitch_std": TEST_PITCH_STD,
        "speaking_rate": TEST_SPEAKING_RATE,
        "dnsmos_ovrl": TEST_DNSMOS,
        "speaker_noised": TEST_SPEAKER_NOISED,
        # Generation
        "cfg_scale": TEST_CFG_SCALE,
        "seed": TEST_SEED,
        "randomize_seed": TEST_RANDOMIZE_SEED,
        "unconditional_keys": TEST_UNCONDITIONAL_KEYS,
        # Sampling
        "linear": TEST_LINEAR,
        "confidence": TEST_CONFIDENCE,
        "quadratic": TEST_QUADRATIC,
        "top_p": TEST_TOP_P,
        "top_k": TEST_TOP_K,
        "min_p": TEST_MIN_P,
        # Model Choice (Assuming server handles default or has a way to specify)
        # "model_choice": "Zyphra/Zonos-v0.1-hybrid" # Example if needed
    }
    # Filter out None values if prefix is optional server-side
    payload = {k: v for k, v in payload.items() if v is not None}

    print(f"Sending POST request with payload:")
    import json
    print(json.dumps(payload, indent=2)) # Pretty print the payload

    try:
        response = requests.post(SERVER_URL, json=payload, timeout=60) # Add timeout

        # 1. Check Status Code
        print(f"Received status code: {response.status_code}")
        if response.status_code != 200:
            print(f"Error: Expected status code 200, but got {response.status_code}")
            try:
                error_details = response.json()
                print(f"Server error details: {error_details}")
            except requests.exceptions.JSONDecodeError:
                print(f"Server response content: {response.text}")
            return False

        # 2. Check Content-Type Header
        content_type = response.headers.get('Content-Type')
        print(f"Received Content-Type: {content_type}")
        if content_type != 'audio/wav':
            print(f"Error: Expected Content-Type 'audio/wav', but got '{content_type}'")
            return False

        # 3. Check if content is non-empty
        if not response.content:
            print("Error: Received empty response content.")
            return False
        print(f"Received {len(response.content)} bytes of audio data.")

        # 4. (Optional but recommended) Basic WAV validation
        try:
            wav_buffer = io.BytesIO(response.content)
            with wave.open(wav_buffer, 'rb') as wf:
                print(f"Successfully opened response as WAV file.")
                print(f"  Format: {wf.getcompname()}")
                print(f"  Channels: {wf.getnchannels()}")
                print(f"  Frame Rate: {wf.getframerate()}")
                print(f"  Frames: {wf.getnframes()}")
        except wave.Error as e:
            print(f"Error: Could not validate response content as a WAV file. Error: {e}")
            return False

        # 5. (Optional) Save the output file
        try:
            with open(OUTPUT_FILENAME, 'wb') as f:
                f.write(response.content)
            print(f"Successfully saved received audio to '{OUTPUT_FILENAME}'")
        except IOError as e:
            print(f"Warning: Could not save output file '{OUTPUT_FILENAME}'. Error: {e}")


        print("--- Test Passed ---")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        print("Is the server running at {SERVER_URL}?")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during the test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if run_test():
        print("\nIntegration test completed successfully.")
    else:
        print("\nIntegration test failed.")
        exit(1) # Exit with a non-zero code to indicate failure
