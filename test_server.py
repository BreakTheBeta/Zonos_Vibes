import requests
import os
import wave
import io
import zipfile # Added for handling ZIP responses
import json # Added for pretty printing payload
import shutil # For removing directory if needed
import subprocess # Added for playing audio

# --- Configuration ---
SERVER_URL = "http://192.168.1.128:5000/tts"

TEST_TEXT = "This is an integration test for the TTS server. End."
TEST_TEXT = """
No other items are to be placed on this table. If SCP-717 shows signs of movement, one staff member is to sit at the table and remain in the room with it. The staff member present must turn on the lamp and point at the word "WAIT" on the modified Ouija board, then immediately shut the lamp off. Staff is advised to remain quiet and breathe steadily until relieved.

SCP-717 is sealed behind a titanium alloy vault door, lined with a plating of [REDACTED] alloy. It is mounted on the wall behind SCP-717. The vault is to remain sealed per mutual agreement. If any whistling is heard from the vault, maintenance must be performed immediately to prevent a breach.

If the SCP-717 vault is breached from the far side, it is to be considered a hostile act. Communication with SCP-717 is to immediately cease and staff are to equip weapons to repel invaders. END.
'"""
TEST_TEXT = """
I don't believe so, AD-394 could certainly be contributing to what we see in AD-644. I think AD-644 is more due to a single ad request having to wait for multiple (and larger) queries to complete before returning a response. Can the queries be run in parallel? They are hitting the same table though. The only workaround at the moment is to ask the retailer to split requests into multiple requests, one for each slot. Scaling up the Ads Replicas makes no real improvement.
"""

# --- Speaker/Prefix ---
TEST_SPEAKER_PATH = "short.wav" # Use an existing audio file for speaker cloning
TEST_PREFIX_PATH = "assets/silence_100ms.wav" # Optional: Audio to continue from

# --- Conditioning Parameters ---
# Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral
TEST_EMOTIONS = [0.75, 0.35, 0.05, 0.05, 0.55, 0.95, 0.1, 0.2] 
TEST_VQ_SCORE = 0.78
TEST_FMAX = 20000.0
TEST_PITCH_STD = 45.0
TEST_SPEAKING_RATE = 15.0
TEST_DNSMOS = 4.0
TEST_SPEAKER_NOISED = True # Denoise speaker embedding?
TEST_LANGUAGE = "en-us" # Language code

# --- Generation Parameters ---
TEST_CFG_SCALE = 2.0
TEST_SEED = 420
TEST_RANDOMIZE_SEED = False
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
OUTPUTS_DIR = "outputs" # Directory to store all test outputs
OUTPUT_FILENAME = os.path.join(OUTPUTS_DIR, "test_output_combined.wav") # Combined WAV path
OUTPUT_ZIP_FILENAME = os.path.join(OUTPUTS_DIR, "test_output_chunks.zip") # Separate chunks ZIP path

def ensure_output_dir():
    """Creates the output directory if it doesn't exist."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    print(f"Ensured output directory exists: '{OUTPUTS_DIR}'")

def run_test_combined():
    """Runs the integration test requesting a single combined WAV file."""
    ensure_output_dir() # Make sure output dir exists
    print(f"\n--- Running Test: Combined Output (Default) ---")
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
        #"emotion": TEST_EMOTIONS,
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
        #"unconditional_keys": TEST_UNCONDITIONAL_KEYS,
        # Sampling
        "linear": TEST_LINEAR,
        "confidence": TEST_CONFIDENCE,
        "quadratic": TEST_QUADRATIC,
        "top_p": TEST_TOP_P,
        "top_k": TEST_TOP_K,
        "min_p": TEST_MIN_P,
        # Model Choice (Assuming server handles default or has a way to specify)
        # "model_choice": "Zyphra/Zonos-v0.1-hybrid" # Example if needed
        # "combine_chunks": True # Explicitly request combined output
    }
    # Filter out None values if prefix is optional server-side
    payload = {k: v for k, v in payload.items() if v is not None}

    print(f"Sending POST request with payload:")
    print(json.dumps(payload, indent=2)) # Pretty print the payload

    try:
        response = requests.post(SERVER_URL, json=payload, timeout=180) # Increased timeout to 180 seconds

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

        # 5. Save the output file to the outputs directory
        try:
            with open(OUTPUT_FILENAME, 'wb') as f:
                f.write(response.content)
            print(f"Successfully saved combined audio to '{OUTPUT_FILENAME}'")

            # 6. Play the saved audio file
            try:
                print(f"Attempting to play audio file: '{OUTPUT_FILENAME}'")
                # Use check=True to raise an exception if afplay fails
                subprocess.run(["afplay", OUTPUT_FILENAME], check=True, capture_output=True)
                print(f"Successfully played audio file.")
            except FileNotFoundError:
                print(f"Error: 'afplay' command not found. Is it installed and in your PATH?")
                # Decide if this should be a test failure
            except subprocess.CalledProcessError as e:
                print(f"Error playing audio file with afplay: {e}")
                print(f"afplay stderr: {e.stderr.decode()}")
                # Decide if this should be a test failure
            except Exception as e:
                print(f"An unexpected error occurred during audio playback: {e}")
                # Decide if this should be a test failure

        except IOError as e:
            print(f"Error: Could not save combined output file '{OUTPUT_FILENAME}'. Error: {e}")
            # Consider this a failure? For now, just warn.


        print("--- Combined Output Test Passed ---")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        print("Is the server running at {SERVER_URL}?")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during the combined output test: {e}")
        import traceback
        traceback.print_exc()
        return False

# Removed run_test_separate_chunks() function

if __name__ == "__main__":
    # Optional: Clean previous outputs
    # if os.path.exists(OUTPUTS_DIR):
    #     print(f"Removing previous output directory: {OUTPUTS_DIR}")
    #     shutil.rmtree(OUTPUTS_DIR)

    print("Starting TTS Server Integration Test (Combined Output Only)...")
    combined_test_passed = run_test_combined()

    print("\n--- Test Summary ---")
    print(f"Combined Output Test: {'PASSED' if combined_test_passed else 'FAILED'}")

    if combined_test_passed:
        print("\nIntegration test completed successfully.")
        exit(0)
    else:
        print("\nIntegration test failed.")
        exit(1) # Exit with a non-zero code to indicate failure
