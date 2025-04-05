import requests
import os
import wave
import io
import zipfile # Added for handling ZIP responses
import json # Added for pretty printing payload
import shutil # For removing directory if needed

# --- Configuration ---
SERVER_URL = "http://192.168.1.128:5000/tts"

TEST_TEXT = "This is an integration test for the TTS server. End."
TEST_TEXT = """
Let me tell you the story when the level 600 school gyatt walked passed me, I was in class drinking my grimace rizz shake from ohio during my rizzonomics class when all of the sudden this crazy ohio bing chilling gyatt got sturdy, past my class. I was watching kai cenat hit the griddy on twitch. This is when I let my rizz take over and I became the rizzard of oz. I screamed, look at this bomboclat gyatt
'"""

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
        # "combine_chunks": True # Explicitly setting default for clarity (optional)
    }
    # Filter out None values if prefix is optional server-side
    payload = {k: v for k, v in payload.items() if v is not None}

    print(f"Sending POST request with payload:")
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

        # 5. Save the output file to the outputs directory
        try:
            with open(OUTPUT_FILENAME, 'wb') as f:
                f.write(response.content)
            print(f"Successfully saved combined audio to '{OUTPUT_FILENAME}'")
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

def run_test_separate_chunks():
    """Runs the integration test requesting separate WAV files in a ZIP."""
    ensure_output_dir() # Make sure output dir exists
    print(f"\n--- Running Test: Separate Chunks Output (ZIP) ---")
    print(f"Target URL: {SERVER_URL}")

    if not os.path.exists(TEST_SPEAKER_PATH):
        print(f"Error: Test speaker audio file not found at '{TEST_SPEAKER_PATH}'.")
        return False

    payload = {
        # Core
        "text": TEST_TEXT, # Use the same long text to ensure chunking happens
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
        "randomize_seed": TEST_RANDOMIZE_SEED, # Keep consistent for comparison if needed
        "unconditional_keys": TEST_UNCONDITIONAL_KEYS,
        # Sampling
        "linear": TEST_LINEAR,
        "confidence": TEST_CONFIDENCE,
        "quadratic": TEST_QUADRATIC,
        "top_p": TEST_TOP_P,
        "top_k": TEST_TOP_K,
        "min_p": TEST_MIN_P,
        # --- Key Change: Request separate chunks ---
        "combine_chunks": False
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    print(f"Sending POST request with payload (combine_chunks=False):")
    print(json.dumps(payload, indent=2))

    try:
        response = requests.post(SERVER_URL, json=payload, timeout=120) # Increase timeout for potentially longer processing

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

        # 2. Check Content-Type Header for ZIP
        content_type = response.headers.get('Content-Type')
        print(f"Received Content-Type: {content_type}")
        if content_type != 'application/zip':
            print(f"Error: Expected Content-Type 'application/zip', but got '{content_type}'")
            return False

        # 3. Check if content is non-empty
        if not response.content:
            print("Error: Received empty response content.")
            return False
        print(f"Received {len(response.content)} bytes of ZIP data.")

        # 4. Validate ZIP content
        try:
            zip_buffer = io.BytesIO(response.content)
            with zipfile.ZipFile(zip_buffer, 'r') as zipf:
                print(f"Successfully opened response as ZIP file.")
                file_list = zipf.namelist()
                if not file_list:
                    print("Error: ZIP file is empty.")
                    return False
                print(f"  Files in ZIP: {file_list}")

                # Validate each file within the ZIP
                for filename in file_list:
                    if not filename.lower().endswith('.wav'):
                        print(f"Error: File '{filename}' in ZIP is not a WAV file.")
                        return False

                    try:
                        wav_data = zipf.read(filename)
                        if not wav_data:
                             print(f"Error: WAV file '{filename}' in ZIP is empty.")
                             return False

                        chunk_wav_buffer = io.BytesIO(wav_data)
                        with wave.open(chunk_wav_buffer, 'rb') as wf_chunk:
                            print(f"    Validated '{filename}': Channels={wf_chunk.getnchannels()}, Rate={wf_chunk.getframerate()}, Frames={wf_chunk.getnframes()}")
                    except wave.Error as e_wav:
                        print(f"Error: Could not validate WAV file '{filename}' within ZIP. Error: {e_wav}")
                        return False
                    except Exception as e_read:
                         print(f"Error reading or processing file '{filename}' from ZIP. Error: {e_read}")
                         return False

        except zipfile.BadZipFile as e_zip:
            print(f"Error: Could not open response content as a ZIP file. Error: {e_zip}")
            return False
        except Exception as e_zip_other:
            print(f"Error processing ZIP file content. Error: {e_zip_other}")
            return False


        # 5. Save the output ZIP file and extract its contents
        try:
            # Save the ZIP itself
            with open(OUTPUT_ZIP_FILENAME, 'wb') as f:
                f.write(response.content)
            print(f"Successfully saved received ZIP archive to '{OUTPUT_ZIP_FILENAME}'")

            # Extract the contents of the saved ZIP into the outputs directory
            print(f"Extracting contents of '{OUTPUT_ZIP_FILENAME}' to '{OUTPUTS_DIR}'...")
            with zipfile.ZipFile(OUTPUT_ZIP_FILENAME, 'r') as zip_ref:
                zip_ref.extractall(OUTPUTS_DIR)
            print(f"Successfully extracted ZIP contents.")

        except IOError as e:
            print(f"Error: Could not save output ZIP file '{OUTPUT_ZIP_FILENAME}'. Error: {e}")
            return False # Treat failure to save/extract as test failure
        except zipfile.BadZipFile as e_zip:
             print(f"Error: Failed to extract '{OUTPUT_ZIP_FILENAME}' (invalid ZIP). Error: {e_zip}")
             return False
        except Exception as e_extract:
             print(f"Error during ZIP extraction. Error: {e_extract}")
             return False


        print("--- Separate Chunks Test Passed ---")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        print("Is the server running at {SERVER_URL}?")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during the separate chunks test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Optional: Clean previous outputs
    # if os.path.exists(OUTPUTS_DIR):
    #     print(f"Removing previous output directory: {OUTPUTS_DIR}")
    #     shutil.rmtree(OUTPUTS_DIR)

    print("Starting TTS Server Integration Tests...")
    combined_test_passed = run_test_combined()
    separate_chunks_test_passed = run_test_separate_chunks()

    print("\n--- Test Summary ---")
    print(f"Combined Output Test: {'PASSED' if combined_test_passed else 'FAILED'}")
    print(f"Separate Chunks Test: {'PASSED' if separate_chunks_test_passed else 'FAILED'}")

    if combined_test_passed and separate_chunks_test_passed:
        print("\nAll integration tests completed successfully.")
        exit(0)
    else:
        print("\nOne or more integration tests failed.")
        exit(1) # Exit with a non-zero code to indicate failure
