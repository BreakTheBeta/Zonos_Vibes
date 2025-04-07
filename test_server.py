import os
import shutil
import requests # Keep for RequestException handling
from client import TTSClient, TTSClientError, play_audio_bytes # Import from new client module

# --- Configuration ---
SERVER_URL = "http://192.168.1.128:5000/tts" # Keep server URL config

TEST_TEXT = """
Yo watching both of them discuss if *insert entity* is fascist or not, while neither of them has an actual clear definition of the word fascism FOR LITERALLY AN HOUR+ was annoying. Therefore for the sake of saving time in the future i'll make this effort post! I'll try to discuss fascism here and give some clear insight into what modern academics understand the term fascism to mean... (edit = writing mistakes and minor elaboration)

(first of all a little preface: i'm a history student currently doing my master's thesis, while my masters is not related to fascism at all i wrote my bachelor's thesis on Russian fascism and Eurasianism. Due to that and some courses I've taken i believe i have the ability to at least provide some clarity around the question "is trump a fascist". Also English is my second language so be kind!
"""

# --- Test Specific Configuration ---
USE_SPEAKER_AUDIO = True # Override client default (False) for this test
TEST_SPEAKER_PATH = "short_2.wav" # Specify the speaker audio for this test

# --- Output ---
OUTPUTS_DIR = "outputs"
# Note: Most other parameters now use defaults defined in client.py
OUTPUT_FILENAME = os.path.join(OUTPUTS_DIR, "test_output_client.wav") # Updated filename

def ensure_output_dir():
    """Creates the output directory if it doesn't exist."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    print(f"Ensured output directory exists: '{OUTPUTS_DIR}'")

def run_test_with_client():
    """Runs the integration test using the TTSClient."""
    ensure_output_dir()
    print(f"\n--- Running Test via TTSClient ---")

    client = TTSClient(SERVER_URL)

    try:
        # Call the client's synthesize method.
        # Most parameters will use the defaults set in client.py.
        # We only need to specify parameters that differ from the defaults
        # or are required (like text).
        audio_bytes = client.synthesize(
            text=TEST_TEXT,
            use_speaker_audio=USE_SPEAKER_AUDIO, # Explicitly True for this test
            speaker_audio_path=TEST_SPEAKER_PATH  # Explicitly set for this test
            # All other parameters (language, prefix, conditioning, generation, sampling)
            # will use the defaults defined in the TTSClient.synthesize method.
        )

        print("--- Synthesis Request Successful ---")

        # Save the output file
        try:
            with open(OUTPUT_FILENAME, 'wb') as f:
                f.write(audio_bytes)
            print(f"Successfully saved client audio output to '{OUTPUT_FILENAME}'")

            # Play the saved audio file using the client's helper.
            # This will create a separate temporary file for playback and delete it,
            # leaving OUTPUT_FILENAME untouched.
            play_audio_bytes(audio_bytes)

        except IOError as e:
            print(f"Error: Could not save client output file '{OUTPUT_FILENAME}'. Error: {e}")
            # Decide if this should be a test failure

        return True # Indicate success

    except TTSClientError as e:
        print(f"TTS Client Error: {e}")
        return False # Indicate failure
    except requests.exceptions.RequestException as e:
        # Catch potential network errors not handled within the client's synthesize
        print(f"Network Error during request: {e}")
        print(f"Is the server running at {client.server_url}?")
        return False # Indicate failure
    except Exception as e:
        print(f"An unexpected error occurred during the client test: {e}")
        import traceback
        traceback.print_exc()
        return False # Indicate failure


if __name__ == "__main__":
    # Optional: Clean previous outputs
    # if os.path.exists(OUTPUTS_DIR):
    #     print(f"Removing previous output directory: {OUTPUTS_DIR}")
    #     shutil.rmtree(OUTPUTS_DIR)

    print("Starting TTS Server Integration Test via Client...")
    test_passed = run_test_with_client()

    print("\n--- Test Summary ---")
    print(f"Client Test: {'PASSED' if test_passed else 'FAILED'}")

    if test_passed:
        print("\nIntegration test completed successfully.")
        exit(0)
    else:
        print("\nIntegration test failed.")
        exit(1)
