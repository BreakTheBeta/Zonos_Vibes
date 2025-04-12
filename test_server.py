import os
import shutil
import requests # Keep for RequestException handling
from client import TTSClient, TTSClientError, play_audio_bytes # Import from new client module

# --- Configuration ---
SERVER_URL = "http://192.168.1.128:5000/tts" # Keep server URL config

TEST_TEXT = """
Everything is priced in, don't even ask the question. The answer is yes—it's priced in. Think Amazon will beat the next earnings? That's already been priced in. You work at the drive-thru for Mickey D's and found out that the burgers are made of human meat? Priced in. You think insiders don't already know that? The market is an all-powerful, all-encompassing being that knows the very inner workings of your subconscious before you were even born.

Your very existence was priced in decades ago when the market was valuing Standard Oil's expected future earnings based on population growth that would lead to your birth. It considered what age you'd get a car, how many times you'd drive your car every week, and how many times you'd take the bus or train. Anything you can think of has already been priced in—even the things you aren't thinking of.

You have no original thoughts. Your consciousness is just an illusion, a product of the omniscient market. Free will is a myth. The market sees all, knows all, and will be there from the beginning of time until the end of the universe (the market has already priced in the heat death of the universe).

So, before you make a post on WSB asking whether AAPL has priced in earpod 11 sales or whatever—know that it's already been priced in. And don't ask such a dumb fucking question again.
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
