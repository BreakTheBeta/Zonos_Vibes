import requests
import os
import wave
import io

# --- Configuration ---
SERVER_URL = "http://192.168.1.128:5000/tts"

TEST_TEXT = "This is an integration test for the TTS server. End."
TEST_TEXT = "Seriously, with this large of an explosive diarrhea dump the market has taken in the last 48 hours, there's no way to believe that big corporations are actually in charge of everything. Do we think they'd want their bottom line destroyed like this? If so then I guess they aren't that greedy after all. End."

# Use an existing audio file in the project directory for the speaker
TEST_SPEAKER_PATH = "short.wav"
TEST_SPEAKING_RATE = 14.0
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
        "text": TEST_TEXT,
        "speaker_audio_path": TEST_SPEAKER_PATH,
        "speaking_rate": TEST_SPEAKING_RATE
    }
    print(f"Sending POST request with payload: {payload}")

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
