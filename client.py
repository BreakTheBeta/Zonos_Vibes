import requests
import os
import wave
import io
import json
import subprocess

class TTSClientError(Exception):
    """Custom exception for TTS client errors."""
    pass

class TTSClient:
    """Client for interacting with the Zonos TTS server."""

    def __init__(self, server_url: str):
        """
        Initializes the TTS client.

        Args:
            server_url: The base URL of the TTS server (e.g., "http://localhost:5000/tts").
        """
        if not server_url.endswith('/tts'):
             # Ensure the URL points to the correct endpoint if not already specified
             self.server_url = server_url.rstrip('/') + '/tts'
             print(f"Warning: Provided server_url did not end with /tts. Appended it: {self.server_url}")
        else:
            self.server_url = server_url
        print(f"TTS Client initialized for server: {self.server_url}")

    def _validate_speaker_audio(self, use_speaker_audio: bool, speaker_audio_path: str | None) -> str:
        """Validates speaker audio path based on configuration."""
        default_silence = "assets/silence_100ms.wav" # Define default silence path

        if use_speaker_audio:
            if not speaker_audio_path:
                print(f"Warning: use_speaker_audio is True, but speaker_audio_path is empty. Using default silence: '{default_silence}'")
                return default_silence
            if not os.path.exists(speaker_audio_path):
                print(f"Warning: use_speaker_audio is True, but speaker audio file not found at '{speaker_audio_path}'. Using default silence: '{default_silence}'")
                return default_silence
            print(f"Using speaker audio: '{speaker_audio_path}'")
            return speaker_audio_path
        else:
            print(f"use_speaker_audio is False. Using default silence for speaker: '{default_silence}'")
            return default_silence

    def synthesize(
        self,
        text: str,
        language: str = "en-us",
        use_speaker_audio: bool = False, # Default to not using custom speaker
        speaker_audio_path: str | None = None, # Default to None, logic handles silence
        prefix_audio_path: str | None = "assets/silence_100ms.wav", # Default prefix
        # Conditioning
        emotion: list[float] | None = None, # No default emotion
        vq_score: float | None = 0.78,
        fmax: float | None = 20000.0,
        pitch_std: float | None = 45.0,
        speaking_rate: float | None = 15.0,
        dnsmos_ovrl: float | None = 4.0,
        speaker_noised: bool | None = True,
        # Generation
        cfg_scale: float | None = 2.0,
        seed: int | None = 420,
        randomize_seed: bool | None = False,
        unconditional_keys: list[str] | None = None, # No default unconditional keys
        # Sampling
        linear: float | None = 0.5,
        confidence: float | None = 0.40,
        quadratic: float | None = 0.00,
        top_p: float | None = 0.0,
        top_k: int | None = 0,
        min_p: float | None = 0.0,
        timeout: int = 580
    ) -> bytes:
        """
        Sends a request to the TTS server to synthesize audio.

        Args:
            text: The text to synthesize.
            language: Language code (e.g., "en-us").
            use_speaker_audio: Whether to use the provided speaker audio file.
            speaker_audio_path: Path to the speaker audio file (WAV). Used if use_speaker_audio is True. Defaults to silence if invalid or use_speaker_audio is False.
            prefix_audio_path: Path to an optional prefix audio file (WAV).
            emotion: List of emotion values.
            vq_score: VQ score conditioning.
            fmax: Fmax conditioning.
            pitch_std: Pitch standard deviation conditioning.
            speaking_rate: Speaking rate conditioning.
            dnsmos_ovrl: DNSMOS overall score conditioning.
            speaker_noised: Whether speaker embedding was derived from noisy audio.
            cfg_scale: Classifier-Free Guidance scale.
            seed: Random seed for generation.
            randomize_seed: Whether to randomize the seed.
            unconditional_keys: List of conditioning keys to make unconditional.
            linear: Linear sampling parameter.
            confidence: Confidence sampling parameter.
            quadratic: Quadratic sampling parameter.
            top_p: Top-P sampling parameter.
            top_k: Top-K sampling parameter.
            min_p: Min-P sampling parameter.
            timeout: Request timeout in seconds.

        Returns:
            The raw WAV audio data as bytes.

        Raises:
            TTSClientError: If the server returns an error or the response is invalid.
            requests.exceptions.RequestException: For network-related errors.
        """
        validated_speaker_path = self._validate_speaker_audio(use_speaker_audio, speaker_audio_path)

        payload = {
            "text": text,
            "language": language,
            "speaker_audio_path": validated_speaker_path,
            "prefix_audio_path": prefix_audio_path,
            "emotion": emotion,
            "vq_score": vq_score,
            "fmax": fmax,
            "pitch_std": pitch_std,
            "speaking_rate": speaking_rate,
            "dnsmos_ovrl": dnsmos_ovrl,
            "speaker_noised": speaker_noised,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "randomize_seed": randomize_seed,
            "unconditional_keys": unconditional_keys,
            "linear": linear,
            "confidence": confidence,
            "quadratic": quadratic,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
        }

        # Filter out None values before sending
        payload = {k: v for k, v in payload.items() if v is not None}

        print(f"Sending POST request to {self.server_url} with payload:")
        print(json.dumps(payload, indent=2))

        try:
            response = requests.post(self.server_url, json=payload, timeout=timeout)

            print(f"Received status code: {response.status_code}")
            if response.status_code != 200:
                error_details = f"Status Code: {response.status_code}"
                try:
                    error_json = response.json()
                    error_details += f", Details: {error_json}"
                except requests.exceptions.JSONDecodeError:
                    error_details += f", Content: {response.text}"
                raise TTSClientError(f"Server error: {error_details}")

            content_type = response.headers.get('Content-Type')
            print(f"Received Content-Type: {content_type}")
            if content_type != 'audio/wav':
                raise TTSClientError(f"Expected Content-Type 'audio/wav', but got '{content_type}'")

            if not response.content:
                raise TTSClientError("Received empty response content.")
            print(f"Received {len(response.content)} bytes of audio data.")

            # Basic WAV validation
            try:
                wav_buffer = io.BytesIO(response.content)
                with wave.open(wav_buffer, 'rb') as wf:
                    print(f"Successfully validated response as WAV file.")
                    print(f"  Format: {wf.getcompname()}")
                    print(f"  Channels: {wf.getnchannels()}")
                    print(f"  Frame Rate: {wf.getframerate()}")
                    print(f"  Frames: {wf.getnframes()}")
            except wave.Error as e:
                raise TTSClientError(f"Could not validate response content as a WAV file. Error: {e}")

            return response.content

        except requests.exceptions.RequestException as e:
            print(f"Network error during request: {e}")
            raise # Re-raise network errors
        except TTSClientError:
            raise # Re-raise client-specific errors
        except Exception as e:
            print(f"An unexpected error occurred during synthesis request: {e}")
            import traceback
            traceback.print_exc()
            raise TTSClientError(f"Unexpected error: {e}") from e

def play_audio_bytes(audio_bytes: bytes, temp_filename: str = "temp_playback.wav"):
    """
    Saves audio bytes to a temporary file and plays it using afplay.

    Args:
        audio_bytes: The raw WAV audio data.
        temp_filename: The name for the temporary WAV file.
    """
    try:
        with open(temp_filename, 'wb') as f:
            f.write(audio_bytes)
        print(f"Temporarily saved audio to '{temp_filename}' for playback.")

        print(f"Attempting to play audio file: '{temp_filename}'")
        subprocess.run(["afplay", temp_filename], check=True, capture_output=True)
        print(f"Successfully played audio file.")

    except FileNotFoundError:
        print(f"Error: 'afplay' command not found. Cannot play audio.")
    except subprocess.CalledProcessError as e:
        print(f"Error playing audio file with afplay: {e}")
        print(f"afplay stderr: {e.stderr.decode()}")
    except IOError as e:
        print(f"Error writing temporary audio file '{temp_filename}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred during audio playback: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
                print(f"Removed temporary playback file: '{temp_filename}'")
            except OSError as e:
                print(f"Error removing temporary file '{temp_filename}': {e}")

# Example usage (can be run directly for basic testing)
if __name__ == "__main__":
    print("Running basic client test...")
    # Replace with your actual server URL if different
    test_url = "http://127.0.0.1:5000/tts"
    client = TTSClient(test_url)

    try:
        # Simple synthesis request
        audio_data = client.synthesize(
            text="Hello from the client.",
            use_speaker_audio=False # Use default silence
        )
        print(f"Successfully received {len(audio_data)} bytes.")
        play_audio_bytes(audio_data)

    except (TTSClientError, requests.exceptions.RequestException) as e:
        print(f"Client test failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during client test: {e}")
