import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import wave
import array

def trim_tts_audio(input_wav, output_wav, min_silence_len=70, silence_thresh=-60, padding_ms=200):
    """
    Trims a TTS audio file by removing the "Start" and "End" markers,
    keeping only the content between them.
    
    Parameters:
    -----------
    input_wav : str
        Path to the input WAV file
    output_wav : str
        Path to save the trimmed WAV file
    min_silence_len : int
        Minimum length of silence (in ms) to be considered a break
    silence_thresh : int
        Threshold (in dBFS) below which is considered silence
    padding_ms : int
        Padding to add around non-silent sections (in ms)
    
    Returns:
    --------
    tuple:
        Start time and end time (in seconds) of the trimmed section
    """
    # Load the audio file using pydub
    audio = AudioSegment.from_wav(input_wav)
    
    # Find non-silent chunks
    non_silent_chunks = detect_nonsilent(
        audio, 
        min_silence_len=min_silence_len, 
        silence_thresh=silence_thresh
    )
    
    # Make sure we found at least two chunks (Start and End markers)
    if len(non_silent_chunks) < 2:
        raise ValueError("Could not identify distinct Start and End markers in the audio")
    
    # Get the end of the first chunk (Start marker) and the start of the last chunk (End marker)
    start_end_time = non_silent_chunks[0][1] + padding_ms  # End of "Start" + padding
    end_start_time = non_silent_chunks[-1][0] - padding_ms  # Start of "End" - padding
    
    # Ensure we're not creating an invalid segment
    if start_end_time >= end_start_time:
        raise ValueError("Could not properly identify the content between markers")
    
    # Extract the audio between Start and End
    trimmed_audio = audio[start_end_time:end_start_time]
    
    # Export the trimmed audio
    trimmed_audio.export(output_wav, format="wav")
    
    return start_end_time/1000, end_start_time/1000  # Return timestamps in seconds


# Example usage
if __name__ == "__main__":
    # Using the first method with pydub
    start_time, end_time = trim_tts_audio("outputs/output_chunk_2.wav", "outputs/output_chunk_2_trimmed.wav")
    print(f"Trimmed audio from {start_time:.2f}s to {end_time:.2f}s")
    
    # Using the alternative method with wave and numpy
    # start_time, end_time = trim_tts_audio_alternative("input.wav", "output_trimmed_alt.wav")
    # print(f"Trimmed audio from {start_time:.2f}s to {end_time:.2f}s")