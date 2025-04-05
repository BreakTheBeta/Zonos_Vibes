import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import wave
import array

def trim_tts_audio(input_wav, output_wav, min_silence_len=80, silence_thresh=-60, padding_ms=200):
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

def trim_tts_audio_alternative(input_wav, output_wav, threshold=1000, min_silence_duration=0.5):
    """
    Alternative approach using numpy and wave modules to trim a TTS audio file.
    
    Parameters:
    -----------
    input_wav : str
        Path to the input WAV file
    output_wav : str
        Path to save the trimmed WAV file
    threshold : int
        Amplitude threshold for detecting speech
    min_silence_duration : float
        Minimum duration of silence (in seconds) to be considered a break
    """
    # Open the WAV file
    with wave.open(input_wav, 'rb') as wf:
        # Get the file parameters
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        
        # Read all frames
        frames = wf.readframes(n_frames)
    
    # Convert binary data to numpy array
    if sample_width == 2:
        fmt = 'h'  # short (16-bit)
    elif sample_width == 4:
        fmt = 'i'  # int (32-bit)
    else:
        fmt = 'B'  # unsigned char (8-bit)
    
    # Convert to array
    data = array.array(fmt, frames)
    
    # If stereo, convert to mono by averaging channels
    if n_channels == 2:
        left = data[0::2]
        right = data[1::2]
        data = [(l + r) // 2 for l, r in zip(left, right)]
    
    # Convert to numpy array for easier processing
    data = np.array(data)
    
    # Calculate the absolute values
    abs_data = np.abs(data)
    
    # Find regions above the threshold (speech)
    speech_regions = abs_data > threshold
    
    # Convert to sample indices
    min_silence_samples = int(min_silence_duration * framerate)
    
    # Find transitions from silence to speech and vice versa
    speech_starts = np.where(np.logical_and(~speech_regions[:-1], speech_regions[1:]))[0]
    speech_ends = np.where(np.logical_and(speech_regions[:-1], ~speech_regions[1:]))[0]
    
    # If the file starts with speech, add a start point at 0
    if speech_regions[0]:
        speech_starts = np.insert(speech_starts, 0, 0)
    
    # If the file ends with speech, add an end point at the end
    if speech_regions[-1]:
        speech_ends = np.append(speech_ends, len(data) - 1)
    
    # Group continuous speech regions
    speech_segments = []
    current_start = speech_starts[0]
    
    for i in range(len(speech_ends)):
        if i < len(speech_starts) - 1 and speech_starts[i+1] - speech_ends[i] <= min_silence_samples:
            # This is a short silence, continue the segment
            continue
        else:
            # End of a segment
            speech_segments.append((current_start, speech_ends[i]))
            if i < len(speech_starts) - 1:
                current_start = speech_starts[i+1]
    
    # Ensure we have at least two segments (Start and End markers)
    if len(speech_segments) < 2:
        raise ValueError("Could not identify distinct Start and End markers in the audio")
    
    # Assume the first segment is "Start" and the last is "End"
    start_end = speech_segments[0][1]
    end_start = speech_segments[-1][0]
    
    # Extract the audio between Start and End
    trimmed_data = data[start_end:end_start]
    
    # Convert back to binary data
    trimmed_frames = array.array(fmt, trimmed_data).tobytes()
    
    # Write to a new WAV file
    with wave.open(output_wav, 'wb') as wf:
        wf.setnchannels(1)  # Mono output
        wf.setsampwidth(sample_width)
        wf.setframerate(framerate)
        wf.writeframes(trimmed_frames)
    
    return start_end/framerate, end_start/framerate  # Return timestamps in seconds

# Example usage
if __name__ == "__main__":
    # Using the first method with pydub
    start_time, end_time = trim_tts_audio("outputs/output_chunk_2.wav", "outputs/output_chunk_2_trimmed.wav")
    print(f"Trimmed audio from {start_time:.2f}s to {end_time:.2f}s")
    
    # Using the alternative method with wave and numpy
    # start_time, end_time = trim_tts_audio_alternative("input.wav", "output_trimmed_alt.wav")
    # print(f"Trimmed audio from {start_time:.2f}s to {end_time:.2f}s")