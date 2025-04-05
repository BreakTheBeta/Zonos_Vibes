import re
import math

# Average reading speed (words per second) - adjust as needed
WORDS_PER_SECOND = 2.5
MAX_CHUNK_SECONDS = 30

def estimate_reading_time_seconds(text: str) -> float:
    """Estimates the reading time for a given text in seconds."""
    words = text.split()
    word_count = len(words)
    if word_count == 0:
        return 0.0
    return word_count / WORDS_PER_SECOND

def _base_clean_text(text: str) -> str:
    """Core cleaning logic without start/end markers."""
    # 1. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. Replace markdown code blocks
    text = re.sub(r'```.*?```', ' File: ', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]+`', ' File: ', text)
    text = re.sub(r'\s+', ' ', text).strip() # Normalize again


    # Optional: Further cleanups can be added here
    return text

def clean_and_split_text(text: str, max_seconds: float = MAX_CHUNK_SECONDS) -> list[str]:
    """
    Cleans text, estimates reading time, and splits it into chunks
    where each chunk is estimated to be read in less than max_seconds.
    Adds 'Start...' and '...End.' markers to each chunk.
    """
    base_cleaned_text = _base_clean_text(text)
    if not base_cleaned_text:
        return []

    # Split into potential sentences or segments (simple split by ., !, ?)
    # This is a basic approach; more sophisticated sentence boundary detection could be used.
    sentences = re.split(r'(?<=[.!?])\s+', base_cleaned_text)
    sentences = [s.strip() for s in sentences if s.strip()] # Remove empty strings

    print("Cleaned Ouput:", base_cleaned_text)

    chunks = []
    current_chunk = ""
    current_chunk_time = 0.0

    for sentence in sentences:
        sentence_time = estimate_reading_time_seconds(sentence)

        # If a single sentence is already too long, split it further (simple word split)
        if sentence_time > max_seconds:
            words = sentence.split()
            words_per_chunk = math.floor(max_seconds * WORDS_PER_SECOND)
            sub_sentences = [' '.join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]
            
            # Process the sub-sentences
            for sub_sentence in sub_sentences:
                sub_sentence_time = estimate_reading_time_seconds(sub_sentence)
                # If the current chunk is not empty, finalize it
                if current_chunk:
                    chunks.append(f"Stararart... {current_chunk.strip()}... Stararat.")
                    current_chunk = ""
                    current_chunk_time = 0.0
                # Add the long sub-sentence as its own chunk
                chunks.append(f"Stararart... {sub_sentence.strip()}... Stararat.")
            continue # Move to the next original sentence

        # Check if adding the current sentence exceeds the max time
        if current_chunk and current_chunk_time + sentence_time > max_seconds:
            # Finalize the current chunk
            chunks.append(f"Stararart... {current_chunk.strip()}... Stararart.")
            # Start a new chunk with the current sentence
            current_chunk = sentence
            current_chunk_time = sentence_time
        else:
            # Add the sentence to the current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_chunk_time += sentence_time

    # Add the last remaining chunk if it exists
    if current_chunk:
        chunks.append(f"Start... {current_chunk.strip()}... End.")

    return chunks

# --- Original function kept for potential backward compatibility or other uses ---
def clean_text_for_tts(text: str) -> str:
    """
    Original cleaning function. Cleans text and adds start/end markers
    to the *entire* block. Does not split.
    """
    base_cleaned_text = _base_clean_text(text)
    if base_cleaned_text:
        cleaned_text = f"Start... {base_cleaned_text}... End."
    else:
        cleaned_text = ""
    return cleaned_text
# --- End of original function ---


# Example Usage
if __name__ == "__main__":
    input_text_long = """
    This is the first sentence. It should be fairly short.
    This is the second sentence, which is a bit longer to test the timing estimation. Maybe add a few more words.
    A third sentence follows. Short and sweet.
    Now for a much, much longer sentence designed specifically to exceed the thirty-second limit all by itself, requiring it to be split into multiple smaller pieces based on the word count calculation derived from the estimated words per second reading speed. We need to ensure this part gets broken down correctly. Let's add even more words to be absolutely sure it triggers the splitting logic within the function. This sentence just keeps going and going, filling up space and time.
    Another short one.
    And a final sentence to wrap things up. This might start a new chunk or be added to the previous one depending on timing.
    ```python
    print("This code block will be replaced.")
    ```
    Followed by text after the code block.
    `inline code` here.
    """
    print("--- Testing clean_and_split_text ---")
    split_chunks = clean_and_split_text(input_text_long)
    print(f"Original Long Text:\n---\n{input_text_long}\n---")
    print("\nSplit Chunks:")
    for i, chunk in enumerate(split_chunks):
        estimated_time = estimate_reading_time_seconds(chunk.replace("Start... ", "").replace("... End.", ""))
        print(f"--- Chunk {i+1} (Est: {estimated_time:.2f}s) ---")
        print(chunk)
    print("--- End of Split Chunks ---\n")

    input_text_short = "This is a very short text. It won't need splitting."
    split_chunks_short = clean_and_split_text(input_text_short)
    print(f"Original Short Text:\n---\n{input_text_short}\n---")
    print("\nSplit Chunks (Short):")
    for i, chunk in enumerate(split_chunks_short):
        estimated_time = estimate_reading_time_seconds(chunk.replace("Start... ", "").replace("... End.", ""))
        print(f"--- Chunk {i+1} (Est: {estimated_time:.2f}s) ---")
        print(chunk)
    print("--- End of Split Chunks (Short) ---\n")

    input_text_empty = "   "
    split_chunks_empty = clean_and_split_text(input_text_empty)
    print(f"Original Empty Text:\n---\n{input_text_empty}\n---")
    print("\nSplit Chunks (Empty):")
    print(split_chunks_empty)
    print("--- End of Split Chunks (Empty) ---")

    # Test the original function remains
    print("\n--- Testing original clean_text_for_tts ---")
    cleaned_original = clean_text_for_tts(input_text_short)
    print(f"Original Short Text:\n---\n{input_text_short}\n---")
    print(f"\nCleaned (Original Method):\n---\n{cleaned_original}\n---")
    print("--- End of Original Test ---")
