import re

def clean_text_for_tts(text: str) -> str:
    """
    Cleans and prepares text for a text-to-speech (TTS) model.

    - Replaces markdown code blocks with a generic phrase.
    - Adds 'start dot' and 'end dot' markers to sentences.
    - Normalizes whitespace.
    """

    # 1. Normalize whitespace (replace multiple spaces/newlines with single space)
    # Also handle potential leading/trailing whitespace introduced by replacements
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. Replace markdown code blocks (multiline and inline)
    # Replace multiline ```...``` blocks - ensure space padding
    text = re.sub(r'```.*?```', ' Code block content ', text, flags=re.DOTALL)
    # Replace inline `...` blocks - ensure space padding
    text = re.sub(r'`[^`]+`', ' Code block content ', text)
    # Normalize spaces again after replacements
    text = re.sub(r'\s+', ' ', text).strip()


    # 3. Split into sentences (simple split by ., ?, !)
    # Use lookbehind and lookahead to keep delimiters attached to sentences
    # This handles cases like "Mr. Smith" better but still not perfectly.
    # A more robust sentence tokenizer (like NLTK) could be used for complex cases.
    sentences = re.split(r'(?<=[.?!])\s+', text)

    processed_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence: # Only process non-empty sentences
            # Check if sentence already ends with punctuation handled by the split
            if sentence.endswith(('.', '?', '!')):
                 # Add markers around the existing sentence and punctuation
                 processed_sentences.append(f"start dot {sentence} end dot")
            else:
                 # If no punctuation (e.g., last sentence fragment), add markers
                 # We might want to add a default period here, but sticking to reqs
                 processed_sentences.append(f"start dot {sentence} end dot")


    # 4. Join processed sentences
    # Add spaces between the "end dot" of one sentence and "start dot" of the next.
    cleaned_text = " ".join(processed_sentences)

    # Optional: Further cleanups can be added here
    # e.g., expanding abbreviations, handling numbers, removing special chars

    return cleaned_text

# Example Usage
if __name__ == "__main__":
    input_text_1 = """
    Here is some text. This is the first sentence!
    And here is another one?
    Let's include `inline code` example.
    ```python
    print("Hello, World!")
    # This is a comment
    ```
    This is the final sentence.
    """
    cleaned_text_1 = clean_text_for_tts(input_text_1)
    print(f"Original 1:\n---\n{input_text_1}\n---")
    print(f"\nCleaned 1:\n---\n{cleaned_text_1}\n---\n")

    input_text_2 = "Just one sentence without punctuation"
    cleaned_text_2 = clean_text_for_tts(input_text_2)
    print(f"Original 2:\n---\n{input_text_2}\n---")
    print(f"\nCleaned 2:\n---\n{cleaned_text_2}\n---\n")

    input_text_3 = "Mr. Smith went to Washington. He saw the `important_file.txt`."
    cleaned_text_3 = clean_text_for_tts(input_text_3)
    print(f"Original 3:\n---\n{input_text_3}\n---")
    print(f"\nCleaned 3:\n---\n{cleaned_text_3}\n---\n")

    input_text_4 = "```bash\nls -l\n``` This follows the code."
    cleaned_text_4 = clean_text_for_tts(input_text_4)
    print(f"Original 4:\n---\n{input_text_4}\n---")
    print(f"\nCleaned 4:\n---\n{cleaned_text_4}\n---\n")

    input_text_5 = "Sentence one. Sentence two! Sentence three?"
    cleaned_text_5 = clean_text_for_tts(input_text_5)
    print(f"Original 5:\n---\n{input_text_5}\n---")
    print(f"\nCleaned 5:\n---\n{cleaned_text_5}\n---\n")
