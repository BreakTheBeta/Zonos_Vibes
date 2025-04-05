import re

def clean_text_for_tts(text: str) -> str:
    """
    Cleans and prepares text for a text-to-speech (TTS) model.

    - Replaces markdown code blocks with a generic phrase.
    - Adds 'start dot' and 'end dot' markers to the beginning and end of the entire text block.
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

    # 3. Add start/end markers to the entire block
    # Ensure there's content before adding markers
    if text:
        cleaned_text = f"start dot {text} end dot"
    else:
        cleaned_text = "" # Return empty if input was empty/whitespace only

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
