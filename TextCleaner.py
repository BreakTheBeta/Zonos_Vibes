import re

# Constants removed as splitting logic is gone

def _base_clean_text(text: str) -> str:
    """Core cleaning logic without start/end markers."""
    # 1. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. Replace markdown code blocks
    # Use a function for replacement to handle potential leading/trailing whitespace in capture
    def replace_code_block(match):
        content = match.group(1).strip() # Capture group 1 and strip whitespace
        return f"File: {content}."

    text = re.sub(r'```(.*?)```', replace_code_block, text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', replace_code_block, text)
    text = re.sub(r'\s+', ' ', text).strip() # Normalize again


    # Optional: Further cleanups can be added here
    return text

# Splitting logic removed.

def clean_text(text: str) -> str:
    """
    Cleans text using the base cleaning logic. Does not split or add markers.
    """
    return _base_clean_text(text)


# Example Usage
if __name__ == "__main__":
    input_text_example = """
    This is a sentence with some   extra   whitespace.
    It also includes a code block:
    ```python
    print("Hello, World!") # Some comment
    ```
    And some `inline code`.
    Followed by more text.
    """
    print("--- Testing clean_text ---")
    cleaned_text = clean_text(input_text_example)
    print(f"Original Text:\n---\n{input_text_example}\n---")
    print(f"\nCleaned Text:\n---\n{cleaned_text}\n---")
    print("--- End of Test ---")
