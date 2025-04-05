# Makefile for Zonos Vibes project

.PHONY: test

# Ensure uv is available and dependencies are synced before running tests
test:
	uv run python -m unittest test_text_cleaner.py

# Add other common targets like install, lint, format etc. as needed
# e.g., install: uv sync
