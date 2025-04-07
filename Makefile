# Makefile for Zonos Vibes project

.PHONY: test deploy_beta1l

# Ensure uv is available and dependencies are synced before running tests
test:
	uv run python -m unittest test_text_cleaner.py

# Deploy and run the server on beta1l using the dedicated script
deploy_beta1l:
	$(eval BRANCH := $(shell git rev-parse --abbrev-ref HEAD))
	@echo "Running deployment script for branch $(BRANCH)..."
	./deploy_beta1l.sh $(BRANCH)

# Add other common targets like install, lint, format etc. as needed
# e.g., install: uv sync
