# Makefile for Zonos Vibes project
.PHONY: test test_server deploy_beta1l list status

list:
	@echo "Available make commands:"
	@grep "^[^#[:space:]].*:" Makefile | cut -d: -f1 | sort

# Ensure uv is available and dependencies are synced before running tests
test:
	uv run python -m unittest test_text_cleaner.py

# Run the server tests
test_server:
	uv run python test_server.py

# Deploy and run the server on beta1l using the dedicated script
deploy_beta1l:
	$(eval BRANCH := $(shell git rev-parse --abbrev-ref HEAD))
	@echo "Running deployment script for branch $(BRANCH)..."
	./deploy_beta1l.sh $(BRANCH)

# Get the status of the beta1l deployment
status:
	@echo "Getting status of beta1l deployment..."
	./get_status_of_beta1.sh

# Add other common targets like install, lint, format etc. as needed
# e.g., install: uv sync
