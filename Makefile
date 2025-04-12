# Makefile for Zonos Vibes project
.PHONY: test test_server deploy_beta1l stop_beta1l list status server

list:
	@echo "Available make commands:"
	@grep "^[^#[:space:]].*:" Makefile | cut -d: -f1 | sort

test:
	uv run python -m unittest test_text_cleaner.py

test_server:
	uv run python test_server.py

deploy_beta1l:
	$(eval BRANCH := $(shell git rev-parse --abbrev-ref HEAD))
	@echo "Running deployment script for branch $(BRANCH)..."
	./deploy_beta1l.sh $(BRANCH)

stop_beta1l:
	@echo "Running stop script for beta1l..."
	./stop_beta1l.sh

status:
	@echo "Getting status of beta1l deployment..."
	./get_status_of_beta1.sh

server:
	uv run server.py

 e.g., install: uv sync
