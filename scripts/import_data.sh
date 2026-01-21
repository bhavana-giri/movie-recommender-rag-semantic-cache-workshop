#!/bin/bash
# =============================================================================
# RIOT Data Import Script
# Imports movies.json into Redis Cloud using RIOT CLI
# =============================================================================

set -e

# Get the script directory and project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/.env"

# Load REDIS_URL from .env file if not already set
if [ -z "$REDIS_URL" ]; then
    if [ -f "$ENV_FILE" ]; then
        echo "Loading REDIS_URL from .env file..."
        # Extract REDIS_URL from .env file (handles quotes and spaces)
        REDIS_URL=$(grep -E "^REDIS_URL=" "$ENV_FILE" | cut -d '=' -f2- | tr -d '"' | tr -d "'")
        export REDIS_URL
    fi
fi

# Check if REDIS_URL is set
if [ -z "$REDIS_URL" ]; then
    echo "Error: REDIS_URL not found."
    echo ""
    echo "Please either:"
    echo "  1. Add REDIS_URL to your .env file:"
    echo "     REDIS_URL=redis://default:YOUR_PASSWORD@YOUR_ENDPOINT:PORT"
    echo ""
    echo "  2. Or export it as an environment variable:"
    echo "     export REDIS_URL=redis://default:YOUR_PASSWORD@YOUR_ENDPOINT:PORT"
    exit 1
fi

# Check if RIOT is installed
if ! command -v riot &> /dev/null; then
    echo "Error: RIOT CLI is not installed."
    echo ""
    echo "Install RIOT using one of these methods:"
    echo ""
    echo "  macOS (Homebrew):"
    echo "    brew install redis/tap/riot"
    echo ""
    echo "  Docker:"
    echo "    docker run riotx/riot --help"
    echo ""
    echo "  Or download from: https://github.com/redis/riot"
    exit 1
fi

# Data file path
DATA_FILE="$PROJECT_DIR/resources/movies.json"

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file not found: $DATA_FILE"
    exit 1
fi

echo "=== RIOT Data Import ==="
echo "Importing: $DATA_FILE"
echo "Target: Redis Cloud"
echo ""

# Import movies.json into Redis as Hash keys with prefix "movie:"
# Each movie will be stored with key pattern: movie:1, movie:2, etc.
riot file-import --uri="$REDIS_URL" "$DATA_FILE" hset --keyspace movie --key id

echo ""
echo "=== Import Complete ==="
echo "Movies have been imported with key prefix 'movie:' and sequential IDs"
echo ""
echo "Next step: Create embeddings and search index by calling:"
echo "  curl -X POST http://localhost:8000/api/create-index"

