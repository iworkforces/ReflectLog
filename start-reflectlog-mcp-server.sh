#!/bin/bash

# ReflectLog Server Startup Script
# This script starts the ReflectLog server

set -e

# Global variable to track server PID
SERVER_PID=""
SHUTDOWN_INITIATED=0
GRACEFUL_TIMEOUT=20
TERM_TIMEOUT=10

# Graceful shutdown function
shutdown_server() {
    local signal="${1:-SIGTERM}"

    if [[ $SHUTDOWN_INITIATED -eq 1 ]]; then
        if [[ "$signal" != "EXIT" ]]; then
            return
        fi
    fi
    SHUTDOWN_INITIATED=1

    local effective_signal="$signal"
    if [[ "$effective_signal" == "EXIT" ]]; then
        effective_signal="SIGTERM"
    fi

    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo -e "${YELLOW}🛑 Shutting down ReflectLog Server Gracefully...${NC}"
        echo -e "${CYAN}   Persisting USearch and Tantivy data to disk...${NC}"

        local target_pid="$SERVER_PID"
        local pgid=""
        pgid="$(ps -o pgid= -p "$SERVER_PID" 2>/dev/null | tr -d ' ' || true)"
        if [[ -n "$pgid" && "$pgid" != "$$" ]]; then
            target_pid="-$pgid"
        fi

        if ! kill -"${effective_signal}" "$target_pid" 2>/dev/null; then
            kill -"${effective_signal}" "$SERVER_PID" 2>/dev/null || true
        fi

        local elapsed=0
        while kill -0 "$SERVER_PID" 2>/dev/null && [[ $elapsed -lt $GRACEFUL_TIMEOUT ]]; do
            sleep 1
            elapsed=$((elapsed + 1))
        done

        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo -e "${YELLOW}⏱️  Graceful shutdown taking longer than ${GRACEFUL_TIMEOUT}s, sending SIGTERM...${NC}"
            if ! kill -TERM "$target_pid" 2>/dev/null; then
                kill -TERM "$SERVER_PID" 2>/dev/null || true
            fi

            elapsed=0
            while kill -0 "$SERVER_PID" 2>/dev/null && [[ $elapsed -lt $TERM_TIMEOUT ]]; do
                sleep 1
                elapsed=$((elapsed + 1))
            done
        fi

        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo -e "${RED}⚠️  Force stopping server...${NC}"
            if ! kill -KILL "$target_pid" 2>/dev/null; then
                kill -KILL "$SERVER_PID" 2>/dev/null || true
            fi
        fi

        set +e
        wait "$SERVER_PID" 2>/dev/null
        set -e

        echo -e "${GREEN}✅ Server stopped successfully${NC}"
    fi

    if [[ "$signal" != "EXIT" ]]; then
        exit 0
    fi
}

# Set up signal handlers
trap 'shutdown_server SIGINT' SIGINT
trap 'shutdown_server SIGTERM' SIGTERM
trap 'shutdown_server SIGQUIT' SIGQUIT
trap 'shutdown_server EXIT' EXIT

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Usage/help function
show_usage() {
    cat << EOF
${BLUE}ReflectLog Server${NC}
Usage: $(basename "$0") [OPTIONS]

Options:
  --help, -h             Show this help message
EOF
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            echo ""
            show_usage
            exit 1
            ;;
    esac
done

echo -e "${BLUE}🚀 ReflectLog Server${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ uv is not installed${NC}"
    echo -e "${YELLOW}Please install uv: https://docs.astral.sh/uv/getting-started/installation/${NC}"
    exit 1
fi

echo -e "${GREEN}✅ uv is available${NC}"
echo -e "${CYAN}uv version: $(uv --version)${NC}"
echo -e "${CYAN}Python version: $(uv run python --version 2>&1)${NC}"
echo ""

# Check if we're in the correct directory
if [ ! -f "reflectlog/server.py" ]; then
    echo -e "${RED}❌ reflectlog/server.py not found${NC}"
    echo -e "${YELLOW}Please run this script from the ReflectLog project root directory${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Found reflectlog/server.py${NC}"
echo ""

# Check if pyproject.toml exists
if [ ! -f "pyproject.toml" ]; then
    echo -e "${YELLOW}⚠️  pyproject.toml not found${NC}"
    echo -e "${YELLOW}Make sure dependencies are properly configured${NC}"
fi

# Install dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
echo -e "${CYAN}Command: uv sync${NC}"
if uv sync; then
    echo -e "${GREEN}✅ Dependencies installed successfully${NC}"
else
    echo -e "${RED}❌ Failed to install dependencies${NC}"
    exit 1
fi
echo ""

# Load .env file if it exists
if [ -f ".env" ]; then
    echo -e "${BLUE}Loading environment variables from .env...${NC}"
    # Export all variables from .env file
    # set -a makes all variables automatically exported
    set -a
    source .env
    set +a
    echo -e "${GREEN}✅ Environment variables loaded from .env${NC}"
    echo ""
else
    echo -e "${YELLOW}⚠️  .env file not found, using environment variables only${NC}"
    echo ""
fi

# Display startup information
echo -e "${BLUE}Starting ReflectLog Server...${NC}"
if [ ! -z "$MCP_TRANSPORT" ]; then
    echo -e "${CYAN}Transport: $MCP_TRANSPORT${NC}"
else
    echo -e "${CYAN}Transport: stdio (default)${NC}"
fi
echo -e "${CYAN}Command: uv run reflectlog${NC}"
echo ""

# Start the server
echo -e "${YELLOW}Press Ctrl+C to stop the server gracefully (data will be persisted)${NC}"
echo ""

# Start the server in the background and capture its PID
# Let server.py read MCP_* environment variables from .env
uv run reflectlog &
SERVER_PID=$!

# Wait for the server process to complete
wait $SERVER_PID
