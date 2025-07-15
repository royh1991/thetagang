#!/bin/bash
# Wrapper script to run backtests with automatic IB Gateway recovery
# This handles connection issues automatically

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CHECK_SCRIPT="$SCRIPT_DIR/check_ib_gateway.sh"
MAX_RETRIES=3
RETRY_DELAY=10

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to run backtest
run_backtest() {
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    python backtest2/run_backtest.py "$@"
}

# Main execution
echo -e "${YELLOW}=== Backtest Runner with Auto-Recovery ===${NC}"

# First, ensure IB Gateway is healthy
echo "Checking IB Gateway health..."
if ! "$CHECK_SCRIPT" check; then
    echo -e "${YELLOW}IB Gateway needs restart, attempting recovery...${NC}"
    if ! "$CHECK_SCRIPT" restart; then
        echo -e "${RED}Failed to recover IB Gateway. Please check Docker logs.${NC}"
        exit 1
    fi
fi

# Try to run the backtest with retries
attempt=1
while [ $attempt -le $MAX_RETRIES ]; do
    echo -e "\n${YELLOW}Running backtest (attempt $attempt of $MAX_RETRIES)...${NC}"
    
    if run_backtest "$@"; then
        echo -e "${GREEN}✓ Backtest completed successfully${NC}"
        exit 0
    else
        exit_code=$?
        echo -e "${RED}✗ Backtest failed with exit code: $exit_code${NC}"
        
        # Check if it's a connection issue
        if [ $attempt -lt $MAX_RETRIES ]; then
            echo "Checking if IB Gateway needs restart..."
            
            # Try to recover IB Gateway
            if ! "$CHECK_SCRIPT" check >/dev/null 2>&1; then
                echo "IB Gateway connection issue detected, attempting recovery..."
                if "$CHECK_SCRIPT" restart; then
                    echo "IB Gateway restarted, retrying backtest in ${RETRY_DELAY} seconds..."
                    sleep $RETRY_DELAY
                else
                    echo -e "${RED}Failed to restart IB Gateway${NC}"
                    exit 1
                fi
            else
                echo "IB Gateway appears healthy, waiting ${RETRY_DELAY} seconds before retry..."
                sleep $RETRY_DELAY
            fi
        fi
    fi
    
    attempt=$((attempt + 1))
done

echo -e "${RED}Failed to complete backtest after $MAX_RETRIES attempts${NC}"
exit 1