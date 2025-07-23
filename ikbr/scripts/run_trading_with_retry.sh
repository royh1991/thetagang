#!/bin/bash

# Wrapper script to run trading bot with automatic IB Gateway recovery
# Handles connection failures and restarts as needed

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
HEALTH_CHECK_SCRIPT="$SCRIPT_DIR/check_ib_gateway.sh"
MAX_TRADING_RETRIES=3
RETRY_DELAY=30

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] ${message}${NC}"
}

# Function to run the trading bot
run_trading_bot() {
    cd "$PROJECT_DIR"
    
    # Activate virtual environment if it exists
    if [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
        source "$PROJECT_DIR/venv/bin/activate"
    elif [ -f "$PROJECT_DIR/../venv/bin/activate" ]; then
        source "$PROJECT_DIR/../venv/bin/activate"
    fi
    
    # Run the trading bot with all passed arguments
    python backtest2/main.py "$@"
}

# Main execution
main() {
    print_status $GREEN "Starting trading bot with automatic recovery..."
    
    # First ensure IB Gateway is healthy
    print_status $YELLOW "Checking IB Gateway health..."
    if ! "$HEALTH_CHECK_SCRIPT" auto; then
        print_status $RED "Failed to establish healthy IB Gateway connection"
        exit 1
    fi
    
    # Now run the trading bot with retries
    local retry_count=0
    
    while [ $retry_count -lt $MAX_TRADING_RETRIES ]; do
        print_status $GREEN "Starting trading bot (attempt $((retry_count + 1))/$MAX_TRADING_RETRIES)..."
        
        # Run trading bot and capture exit code
        set +e
        run_trading_bot "$@"
        local exit_code=$?
        set -e
        
        if [ $exit_code -eq 0 ]; then
            print_status $GREEN "Trading bot exited normally"
            exit 0
        fi
        
        print_status $YELLOW "Trading bot exited with code $exit_code"
        
        # Check if it was a connection error
        if grep -q "API connection failed\|No security definition\|Peer closed connection" backtest2/logs/*.log 2>/dev/null; then
            print_status $YELLOW "Detected connection issue, checking IB Gateway..."
            
            # Try to recover IB Gateway
            if "$HEALTH_CHECK_SCRIPT" auto; then
                retry_count=$((retry_count + 1))
                
                if [ $retry_count -lt $MAX_TRADING_RETRIES ]; then
                    print_status $YELLOW "Waiting $RETRY_DELAY seconds before retry..."
                    sleep $RETRY_DELAY
                fi
            else
                print_status $RED "Failed to recover IB Gateway"
                exit 1
            fi
        else
            # Non-connection error, don't retry
            print_status $RED "Trading bot failed with non-connection error"
            exit $exit_code
        fi
    done
    
    print_status $RED "Maximum retry attempts reached"
    exit 1
}

# Check if arguments were provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 [trading bot arguments]"
    echo "Example: $0 --symbol TSLA --strategy nick"
    exit 1
fi

# Run main function with all arguments
main "$@"