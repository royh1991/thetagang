#!/bin/bash

# IB Gateway Health Check and Auto-Restart Script
# Based on patterns from llm_instruct.txt

set -e

# Configuration
CONTAINER_NAME="ibkr-gateway"
MAX_RETRIES=3
WAIT_TIME=120  # seconds to wait for authentication
CHECK_INTERVAL=10  # seconds between checks

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] ${message}${NC}"
}

# Function to check if container is running
check_container_running() {
    docker ps --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "$CONTAINER_NAME"
}

# Function to check if IB Gateway is authenticated
check_authenticated() {
    docker logs "$CONTAINER_NAME" 2>&1 | tail -50 | grep -q "Login has completed"
}

# Function to check if configuration tasks completed
check_config_completed() {
    docker logs "$CONTAINER_NAME" 2>&1 | tail -50 | grep -q "Configuration tasks completed"
}

# Function to test API connection
test_api_connection() {
    # Simple Python script to test connection
    python3 - <<EOF
import asyncio
from ib_async import IB

async def test():
    ib = IB()
    try:
        await asyncio.wait_for(
            ib.connectAsync('localhost', 4102, clientId=999),
            timeout=10
        )
        print("API_CONNECTION_SUCCESS")
        ib.disconnect()
        return True
    except Exception as e:
        print(f"API_CONNECTION_FAILED: {e}")
        return False

asyncio.run(test())
EOF
}

# Function to restart container
restart_container() {
    print_status $YELLOW "Restarting $CONTAINER_NAME..."
    docker restart "$CONTAINER_NAME"
    sleep 5
}

# Function to perform full health check
health_check() {
    print_status $GREEN "Starting IB Gateway health check..."
    
    # Check 1: Container running
    if ! check_container_running; then
        print_status $RED "Container $CONTAINER_NAME is not running!"
        return 1
    fi
    print_status $GREEN "✓ Container is running"
    
    # Check 2: Authentication completed
    if ! check_authenticated; then
        print_status $RED "IB Gateway not authenticated"
        return 1
    fi
    print_status $GREEN "✓ Authentication completed"
    
    # Check 3: Configuration completed
    if ! check_config_completed; then
        print_status $RED "Configuration tasks not completed"
        return 1
    fi
    print_status $GREEN "✓ Configuration completed"
    
    # Check 4: API connectivity
    if test_api_connection | grep -q "API_CONNECTION_SUCCESS"; then
        print_status $GREEN "✓ API connection successful"
        return 0
    else
        print_status $RED "API connection failed"
        return 1
    fi
}

# Function to wait for authentication with timeout
wait_for_auth() {
    local elapsed=0
    
    print_status $YELLOW "Waiting for IB Gateway authentication..."
    
    while [ $elapsed -lt $WAIT_TIME ]; do
        if check_authenticated && check_config_completed; then
            print_status $GREEN "Authentication completed!"
            return 0
        fi
        
        sleep $CHECK_INTERVAL
        elapsed=$((elapsed + CHECK_INTERVAL))
        print_status $YELLOW "Waiting... ($elapsed/$WAIT_TIME seconds)"
    done
    
    print_status $RED "Authentication timeout after $WAIT_TIME seconds"
    return 1
}

# Main function
main() {
    local mode=${1:-check}
    
    case $mode in
        check)
            # Just perform health check
            if health_check; then
                print_status $GREEN "IB Gateway is healthy"
                exit 0
            else
                print_status $RED "IB Gateway health check failed"
                exit 1
            fi
            ;;
            
        restart)
            # Force restart and wait
            restart_container
            if wait_for_auth && health_check; then
                print_status $GREEN "IB Gateway restarted successfully"
                exit 0
            else
                print_status $RED "IB Gateway restart failed"
                exit 1
            fi
            ;;
            
        auto)
            # Auto mode with retries
            local retry_count=0
            
            while [ $retry_count -lt $MAX_RETRIES ]; do
                if health_check; then
                    print_status $GREEN "IB Gateway is healthy"
                    exit 0
                fi
                
                retry_count=$((retry_count + 1))
                print_status $YELLOW "Health check failed, attempting restart ($retry_count/$MAX_RETRIES)..."
                
                restart_container
                
                if wait_for_auth && health_check; then
                    print_status $GREEN "IB Gateway recovered successfully"
                    exit 0
                fi
            done
            
            print_status $RED "Failed to recover IB Gateway after $MAX_RETRIES attempts"
            exit 1
            ;;
            
        *)
            echo "Usage: $0 [check|restart|auto]"
            echo "  check   - Check health status (default)"
            echo "  restart - Force restart and wait for ready"
            echo "  auto    - Check health and auto-restart if needed"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"