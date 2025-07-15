#!/bin/bash
# IB Gateway Health Check and Auto-Restart Script
# This script checks if IB Gateway is properly authenticated and restarts if needed

set -e

# Configuration
CONTAINER_NAME="ibkr-gateway"
MAX_WAIT_TIME=120  # Maximum seconds to wait for authentication
CHECK_INTERVAL=5   # Seconds between checks
MAX_RESTART_ATTEMPTS=3
LOG_FILE="ib_gateway_health.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if container is running
check_container_running() {
    if docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        return 0
    else
        return 1
    fi
}

# Check if IB Gateway is authenticated
check_authentication() {
    local auth_status=$(docker logs "$CONTAINER_NAME" 2>&1 | tail -100 | grep -E "(Login has completed|Configuration tasks completed)" | wc -l)
    if [ "$auth_status" -ge 2 ]; then
        return 0
    else
        return 1
    fi
}

# Check for connection errors
check_for_errors() {
    local error_count=$(docker logs "$CONTAINER_NAME" 2>&1 | tail -50 | grep -E "(Connection refused|TimeoutError|will exit if login dialog)" | wc -l)
    if [ "$error_count" -gt 0 ]; then
        return 1
    else
        return 0
    fi
}

# Test API connection
test_api_connection() {
    # Use Python to test the actual API connection
    python3 -c "
import asyncio
from ib_async import IB
import sys

async def test_connection():
    ib = IB()
    try:
        await ib.connectAsync('localhost', 4102, clientId=999, timeout=10)
        await ib.disconnectAsync()
        print('SUCCESS')
        return True
    except Exception as e:
        print(f'FAILED: {e}')
        return False

try:
    result = asyncio.run(test_connection())
    sys.exit(0 if result else 1)
except:
    sys.exit(1)
" 2>/dev/null
}

# Restart container
restart_container() {
    log "Restarting IB Gateway container..."
    docker restart "$CONTAINER_NAME" >/dev/null 2>&1
    sleep 10  # Initial wait for container to start
}

# Main health check function
perform_health_check() {
    echo -e "${YELLOW}=== IB Gateway Health Check ===${NC}"
    
    # Check if container is running
    if ! check_container_running; then
        log "ERROR: Container $CONTAINER_NAME is not running"
        echo -e "${RED}✗ Container not running${NC}"
        return 1
    fi
    echo -e "${GREEN}✓ Container is running${NC}"
    
    # Check authentication status
    if check_authentication; then
        echo -e "${GREEN}✓ IB Gateway is authenticated${NC}"
        
        # Test API connection
        echo -n "Testing API connection... "
        if test_api_connection; then
            echo -e "${GREEN}✓ API connection successful${NC}"
            return 0
        else
            echo -e "${RED}✗ API connection failed${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ IB Gateway not authenticated${NC}"
        
        # Check for specific errors
        if ! check_for_errors; then
            echo -e "${RED}✗ Connection errors detected${NC}"
        fi
        return 1
    fi
}

# Auto-restart with retry logic
auto_restart_with_retry() {
    local attempt=1
    
    while [ $attempt -le $MAX_RESTART_ATTEMPTS ]; do
        log "Restart attempt $attempt of $MAX_RESTART_ATTEMPTS"
        
        # Restart the container
        restart_container
        
        # Wait for authentication
        local wait_time=0
        while [ $wait_time -lt $MAX_WAIT_TIME ]; do
            echo -ne "\rWaiting for authentication... ${wait_time}s / ${MAX_WAIT_TIME}s"
            
            if check_authentication; then
                echo -e "\n${GREEN}✓ Authentication successful${NC}"
                
                # Additional wait for API to be ready
                sleep 5
                
                # Test API connection
                if test_api_connection; then
                    log "IB Gateway successfully restarted and API is accessible"
                    return 0
                fi
            fi
            
            sleep $CHECK_INTERVAL
            wait_time=$((wait_time + CHECK_INTERVAL))
        done
        
        echo -e "\n${RED}✗ Authentication timeout${NC}"
        log "Authentication failed after ${MAX_WAIT_TIME} seconds"
        
        attempt=$((attempt + 1))
    done
    
    log "ERROR: Failed to restart IB Gateway after $MAX_RESTART_ATTEMPTS attempts"
    return 1
}

# Main script logic
main() {
    case "${1:-check}" in
        check)
            if perform_health_check; then
                echo -e "\n${GREEN}IB Gateway is healthy${NC}"
                exit 0
            else
                echo -e "\n${RED}IB Gateway health check failed${NC}"
                exit 1
            fi
            ;;
        
        restart)
            log "Manual restart requested"
            if auto_restart_with_retry; then
                echo -e "\n${GREEN}IB Gateway restarted successfully${NC}"
                exit 0
            else
                echo -e "\n${RED}Failed to restart IB Gateway${NC}"
                exit 1
            fi
            ;;
        
        auto)
            # Continuous monitoring mode
            log "Starting continuous monitoring mode"
            while true; do
                if ! perform_health_check >/dev/null 2>&1; then
                    log "Health check failed, initiating auto-restart"
                    auto_restart_with_retry
                fi
                sleep 60  # Check every minute
            done
            ;;
        
        *)
            echo "Usage: $0 [check|restart|auto]"
            echo "  check   - Check IB Gateway health (default)"
            echo "  restart - Force restart IB Gateway"
            echo "  auto    - Continuous monitoring with auto-restart"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"