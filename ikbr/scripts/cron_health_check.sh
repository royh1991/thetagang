#!/bin/bash

# Cron health check script for IBKR trading bot
# Add to crontab with: */5 * * * * /home/royhu91/thetagang/ikbr/scripts/cron_health_check.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/home/royhu91/trading_health_check.log"
PIDFILE="/tmp/ikbr_trading.pid"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Check if trading bot is running
check_trading_bot() {
    if [ -f "$PIDFILE" ]; then
        local pid=$(cat "$PIDFILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            # Check if it's actually our Python process
            if ps -p "$pid" -o comm= | grep -q "python"; then
                return 0
            fi
        fi
    fi
    return 1
}

# Start trading bot
start_trading_bot() {
    cd "$PROJECT_DIR"
    
    # Find and use the correct Python from venv
    local PYTHON_CMD="python"
    if [ -f "$PROJECT_DIR/venv/bin/python" ]; then
        PYTHON_CMD="$PROJECT_DIR/venv/bin/python"
    elif [ -f "$PROJECT_DIR/../venv/bin/python" ]; then
        PYTHON_CMD="$PROJECT_DIR/../venv/bin/python"
    else
        log "ERROR: Virtual environment not found!"
        return 1
    fi
    
    # Start in background and save PID
    nohup $PYTHON_CMD backtest2/main.py --symbol TSLA --strategy nick \
        >> /home/royhu91/trading.log 2>&1 &
    
    echo $! > "$PIDFILE"
    log "Started trading bot with PID $!"
}

# Main logic
main() {
    # First check IB Gateway health
    if ! "$SCRIPT_DIR/check_ib_gateway.sh" check > /dev/null 2>&1; then
        log "IB Gateway health check failed, attempting recovery..."
        
        if "$SCRIPT_DIR/check_ib_gateway.sh" auto >> "$LOG_FILE" 2>&1; then
            log "IB Gateway recovered successfully"
        else
            log "ERROR: Failed to recover IB Gateway"
            exit 1
        fi
    fi
    
    # Check if trading bot is running
    if ! check_trading_bot; then
        log "Trading bot not running, starting it..."
        start_trading_bot
    else
        # Bot is running, check if it's responsive by looking at recent logs
        local last_log_time=$(tail -1 /home/royhu91/trading.log 2>/dev/null | grep -oE '\d{2}:\d{2}:\d{2}' | head -1)
        if [ -n "$last_log_time" ]; then
            # Convert to seconds for comparison
            local current_epoch=$(date +%s)
            local log_epoch=$(date -d "$last_log_time" +%s 2>/dev/null || echo 0)
            local time_diff=$((current_epoch - log_epoch))
            
            # If no logs for more than 10 minutes, restart
            if [ $time_diff -gt 600 ]; then
                log "Trading bot appears stuck (no logs for $time_diff seconds), restarting..."
                kill $(cat "$PIDFILE") 2>/dev/null
                sleep 5
                start_trading_bot
            fi
        fi
    fi
}

# Run main function
main