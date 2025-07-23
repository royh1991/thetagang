# IBKR Trading Bot Scripts

This directory contains utility scripts for managing the IBKR trading bot.

## Scripts Overview

### 1. `check_ib_gateway.sh`
Health check and auto-restart script for IB Gateway container.

**Usage:**
```bash
# Check health status
./check_ib_gateway.sh check

# Force restart and wait for ready
./check_ib_gateway.sh restart

# Auto mode - check health and restart if needed
./check_ib_gateway.sh auto
```

### 2. `run_trading_with_retry.sh`
Wrapper script that runs the trading bot with automatic IB Gateway recovery.

**Usage:**
```bash
# Run with default parameters
./run_trading_with_retry.sh --symbol TSLA --strategy nick

# Run with custom parameters
./run_trading_with_retry.sh --symbol TSLA --strategy nick --position-size 0.5 --adx-threshold 25
```

### 3. `cron_health_check.sh`
Cron-based health check that monitors both IB Gateway and the trading bot.

**Setup:**
```bash
# Add to crontab (runs every 5 minutes)
crontab -e
*/5 * * * * /home/royhu91/thetagang/ikbr/scripts/cron_health_check.sh
```

### 4. `ikbr-trading.service`
Systemd service file for automatic startup and restart.

**Installation:**
```bash
# Run the install script
sudo ./install_service.sh

# Or manually:
sudo cp ikbr-trading.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ikbr-trading
sudo systemctl start ikbr-trading
```

**Management:**
```bash
# Start/stop/restart
sudo systemctl start ikbr-trading
sudo systemctl stop ikbr-trading
sudo systemctl restart ikbr-trading

# Check status
sudo systemctl status ikbr-trading

# View logs
sudo journalctl -u ikbr-trading -f
```

## Deployment on Google Cloud VM

1. **Initial Setup:**
   ```bash
   # Clone repository
   git clone <your-repo> ~/thetagang
   cd ~/thetagang/ikbr
   
   # Create virtual environment
   python3 -m venv ../venv
   source ../venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configure Docker:**
   ```bash
   # Ensure docker-compose.yml is configured
   # Start IB Gateway container
   docker-compose up -d ib-gateway
   ```

3. **Choose Deployment Method:**

   **Option A: Systemd (Recommended)**
   ```bash
   # Install service
   sudo ./scripts/install_service.sh
   
   # Start service
   sudo systemctl start ikbr-trading
   ```

   **Option B: Cron + Health Check**
   ```bash
   # Add to crontab
   crontab -e
   */5 * * * * /home/royhu91/thetagang/ikbr/scripts/cron_health_check.sh
   
   # Start manually first time
   ./scripts/run_trading_with_retry.sh --symbol TSLA --strategy nick &
   ```

4. **Monitor Logs:**
   ```bash
   # Trading logs
   tail -f ~/trading.log
   
   # Health check logs (if using cron)
   tail -f ~/trading_health_check.log
   
   # Systemd logs (if using systemd)
   sudo journalctl -u ikbr-trading -f
   ```

## Troubleshooting

### Connection Issues
If you see "API connection failed" or "No security definition":
1. Check IB Gateway health: `./check_ib_gateway.sh check`
2. Restart IB Gateway: `./check_ib_gateway.sh restart`
3. Check Docker logs: `docker logs ibkr-gateway`

### Position Sizing Errors
The "NoneType object has no attribute 'last'" error has been fixed by:
- Adding ticker subscription on connect
- Validating ticker data before use
- Using bid/ask midpoint as fallback

### Authentication Delays
IB Gateway can take 1-2 minutes to authenticate. The scripts handle this by:
- Waiting up to 2 minutes for authentication
- Checking for "Login has completed" in logs
- Automatic retry with exponential backoff

## Best Practices

1. **Always use wrapper scripts** instead of running main.py directly
2. **Monitor logs regularly** for any issues
3. **Set up alerts** for extended downtime
4. **Review trades daily** to ensure strategy is working as expected
5. **Keep position sizes conservative** (10% default)

## Security Notes

- Never commit credentials to git
- Use environment variables for sensitive data
- Restrict file permissions on scripts
- Use paper trading mode for testing
- Always confirm before switching to live trading