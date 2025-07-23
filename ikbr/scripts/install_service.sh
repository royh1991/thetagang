#!/bin/bash

# Script to install the IBKR trading bot as a systemd service

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="ikbr-trading.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_FILE"

echo "Installing IBKR Trading Bot systemd service..."

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo: sudo $0"
    exit 1
fi

# Copy service file
echo "Copying service file to $SERVICE_PATH..."
cp "$SCRIPT_DIR/$SERVICE_FILE" "$SERVICE_PATH"

# Reload systemd
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Enable the service
echo "Enabling service..."
systemctl enable $SERVICE_FILE

echo "Service installed successfully!"
echo ""
echo "To manage the service, use:"
echo "  sudo systemctl start ikbr-trading    # Start the service"
echo "  sudo systemctl stop ikbr-trading     # Stop the service"
echo "  sudo systemctl status ikbr-trading   # Check status"
echo "  sudo systemctl restart ikbr-trading  # Restart the service"
echo "  sudo journalctl -u ikbr-trading -f   # View logs"
echo ""
echo "The service will automatically start on boot and restart on failures."