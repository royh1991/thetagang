# Deployment Guide for Google Cloud VM

## Option 1: SSH Key + Git Clone (Recommended)

### 1. Set up SSH key on your local machine (if you don't have one)
```bash
ssh-keygen -t ed25519 -C "royhu91@gmail.com"
```

### 2. Add your SSH public key to GitHub
- Copy your public key: `cat ~/.ssh/id_ed25519.pub`
- Go to GitHub Settings → SSH and GPG keys → New SSH key
- Paste the key and save

### 3. SSH into your Google VM
```bash
gcloud compute ssh YOUR_VM_NAME --zone=YOUR_ZONE
# or
ssh username@VM_EXTERNAL_IP
```

### 4. On the VM, generate an SSH key for GitHub access
```bash
ssh-keygen -t ed25519 -C "vm-deploy-key"
cat ~/.ssh/id_ed25519.pub
```

  sudo rm /var/lib/dpkg/lock-frontend
  sudo rm /var/lib/dpkg/lock
  sudo rm /var/cache/apt/archives/lock
### 5. Add the VM's public key to your GitHub repo
- Go to your repo → Settings → Deploy keys
- Add new deploy key (read-only is fine)
- Paste the VM's public key

### 6. Clone the repository on the VM
```bash
git clone git@github.com:YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO/ikbr
```

## Option 2: Direct Transfer via SCP

If the repo is private or you want a quick one-time transfer:

```bash
# From your local machine
cd /Users/rhu/thetagang
tar -czf ikbr.tar.gz ikbr/

# Transfer to VM
gcloud compute scp ikbr.tar.gz YOUR_VM_NAME:~/ --zone=YOUR_ZONE
# or
scp ikbr.tar.gz username@VM_EXTERNAL_IP:~/

# On the VM
tar -xzf ikbr.tar.gz
cd ikbr/
```

## Setting Up the Environment on the VM

### 1. Install Python and dependencies
```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python 3.11+ and pip
sudo apt-get install -y python3.11 python3.11-venv python3-pip

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Install Docker for IB Gateway
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Log out and back in for group changes

# Start IB Gateway
docker-compose up -d
```

### 3. Configure IB Gateway Connection
```bash
# Create .env file for configuration
cat > .env << EOF
IB_GATEWAY_HOST=localhost
IB_GATEWAY_PORT=4102
IB_GATEWAY_PORT_LIVE=4101
IB_CLIENT_ID=99
EOF
```

### 4. Test the connection
```bash
python backtest2/test_live_connection.py
```

## Running on the VM

### For Backtesting
```bash
python backtest2/run_backtest.py --symbol TSLA --days 100 --strategy nick
```

### For Paper Trading
```bash
# Run in screen/tmux for persistence
screen -S trading
python backtest2/main.py --symbol TSLA --strategy nick

# Detach with Ctrl+A, D
# Reattach with: screen -r trading
```

### For Production Trading
```bash
python backtest2/main.py --symbol TSLA --strategy nick --trading-mode live
```

## Monitoring and Logs

- Backtest results: `backtest2/runs/`
- Live trading logs: `backtest2/logs/`
- IB Gateway logs: `docker-compose logs ib-gateway`

docker compose logs -f ib-gateway


## Security Considerations

1. **Firewall Rules**: Only allow SSH (port 22) from your IP
2. **IB Gateway Ports**: Keep 4101/4102 closed to external traffic
3. **Use strong passwords** for IB Gateway
4. **Monitor resource usage**: Trading can be CPU/memory intensive
5. **Set up alerts** for errors or unexpected behavior

## Automated Startup (Optional)

Create a systemd service for automatic startup:

```bash
sudo nano /etc/systemd/system/trading-bot.service
```

```ini
[Unit]
Description=Trading Bot
After=network.target docker.service

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/ikbr
Environment="PATH=/home/YOUR_USERNAME/ikbr/venv/bin"
ExecStart=/home/YOUR_USERNAME/ikbr/venv/bin/python /home/YOUR_USERNAME/ikbr/backtest2/main.py --symbol TSLA --strategy nick
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable trading-bot
sudo systemctl start trading-bot
sudo systemctl status trading-bot
```