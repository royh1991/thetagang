#!/usr/bin/env python3
"""Quick backtest runner"""

import subprocess
import sys

# Run the backtest with 30 days of data
cmd = [
    sys.executable,
    "run_backtest.py",
    "--mode", "backtest",
    "--strategy", "enhanced_momentum",
    "--symbols", "AAPL", "TSLA", "NVDA",
    "--days", "30"
]

print("Running backtest with command:")
print(" ".join(cmd))
print("="*60)

# Run the command
result = subprocess.run(cmd, cwd="/Users/rhu/thetagang/ikbr")

sys.exit(result.returncode)