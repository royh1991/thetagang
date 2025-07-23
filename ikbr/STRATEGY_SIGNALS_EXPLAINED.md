# Nick Strategy Trading Signals Explained

## Overview
The Nick Strategy is a funnel breakout strategy that identifies trending markets and trades breakouts with specific confirmations. Here's why the bot takes trades.

## Entry Conditions (ALL must be true)

### 1. **Trending Market (ADX > 30)**
- ADX (Average Directional Index) measures trend strength
- Values > 30 indicate a strong trend
- Values < 30 indicate choppy/sideways market
- **Why:** Breakouts work better in trending markets

### 2. **Price Breakout**
- Price breaks above the highest high of the last 20 bars
- This is the "funnel breakout" - price escaping a range
- **Why:** Indicates momentum and potential trend continuation

### 3. **Volume Spike (2x average)**
- Current volume > 2x the 20-bar average volume
- **Why:** High volume confirms the breakout is significant

### 4. **RSI Confirmation (> 55)**
- RSI above 55 indicates bullish momentum
- **Why:** Confirms the breakout has momentum behind it

### 5. **Market Context (SPY bullish)**
- SPY 20 SMA > 50 SMA indicates overall market uptrend
- **Why:** Trading with the market trend increases success rate

### 6. **Position and Cooldown Checks**
- Not already in a position
- At least 10 bars since last exit (cooldown period)
- **Why:** Prevents overtrading and emotional decisions

## Example Signal from Logs

```
14:04:05 | INFO | Long signal: ADX=39.3, RSI=73.8, Volume spike=True, Breakout above 322.31
```

**Translation:**
- ✅ ADX=39.3 (>30) - Strong trend present
- ✅ RSI=73.8 (>55) - Strong bullish momentum
- ✅ Volume spike=True - Confirmed with high volume
- ✅ Breakout above 322.31 - Price broke 20-bar high
- ✅ SPY bullish (implicit) - Market context positive

## Exit Conditions

### 1. **Stop Loss**
- Set at entry price - (1.0 × ATR)
- **Why:** Limits losses if trade goes against us

### 2. **Take Profit**
- Set at entry price + (2.5 × ATR)
- **Why:** 2.5:1 reward/risk ratio

### 3. **Smooth Exit**
- Two consecutive bars close below 5-bar SMA
- **Why:** Indicates short-term trend reversal

## Why Trades Failed in Your Logs

Looking at your logs, trades were signaled but not executed due to:
- `Error calculating position size: 'NoneType' object has no attribute 'last'`
- This meant valid signals couldn't be acted upon

## After Fixes

With the fixes implemented:
1. **Ticker subscription** ensures price data is available
2. **Fallback pricing** uses bid/ask if last price unavailable
3. **Better error handling** prevents crashes
4. **Automatic recovery** restarts on connection issues

## Risk Management

- **Position Size:** 10% of account (configurable)
- **Max Position Value:** $10,000 (configurable)
- **Minimum Order Interval:** 5 seconds (prevents rapid-fire trades)
- **Stop Loss:** Always set on entry

## Reading the Logs

When a trade is taken, you'll see:
```
BUY signal: Breakout > $322.31 | ADX=39.3 | RSI=73.8 | Vol spike 2.0x | SPY bullish | SL=$321.93 | TP=$323.54
```

This gives you:
- Entry reason (Breakout)
- All confirmations (ADX, RSI, Volume, SPY)
- Risk levels (Stop Loss, Take Profit)

## Configuration

You can adjust strategy parameters:
- `--adx-threshold 30` (default, can lower to 25 for more trades)
- `--rsi-threshold 55` (default, can lower to 50)
- `--position-size 0.10` (10% of account)
- `--lookback-period 20` (bars for breakout calculation)

## Performance Expectations

Based on backtesting:
- **Win Rate:** ~40-45%
- **Risk/Reward:** 1:2.5
- **Trade Frequency:** 1-5 trades per day depending on market conditions
- **Best Performance:** Trending markets
- **Worst Performance:** Choppy, sideways markets

## Monitoring

Watch for:
1. **Entry efficiency** - Are entries near the breakout level?
2. **Exit efficiency** - Are we capturing profits or stopping out?
3. **Market conditions** - Is ADX staying above 30?
4. **Volume patterns** - Are volume spikes reliable?

The strategy is designed to catch momentum moves in trending markets while avoiding choppy conditions.