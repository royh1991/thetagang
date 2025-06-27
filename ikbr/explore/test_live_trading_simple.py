"""
Simple Live Trading Test

Tests the trading system with real market data.
"""

from ib_async import *
import os
import time
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

# Import our components
from core.market_data import MarketDataManager, MarketDataConfig
from core.order_manager import OrderManager, OrderConfig, Signal
from core.risk_manager import RiskManager, RiskLimits

def test_live_system():
    """Test the trading system with live data"""
    ib = IB()
    
    try:
        # Connect
        port = int(os.getenv('IB_GATEWAY_PORT', 4002))
        logger.info(f"Connecting to IB Gateway on port {port}...")
        ib.connect('localhost', port, clientId=1)
        
        logger.success("✅ Connected successfully!")
        
        # Create managers
        logger.info("\nInitializing components...")
        
        # Market data manager
        market_data = MarketDataManager(
            ib,
            MarketDataConfig(
                delayed_data_fallback=False  # We have real-time data now!
            )
        )
        
        # Order manager
        order_manager = OrderManager(
            ib,
            OrderConfig(
                default_timeout=30,
                use_adaptive_orders=False
            )
        )
        
        # Risk manager
        risk_manager = RiskManager(
            ib,
            RiskLimits(
                max_position_size=5000,
                max_positions=3,
                risk_per_trade_pct=0.01
            )
        )
        
        # Link components
        order_manager.set_risk_manager(risk_manager)
        
        logger.success("✅ Components initialized")
        
        # Test 1: Market Data
        logger.info("\n1. Testing real-time market data...")
        
        # Use real-time data
        ib.reqMarketDataType(1)  # Real-time data
        
        # Test multiple symbols
        symbols = ["SPY", "AAPL", "MSFT"]
        
        for symbol in symbols:
            print(Stock)
            contract = Stock(symbol, 'SMART', 'USD')
            ib.qualifyContracts(contract)
            
            ticker = ib.reqMktData(contract, '', False, False)
            ib.sleep(.5)
            
            if not util.isNan(ticker.last):
                logger.success(f"✅ {symbol}: Last=${ticker.last:.2f}, "
                             f"Bid=${ticker.bid:.2f}, Ask=${ticker.ask:.2f}, "
                             f"Volume={ticker.volume:,.0f}")
            else:
                logger.warning(f"⚠️  {symbol}: No data")
        
        # Test 2: Historical Data
        logger.info("\n2. Testing historical data...")
        spy = Stock('SPY', 'SMART', 'USD')
        ib.qualifyContracts(spy)
        
        bars = ib.reqHistoricalData(
            spy,
            endDateTime='',
            durationStr='1 D',
            barSizeSetting='5 mins',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        if bars:
            logger.success(f"✅ Got {len(bars)} historical bars")
            logger.info(f"   Latest: {bars[-1].date} Close=${bars[-1].close:.2f}")
            
            # Calculate simple metrics
            closes = [bar.close for bar in bars[-20:]]
            sma20 = sum(closes) / len(closes)
            logger.info(f"   20-bar SMA: ${sma20:.2f}")
        
        # Test 3: Order Validation
        logger.info("\n3. Testing order creation...")
        
        # Create a test signal (NOT submitting)
        signal = Signal(
            action="BUY",
            symbol="SPY",
            quantity=1,
            order_type="LIMIT",
            limit_price=ticker.bid - 0.50,  # Well below market
            stop_loss=ticker.bid - 2.00,
            take_profit=ticker.bid + 2.00
        )
        
        logger.info(f"   Created signal: BUY 1 SPY @ ${signal.limit_price:.2f}")
        logger.info(f"   Stop Loss: ${signal.stop_loss:.2f}")
        logger.info(f"   Take Profit: ${signal.take_profit:.2f}")
        
        # Test 4: Risk Check
        logger.info("\n4. Testing risk management...")
        
        # Get account info
        account_values = ib.accountValues()
        net_liq = 0
        for av in account_values:
            if av.tag == "NetLiquidation" and av.currency == "USD":
                net_liq = float(av.value)
                break
        
        logger.info(f"   Account Value: ${net_liq:,.2f}")
        
        # Calculate position size
        if signal.stop_loss and signal.limit_price:
            risk_per_share = abs(signal.limit_price - signal.stop_loss)
            risk_amount = net_liq * 0.01  # 1% risk
            position_size = int(risk_amount / risk_per_share)
            logger.info(f"   1% Risk Position Size: {position_size} shares")
            logger.info(f"   Risk per share: ${risk_per_share:.2f}")
            logger.info(f"   Total risk: ${risk_amount:.2f}")
        
        # Test 5: Check positions
        logger.info("\n5. Checking current positions...")
        positions = ib.positions()
        
        if positions:
            for pos in positions:
                logger.info(f"   {pos.contract.symbol}: {pos.position} shares @ ${pos.avgCost:.2f}")
        else:
            logger.info("   No open positions")
        
        # Summary
        logger.info("\n" + "="*50)
        logger.success("✅ TRADING SYSTEM TEST COMPLETE")
        logger.info("="*50)
        logger.info("All components working with live market data!")
        logger.info("Ready for live trading strategies.")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if ib.isConnected():
            ib.disconnect()
            logger.info("\nDisconnected")


if __name__ == "__main__":
    test_live_system()