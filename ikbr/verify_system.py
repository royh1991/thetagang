"""
System Verification Script

Verifies all components are working correctly without requiring IB Gateway.
"""

import asyncio
import sys
import importlib
from pathlib import Path
from loguru import logger

sys.path.insert(0, '.')


class SystemVerifier:
    """Verify trading system components"""
    
    def __init__(self):
        self.results = {
            'imports': {},
            'unit_tests': {},
            'components': {},
            'strategies': {}
        }
    
    async def verify_all(self):
        """Run all verification checks"""
        logger.info("Starting system verification...")
        
        # 1. Check imports
        self.verify_imports()
        
        # 2. Run unit tests
        await self.run_unit_tests()
        
        # 3. Test component initialization
        await self.test_components()
        
        # 4. Test strategies
        await self.test_strategies()
        
        # Print results
        self.print_results()
    
    def verify_imports(self):
        """Verify all modules can be imported"""
        logger.info("\nVerifying imports...")
        
        modules = [
            'core.event_bus',
            'core.market_data', 
            'core.order_manager',
            'core.risk_manager',
            'strategies.base_strategy',
            'strategies.examples.momentum_strategy',
            'strategies.examples.mean_reversion_strategy',
            'backtest.engine',
            'backtest.data_provider',
            'backtest.mock_broker'
        ]
        
        for module in modules:
            try:
                importlib.import_module(module)
                self.results['imports'][module] = True
                logger.success(f"✓ {module}")
            except Exception as e:
                self.results['imports'][module] = False
                logger.error(f"✗ {module}: {e}")
    
    async def run_unit_tests(self):
        """Run unit tests"""
        logger.info("\nRunning unit tests...")
        
        test_files = [
            'tests/unit/test_event_bus.py',
            'tests/unit/test_market_data.py',
            'tests/unit/test_order_manager.py',
            'tests/unit/test_risk_manager.py'
        ]
        
        for test_file in test_files:
            if Path(test_file).exists():
                try:
                    # Run pytest programmatically
                    import subprocess
                    result = subprocess.run(
                        [sys.executable, '-m', 'pytest', test_file, '-v', '--tb=short'],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        self.results['unit_tests'][test_file] = True
                        logger.success(f"✓ {test_file}")
                    else:
                        self.results['unit_tests'][test_file] = False
                        logger.error(f"✗ {test_file}")
                        if result.stderr:
                            logger.debug(result.stderr)
                            
                except Exception as e:
                    self.results['unit_tests'][test_file] = False
                    logger.error(f"✗ {test_file}: {e}")
            else:
                logger.warning(f"Test file not found: {test_file}")
    
    async def test_components(self):
        """Test component initialization"""
        logger.info("\nTesting components...")
        
        try:
            from core.event_bus import EventBus, Event, EventTypes
            
            # Test event bus
            bus = EventBus("test")
            await bus.start()
            
            # Test event emission
            received = []
            async def handler(event):
                received.append(event)
            
            bus.subscribe("test.event", handler)
            await bus.emit(Event("test.event", {"data": "test"}))
            await asyncio.sleep(0.1)
            
            if received:
                self.results['components']['event_bus'] = True
                logger.success("✓ Event Bus")
            else:
                self.results['components']['event_bus'] = False
                logger.error("✗ Event Bus - no events received")
            
            await bus.stop()
            
        except Exception as e:
            self.results['components']['event_bus'] = False
            logger.error(f"✗ Event Bus: {e}")
        
        # Test other components with mocks
        try:
            from unittest.mock import Mock
            from core.market_data import MarketDataManager, TickBuffer
            from core.order_manager import OrderManager, Signal
            from core.risk_manager import RiskManager, RiskLimits
            
            # Test tick buffer
            buffer = TickBuffer(size=100)
            buffer.add(1.0, 100.0, 1000)
            buffer.add(2.0, 101.0, 1100)
            timestamps, prices, volumes = buffer.get_recent(2)
            
            if len(prices) == 2:
                self.results['components']['tick_buffer'] = True
                logger.success("✓ Tick Buffer")
            else:
                self.results['components']['tick_buffer'] = False
                logger.error("✗ Tick Buffer")
            
            # Test signal validation
            from core.order_manager import validate_signal
            valid_signal = Signal("BUY", "SPY", 100, "MARKET")
            if validate_signal(valid_signal):
                self.results['components']['signal_validation'] = True
                logger.success("✓ Signal Validation")
            else:
                self.results['components']['signal_validation'] = False
                logger.error("✗ Signal Validation")
            
        except Exception as e:
            logger.error(f"Component test error: {e}")
    
    async def test_strategies(self):
        """Test strategy initialization"""
        logger.info("\nTesting strategies...")
        
        try:
            from unittest.mock import Mock
            from strategies.examples.momentum_strategy import MomentumStrategy, MomentumConfig
            from strategies.examples.mean_reversion_strategy import MeanReversionStrategy, MeanReversionConfig
            
            # Create mocks
            mock_market_data = Mock()
            mock_order_manager = Mock()
            mock_risk_manager = Mock()
            
            # Test momentum strategy
            momentum_config = MomentumConfig(["SPY", "AAPL"])
            momentum = MomentumStrategy(
                momentum_config,
                mock_market_data,
                mock_order_manager,
                mock_risk_manager
            )
            
            if momentum.config.name == "Momentum":
                self.results['strategies']['momentum'] = True
                logger.success("✓ Momentum Strategy")
            
            # Test mean reversion strategy
            mr_config = MeanReversionConfig(["SPY"])
            mean_rev = MeanReversionStrategy(
                mr_config,
                mock_market_data,
                mock_order_manager,
                mock_risk_manager
            )
            
            if mean_rev.config.name == "MeanReversion":
                self.results['strategies']['mean_reversion'] = True
                logger.success("✓ Mean Reversion Strategy")
                
        except Exception as e:
            logger.error(f"Strategy test error: {e}")
    
    def print_results(self):
        """Print verification results"""
        logger.info("\n" + "="*60)
        logger.info("SYSTEM VERIFICATION RESULTS")
        logger.info("="*60)
        
        # Count totals
        total_checks = 0
        passed_checks = 0
        
        for category, results in self.results.items():
            logger.info(f"\n{category.upper().replace('_', ' ')}:")
            for item, passed in results.items():
                total_checks += 1
                if passed:
                    passed_checks += 1
                status = "✓ PASS" if passed else "✗ FAIL"
                logger.info(f"  {item}: {status}")
        
        logger.info("\n" + "-"*60)
        success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        logger.info(f"Total: {passed_checks}/{total_checks} checks passed ({success_rate:.1f}%)")
        
        if passed_checks == total_checks:
            logger.success("\n✓ All systems operational!")
        else:
            logger.warning(f"\n⚠ {total_checks - passed_checks} checks failed")
        
        logger.info("="*60)


async def main():
    """Run system verification"""
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    verifier = SystemVerifier()
    await verifier.verify_all()
    
    logger.info("\nTo run paper trading test:")
    logger.info("1. Start IB Gateway: docker-compose up -d")
    logger.info("2. Run: python test_paper_trading.py")
    
    logger.info("\nTo run backtest:")
    logger.info("Run: python run_backtest.py")


if __name__ == "__main__":
    asyncio.run(main())