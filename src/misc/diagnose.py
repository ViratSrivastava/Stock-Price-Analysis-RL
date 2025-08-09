#!/usr/bin/env python3
"""
Comprehensive diagnosis script for the Stock Trading DQN System
"""

import os
import sys
import logging
import time
from datetime import datetime

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'diagnosis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

def test_environment():
    """Test basic environment"""
    logger.info("=== Testing Environment ===")

    # Check Python version
    logger.info(f"Python version: {sys.version}")

    # Check current directory
    logger.info(f"Current directory: {os.getcwd()}")

    # Check required files
    required_files = [
        'src/api_connector.py',
        'src/training.py',
        'src/main.py',
        '.env'
    ]

    for file in required_files:
        if os.path.exists(file):
            logger.info(f"✅ {file} exists")
        else:
            logger.error(f"❌ {file} missing")

    # Check .env file content
    if os.path.exists('.env'):
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            if api_key:
                logger.info(f"✅ API key found: {api_key[:10]}...")
            else:
                logger.error("❌ API key not found in .env")
        except Exception as e:
            logger.error(f"❌ Error loading .env: {e}")

def test_imports():
    """Test all imports"""
    logger.info("=== Testing Imports ===")

    required_packages = [
        'torch',
        'pandas', 
        'numpy',
        'requests',
        'sklearn',
        'talib',
        'gym',
        'matplotlib'
    ]

    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} imported successfully")
        except ImportError as e:
            logger.error(f"❌ {package} import failed: {e}")

def test_api_connection():
    """Test API connection step by step"""
    logger.info("=== Testing API Connection ===")

    try:
        # Add src to path
        sys.path.insert(0, 'src')

        # Test improved API connector first
        logger.info("Testing improved API connector...")
        from misc.api_connector_fixed import AlphaVantageAPI as ImprovedAPI

        api = ImprovedAPI()

        # Test quote
        logger.info("Testing quote...")
        quote = api.get_quote("AAPL")
        if quote:
            logger.info(f"✅ Quote successful: {quote}")
        else:
            logger.error("❌ Quote failed")
            return False

        # Test compact intraday data
        logger.info("Testing compact intraday data...")
        data = api.get_intraday_data("AAPL", interval='5min', outputsize='compact')
        if data is not None and len(data) > 0:
            logger.info(f"✅ Compact data successful: {len(data)} rows")
            logger.info(f"Data columns: {list(data.columns)}")
            logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
            logger.info(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
            return True
        else:
            logger.error("❌ Compact data failed")
            return False

    except Exception as e:
        logger.error(f"❌ API test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_model_components():
    """Test model components"""
    logger.info("=== Testing Model Components ===")

    try:
        sys.path.insert(0, 'src')

        # Test technical indicators
        logger.info("Testing technical indicators...")
        from technical_indicators import TechnicalIndicators
        logger.info("✅ Technical indicators imported")

        # Test trading environment with dummy data
        logger.info("Testing trading environment...")
        import pandas as pd
        import numpy as np

        # Create dummy data
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        dummy_data = pd.DataFrame({
            'open': np.random.uniform(150, 200, 100),
            'high': np.random.uniform(155, 205, 100),
            'low': np.random.uniform(145, 195, 100),
            'close': np.random.uniform(150, 200, 100),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)

        from trading_environment import StockTradingEnvironment
        env = StockTradingEnvironment(dummy_data)
        logger.info("✅ Trading environment created")

        # Test a few steps
        state = env.reset()
        logger.info(f"✅ Environment reset, state shape: {state.shape}")

        for i in range(3):
            action = i % 3  # Test each action
            next_state, reward, done, info = env.step(action)
            logger.info(f"✅ Step {i}: action={action}, reward={reward:.4f}")

        return True

    except Exception as e:
        logger.error(f"❌ Model components test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_gpu():
    """Test GPU availability"""
    logger.info("=== Testing GPU ===")

    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

            # Test simple tensor operation
            x = torch.randn(10, 10).cuda()
            y = x @ x.T
            logger.info("✅ GPU tensor operations working")
            return True
        else:
            logger.warning("⚠️ CUDA not available, will use CPU")
            return False
    except Exception as e:
        logger.error(f"❌ GPU test failed: {e}")
        return False

def main():
    """Run all diagnostics"""
    logger.info("🔍 Starting comprehensive diagnosis...")

    results = {
        'environment': test_environment(),
        'imports': test_imports(),  
        'api': test_api_connection(),
        'components': test_model_components(),
        'gpu': test_gpu()
    }

    logger.info("=== DIAGNOSIS SUMMARY ===")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name.upper()}: {status}")

    if all(results.values()):
        logger.info("🎉 All tests passed! System should work.")
    else:
        logger.error("❌ Some tests failed. Check logs above.")

    return results

if __name__ == "__main__":
    main()
