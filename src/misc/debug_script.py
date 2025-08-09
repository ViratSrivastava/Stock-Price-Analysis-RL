#!/usr/bin/env python
"""
Debug script to investigate training issues
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from misc.api_connector import AlphaVantageAPI
from trading_environment import StockTradingEnvironment
from data_processor import prepare_data
from debug_utils import validate_dataset, debug_step_execution

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    """Main debug function"""
    symbol = "AAPL"
    logger.info(f"Starting debug process for {symbol}")
    
    try:
        # 1. Test API connection
        api = AlphaVantageAPI()
        logger.info("Testing API connection...")
        if api.test_connection():
            logger.info("✓ API connection successful")
        else:
            logger.error("✗ API connection failed")
            return
        
        # 2. Get data and validate
        logger.info(f"Fetching data for {symbol}...")
        data = api.get_intraday_data(symbol, interval='5min', outputsize='full')
        
        if data is None or data.empty:
            logger.error("No data received from API")
            return
            
        logger.info(f"Received {len(data)} data points")
        
        # 3. Validate raw data
        logger.info("Validating raw data...")
        raw_stats = validate_dataset(data, logger)
        
        # 4. Prepare data for training
        logger.info("Preparing data for training...")
        try:
            from data_processor import prepare_data
            processed_data = prepare_data(data)
            logger.info(f"Data prepared with shape: {processed_data.shape}")
            
            # Validate processed data
            processed_stats = validate_dataset(processed_data, logger)
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return
            
        # 5. Test environment step execution
        logger.info("Testing environment step execution...")
        env = StockTradingEnvironment(processed_data)
        debug_step_execution(env, n_steps=20, logger=logger)
        
        logger.info("Debug process completed successfully")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
