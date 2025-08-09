# api_connector.py
"""
Alpha Vantage API connector for real-time stock data
Handles all external API communications
"""

import requests
import pandas as pd
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

class AlphaVantageAPI:
    """Alpha Vantage API wrapper for real-time stock data"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12  # seconds between requests for free tier
        self.last_request_time = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _wait_for_rate_limit(self):
        """Ensure we don't exceed API rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            self.logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
        
    def get_intraday_data(self, symbol: str, interval: str = '5min', 
                         outputsize: str = 'compact') -> Optional[pd.DataFrame]:
        """Get intraday stock data"""
        self._wait_for_rate_limit()
        
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'apikey': self.api_key,
            'outputsize': outputsize
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                self.logger.error(f"API Error: {data['Error Message']}")
                return None
                
            if 'Note' in data:
                self.logger.warning(f"API Note: {data['Note']}")
                return None
            
            if f'Time Series ({interval})' in data:
                time_series = data[f'Time Series ({interval})']
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                df = df.astype(float)
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                
                # Remove any zero-price rows (invalid data)
                df = df[df['close'] > 0]
                
                self.logger.info(f"Successfully fetched {len(df)} data points for {symbol}")
                return df
            else:
                self.logger.error(f"Unexpected response format: {list(data.keys())}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote"""
        self._wait_for_rate_limit()
        
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if 'Global Quote' in data:
                quote = data['Global Quote']
                return {
                    'symbol': quote['01. symbol'],
                    'price': float(quote['05. price']),
                    'change': float(quote['09. change']),
                    'change_percent': quote['10. change percent'],
                    'timestamp': datetime.now()
                }
            return None
        except Exception as e:
            self.logger.error(f"Error fetching quote for {symbol}: {e}")
            return None

    def get_daily_data(self, symbol: str, outputsize: str = 'compact') -> Optional[pd.DataFrame]:
        """Get daily stock data"""
        self._wait_for_rate_limit()
        
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': outputsize
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                time_series = data['Time Series (Daily)']
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                df = df.astype({'open': float, 'high': float, 'low': float, 
                              'close': float, 'volume': int})
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                return df
            return None
        except Exception as e:
            self.logger.error(f"Error fetching daily data for {symbol}: {e}")
            return None

    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            quote = self.get_quote('AAPL')
            return quote is not None
        except (requests.RequestException, ValueError) as e:
            self.logger.error(f"Connection test failed: {e}")
            return False