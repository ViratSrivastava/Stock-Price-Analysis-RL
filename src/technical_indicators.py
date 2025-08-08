# technical_indicators.py
"""
Technical indicators calculator for stock market analysis
Contains various technical analysis functions used by the DQN model
"""

import pandas as pd
import numpy as np
import talib
import logging
from typing import Optional

class TechnicalIndicators:
    """Technical indicators calculator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators for DQN input"""
        if df.empty or len(df) < 50:  # Need minimum data for indicators
            return df
            
        indicators = df.copy()
        
        try:
            # Price-based indicators
            indicators['sma_5'] = talib.SMA(df['close'], timeperiod=5)
            indicators['sma_10'] = talib.SMA(df['close'], timeperiod=10)
            indicators['sma_20'] = talib.SMA(df['close'], timeperiod=20)
            indicators['sma_50'] = talib.SMA(df['close'], timeperiod=50)
            # Long-term moving averages
            indicators['sma_100'] = talib.SMA(df['close'], timeperiod=100)
            
            indicators['ema_5'] = talib.EMA(df['close'], timeperiod=5)
            indicators['ema_10'] = talib.EMA(df['close'], timeperiod=10)
            indicators['ema_20'] = talib.EMA(df['close'], timeperiod=20)
            indicators['ema_50'] = talib.EMA(df['close'], timeperiod=50)
            
            # Momentum indicators
            indicators['rsi'] = talib.RSI(df['close'], timeperiod=14)
            indicators['rsi_30'] = talib.RSI(df['close'], timeperiod=30)
            
            # MACD
            indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = talib.MACD(df['close'])
            
            # Stochastic Oscillator
            indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
            
            # Williams %R
            indicators['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'])
            
            # Volatility indicators
            indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = talib.BBANDS(df['close'])
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']
            indicators['bb_position'] = (df['close'] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
            # Bollinger %B
            bb_u, bb_m, bb_l = talib.BBANDS(df['close'], timeperiod=20)
            indicators['bb_percentb'] = (df['close'] - bb_l) / (bb_u - bb_l)
            
            indicators['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            indicators['natr'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
            # ATR ratio (normalized volatility)
            indicators['atr_ratio'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14) / df['close']
            
            # Volume indicators
            indicators['obv'] = talib.OBV(df['close'], df['volume'])
            indicators['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
            indicators['adosc'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])
            
            # Volume moving averages
            indicators['volume_sma_10'] = talib.SMA(df['volume'], timeperiod=10)
            indicators['volume_sma_20'] = talib.SMA(df['volume'], timeperiod=20)
            indicators['volume_ratio'] = df['volume'] / indicators['volume_sma_20']
            
            # Price patterns and momentum
            indicators['price_change'] = df['close'].pct_change()
            indicators['price_change_5'] = df['close'].pct_change(5)
            indicators['price_momentum'] = df['close'] / df['close'].shift(10) - 1
            # Multi-horizon returns
            indicators['return_10d'] = df['close'].pct_change(10)
            indicators['return_30d'] = df['close'].pct_change(30)
            
            indicators['high_low_ratio'] = df['high'] / df['low']
            indicators['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # Trend indicators
            indicators['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            indicators['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Additional patterns
            indicators['doji'] = TechnicalIndicators._calculate_doji(df)
            indicators['hammer'] = TechnicalIndicators._calculate_hammer(df)
            indicators['gap'] = TechnicalIndicators._calculate_gaps(df)
            
            # Volatility measures
            indicators['true_range'] = talib.TRANGE(df['high'], df['low'], df['close'])
            indicators['volatility'] = indicators['price_change'].rolling(window=20).std()
            
            # Support and resistance levels
            indicators['resistance'] = df['high'].rolling(window=20).max()
            indicators['support'] = df['low'].rolling(window=20).min()
            indicators['resistance_distance'] = (indicators['resistance'] - df['close']) / df['close']
            indicators['support_distance'] = (df['close'] - indicators['support']) / df['close']
            
            # Volume-weighted average price (VWAP) approximation
            cum_vol_price = (df['close'] * df['volume']).cumsum()
            cum_vol = df['volume'].cumsum()
            indicators['vwap'] = cum_vol_price / cum_vol
            
        except Exception as e:
            logging.error(f"Error calculating indicators: {e}")
            
        return indicators.fillna(0)

    @staticmethod
    def get_feature_columns() -> list:
        """Get list of feature columns for model training"""
        return [
            'open', 'high', 'low', 'close', 'volume',
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100',
            'ema_5', 'ema_10', 'ema_20', 'ema_50',
            'rsi', 'rsi_30',
            'macd', 'macd_signal', 'macd_hist',
            'stoch_k', 'stoch_d', 'williams_r',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position', 'bb_percentb',
            'atr', 'natr', 'atr_ratio',
            'obv', 'ad', 'adosc',
            'volume_sma_10', 'volume_sma_20', 'volume_ratio',
            'price_change', 'price_change_5', 'price_momentum',
            'return_10d', 'return_30d',
            'high_low_ratio', 'close_position',
            'adx', 'cci',
            'doji', 'hammer', 'gap',
            'true_range', 'volatility',
            'resistance_distance', 'support_distance',
            'vwap'
        ]
    
    @staticmethod
    def _calculate_doji(df: pd.DataFrame) -> pd.Series:
        """Calculate Doji candlestick pattern"""
        body = abs(df['close'] - df['open'])
        wick = df['high'] - df['low']
        return (body / wick < 0.1).astype(int)
    
    @staticmethod
    def _calculate_hammer(df: pd.DataFrame) -> pd.Series:
        """Calculate Hammer candlestick pattern"""
        body = abs(df['close'] - df['open'])
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        
        hammer_condition = (
            (lower_wick > 2 * body) & 
            (upper_wick < 0.1 * body) &
            (body > 0)
        )
        return hammer_condition.astype(int)
    
    @staticmethod
    def _calculate_gaps(df: pd.DataFrame) -> pd.Series:
        """Calculate price gaps"""
        gap_up = (df['open'] > df['high'].shift(1)).astype(int)
        gap_down = (df['open'] < df['low'].shift(1)).astype(int)
        return gap_up - gap_down
    
    @staticmethod
    def prepare_features_for_model(df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model input with proper selection and cleaning"""
        feature_cols = TechnicalIndicators.get_feature_columns()
        
        # Select only available features
        available_cols = [col for col in feature_cols if col in df.columns]
        features_df = df[available_cols].copy()
        
        # Handle infinite values
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        # Fill any remaining NaN values
        features_df = features_df.fillna(0)
        
        return features_df