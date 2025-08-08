# main.py

"""
Main entry point for the Stock Trading DQN System
CSV-based version (no API required)
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
import argparse
from training import ModelTrainer
from validation import ModelValidator

def configure_logging():
    """Configure logging to both file and console"""
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = f"logs/run_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(logfile, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logging.getLogger().info(f"Logging to {logfile}")

class StockTradingSystem:
    """Main system orchestrator for CSV-based training"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.trainer = ModelTrainer(data_dir=data_dir)
        self.validator = ModelValidator(data_dir=data_dir)
        self.logger = logging.getLogger(__name__)

    def list_available_symbols(self):
        """List available symbols from CSV files"""
        if not os.path.exists(self.data_dir):
            self.logger.error(f"Data directory not found: {self.data_dir}")
            return []

        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('_daily.csv')]
        symbols = [f.replace('_daily.csv', '') for f in csv_files]

        self.logger.info(f"Available symbols: {symbols}")
        return symbols

    def train(self, symbol: str, episodes: int, freq: str = "daily"):
        self.logger.info(f"🚀 Starting training for {symbol} ({episodes} episodes) on {freq}")
        suffix = "daily" if freq == "daily" else "intraday_5m"
        csv_path = os.path.join(self.data_dir, f"{symbol}_{suffix}.csv")

        # Check if CSV file exists
        csv_path = os.path.join(self.data_dir, f"{symbol}_daily.csv")
        if not os.path.exists(csv_path):
            self.logger.error(f"❌ CSV file not found: {csv_path}")
            available_symbols = self.list_available_symbols()
            if available_symbols:
                self.logger.info(f"Available symbols: {available_symbols}")
            return

        try:
            metrics = self.trainer.train_model(symbol, episodes)
            self.logger.info(f"✅ Training completed for {symbol}")
            self.logger.info(f"📊 Final metrics: {metrics}")
            self.trainer.plot_training_progress(symbol)

        except Exception as e:
            self.logger.error(f"❌ Training failed for {symbol}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def validate(self, symbol: str):
        """Backtest a trained model"""
        self.logger.info(f"🔍 Validating model for {symbol}")

        try:
            backtest = self.validator.backtest_model(symbol)
            self.logger.info(f"📈 Backtest results: {backtest}")

        except Exception as e:
            self.logger.error(f"❌ Validation failed: {e}")

    def _show_csv_info(self, symbol: str):
        """Show information about CSV data for a symbol"""
        try:
            data = self.trainer.load_csv_data(symbol)
            if data is not None:
                self.logger.info(f"📊 Data: {len(data)} rows from {data.index.min()} to {data.index.max()}")
                self.logger.info(f"📈 Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        except Exception as e:
            self.logger.error(f"Error reading CSV: {e}")

    def _show_model_info(self, symbol: str):
        """Show information about trained models for a symbol"""
        if not os.path.exists("models"):
            return
            
        model_files = [f for f in os.listdir("models") if f.startswith(symbol) and f.endswith('.pth')]
        if model_files:
            self.logger.info(f"🤖 Available models: {len(model_files)}")
            for model in sorted(model_files)[-3:]:
                self.logger.info(f"  • {model}")
        else:
            self.logger.info("🤖 No trained models found")

    def _show_all_symbols(self):
        """Show information about all available symbols"""
        self.logger.info("📁 System Information")
        available_symbols = self.list_available_symbols()

        if not available_symbols:
            self.logger.error("❌ No CSV data files found in data/ directory")
            self.logger.info("Expected format: data/SYMBOL_daily.csv")
            return

        self.logger.info(f"📊 Available data files: {len(available_symbols)}")
        for sym in available_symbols:
            self.logger.info(f"  • {sym}")

    def info(self, symbol: str = None):
        """Show information about available data and models"""
        if not symbol:
            self._show_all_symbols()
            return

        self.logger.info(f"📁 Information for {symbol}")

        csv_path = os.path.join(self.data_dir, f"{symbol}_daily.csv")

        if os.path.exists(csv_path):
            self._show_csv_info(symbol)    # ❌ This line is wrong
            self._show_model_info(symbol)
        else:
            self.logger.error(f"❌ CSV file not found: {csv_path}")

    def train_all(self, episodes: int = 1000):
        """Train models for all available symbols"""
        available_symbols = self.list_available_symbols()

        if not available_symbols:
            self.logger.error("❌ No symbols found to train")
            return

        self.logger.info(f"🚀 Starting training for all symbols: {available_symbols}")

        for symbol in available_symbols:
            self.logger.info(f"📈 Training {symbol}...")
            self.train(symbol, episodes)
            self.logger.info(f"✅ Completed {symbol}")

def main():
    configure_logging()
    
    parser = argparse.ArgumentParser(description="Stock Trading DQN System (CSV-based)")
    parser.add_argument("command", choices=["train", "validate", "info", "train-all"],
                        help="Operation to perform")
    parser.add_argument("--symbol", "-s", type=str, default="AAPL",
                        help="Stock symbol (default: AAPL)")
    parser.add_argument("--episodes", "-e", type=int, default=1000,
                        help="Training episodes (default: 1000)")
    parser.add_argument("--freq", "-f", choices=["daily", "intraday"], default="daily",
                        help="Data frequency to use: daily or intraday (default: daily)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing CSV files (default: data)")
    args = parser.parse_args()

    system = StockTradingSystem(data_dir=args.data_dir)

    try:
        if args.command == "train":
            system.train(args.symbol, args.episodes, args.freq)
        elif args.command == "train-all":
            system.train_all(args.episodes)
        elif args.command == "validate":
            system.validate(args.symbol)
        elif args.command == "info":
            system.info(args.symbol if args.symbol != "AAPL" else None)  # Don't default to AAPL for info

    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
