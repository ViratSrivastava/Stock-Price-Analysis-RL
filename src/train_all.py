#!/usr/bin/env python3

"""
Simple training script for CSV-based stock trading DQN
Run this to train models on all available stocks
"""

import os
import sys
import logging
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training import ModelTrainer
from data_processor import StockDataProcessor

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )

def main():
    """Main training function"""
    print("🚀 Stock Trading DQN - CSV Training Script")
    print("=" * 50)

    setup_logging()
    logger = logging.getLogger(__name__)

    # Initialize components
    trainer = ModelTrainer()
    processor = StockDataProcessor()

    # Check available data
    logger.info("📊 Checking available data...")
    validation_results = processor.validate_all_data()

    available_symbols = []
    for symbol, result in validation_results.items():
        if 'error' not in result:
            available_symbols.append(symbol)
            logger.info(f"✅ {symbol}: {result['clean_rows']} rows "
                       f"({result['date_range']['start']} to {result['date_range']['end']})")
        else:
            logger.error(f"❌ {symbol}: {result['error']}")

    if not available_symbols:
        logger.error("❌ No valid data files found!")
        print("\nMake sure your CSV files are in the 'data/' directory")
        print("Expected format: data/SYMBOL_daily.csv")
        return

    print(f"\n📈 Found {len(available_symbols)} symbols to train: {available_symbols}")

    # Training configuration
    episodes = 500  # Reduced for testing

    # Train each symbol
    for i, symbol in enumerate(available_symbols, 1):
        print(f"\n🎯 Training {symbol} ({i}/{len(available_symbols)})")
        print("-" * 30)

        try:
            # Train model
            metrics = trainer.train_model(symbol, episodes=episodes)

            if 'error' in metrics:
                logger.error(f"❌ Training failed for {symbol}: {metrics['error']}")
                continue

            # Log results
            logger.info(f"✅ Training completed for {symbol}")
            logger.info(f"   Final Reward: {metrics.get('final_reward', 0):.4f}")
            logger.info(f"   Best Reward: {metrics.get('best_reward', 0):.4f}")
            logger.info(f"   Avg Profit: {metrics.get('avg_profit', 0):.4f}")
            logger.info(f"   Training Time: {metrics.get('training_time', 0):.1f}s")

            # Generate training plot
            try:
                trainer.plot_training_progress(symbol)
                logger.info(f"📊 Training plot saved for {symbol}")
            except Exception as e:
                logger.warning(f"⚠️ Could not generate plot for {symbol}: {e}")

        except KeyboardInterrupt:
            logger.info("🛑 Training interrupted by user")
            break
        except Exception as e:
            logger.error(f"❌ Unexpected error training {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    print("\n🎉 Training session completed!")
    print("\nTrained models saved in: models/")
    print("Training logs saved in: logs/")

    # Summary
    model_files = []
    if os.path.exists("models"):
        model_files = [f for f in os.listdir("models") if f.endswith('.pth')]

    print(f"\n📁 {len(model_files)} model files created")

    # Show how to validate
    print("\n🔍 To validate your models, run:")
    for symbol in available_symbols:
        print(f"  python main.py validate --symbol {symbol}")

if __name__ == "__main__":
    main()
