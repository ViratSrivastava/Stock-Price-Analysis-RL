
import os
import sys
import logging
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from misc.api_connector import AlphaVantageAPI
from training import ModelTrainer

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

def minimal_train():
    """Minimal training test"""
    try:
        logger.info("=== Starting Minimal Training Test ===")

        # Load API key
        load_dotenv()
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        logger.info(f"API key loaded: {api_key[:10] if api_key else 'None'}...")

        # Test API connection first
        logger.info("Testing API connection...")
        api = AlphaVantageAPI(api_key)

        # Try a quick quote test
        logger.info("Testing quote...")
        quote = api.get_quote("AAPL")
        if quote:
            logger.info(f"✅ Quote successful: ${quote['price']}")
        else:
            logger.error("❌ Quote failed - stopping")
            return

        # Try minimal data fetch
        logger.info("Testing data fetch...")
        data = api.get_intraday_data("AAPL", interval='5min', outputsize='compact')
        if data is not None and len(data) > 0:
            logger.info(f"✅ Data fetch successful: {len(data)} rows")
            logger.info(f"Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
        else:
            logger.error("❌ Data fetch failed - stopping")
            return

        # Try creating trainer
        logger.info("Creating trainer...")
        trainer = ModelTrainer(api_key)
        logger.info("✅ Trainer created successfully")

        # Try preparing training data
        logger.info("Preparing training data...")
        training_data = trainer.prepare_training_data("AAPL")
        if training_data is not None:
            logger.info(f"✅ Training data prepared: {training_data.shape}")
        else:
            logger.error("❌ Training data preparation failed")
            return

        logger.info("=== All tests passed! ===")

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    minimal_train()
