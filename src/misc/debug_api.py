
import os
import logging
from dotenv import load_dotenv
from misc.api_connector import AlphaVantageAPI

# Setup basic logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_api():
    """Simple API test"""
    try:
        # Load API key
        load_dotenv()
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        logger.info(f"API key loaded: {api_key[:10]}...")

        # Test connection
        api = AlphaVantageAPI(api_key)
        logger.info("Testing simple quote first...")

        # Try a simple quote first
        quote = api.get_quote("AAPL")
        if quote:
            logger.info(f"Quote successful: {quote}")
        else:
            logger.error("Quote failed")

        logger.info("Testing intraday data...")
        # Try intraday data with compact size first
        data = api.get_intraday_data("AAPL", interval='5min', outputsize='compact')
        if data is not None:
            logger.info(f"Intraday data successful: {len(data)} rows")
        else:
            logger.error("Intraday data failed")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    test_api()
