import pandas as pd
import logging
import traceback

def validate_dataset(data, logger=None):
    """Validate a dataset for training, identifying potential issues"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Dataset validation - shape: {data.shape}")
    
    # Check for zero or negative prices
    zero_prices = data[data['close'] <= 0].shape[0]
    if zero_prices > 0:
        logger.warning(f"Found {zero_prices} rows with zero or negative prices")
        
    # Check for NaN values
    nan_count = data.isna().sum()
    if nan_count.sum() > 0:
        logger.warning(f"Found NaN values in dataset: {nan_count}")
    
    # Check for very low prices (potential errors)
    low_prices = data[data['close'] < 0.1].shape[0]
    if low_prices > 0:
        logger.warning(f"Found {low_prices} rows with suspiciously low prices (<0.1)")
        
    # Show price range statistics
    logger.info(f"Price range: min={data['close'].min():.4f}, max={data['close'].max():.4f}, mean={data['close'].mean():.4f}")
    
    return {
        "zero_prices": zero_prices,
        "nan_values": nan_count.sum(),
        "low_prices": low_prices,
        "total_rows": data.shape[0]
    }

def debug_step_execution(env, n_steps=10, logger=None):
    """Debug environment step execution"""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    logger.info(f"Starting step execution debug ({n_steps} steps)")
    
    # Reset environment
    state = env.reset()
    logger.info(f"Initial state shape: {state.shape}")
    
    total_reward = 0
    
    # Execute steps with different actions
    for i in range(n_steps):
        try:
            # Try each action in sequence
            action = i % 3  # Cycle through 0, 1, 2
            
            logger.info(f"Step {i} - Executing action {action}")
            state, reward, done, info = env.step(action)
            
            logger.info(f"  Price: {env.data.iloc[env.current_step-1]['close']:.4f}")
            logger.info(f"  Reward: {reward:.4f}")
            logger.info(f"  Portfolio: {info['net_worth']:.2f}")
            
            total_reward += reward
            
            if done:
                logger.info(f"Environment signaled done at step {i}")
                break
                
        except Exception as e:
            logger.error(f"Error at step {i}: {e}")
            logger.error(traceback.format_exc())
            break
            
    logger.info(f"Debug execution complete. Total reward: {total_reward:.4f}")
    return total_reward
