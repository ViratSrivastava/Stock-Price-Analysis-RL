# training.py

"""
Model training module for DQN stock trading
Handles training loop, model persistence, and performance tracking
CSV-based data loading (no API required)
"""

import numpy as np
import pandas as pd
import logging
import os
import json
import time
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from technical_indicators import TechnicalIndicators
from trading_environment import StockTradingEnvironment
from dqn_model import DQNAgent


# Hyperparameter grid (tweak as needed)
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.01

class ModelTrainer:
    """Handles training of DQN models for stock trading using CSV data"""

    def __init__(self, data_dir: str = "data", model_dir: str = "models", use_gpu: bool = True,
                num_workers: int = 4):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.use_gpu = use_gpu
        self.num_workers = num_workers
        self.scaler = MinMaxScaler()
        # Set a larger default batch size for GPU training
        self.default_batch_size = 512 if use_gpu else 128

        # GPU setup
        if use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache() # Clear GPU memory
            print(f"🚀 GPU Training Enabled: {torch.cuda.get_device_name(0)}")

        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(f"{model_dir}/checkpoints", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Training metrics
        self.training_history = {}

    def load_csv_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load stock data from CSV file"""
        csv_path = os.path.join(self.data_dir, f"{symbol}_daily.csv")
        
        if not os.path.exists(csv_path):
            self.logger.error(f"CSV file not found: {csv_path}")
            return None
            
        try:
            # Read CSV with proper handling of the header structure
            df = pd.read_csv(csv_path)
            
            # Skip the ticker and empty rows, start from actual data
            # Find the first row with a valid date
            data_start_idx = None
            for idx, row in df.iterrows():
                try:
                    pd.to_datetime(row.iloc[0])  # Try to parse first column as date
                    data_start_idx = idx
                    break
                except (ValueError, TypeError):
                    continue
            
            if data_start_idx is None:
                self.logger.error(f"Could not find valid date data in {csv_path}")
                return None
            
            # Extract data from the valid rows
            data_df = df.iloc[data_start_idx:].copy()
            
            # Rename columns properly
            data_df.columns = ['date', 'close', 'high', 'low', 'open', 'volume']
            
            # Convert date column and set as index
            data_df['date'] = pd.to_datetime(data_df['date'])
            data_df.set_index('date', inplace=True)
            
            # Convert price columns to float
            price_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in price_columns:
                data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
            
            # Sort by date
            data_df = data_df.sort_index()
            
            self.logger.info(f"Loaded {len(data_df)} rows from {csv_path}")
            self.logger.info(f"Date range: {data_df.index.min()} to {data_df.index.max()}")
            
            return data_df
            
        except Exception as e:
            self.logger.error(f"Error loading CSV {csv_path}: {e}")
            return None

    def prepare_training_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Prepare training data with technical indicators"""
        self.logger.info(f"Preparing training data for {symbol}")
        
        raw_data = self.load_csv_data(symbol)
        if raw_data is None:
            return None

        # Data cleaning
        initial_len = len(raw_data)
        raw_data = raw_data[
            (raw_data['close'] > 0) &
            (raw_data['open'] > 0) &
            (raw_data['high'] > 0) &
            (raw_data['low'] > 0) &
            (raw_data['close'].notna()) &
            (raw_data['volume'] > 0)
        ]

        cleaned_len = len(raw_data)
        if cleaned_len < 100:
            self.logger.error(f"Insufficient clean data for {symbol}: only {cleaned_len} rows")
            return None

        self.logger.info(f"Cleaned data: {initial_len} -> {cleaned_len} rows ({(1-cleaned_len/initial_len)*100:.1f}% removed)")

        # Calculate technical indicators
        data_with_indicators = TechnicalIndicators.calculate_indicators(raw_data)

        # Prepare features for model
        prepared_data = TechnicalIndicators.prepare_features_for_model(data_with_indicators)

        # Final validation after indicator calculation
        prepared_data = prepared_data.dropna()
        prepared_data = prepared_data[prepared_data['close'] > 0]

        if len(prepared_data) < 100:
            self.logger.error(f"Insufficient data after indicator calculation: only {len(prepared_data)} rows")
            return None

        self.logger.info(f"Prepared {len(prepared_data)} data points with {len(prepared_data.columns)} features")
        return prepared_data

    def normalize_data(self, data: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """Normalize data for training"""
        if fit_scaler:
            normalized_data = pd.DataFrame(
                self.scaler.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
        else:
            normalized_data = pd.DataFrame(
                self.scaler.transform(data),
                columns=data.columns,
                index=data.index
            )
        return normalized_data

    def train_model(self, symbol: str, episodes: int = 1_000, save_freq: int = 100,
                    initial_balance: float = 10_000, transaction_cost: float = 0.001) -> Dict:
        """Train a DQN model for one symbol and return training metrics."""
        self.logger.info("Starting training for %s – %d episodes", symbol, episodes)
        
        training_data = self._prepare_dataset(symbol)
        if training_data is None:
            return {"error": "Failed to prepare training data"}

        env, agent = self._build_env_and_agent(training_data, initial_balance, transaction_cost)

        # Metrics container
        rewards, profits, trades = [], [], []
        best_reward, start = float("-inf"), time.time()

        # Main training loop
        for episode in range(episodes):
            ep_reward, info = self._run_episode(env, agent)
            rewards.append(ep_reward)
            profits.append(info.get("portfolio_return", 0))
            trades.append(info.get("total_trades", 0))

            best_reward = self._maybe_save_best(agent, symbol, episode, ep_reward, best_reward)

            if episode % save_freq == 0:
                self._save_checkpoint(agent, symbol, episode)

            self._log_progress(episode, episodes, rewards, profits, agent, best_reward)

            # Experience replay and target network update
            if len(agent.memory) > agent.batch_size:
                agent.replay()

            if episode % 10 == 0:
                agent.update_target_network()

        # Wrap-up
        final_metrics = self._finalise_training(
            symbol, rewards, profits, trades, time.time() - start, episodes, agent
        )

        return final_metrics

    # Helper methods
    def _prepare_dataset(self, symbol: str) -> Optional[pd.DataFrame]:
        data = self.prepare_training_data(symbol)
        return self.normalize_data(data, fit_scaler=True) if data is not None else None

    def _build_env_and_agent(self, data: pd.DataFrame, initial_balance: float,
                        transaction_cost: float) -> Tuple[StockTradingEnvironment, DQNAgent]:
        """Build environment and agent with GPU support"""
        env = StockTradingEnvironment(data, initial_balance, transaction_cost)

        # Create GPU-enabled agent with hyperparameters from config
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            epsilon_decay=EPSILON_DECAY,
            epsilon_min=EPSILON_MIN,
            use_gpu=self.use_gpu,
            num_workers=self.num_workers,
            batch_size=self.default_batch_size,
            memory_size=40000 if self.use_gpu else 20000
        )

        return env, agent

    def _run_episode(self, env: StockTradingEnvironment, agent: DQNAgent) -> Tuple[float, Dict]:
        """Run a single training episode with immediate learning"""
        state = env.reset()
        total_reward = 0.0
        
        while True:
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            # Learn immediately from new experience:
            if len(agent.memory) > agent.batch_size:
                # Replay N times per step for faster convergence
                for _ in range(3):
                    agent.replay()

            state = next_state
            total_reward += reward

            if done:
                return total_reward, info

    def _maybe_save_best(self, agent: DQNAgent, symbol: str,
                        episode: int, reward: float, best: float) -> float:
        if reward > best:
            self.save_model(agent, symbol, episode, "best")
            return reward
        return best

    def _save_checkpoint(self, agent: DQNAgent, symbol: str, episode: int) -> None:
        self.save_model(agent, symbol, episode, "checkpoint")

    def _log_progress(self, ep: int, total_ep: int, rewards: List[float],
                     profits: List[float], agent: DQNAgent, best: float) -> None:
        self.logger.info(
            "Episode %d/%d | AvgReward %.4f | AvgProfit %.4f | ε %.4f | BestReward %.4f",
            ep, total_ep, np.mean(rewards[-100:]), np.mean(profits[-100:]),
            agent.epsilon, best,
        )

    def _finalise_training(self, symbol: str, rewards: List[float], profits: List[float],
                          trades: List[int], elapsed: float, episodes: int, agent: DQNAgent) -> Dict:
        """Finalize training and save results"""
        self.logger.info("Training completed in %.2f s", elapsed)
        self.save_model(agent=agent, symbol=symbol, episode=episodes, suffix="final")
        
        metrics = self.calculate_training_metrics(rewards, profits, trades, elapsed)
        
        self.save_training_history(symbol, {
            "episode_rewards": rewards,
            "episode_profits": profits,
            "episode_trades": trades,
            "metrics": metrics,
        })
        
        return metrics

    def calculate_training_metrics(self, rewards: List[float], profits: List[float],
                                  trades: List[int], training_time: float) -> Dict:
        """Calculate comprehensive training metrics"""
        metrics = {
            'total_episodes': len(rewards),
            'training_time': training_time,
            'final_reward': rewards[-1] if rewards else 0,
            'best_reward': max(rewards) if rewards else 0,
            'avg_reward': np.mean(rewards) if rewards else 0,
            'reward_std': np.std(rewards) if rewards else 0,
            'final_profit': profits[-1] if profits else 0,
            'best_profit': max(profits) if profits else 0,
            'avg_profit': np.mean(profits) if profits else 0,
            'profit_std': np.std(profits) if profits else 0,
            'avg_trades_per_episode': np.mean(trades) if trades else 0,
            'improvement_rate': self.calculate_improvement_rate(profits),
            'convergence_episode': self.find_convergence_episode(rewards)
        }
        return metrics

    def calculate_improvement_rate(self, profits: List[float]) -> float:
        """Calculate rate of improvement during training"""
        if len(profits) < 10:
            return 0

        first_half = profits[:len(profits)//2]
        second_half = profits[len(profits)//2:]

        if not first_half or not second_half:
            return 0

        return np.mean(second_half) - np.mean(first_half)

    def find_convergence_episode(self, rewards: List[float], window: int = 100) -> Optional[int]:
        """Find episode where model converged"""
        if len(rewards) < window * 2:
            return None

        for i in range(window, len(rewards) - window):
            current_window = rewards[i:i+window]
            std_current = np.std(current_window)
            if std_current < 0.1:
                return i

        return None

    def save_model(self, agent: DQNAgent, symbol: str, episode: int, suffix: str):
        """Save model with metadata"""
        model_path = f"{self.model_dir}/{symbol}_{suffix}_episode_{episode}.pth"
        agent.save_model(model_path)

        # Save metadata
        metadata = {
            'symbol': symbol,
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'state_dim': agent.state_dim,
            'action_dim': agent.action_dim,
            'epsilon': agent.epsilon,
            'total_experiences': len(agent.memory),
            'average_loss': np.mean(agent.training_losses[-100:]) if agent.training_losses else 0
        }

        metadata_path = f"{self.model_dir}/{symbol}_{suffix}_episode_{episode}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def save_training_history(self, symbol: str, history: Dict):
        """Save training history for analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = f"logs/{symbol}_training_history_{timestamp}.json"

        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, value in history.items():
            if isinstance(value, np.ndarray):
                serializable_history[key] = value.tolist()
            elif isinstance(value, list) and value and isinstance(value[0], (np.float64, np.float32)):
                serializable_history[key] = [float(x) for x in value]
            else:
                serializable_history[key] = value

        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)

        self.logger.info(f"Training history saved to {history_path}")

    def load_model(self, symbol: str, model_type: str = "best") -> Optional[DQNAgent]:
        """Load trained model"""
        try:
            # Find the most recent model of the specified type
            model_files = [f for f in os.listdir(self.model_dir)
                          if f.startswith(f"{symbol}_{model_type}") and f.endswith('.pth')]

            if not model_files:
                self.logger.error(f"No {model_type} model found for {symbol}")
                return None

            # Sort by episode number and take the latest
            model_files.sort(key=lambda x: int(x.split('_episode_')[1].split('.')[0]))
            latest_model = model_files[-1]
            model_path = os.path.join(self.model_dir, latest_model)

            # Load metadata to get state dimensions
            metadata_path = model_path.replace('.pth', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                state_dim = metadata['state_dim']
            else:
                self.logger.warning("No metadata found, using default state dimension")
                state_dim = 52  # Adjust based on your feature count

            # Create agent with GPU support and load model
            agent = DQNAgent(
                state_dim=state_dim,
                use_gpu=self.use_gpu,
                num_workers=self.num_workers
            )

            agent.load_model(model_path)
            self.logger.info(f"Loaded model: {latest_model}")
            return agent

        except Exception as e:
            self.logger.error(f"Error loading model for {symbol}: {e}")
            return None

    def plot_training_progress(self, symbol: str, save_plot: bool = True):
        """Plot training progress"""
        try:
            # Find latest training history
            history_files = [f for f in os.listdir('logs')
                           if f.startswith(f"{symbol}_training_history") and f.endswith('.json')]

            if not history_files:
                self.logger.error(f"No training history found for {symbol}")
                return

            latest_history = sorted(history_files)[-1]
            with open(f'logs/{latest_history}', 'r') as f:
                history = json.load(f)

            # Create plots
            _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # Rewards plot
            ax1.plot(history['episode_rewards'])
            ax1.set_title(f'{symbol} - Episode Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.grid(True)

            # Profits plot
            ax2.plot(history['episode_profits'])
            ax2.set_title(f'{symbol} - Portfolio Returns')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Portfolio Return')
            ax2.grid(True)

            # Trades plot
            ax3.plot(history['episode_trades'])
            ax3.set_title(f'{symbol} - Trades per Episode')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Number of Trades')
            ax3.grid(True)

            # Moving average of rewards
            window = min(50, len(history['episode_rewards']) // 10)
            if window > 1:
                moving_avg = pd.Series(history['episode_rewards']).rolling(window=window).mean()
                ax4.plot(moving_avg)
                ax4.set_title(f'{symbol} - Reward Moving Average (window={window})')
                ax4.set_xlabel('Episode')
                ax4.set_ylabel('Average Reward')
                ax4.grid(True)

            plt.tight_layout()

            if save_plot:
                plot_path = f"logs/{symbol}_training_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Training plot saved to {plot_path}")

            plt.show()

        except Exception as e:
            self.logger.error(f"Error plotting training progress: {e}")


if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer()
    
    # Train model for AAPL
    metrics = trainer.train_model("AAPL", episodes=1000)
    print("Training metrics:", metrics)
    
    # Plot results
    trainer.plot_training_progress("AAPL")
