import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import requests
import json
import time
import warnings
from collections import deque
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import talib
import gym
from gym import spaces
from dotenv import load_dotenv 

warnings.filterwarnings('ignore')

class AlphaVantageAPI:
    """Alpha Vantage API wrapper for real-time stock data"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        
    def get_intraday_data(self, symbol, interval='5min', outputsize='compact'):
        """Get intraday stock data"""
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
            
            if f'Time Series ({interval})' in data:
                time_series = data[f'Time Series ({interval})']
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                df = df.astype(float)
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                return df
            else:
                print(f"Error: {data}")
                return None
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def get_quote(self, symbol):
        """Get real-time quote"""
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
                    'change_percent': quote['10. change percent']
                }
            return None
            
        except Exception as e:
            print(f"Error fetching quote: {e}")
            return None

class TechnicalIndicators:
    """Technical indicators calculator"""
    
    @staticmethod
    def calculate_indicators(df):
        """Calculate technical indicators"""
        indicators = df.copy()
        
        # Price-based indicators
        indicators['sma_5'] = talib.SMA(df['close'], timeperiod=5)
        indicators['sma_10'] = talib.SMA(df['close'], timeperiod=10)
        indicators['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        indicators['ema_5'] = talib.EMA(df['close'], timeperiod=5)
        indicators['ema_10'] = talib.EMA(df['close'], timeperiod=10)
        
        # Momentum indicators
        indicators['rsi'] = talib.RSI(df['close'], timeperiod=14)
        indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = talib.MACD(df['close'])
        
        # Volatility indicators
        indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = talib.BBANDS(df['close'])
        indicators['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Volume indicators
        indicators['obv'] = talib.OBV(df['close'], df['volume'])
        indicators['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        
        # Price patterns
        indicators['price_change'] = df['close'].pct_change()
        indicators['high_low_ratio'] = df['high'] / df['low']
        indicators['volume_sma'] = talib.SMA(df['volume'], timeperiod=10)
        
        return indicators.fillna(0)

class StockTradingEnvironment(gym.Env):
    """Custom gym environment for stock trading"""
    
    def __init__(self, data, initial_balance=10000):
        super(StockTradingEnvironment, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.max_steps = len(data) - 1
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: technical indicators + portfolio info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(self.data.columns) + 3,), 
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        
        return self._get_observation()
    
    def _get_observation(self):
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
            
        obs = self.data.iloc[self.current_step].values
        
        # Add portfolio information
        portfolio_info = np.array([
            self.balance / self.initial_balance,
            self.shares_held,
            self.net_worth / self.initial_balance
        ])
        
        return np.concatenate([obs, portfolio_info]).astype(np.float32)
    
    def step(self, action):
        current_price = self.data.iloc[self.current_step]['close']
        
        if action == 1:  # Buy
            shares_to_buy = self.balance // current_price
            if shares_to_buy > 0:
                self.shares_held += shares_to_buy
                self.balance -= shares_to_buy * current_price
                
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0
        
        # Calculate net worth
        self.net_worth = self.balance + self.shares_held * current_price
        
        # Calculate reward
        reward = (self.net_worth - self.max_net_worth) / self.initial_balance
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_observation(), reward, done, {}

class DQN(nn.Module):
    """Deep Q-Network"""
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=3):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    """DQN Agent for stock trading"""
    
    def __init__(self, state_dim, action_dim=3, learning_rate=0.001, 
                 gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = np.random.default_rng(seed=42)  # Using 42 as a fixed seed for reproducibility
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_dim, output_dim=action_dim).to(self.device)
        self.target_network = DQN(state_dim, output_dim=action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Update target network
        self.update_target_network()
    
    def update_target_network(self):
        """Update target network with main network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if self.rng.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """Save model weights"""
        torch.save(self.q_network.state_dict(), filepath)
    
    def load_model(self, filepath):
        """Load model weights"""
        self.q_network.load_state_dict(torch.load(filepath))
        self.update_target_network()

class StockPredictionDQN:
    """Main class for stock price prediction using DQN"""
    
    def __init__(self, api_key, symbol='AAPL'):
        self.api = AlphaVantageAPI(api_key)
        self.symbol = symbol
        self.scaler = MinMaxScaler()
        self.agent = None
        self.environment = None
        self.prediction_history = []
        self.actual_history = []
        
    def prepare_data(self, data):
        """Prepare data with technical indicators"""
        # Calculate technical indicators
        indicators_data = TechnicalIndicators.calculate_indicators(data)
        
        # Select features for training
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'atr',
            'obv', 'ad', 'price_change', 'high_low_ratio', 'volume_sma'
        ]
        
        # Clean and normalize data
        clean_data = indicators_data[feature_columns].fillna(0)
        normalized_data = pd.DataFrame(
            self.scaler.fit_transform(clean_data),
            columns=clean_data.columns,
            index=clean_data.index
        )
        
        return normalized_data
    
    def train_model(self, episodes=1000, update_target_freq=10):
        """Train the DQN model"""
        print("Fetching training data...")
        data = self.api.get_intraday_data(self.symbol, interval='5min', outputsize='full')
        
        if data is None:
            print("Failed to fetch training data")
            return
        
        print(f"Preparing data... Shape: {data.shape}")
        prepared_data = self.prepare_data(data)
        
        # Create environment and agent
        self.environment = StockTradingEnvironment(prepared_data)
        state_dim = self.environment.observation_space.shape[0]
        self.agent = DQNAgent(state_dim)
        
        print(f"Starting training for {episodes} episodes...")
        episode_rewards = []
        
        for episode in range(episodes):
            state = self.environment.reset()
            total_reward = 0
            
            while True:
                action = self.agent.act(state)
                next_state, reward, done, _ = self.environment.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # Train agent
            self.agent.replay()
            
            # Update target network
            if episode % update_target_freq == 0:
                self.agent.update_target_network()
            
            episode_rewards.append(total_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.4f}, "
                      f"Epsilon: {self.agent.epsilon:.4f}")
        
        print("Training completed!")
        return episode_rewards
    
    def predict_live(self):
        """Make live predictions and validate"""
        if self.agent is None:
            print("Model not trained yet!")
            return
        
        print("Fetching live data...")
        current_data = self.api.get_intraday_data(self.symbol, interval='5min', outputsize='compact')
        
        if current_data is None:
            print("Failed to fetch live data")
            return
        
        # Prepare current data
        prepared_data = self.prepare_data(current_data)
        current_price = current_data['close'].iloc[-1]
        
        # Create temporary environment for prediction
        temp_env = StockTradingEnvironment(prepared_data.tail(1))
        state = temp_env.reset()
        
        # Make prediction
        action = self.agent.act(state)
        action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        prediction = action_map[action]
        
        # Get Q-values for confidence
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
        q_values = self.agent.q_network(state_tensor)
        confidence = torch.softmax(q_values, dim=1).max().item()
        
        print("\n--- Live Prediction ---")
        print(f"Symbol: {self.symbol}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Timestamp: {datetime.now()}")
        
        return {
            'symbol': self.symbol,
            'current_price': current_price,
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': datetime.now()
        }
    
    def validate_prediction(self, prediction_result, wait_minutes=15):
        """Validate prediction against actual price movement"""
        if prediction_result is None:
            return
        
        initial_price = prediction_result['current_price']
        prediction = prediction_result['prediction']
        
        print(f"\nWaiting {wait_minutes} minutes for validation...")
        time.sleep(wait_minutes * 60)  # Wait for specified minutes
        
        # Get updated price
        quote = self.api.get_quote(self.symbol)
        if quote is None:
            print("Failed to get validation quote")
            return
        
        new_price = quote['price']
        price_change = new_price - initial_price
        price_change_percent = (price_change / initial_price) * 100
        
        # Determine actual movement
        if price_change > 0:
            actual_movement = 'UP'
        elif price_change < 0:
            actual_movement = 'DOWN'
        else:
            actual_movement = 'FLAT'
        
        # Check prediction accuracy
        prediction_correct = (prediction == 'BUY' and actual_movement == 'UP') or \
                           (prediction == 'SELL' and actual_movement == 'DOWN') or \
                           (prediction == 'HOLD' and actual_movement == 'FLAT')
        
        # Store results
        validation_result = {
            'initial_price': initial_price,
            'new_price': new_price,
            'price_change': price_change,
            'price_change_percent': price_change_percent,
            'predicted': prediction,
            'actual': actual_movement,
            'correct': prediction_correct,
            'timestamp': datetime.now()
        }
        
        self.prediction_history.append(prediction_result)
        self.actual_history.append(validation_result)
        
        print("\n--- Validation Results ---")
        print(f"Initial Price: ${initial_price:.2f}")
        print(f"New Price: ${new_price:.2f}")
        print(f"Price Change: ${price_change:.2f} ({price_change_percent:.2f}%)")
        print(f"Predicted: {prediction}")
        print(f"Actual Movement: {actual_movement}")
        print(f"Prediction Correct: {prediction_correct}")
        
        # Calculate overall accuracy
        if len(self.actual_history) > 0:
            correct_predictions = sum(1 for result in self.actual_history if result['correct'])
            accuracy = correct_predictions / len(self.actual_history)
            print(f"Overall Accuracy: {accuracy:.2%} ({correct_predictions}/{len(self.actual_history)})")
        
        return validation_result
    
    def run_live_trading_simulation(self, duration_hours=1, prediction_interval_minutes=30):
        """Run continuous live trading simulation"""
        print(f"Starting live trading simulation for {duration_hours} hours...")
        print(f"Making predictions every {prediction_interval_minutes} minutes")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        while datetime.now() < end_time:
            try:
                # Make prediction
                prediction = self.predict_live()
                
                # Validate after some time (shorter for simulation)
                validation = self.validate_prediction(prediction, wait_minutes=5)
                
                # Wait for next prediction
                time.sleep(prediction_interval_minutes * 60)
                
            except KeyboardInterrupt:
                print("\nStopping simulation...")
                break
            except Exception as e:
                print(f"Error in simulation: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
        
        print("\nSimulation completed!")
        
        # Print final statistics
        if len(self.actual_history) > 0:
            correct = sum(1 for result in self.actual_history if result['correct'])
            total = len(self.actual_history)
            accuracy = correct / total
            
            print("\n--- Final Statistics ---")
            print(f"Total Predictions: {total}")
            print(f"Correct Predictions: {correct}")
            print(f"Accuracy: {accuracy:.2%}")
            
            # Average price change
            avg_change = np.mean([result['price_change_percent'] for result in self.actual_history])
            print(f"Average Price Change: {avg_change:.2f}%")

# Example usage
if __name__ == "__main__":
    # Initialize the model (replace with your Alpha Vantage API key)
    API_KEY = "ALPHA_VANTAGE_API_KEY"
    
    # Create the model
    model = StockPredictionDQN(API_KEY, symbol='AAPL')
    
    # Train the model
    print("Training DQN model...")
    rewards = model.train_model(episodes=15000)
    
    # Save the trained model
    if model.agent:
        model.agent.save_model('stock_dqn_model.pth')
        print("Model saved!")
    
    # Make live predictions
    print("\nMaking live predictions...")
    prediction = model.predict_live()
    
    # Validate prediction
    if prediction:
        validation = model.validate_prediction(prediction, wait_minutes=15)
    
    # Optional: Run continuous simulation
    # model.run_live_trading_simulation(duration_hours=2, prediction_interval_minutes=30)