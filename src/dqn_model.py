# dqn_model.py - Complete DQNAgent class with proper logger initialization

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional
import os
import pickle
import logging

class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int = 3, learning_rate: float = 0.001,
                 gamma: float = 0.95, epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, memory_size: int = 10000,
                 batch_size: int = 64, use_gpu: bool = True, num_workers: int = 4):
        
        # Initialize random number generator
        self.rng = np.random.default_rng(seed=42)
        
        # Store hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Initialize logger FIRST - This was missing!
        self.logger = logging.getLogger(__name__)
        
        # GPU Setup
        self.device = self._setup_device(use_gpu)
        print(f"Using device: {self.device}")
        
        # Networks
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Initialize target network
        self.update_target_network()
        
        # Experience replay
        self.memory = []
        self.memory_lock = threading.Lock()
        
        # Training metrics
        self.training_losses = []
        
        self.logger.info(f"DQN Agent initialized with state_dim={state_dim}, device={self.device}")
    
    def _setup_device(self, use_gpu: bool) -> torch.device:
        """Setup computing device with GPU support"""
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
        else:
            device = torch.device("cpu")
            if use_gpu:
                print("GPU requested but not available, using CPU")
        
        return device
    
    # Replace _build_network() with this deeper, wider architecture:
    def _build_network(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.state_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.05),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, self.action_dim)
        )

    
    def remember(self, state, action, reward, next_state, done):
        """Thread-safe memory storage"""
        with self.memory_lock:
            if len(self.memory) >= self.memory_size:
                self.memory.pop(0)
            self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training: bool = True) -> int:
        """Make action with GPU acceleration"""
        if training and self.rng.random() <= self.epsilon:
            return self.rng.choice(self.action_dim)
        
        # Convert to tensor and move to GPU
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            
        return q_values.cpu().argmax().item()
    
    def replay(self) -> float:
        """Experience replay with GPU acceleration and multithreading"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        with self.memory_lock:
            batch_indices = self.rng.choice(len(self.memory), self.batch_size, replace=False)
            experiences = [self.memory[i] for i in batch_indices]
            
        # Prepare tensors on GPU
        # Stack into NumPy array first to speed up tensor creation
        states = torch.FloatTensor(np.vstack([e[0] for e in experiences])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in experiences])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in experiences])).to(self.device)
        next_states = torch.FloatTensor(np.vstack([e[3] for e in experiences])).to(self.device)
        dones = torch.BoolTensor([e[4] for e in experiences]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Loss calculation
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimization
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Store loss
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        
        return loss_value
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.logger.debug("Target network updated")
    
    def save_model(self, filepath: str):
        """Save model to file"""
        try:
            model_data = {
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'training_losses': self.training_losses,
                'hyperparameters': {
                    'state_dim': self.state_dim,
                    'action_dim': self.action_dim,
                    'learning_rate': self.learning_rate,
                    'gamma': self.gamma,
                    'epsilon_decay': self.epsilon_decay,
                    'epsilon_min': self.epsilon_min,
                    'memory_size': self.memory_size,
                    'batch_size': self.batch_size
                }
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            torch.save(model_data, filepath)
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: str):
        """Load model from file"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")
            
            # Load model data
            model_data = torch.load(filepath, map_location=self.device)
            
            # Load network states
            self.q_network.load_state_dict(model_data['q_network_state_dict'])
            self.target_network.load_state_dict(model_data['target_network_state_dict'])
            self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
            
            # Load training state
            self.epsilon = model_data.get('epsilon', self.epsilon_min)
            self.training_losses = model_data.get('training_losses', [])
            
            # Move networks to correct device
            self.q_network.to(self.device)
            self.target_network.to(self.device)
            
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def get_q_values(self, state) -> np.ndarray:
        """Get Q-values for a given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.cpu().numpy().flatten()
    
    def get_training_stats(self) -> dict:
        """Get training statistics"""
        return {
            'total_experiences': len(self.memory),
            'epsilon': self.epsilon,
            'average_loss': np.mean(self.training_losses[-100:]) if self.training_losses else 0,
            'total_training_steps': len(self.training_losses),
            'memory_usage': len(self.memory) / self.memory_size * 100
        }
    
    def set_training_mode(self, training: bool = True):
        """Set training mode for networks"""
        if training:
            self.q_network.train()
            self.target_network.train()
        else:
            self.q_network.eval()
            self.target_network.eval()
    
    def clear_memory(self):
        """Clear experience replay memory"""
        with self.memory_lock:
            self.memory.clear()
        self.logger.info("Experience replay memory cleared")
    
    def get_action_probabilities(self, state) -> np.ndarray:
        """Get action probabilities using softmax"""
        q_values = self.get_q_values(state)
        
        # Apply softmax to get probabilities
        exp_values = np.exp(q_values - np.max(q_values))  # Numerical stability
        probabilities = exp_values / np.sum(exp_values)
        
        return probabilities

