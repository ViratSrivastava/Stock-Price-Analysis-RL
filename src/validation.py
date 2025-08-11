# validation.py

"""
Model validation and testing module (CSV-based)
Handles backtesting and performance evaluation using CSV data
"""

import numpy as np
import pandas as pd
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from technical_indicators import TechnicalIndicators
from trading_environment import StockTradingEnvironment
from dqn_model import DQNAgent
from training import ModelTrainer

class ModelValidator:
    """Handles validation and testing of trained DQN models using CSV data"""

    # Class constants for error messages
    ERROR_FAILED_TO_PREPARE_DATA = 'Failed to prepare data'

    def __init__(self, data_dir: str = "data", model_dir: str = "models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.trainer = ModelTrainer(data_dir=data_dir, model_dir=model_dir)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Create validation logs directory
        os.makedirs("validation_logs", exist_ok=True)

    def backtest_model(self, symbol: str, test_split: float = 0.2,
                    initial_balance: float = 10000, model_path: str = None) -> Dict:
        """Backtest model on historical data"""
        self.logger.info(f"Starting backtest for {symbol} - {test_split*100}% of data for testing")

        try:
            # Load trained model - use specific path if provided
            if model_path:
                if not os.path.exists(model_path):
                    self.logger.error(f"Model file not found: {model_path}")
                    return {'error': f'Model file not found: {model_path}'}
                
                self.logger.info(f"Loading specific model from: {model_path}")
                agent = self.trainer.load_model_from_path(model_path)
                model_info = f"Custom model: {os.path.basename(model_path)}"
            else:
                self.logger.info(f"Loading best model for {symbol}")
                agent = self.trainer.load_model(symbol, "best")
                model_info = f"Best model for {symbol}"
            
            if agent is None:
                return {'error': 'No trained model found or failed to load model'}
                
            self.logger.info(f"Successfully loaded model: {model_info}")
            
            # Load and prepare data
            training_data = self.trainer.prepare_training_data(symbol)
            if training_data is None:
                return {'error': self.ERROR_FAILED_TO_PREPARE_DATA}

            # Split data for backtesting (use last portion for testing)
            split_idx = int(len(training_data) * (1 - test_split))
            test_data = training_data.iloc[split_idx:]

            if len(test_data) < 50:
                return {'error': f'Insufficient test data: only {len(test_data)} rows'}

            self.logger.info(f"Using {len(test_data)} days of test data (from {test_data.index[0]} to {test_data.index[-1]})")

            # Normalize test data (using scaler fitted on training data)
            normalized_data = self.trainer.normalize_data(test_data, fit_scaler=True)

            # Create backtest environment
            backtest_env = StockTradingEnvironment(normalized_data, initial_balance=initial_balance)

            # Run backtest
            state = backtest_env.reset()
            total_reward = 0
            actions_taken = []
            step_count = 0
            daily_returns = []
            portfolio_values = []

            self.logger.info("Starting backtest simulation...")
            
            while True:
                action = agent.act(state, training=False)
                next_state, reward, done, info = backtest_env.step(action)

                actions_taken.append(action)
                total_reward += reward
                step_count += 1
                
                # Track portfolio value and daily returns
                portfolio_values.append(backtest_env.net_worth)
                if len(portfolio_values) > 1:
                    daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
                    daily_returns.append(daily_return)
                
                state = next_state

                if done:
                    break

            # Calculate backtest metrics
            performance_metrics = backtest_env.get_performance_metrics()

            # Calculate additional metrics
            action_names = ['HOLD', 'BUY', 'SELL']
            actions_distribution = {
                'HOLD': actions_taken.count(0),
                'BUY': actions_taken.count(1),
                'SELL': actions_taken.count(2)
            }
            
            # Calculate returns and risk metrics
            total_return = (backtest_env.net_worth - initial_balance) / initial_balance
            annualized_return = (1 + total_return) ** (252 / len(test_data)) - 1 if len(test_data) > 0 else 0
            
            backtest_results = {
                'symbol': symbol,
                'model_used': model_info,
                'model_path': model_path if model_path else 'auto-selected',
                'test_split': test_split,
                'test_period_days': len(test_data),
                'test_start_date': test_data.index[0].strftime('%Y-%m-%d'),
                'test_end_date': test_data.index[-1].strftime('%Y-%m-%d'),
                'initial_balance': initial_balance,
                'final_balance': backtest_env.net_worth,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'annualized_return': annualized_return,
                'annualized_return_pct': annualized_return * 100,
                'total_reward': total_reward,
                'avg_reward_per_step': total_reward / step_count if step_count > 0 else 0,
                'total_trades': performance_metrics.get('total_trades', 0),
                'winning_trades': performance_metrics.get('winning_trades', 0),
                'losing_trades': performance_metrics.get('losing_trades', 0),
                'win_rate': performance_metrics.get('win_rate', 0),
                'win_rate_pct': performance_metrics.get('win_rate', 0) * 100,
                'max_drawdown': performance_metrics.get('max_drawdown', 0),
                'max_drawdown_pct': performance_metrics.get('max_drawdown', 0) * 100,
                'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
                'volatility': performance_metrics.get('volatility', 0),
                'volatility_pct': performance_metrics.get('volatility', 0) * 100,
                'actions_distribution': actions_distribution,
                'actions_distribution_pct': {
                    action: (count / len(actions_taken) * 100) if len(actions_taken) > 0 else 0
                    for action, count in actions_distribution.items()
                },
                'total_steps': step_count,
                'avg_portfolio_value': sum(portfolio_values) / len(portfolio_values) if portfolio_values else initial_balance,
                'min_portfolio_value': min(portfolio_values) if portfolio_values else initial_balance,
                'max_portfolio_value': max(portfolio_values) if portfolio_values else initial_balance,
                'timestamp': datetime.now().isoformat()
            }

            # Enhanced logging
            self.logger.info(f"✅ Backtest completed for {symbol}")
            self.logger.info(f"📊 Model: {model_info}")
            self.logger.info(f"📅 Test Period: {backtest_results['test_start_date']} to {backtest_results['test_end_date']} ({len(test_data)} days)")
            self.logger.info(f"💰 Initial Balance: ${initial_balance:,.2f}")
            self.logger.info(f"💰 Final Balance: ${backtest_env.net_worth:,.2f}")
            self.logger.info(f"📈 Total Return: {backtest_results['total_return_pct']:.2f}%")
            self.logger.info(f"📈 Annualized Return: {backtest_results['annualized_return_pct']:.2f}%")
            self.logger.info(f"🎯 Win Rate: {backtest_results['win_rate_pct']:.2f}%")
            self.logger.info(f"📊 Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}")
            self.logger.info(f"📉 Max Drawdown: {backtest_results['max_drawdown_pct']:.2f}%")
            self.logger.info(f"🔄 Total Trades: {backtest_results['total_trades']}")
            self.logger.info(f"⚡ Total Reward: {total_reward:.2f}")
            self.logger.info(f"🎭 Actions: HOLD({actions_distribution['HOLD']}) BUY({actions_distribution['BUY']}) SELL({actions_distribution['SELL']})")

            # Save backtest results with model info
            self.save_backtest_results(symbol, backtest_results)

            return backtest_results

        except Exception as e:
            self.logger.error(f"❌ Error during backtesting for {symbol}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {'error': str(e)}

    def walk_forward_analysis(self, symbol: str, window_size: int = 252,
                             step_size: int = 21, initial_balance: float = 10000) -> Dict:
        """Perform walk-forward analysis"""
        self.logger.info(f"Starting walk-forward analysis for {symbol}")

        try:
            # Load model and data
            agent = self.trainer.load_model(symbol, "best")
            if agent is None:
                return {'error': f'No trained model found for {symbol}'}
            training_data = self.trainer.prepare_training_data(symbol)
            if training_data is None:
                return {'error': self.ERROR_FAILED_TO_PREPARE_DATA}
                return {'error': 'Failed to prepare data'}

            if len(training_data) < window_size * 2:
                return {'error': f'Insufficient data for walk-forward analysis: need at least {window_size * 2} rows'}

            results = []
            start_idx = 0

            while start_idx + window_size < len(training_data):
                end_idx = start_idx + window_size
                window_data = training_data.iloc[start_idx:end_idx]

                # Normalize window data
                normalized_data = self.trainer.normalize_data(window_data, fit_scaler=True)

                # Create environment and run
                env = StockTradingEnvironment(normalized_data, initial_balance=initial_balance)
                state = env.reset()
                total_reward = 0

                while True:
                    action = agent.act(state, training=False)
                    next_state, reward, done, _ = env.step(action)
                    total_reward += reward
                    state = next_state

                    if done:
                        break

                # Store results
                performance_metrics = env.get_performance_metrics()
                results.append({
                    'window_start': start_idx,
                    'window_end': end_idx,
                    'total_return': performance_metrics.get('total_return', 0),
                    'total_trades': performance_metrics.get('total_trades', 0),
                    'win_rate': performance_metrics.get('win_rate', 0),
                    'final_balance': env.net_worth
                })

                start_idx += step_size

            # Calculate summary statistics
            returns = [r['total_return'] for r in results]

            summary = {
                'symbol': symbol,
                'window_size': window_size,
                'step_size': step_size,
                'total_windows': len(results),
                'avg_return': np.mean(returns),
                'std_return': np.std(returns),
                'min_return': min(returns),
                'max_return': max(returns),
                'positive_windows': sum(1 for r in returns if r > 0),
                'win_rate': sum(1 for r in returns if r > 0) / len(returns) if returns else 0,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }

            self.logger.info(f"Walk-forward analysis completed for {symbol}")
            self.logger.info(f"  Average Return: {summary['avg_return']:.2%}")
            self.logger.info(f"  Win Rate: {summary['win_rate']:.2%}")
            self.logger.info(f"  Windows Analyzed: {summary['total_windows']}")

            return summary

        except Exception as e:
            self.logger.error(f"Error during walk-forward analysis for {symbol}: {e}")
            return {'error': str(e)}

    def compare_models(self, symbol: str, test_split: float = 0.2) -> Dict:
        """Compare different model types (best vs final)"""
        self.logger.info(f"Comparing models for {symbol}")

        results = {}

        for model_type in ["best", "final"]:
            try:
                # Load model
                agent = self.trainer.load_model(symbol, model_type)
                if agent is None:
                    results[model_type] = {'error': f'No {model_type} model found'}
                    continue

                # Run backtest
                backtest_result = self.backtest_model_with_agent(
                    agent, symbol, test_split, model_type
                )
                results[model_type] = backtest_result

            except Exception as e:
                results[model_type] = {'error': str(e)}

        # Create comparison summary
        comparison = {
            'symbol': symbol,
            'models_compared': list(results.keys()),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

        # Determine best performing model
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_model = max(valid_results.keys(), 
                           key=lambda k: valid_results[k].get('total_return', -float('inf')))
            comparison['best_performing_model'] = best_model

        self.logger.info(f"Model comparison completed for {symbol}")
        if 'best_performing_model' in comparison:
            self.logger.info(f"  Best performing model: {comparison['best_performing_model']}")

        return comparison

    def backtest_model_with_agent(self, agent: DQNAgent, symbol: str, 
                                 test_split: float, model_type: str) -> Dict:
        """Helper method to backtest with a specific agent"""
        # Load and prepare data
        training_data = self.trainer.prepare_training_data(symbol)
        if training_data is None:
            return {'error': 'Failed to prepare data'}

        # Split data for backtesting
        split_idx = int(len(training_data) * (1 - test_split))
        test_data = training_data.iloc[split_idx:]

        # Normalize test data
        normalized_data = self.trainer.normalize_data(test_data, fit_scaler=False)

        # Create backtest environment
        backtest_env = StockTradingEnvironment(normalized_data)

        # Run backtest
        state = backtest_env.reset()
        total_reward = 0
        actions_taken = []

        while True:
            action = agent.act(state, training=False)
            next_state, reward, done, _ = backtest_env.step(action)

            actions_taken.append(action)
            total_reward += reward
            state = next_state

            if done:
                break

        # Calculate performance metrics
        performance_metrics = backtest_env.get_performance_metrics()

        return {
            'model_type': model_type,
            'total_return': performance_metrics.get('total_return', 0),
            'total_trades': performance_metrics.get('total_trades', 0),
            'win_rate': performance_metrics.get('win_rate', 0),
            'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
            'max_drawdown': performance_metrics.get('max_drawdown', 0),
            'final_balance': backtest_env.net_worth,
            'actions_distribution': {
                'HOLD': actions_taken.count(0),
                'BUY': actions_taken.count(1),
                'SELL': actions_taken.count(2)
            }
        }

    def save_backtest_results(self, symbol: str, results: Dict):
        """Save backtest results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_{symbol}_{timestamp}.json"
        filepath = os.path.join("validation_logs", filename)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Backtest results saved to {filepath}")

    def generate_report(self, symbol: str) -> str:
        """Generate a comprehensive validation report"""
        self.logger.info(f"Generating validation report for {symbol}")

        report_lines = [
            f"# Validation Report for {symbol}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Model Performance",
        ]

        try:
            # Run backtest
            backtest_results = self.backtest_model(symbol)

            if 'error' not in backtest_results:
                report_lines.extend([
                    f"- **Total Return**: {backtest_results['total_return']:.2%}",
                    f"- **Win Rate**: {backtest_results['win_rate']:.2%}",
                    f"- **Sharpe Ratio**: {backtest_results['sharpe_ratio']:.2f}",
                    f"- **Max Drawdown**: {backtest_results['max_drawdown']:.2%}",
                    f"- **Total Trades**: {backtest_results['total_trades']}",
                    f"- **Final Balance**: ${backtest_results['final_balance']:.2f}",
                    "",
                    "## Action Distribution",
                    f"- **HOLD**: {backtest_results['actions_distribution']['HOLD']}",
                    f"- **BUY**: {backtest_results['actions_distribution']['BUY']}",
                    f"- **SELL**: {backtest_results['actions_distribution']['SELL']}",
                ])
            else:
                report_lines.append(f"- **Error**: {backtest_results['error']}")

        except Exception as e:
            report_lines.append(f"- **Error**: {str(e)}")

        # Save report
        report_text = "\n".join(report_lines)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"validation_report_{symbol}_{timestamp}.md"
        report_filepath = os.path.join("validation_logs", report_filename)

        with open(report_filepath, 'w') as f:
            f.write(report_text)

        self.logger.info(f"Validation report saved to {report_filepath}")
        return report_filepath


if __name__ == "__main__":
    # Example usage
    validator = ModelValidator()

    # Run backtest
    backtest_results = validator.backtest_model("AAPL")
    print("Backtest results:", backtest_results)

    # Generate report
    report_path = validator.generate_report("AAPL")
    print(f"Report generated: {report_path}")
