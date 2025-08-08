# Stock-Price-Analysis-RL

Stock-Price-Analysis-RL is a project that applies Deep Q-Learning (DQN), a reinforcement learning technique, to predict and analyze stock prices using various sources of stock market data.

## Features

- **Deep Q-Learning Model:** Utilizes DQN for stock price prediction and trading strategy optimization.
- **Multiple Data Sources:** Supports data acquisition from Yahoo Finance, Alpha Vantage, Quandl, and more.
- **Technical Indicators:** Integrates TA-Lib, pandas-ta, and other libraries for feature engineering.
- **Visualization:** Provides data visualization using Matplotlib, Seaborn, and Plotly.
- **Modular Design:** Easily extendable for new data sources and RL algorithms.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/OpenVanguard/Stock-Price-Analysis-RL.git
    cd Stock-Price-Analysis-RL
    ```
2. Set up the Python environment (recommended: use `stock_rl_env`):
    ```sh
    python -m venv stock_rl_env
    source stock_rl_env/Scripts/activate  # On Windows
    # Or
    source stock_rl_env/bin/activate      # On Linux/Mac
    ```
3. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

- Run the main script:
    ```sh
    python src/main.py
    ```
- Explore data analysis and experiments in [notebooks/dataAnalysis.ipynb](notebooks/dataAnalysis.ipynb).

## Project Structure

- `src/`: Main source code for RL environment and training.
- `data/`: Storage for raw and processed stock data.
- `notebooks/`: Jupyter notebooks for exploratory analysis.
- `stock_rl_env/`: Python virtual environment.

## Requirements

- Python 3.10+
- PyTorch
- Stable Baselines3
- Gym
- yfinance, pandas_datareader, alpha_vantage, quandl, investpy
- TA-Lib, pandas-ta, scikit-learn, matplotlib, seaborn, plotly

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author