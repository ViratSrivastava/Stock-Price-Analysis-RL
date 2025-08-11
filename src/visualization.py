import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------- Configuration -------------
JSON_FILES = {
    "AMZN": os.path.join("logs", "AMZN", "AMZN_training_history_20250808_073753.json"),
    "GOOGL": os.path.join("logs", "GOOGL", "GOOGL_training_history_20250810_204638.json"),
    "MSFT": os.path.join("logs", "MSFT", "MSFT_training_history_20250810_205003.json"),
    "TSLA": os.path.join("logs", "TSLA", "TSLA_training_history_20250810_204659.json"),
}

# Save plots to ../plots so they are outside src
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plots"))
os.makedirs(OUT_DIR, exist_ok=True)

# Plot style
sns.set(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (12, 6)
MA_WINDOW = 50  # moving average window
ROLL_WINDOW = 50  # rolling std/corr window
TOP_N = 20  # top episodes by profit

# ------------- Helpers -------------
def load_symbol_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rewards = np.array(data.get("episode_rewards", []), dtype=float)
    profits = np.array(data.get("episode_profits", []), dtype=float)
    trades = np.array(data.get("episode_trades", []), dtype=float) if "episode_trades" in data else None
    return rewards, profits, trades, data.get("metrics", {})

def to_df(rewards, profits, trades):
    n = max(len(rewards), len(profits), len(trades) if trades is not None else 0)
    ep = np.arange(1, n + 1)
    df = pd.DataFrame({"episode": ep})
    # Reward
    if len(rewards) == n:
        df["reward"] = rewards
    else:
        r = np.full(n, np.nan)
        r[:len(rewards)] = rewards
        df["reward"] = r
    # Profit
    if len(profits) == n:
        df["profit"] = profits
    else:
        p = np.full(n, np.nan)
        p[:len(profits)] = profits
        df["profit"] = p
    # Trades
    if trades is not None:
        if len(trades) == n:
            df["trades"] = trades
        else:
            t = np.full(n, np.nan)
            t[:len(trades)] = trades
            df["trades"] = t
    else:
        df["trades"] = np.nan
    return df

def moving_avg(series, window=MA_WINDOW):
    return pd.Series(series).rolling(window=window, min_periods=1).mean().values

def rolling_std(series, window=ROLL_WINDOW):
    return pd.Series(series).rolling(window=window, min_periods=1).std().values

def rolling_corr(a, b, window=ROLL_WINDOW):
    s1 = pd.Series(a)
    s2 = pd.Series(b)
    return s1.rolling(window=window, min_periods=5).corr(s2).values

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# ------------- Visualization functions -------------
def plot_reward_trend(df, symbol):
    plt.figure()
    plt.plot(df["episode"], df["reward"], alpha=0.35, label="Reward (raw)")
    plt.plot(df["episode"], moving_avg(df["reward"]), color="tab:blue", label=f"Reward MA({MA_WINDOW})")
    plt.title(f"{symbol} - Episode Reward Trend")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    savefig(os.path.join(OUT_DIR, f"{symbol}_reward_trend.png"))

def plot_profit_trend(df, symbol):
    plt.figure()
    plt.plot(df["episode"], df["profit"], alpha=0.35, label="Profit (raw)", color="tab:orange")
    plt.plot(df["episode"], moving_avg(df["profit"]), color="tab:red", label=f"Profit MA({MA_WINDOW})")
    plt.title(f"{symbol} - Episode Profit Trend")
    plt.xlabel("Episode")
    plt.ylabel("Profit")
    plt.legend()
    savefig(os.path.join(OUT_DIR, f"{symbol}_profit_trend.png"))

def plot_rolling_volatility(df, symbol):
    plt.figure()
    plt.plot(df["episode"], rolling_std(df["reward"], ROLL_WINDOW), label=f"Reward rolling std({ROLL_WINDOW})")
    plt.plot(df["episode"], rolling_std(df["profit"], ROLL_WINDOW), label=f"Profit rolling std({ROLL_WINDOW})")
    plt.title(f"{symbol} - Rolling Volatility (Std Dev)")
    plt.xlabel("Episode")
    plt.ylabel("Rolling Std Dev")
    plt.legend()
    savefig(os.path.join(OUT_DIR, f"{symbol}_rolling_volatility.png"))

def plot_hist_kde(df, symbol):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(df["reward"].dropna(), kde=True, ax=axes[0], color="tab:blue")
    axes[0].set_title(f"{symbol} - Reward Distribution")
    axes[0].set_xlabel("Reward")

    sns.histplot(df["profit"].dropna(), kde=True, ax=axes[1], color="tab:orange")
    axes[1].set_title(f"{symbol} - Profit Distribution")
    axes[1].set_xlabel("Profit")

    savefig(os.path.join(OUT_DIR, f"{symbol}_hist_kde.png"))

def plot_scatter_reward_profit(df, symbol):
    plt.figure()
    sns.scatterplot(x="reward", y="profit", data=df, s=12, alpha=0.5)
    plt.title(f"{symbol} - Reward vs Profit")
    plt.xlabel("Reward")
    plt.ylabel("Profit")
    savefig(os.path.join(OUT_DIR, f"{symbol}_scatter_reward_profit.png"))

def plot_scatter_trades_profit(df, symbol):
    if df["trades"].notna().sum() > 5:
        plt.figure()
        sns.scatterplot(x="trades", y="profit", data=df, s=12, alpha=0.5, color="tab:green")
        plt.title(f"{symbol} - Trades vs Profit")
        plt.xlabel("Trades")
        plt.ylabel("Profit")
        savefig(os.path.join(OUT_DIR, f"{symbol}_scatter_trades_profit.png"))

def plot_rolling_corr(df, symbol):
    corr = rolling_corr(df["reward"], df["profit"], ROLL_WINDOW)
    plt.figure()
    plt.plot(df["episode"], corr, color="tab:purple")
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"{symbol} - Rolling Correlation Reward vs Profit (window={ROLL_WINDOW})")
    plt.xlabel("Episode")
    plt.ylabel("Correlation")
    savefig(os.path.join(OUT_DIR, f"{symbol}_rolling_corr.png"))

def plot_cumulative_profit(df, symbol):
    cum = np.nancumsum(df["profit"].values)
    plt.figure()
    plt.plot(df["episode"], cum, color="tab:red")
    plt.title(f"{symbol} - Cumulative Profit over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Profit")
    savefig(os.path.join(OUT_DIR, f"{symbol}_cumulative_profit.png"))

def plot_top_n_profit(df, symbol, top_n=TOP_N):
    top = df.nlargest(top_n, "profit")[["episode", "profit"]].sort_values("profit", ascending=True)
    plt.figure(figsize=(12, max(6, top_n * 0.3)))
    plt.barh([f"Ep {int(e)}" for e in top["episode"]], top["profit"], color="tab:orange")
    plt.title(f"{symbol} - Top {top_n} Episodes by Profit")
    plt.xlabel("Profit")
    savefig(os.path.join(OUT_DIR, f"{symbol}_top{top_n}_profit.png"))

def plot_hexbin_reward_trades(df, symbol):
    if df["trades"].notna().any():
        plt.figure()
        plt.hexbin(df["reward"], df["trades"], gridsize=40, cmap="viridis", mincnt=1)
        cb = plt.colorbar()
        cb.set_label("Count")
        plt.title(f"{symbol} - Reward vs Trades (Hexbin)")
        plt.xlabel("Reward")
        plt.ylabel("Trades")
        savefig(os.path.join(OUT_DIR, f"{symbol}_hexbin_reward_trades.png"))

def plot_small_multiples_compare(all_dfs):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for ax, (symbol, df) in zip(axes, all_dfs.items()):
        ax.plot(df["episode"], moving_avg(df["reward"]), label="Reward MA", color="tab:blue")
        ax.plot(df["episode"], moving_avg(df["profit"]), label="Profit MA", color="tab:red")
        ax.set_title(f"{symbol} - Reward/Profit MA")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Value")
        ax.legend()
    savefig(os.path.join(OUT_DIR, f"COMPARE_reward_profit_MA.png"))

def plot_compare_distributions(all_dfs):
    plt.figure(figsize=(12, 6))
    for symbol, df in all_dfs.items():
        sns.kdeplot(df["profit"].dropna(), label=symbol, fill=False, linewidth=2)
    plt.title("Profit Distribution Comparison (KDE)")
    plt.xlabel("Profit")
    plt.legend()
    savefig(os.path.join(OUT_DIR, f"COMPARE_profit_kde.png"))

    plt.figure(figsize=(12, 6))
    for symbol, df in all_dfs.items():
        sns.kdeplot(df["reward"].dropna(), label=symbol, fill=False, linewidth=2)
    plt.title("Reward Distribution Comparison (KDE)")
    plt.xlabel("Reward")
    plt.legend()
    savefig(os.path.join(OUT_DIR, f"COMPARE_reward_kde.png"))

# ------------- Main -------------
def main():
    all_dfs = {}
    for symbol, path in JSON_FILES.items():
        if not os.path.exists(path):
            print(f"[WARN] Missing file for {symbol}: {path}")
            continue

        rewards, profits, trades, metrics = load_symbol_data(path)
        df = to_df(rewards, profits, trades)
        all_dfs[symbol] = df

        try:
            plot_reward_trend(df, symbol)
            plot_profit_trend(df, symbol)
            plot_rolling_volatility(df, symbol)
            plot_hist_kde(df, symbol)
            plot_scatter_reward_profit(df, symbol)
            plot_scatter_trades_profit(df, symbol)
            plot_rolling_corr(df, symbol)
            plot_cumulative_profit(df, symbol)
            plot_top_n_profit(df, symbol, TOP_N)
            plot_hexbin_reward_trades(df, symbol)
        except Exception as e:
            print(f"[ERROR] Plotting failed for {symbol}: {e}")

    if len(all_dfs) >= 2:
        try:
            plot_small_multiples_compare(all_dfs)
            plot_compare_distributions(all_dfs)
        except Exception as e:
            print(f"[ERROR] Comparison plots failed: {e}")

    print(f"All figures saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
