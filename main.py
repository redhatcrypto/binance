import numpy as np
np.NaN = np.nan
import json
import time
import logging
import ccxt
import pandas as pd
import pandas_ta as ta
import os
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from datetime import datetime

# Load config
CONFIG_PATH = "config.json"
STATE_PATH = "trade_state.json"
if not os.path.exists(CONFIG_PATH):
    default_config = {
        "symbol": "ETHUSDT",
        "timeframe": "15m",
        "trend_timeframe": "1h",
        "limit": 100,
        "leverage": 10,
        "min_leverage": 5,
        "max_leverage": 10,
        "leverage_adjustment_threshold": 0.02,
        "rsi_buy": 50,
        "rsi_sell": 50,
        "stoch_rsi_buy": 90,
        "stoch_rsi_sell": 20,
        "stop_loss": 0.02,
        "take_profit": 0.05,
        "position_size": 0.5,
        "trailing_stop_trigger": 0.04,
        "trailing_stop_pct": 0.015,
        "min_balance": 5,
        "max_fee_ratio": 0.001,
        "min_profit_ratio": 0.005,
        "max_cumulative_fees": 1.0,
        "min_volume_ratio": 0.2,
        "atr_period": 14,
        "atr_multiplier": 2.0,
        "starting_balance": 10
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(default_config, f, indent=4)
    print("Config file created. Please update config.json as needed.")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Initialize logging
LOG_PATH = "bot.log"
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Bot started")

# Connect to Binance Testnet
exchange = ccxt.binance({
    'apiKey': 'd56100648c83c00edb963c1817178991020ea4ede466b6fc557ff239d90d8de0',
    'secret': '66740557a31f8a141a24c5574f75ca9f41c65f2841baf88e40533211438b312c',
    'enableRateLimit': True,
    'options': {'defaultType': 'future'},
    'urls': {'api': 'https://testnet.binancefuture.com'}
})
exchange.set_sandbox_mode(True)

# Set initial leverage
try:
    exchange.set_leverage(config["leverage"], config["symbol"])
    logging.info(f"Initial leverage set to {config['leverage']}x")
except Exception as e:
    logging.error(f"Failed to set initial leverage: {e}")
    exit(1)

# Load or initialize state
def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    return {"positions": {}, "in_position": False, "entry_price": 0, "stop_loss": 0, "take_profit": 0, "quantity": 0, "side": None, "cumulative_fees": 0.0, "partial_quantity": 0, "current_leverage": config["leverage"], "last_price": None}

def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=4)

# Fetch market data with retry logic
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(Exception))
def fetch_data(timeframe, limit=config["limit"]):
    try:
        candles = exchange.fetch_ohlcv(config["symbol"], timeframe, limit=limit)
        if candles is None or not candles:
            logging.error(f"fetch_ohlcv returned None or empty for {config['symbol']} on {timeframe}")
            return None
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None

# Fetch ticker with fallback
def fetch_ticker_with_fallback(state):
    try:
        ticker = exchange.fetch_ticker(config["symbol"])
        if ticker is None or "last" not in ticker:
            logging.error("fetch_ticker_with_fallback: fetch_ticker returned None or missing 'last'")
            return state.get("last_price", None)
        price = float(ticker["last"])
        state["last_price"] = price
        save_state(state)
        return price
    except Exception as e:
        logging.error(f"fetch_ticker_with_fallback error: {e}")
        return state.get("last_price", None)

# Calculate indicators
def calculate_indicators(df):
    if df is None or df.empty:
        logging.error("calculate_indicators received None or empty DataFrame")
        return None
    try:
        df["rsi"] = ta.rsi(df["close"], length=14)
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df["macd"] = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACDs_12_26_9"]
        bb = ta.bbands(df["close"], length=20, std=2)
        df["bb_upper"] = bb["BBU_20_2.0"]
        df["bb_lower"] = bb["BBL_20_2.0"]
        df["ema_50"] = ta.ema(df["close"], length=50)
        df["ema_200"] = ta.ema(df["close"], length=200)
        df["stoch_rsi"] = ta.stochrsi(df["close"], length=14)["STOCHRSIk_14_14_3_3"]
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=config["atr_period"])
        # Check for NaN values in key columns
        required_columns = ["rsi", "macd", "macd_signal", "bb_upper", "bb_lower", "stoch_rsi", "atr"]
        if df[required_columns].iloc[-1].isna().any():
            logging.warning("calculate_indicators: NaN values detected in key columns")
            return None
        return df
    except Exception as e:
        logging.error(f"calculate_indicators error: {e}")
        return None

# Adjust leverage
def adjust_leverage(state):
    df = fetch_data(config["timeframe"], limit=config["atr_period"])
    if df is None:
        logging.error("adjust_leverage: Failed to fetch data for leverage adjustment")
        return state
    df = calculate_indicators(df)
    if df is None:
        logging.error("adjust_leverage: Failed to calculate indicators")
        return state
    latest = df.iloc[-1]
    current_price = fetch_ticker_with_fallback(state)
    if current_price is None:
        logging.error("adjust_leverage: No valid price available")
        return state
    atr_ratio = latest["atr"] / current_price
    account = exchange.fetch_balance({'type': 'future'})
    if account is None or 'total' not in account or 'USDT' not in account['total']:
        logging.error("adjust_leverage: Invalid balance data")
        return state
    balance = float(account['total']['USDT'])

    new_leverage = state["current_leverage"]
    if atr_ratio > config["leverage_adjustment_threshold"] or balance < config["min_balance"] * 2:
        new_leverage = max(config["min_leverage"], min(config["max_leverage"], int(state["current_leverage"] * 0.8)))
    elif atr_ratio < config["leverage_adjustment_threshold"] * 0.5 and balance > config["min_balance"] * 5:
        new_leverage = max(config["min_leverage"], min(config["max_leverage"], int(state["current_leverage"] * 1.2)))

    if new_leverage != state["current_leverage"]:
        try:
            exchange.set_leverage(new_leverage, config["symbol"])
            logging.info(f"Leverage adjusted from {state['current_leverage']}x to {new_leverage}x")
            state["current_leverage"] = new_leverage
        except Exception as e:
            logging.error(f"Failed to adjust leverage: {e}")
    return state

# Place order
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(Exception))
def place_order(side, quantity, price=None, dry_run=False):
    symbol = config["symbol"]
    state = adjust_leverage(load_state())
    save_state(state)
    if dry_run:
        current_price = fetch_ticker_with_fallback(state)
        if current_price is None:
            logging.error("place_order: No valid price available in dry_run")
            return None, None, None
        entry_price = current_price
        logging.info(f"Dry run: Simulated {side} order at {entry_price}, quantity={quantity}, leverage={state['current_leverage']}x")
        return {"price": entry_price, "fee": 0}, None, None

    try:
        order_type = "market" if price is None else "limit"
        order = exchange.create_order(symbol, order_type, side, quantity, price)
        if order is None:
            logging.error("place_order: create_order returned None")
            return None, None, None
        fee = float(order["fee"]["cost"]) if "fee" in order else (quantity * order["price"] / state["current_leverage"]) * 0.0004
        logging.info(f"Order placed: {order}, Fee: {fee}, Leverage: {state['current_leverage']}x")

        entry_price = order["price"] if price else fetch_ticker_with_fallback(state)
        if entry_price is None:
            logging.error("place_order: No valid price available after order")
            return order, None, None
        df = calculate_indicators(fetch_data(config["timeframe"], limit=config["atr_period"]))
        if df is None:
            logging.error("place_order: Failed to calculate indicators")
            return order, None, None
        atr = float(df.iloc[-1]["atr"])
        sl_price = entry_price - (atr * config["atr_multiplier"]) if side == "buy" else entry_price + (atr * config["atr_multiplier"])
        tp_price = entry_price * (1 + config["take_profit"]) if side == "buy" else entry_price * (1 - config["take_profit"])

        sl_order = exchange.create_order(symbol, "stop_market", "sell" if side == "buy" else "buy", quantity, None, {"stopPrice": sl_price})
        tp_order = exchange.create_order(symbol, "take_profit_market", "sell" if side == "buy" else "buy", quantity / 2, None, {"stopPrice": tp_price})
        logging.info(f"Stop-loss: {sl_price}, Take-profit (50%): {tp_price}, Fee: {fee}")
        return order, sl_order, tp_order
    except ccxt.RateLimitExceeded as e:
        logging.error(f"Rate limit exceeded: {e}")
        raise
    except Exception as e:
        logging.error(f"Order placement failed: {e}")
        return None, None, None

# Check open positions
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(Exception))
def check_positions():
    positions = exchange.fetch_positions([config["symbol"]])
    if positions is None:
        logging.error("check_positions: fetch_positions returned None")
        return {}
    return {pos["symbol"]: pos for pos in positions if pos["contracts"] > 0}

# Backtesting function
def backtest(days=30):
    logging.info(f"Starting backtest for {days} days")
    limit = int(days * 24 * 4)
    df = fetch_data(config["timeframe"], limit=limit)
    if df is None:
        logging.error("backtest: Failed to fetch data")
        return config["starting_balance"], []
    df_trend = fetch_data(config["trend_timeframe"], limit=int(limit / 4))
    if df_trend is None:
        logging.error("backtest: Failed to fetch trend data")
        return config["starting_balance"], []
    df = calculate_indicators(df)
    if df is None:
        logging.error("backtest: Failed to calculate indicators")
        return config["starting_balance"], []
    df_trend = calculate_indicators(df_trend)
    if df_trend is None:
        logging.error("backtest: Failed to calculate trend indicators")
        return config["starting_balance"], []

    balance = config["starting_balance"]
    position = 0
    trades = []
    cumulative_fees = 0.0
    current_leverage = config["leverage"]

    for i, row in df.iterrows():
        trend_row = df_trend.iloc[max(0, i // 4 - 1)]
        avg_volume = df["volume"].rolling(window=20).mean().iloc[i] if i >= 20 else df["volume"].iloc[:i+1].mean()
        atr_ratio = row["atr"] / row["close"]
        if atr_ratio > config["leverage_adjustment_threshold"] or balance < config["min_balance"] * 2:
            current_leverage = max(config["min_leverage"], min(config["max_leverage"], int(current_leverage * 0.8)))
        elif atr_ratio < config["leverage_adjustment_threshold"] * 0.5 and balance > config["min_balance"] * 5:
            current_leverage = max(config["min_leverage"], min(config["max_leverage"], int(current_leverage * 1.2)))

        if position == 0 and row["volume"] > avg_volume * config["min_volume_ratio"]:
            buy_conditions = {
                "rsi": row["rsi"] < config["rsi_buy"],
                "stoch_rsi": row["stoch_rsi"] < config["stoch_rsi_buy"],
                "macd": row["macd"] > -10,
                "volume": row["volume"] > avg_volume * config["min_volume_ratio"]
            }
            if not all(buy_conditions.values()):
                logging.debug(f"Buy conditions not met: {buy_conditions}")

            if all(buy_conditions.values()):
                entry_price = row["close"]
                quantity = min((balance * current_leverage) / entry_price, balance * 0.01 / entry_price)
                fee = (quantity * entry_price / current_leverage) * 0.0004
                if fee / (quantity * entry_price / current_leverage) > config["max_fee_ratio"]:
                    quantity *= (config["max_fee_ratio"] / (fee / (quantity * entry_price / current_leverage)))
                    fee = (quantity * entry_price / current_leverage) * 0.0004
                expected_profit = (entry_price * (1 + config["take_profit"]) - entry_price) * quantity / current_leverage - 2 * fee
                if expected_profit / (quantity * entry_price / current_leverage) < config["min_profit_ratio"]:
                    logging.debug(f"Buy rejected: Expected profit ({expected_profit}) below threshold")
                    continue
                position = quantity
                balance -= (position * entry_price / current_leverage)
                cumulative_fees += fee
                trades.append({"time": row["timestamp"], "side": "buy", "price": entry_price, "quantity": quantity, "fee": fee, "leverage": current_leverage})
                logging.info(f"Backtest: Buy at {entry_price}, position={position}, fee={fee}, leverage={current_leverage}x")

            sell_conditions = {
                "rsi": row["rsi"] > config["rsi_sell"],
                "stoch_rsi": row["stoch_rsi"] > config["stoch_rsi_sell"],
                "macd": row["macd"] < 10,
                "volume": row["volume"] > avg_volume * config["min_volume_ratio"]
            }
            if not all(sell_conditions.values()):
                logging.debug(f"Sell conditions not met: {sell_conditions}")

            elif all(sell_conditions.values()):
                entry_price = row["close"]
                quantity = min((balance * current_leverage) / entry_price, balance * 0.01 / entry_price)
                fee = (quantity * entry_price / current_leverage) * 0.0004
                if fee / (quantity * entry_price / current_leverage) > config["max_fee_ratio"]:
                    quantity *= (config["max_fee_ratio"] / (fee / (quantity * entry_price / current_leverage)))
                    fee = (quantity * entry_price / current_leverage) * 0.0004
                expected_profit = (entry_price - entry_price * (1 - config["take_profit"])) * quantity / current_leverage - 2 * fee
                if expected_profit / (quantity * entry_price / current_leverage) < config["min_profit_ratio"]:
                    logging.debug(f"Sell rejected: Expected profit ({expected_profit}) below threshold")
                    continue
                position = -quantity
                balance -= (position * entry_price / current_leverage)
                cumulative_fees += fee
                trades.append({"time": row["timestamp"], "side": "sell", "price": entry_price, "quantity": quantity, "fee": fee, "leverage": current_leverage})
                logging.info(f"Backtest: Short at {entry_price}, position={position}, fee={fee}, leverage={current_leverage}x")
        elif position != 0:
            entry_price = trades[-1]["price"]
            atr = row["atr"]
            sl_price = entry_price - (atr * config["atr_multiplier"]) if position > 0 else entry_price + (atr * config["atr_multiplier"])
            tp_price = entry_price * (1 + config["take_profit"]) if position > 0 else entry_price * (1 - config["take_profit"])
            if (position > 0 and (row["low"] <= sl_price or row["high"] >= tp_price)) or \
               (position < 0 and (row["high"] >= sl_price or row["low"] <= tp_price)):
                exit_price = sl_price if (position > 0 and row["low"] <= sl_price) or (position < 0 and row["high"] >= sl_price) else tp_price
                quantity = trades[-1]["quantity"] / 2 if row["high"] >= tp_price else trades[-1]["quantity"]
                fee = (quantity * exit_price / current_leverage) * 0.0004
                balance += (quantity * exit_price / current_leverage) - fee
                cumulative_fees += fee
                trades.append({"time": row["timestamp"], "side": "exit", "price": exit_price, "quantity": quantity, "fee": fee, "leverage": current_leverage})
                logging.info(f"Backtest: Exit at {exit_price}, balance={balance}, fee={fee}, leverage={current_leverage}x")
                if quantity == trades[-1]["quantity"]:
                    position = 0
                else:
                    position -= quantity if position > 0 else -quantity

    final_value = balance + (position * df["close"].iloc[-1] / current_leverage) if position != 0 else balance
    logging.info(f"Backtest completed. Final balance: {final_value} USDT, Trades: {len(trades)}, Cumulative fees: {cumulative_fees}")
    return final_value, trades

# Main bot loop
def run_bot(dry_run=False):
    state = load_state()
    in_position = state["in_position"]
    entry_price = state["entry_price"]
    stop_loss = state["stop_loss"]
    take_profit = state["take_profit"]
    quantity = state["quantity"]
    partial_quantity = state["partial_quantity"]
    side = state["side"]
    cumulative_fees = state["cumulative_fees"]
    current_leverage = state["current_leverage"]

    while True:
        try:
            # Check balance
            account = exchange.fetch_balance({'type': 'future'})
            if account is None or 'total' not in account or 'USDT' not in account['total']:
                logging.error("run_bot: Invalid balance data")
                time.sleep(60)
                continue
            balance = float(account['total']['USDT'])
            logging.info(f"Current balance: {balance} USDT")
            if balance < config["min_balance"]:
                logging.error(f"Insufficient balance ({balance} USDT). Stopping bot.")
                break

            # Check cumulative fees
            if cumulative_fees > config["max_cumulative_fees"]:
                logging.warning(f"Cumulative fees ({cumulative_fees} USDT) exceed threshold. Consider reducing trade frequency.")
                break

            # Check existing positions
            positions = check_positions()
            state["positions"] = positions

            if not positions and not in_position:
                df = fetch_data(config["timeframe"])
                if df is None:
                    logging.error("run_bot: Failed to fetch data for trading")
                    time.sleep(60)
                    continue
                logging.info("Data fetched successfully for trading timeframe")
                df_trend = fetch_data(config["trend_timeframe"])
                if df_trend is None:
                    logging.error("run_bot: Failed to fetch trend data")
                    time.sleep(60)
                    continue
                logging.info("Data fetched successfully for trend timeframe")
                df = calculate_indicators(df)
                if df is None or df.empty:
                    logging.error("run_bot: Failed to calculate indicators or DataFrame is empty")
                    time.sleep(60)
                    continue
                logging.info("Indicators calculated successfully for trading timeframe")
                df_trend = calculate_indicators(df_trend)
                if df_trend is None or df_trend.empty:
                    logging.error("run_bot: Failed to calculate trend indicators or DataFrame is empty")
                    time.sleep(60)
                    continue
                logging.info("Indicators calculated successfully for trend timeframe")
                latest = df.iloc[-1]
                latest_trend = df_trend.iloc[-1]
                if latest is None or latest_trend is None:
                    logging.error("run_bot: Latest data row is None")
                    time.sleep(60)
                    continue
                current_price = fetch_ticker_with_fallback(state)
                if current_price is None:
                    logging.error("run_bot: No valid price available")
                    time.sleep(60)
                    continue
                avg_volume = df["volume"].rolling(window=20).mean().iloc[-1] if len(df) > 20 else df["volume"].mean()

                # Log current indicator values
                logging.info(f"Current indicators: RSI={latest['rsi']}, Stoch RSI={latest['stoch_rsi']}, MACD={latest['macd']}, MACD Signal={latest['macd_signal']}, Close={latest['close']}, BB Lower={latest['bb_lower']}, BB Upper={latest['bb_upper']}, Volume={latest['volume']}, Avg Volume={avg_volume}")

                # Adjust leverage
                state = adjust_leverage(state)
                current_leverage = state["current_leverage"]
                save_state(state)

                # Calculate dynamic quantity with limit
                quantity = min((config["position_size"] * balance) / current_price, balance * 0.01 / current_price)
                fee = (quantity * current_price / current_leverage) * 0.0004
                if fee / (quantity * current_price / current_leverage) > config["max_fee_ratio"]:
                    quantity *= (config["max_fee_ratio"] / (fee / (quantity * current_price / current_leverage)))
                    fee = (quantity * current_price / current_leverage) * 0.0004
                expected_profit = (current_price * (1 + config["take_profit"]) - current_price) * quantity / current_leverage - 2 * fee
                if expected_profit / (quantity * current_price / current_leverage) < config["min_profit_ratio"]:
                    logging.info(f"Trade skipped: Expected profit ({expected_profit}) below threshold")
                    time.sleep(60)
                    continue
                if latest["volume"] < avg_volume * config["min_volume_ratio"]:
                    logging.info(f"Trade skipped: Volume ({latest['volume']}) below threshold ({avg_volume * config['min_volume_ratio']})")
                    time.sleep(60)
                    continue

                # Buy signal
                buy_conditions = {
                    "rsi": latest["rsi"] < config["rsi_buy"] if pd.notna(latest["rsi"]) else False,
                    "stoch_rsi": latest["stoch_rsi"] < config["stoch_rsi_buy"] if pd.notna(latest["stoch_rsi"]) else False,
                    "macd": latest["macd"] > -10 if pd.notna(latest["macd"]) else False
                }
                if all(buy_conditions.values()):
                    order, sl_order, tp_order = place_order("buy", quantity, dry_run=dry_run)
                    if order:
                        entry_price = order["price"] if dry_run else fetch_ticker_with_fallback(state)
                        if entry_price is None:
                            logging.error("run_bot: No valid price after buy order")
                            time.sleep(60)
                            continue
                        df_atr = calculate_indicators(fetch_data(config["timeframe"], limit=config["atr_period"]))
                        if df_atr is None:
                            logging.error("run_bot: Failed to calculate ATR for stop-loss")
                            time.sleep(60)
                            continue
                        stop_loss = entry_price - (float(df_atr.iloc[-1]["atr"]) * config["atr_multiplier"])
                        take_profit = entry_price * (1 + config["take_profit"])
                        side = "buy"
                        in_position = True
                        partial_quantity = quantity / 2
                        cumulative_fees += order.get("fee", fee)
                        state.update({"in_position": True, "entry_price": entry_price, "stop_loss": stop_loss, "take_profit": take_profit, "quantity": quantity, "partial_quantity": partial_quantity, "side": side, "cumulative_fees": cumulative_fees, "current_leverage": current_leverage})
                        save_state(state)
                else:
                    logging.info(f"Buy conditions not met: {buy_conditions}")

                # Sell signal (shorting)
                sell_conditions = {
                    "rsi": latest["rsi"] > config["rsi_sell"] if pd.notna(latest["rsi"]) else False,
                    "stoch_rsi": latest["stoch_rsi"] > config["stoch_rsi_sell"] if pd.notna(latest["stoch_rsi"]) else False,
                    "macd": latest["macd"] < 10 if pd.notna(latest["macd"]) else False
                }
                if all(sell_conditions.values()):
                    order, sl_order, tp_order = place_order("sell", quantity, dry_run=dry_run)
                    if order:
                        entry_price = order["price"] if dry_run else fetch_ticker_with_fallback(state)
                        if entry_price is None:
                            logging.error("run_bot: No valid price after sell order")
                            time.sleep(60)
                            continue
                        df_atr = calculate_indicators(fetch_data(config["timeframe"], limit=config["atr_period"]))
                        if df_atr is None:
                            logging.error("run_bot: Failed to calculate ATR for stop-loss")
                            time.sleep(60)
                            continue
                        stop_loss = entry_price + (float(df_atr.iloc[-1]["atr"]) * config["atr_multiplier"])
                        take_profit = entry_price * (1 - config["take_profit"])
                        side = "sell"
                        in_position = True
                        partial_quantity = quantity / 2
                        cumulative_fees += order.get("fee", fee)
                        state.update({"in_position": True, "entry_price": entry_price, "stop_loss": stop_loss, "take_profit": take_profit, "quantity": quantity, "partial_quantity": partial_quantity, "side": side, "cumulative_fees": cumulative_fees, "current_leverage": current_leverage})
                        save_state(state)
                else:
                    logging.info(f"Sell conditions not met: {sell_conditions}")

            elif in_position:
                current_price = fetch_ticker_with_fallback(state)
                if current_price is None:
                    logging.error("run_bot: No valid price available during position check")
                    time.sleep(60)
                    continue
                profit_loss = (current_price - entry_price) / entry_price if side == "buy" else (entry_price - current_price) / entry_price

                if profit_loss >= config["trailing_stop_trigger"] and quantity > partial_quantity:
                    new_stop = current_price * (1 - config["trailing_stop_pct"]) if side == "buy" else current_price * (1 + config["trailing_stop_pct"])
                    if (side == "buy" and new_stop > stop_loss) or (side == "sell" and new_stop < stop_loss):
                        stop_loss = new_stop
                        state["stop_loss"] = stop_loss
                        save_state(state)
                        logging.info(f"Trailing stop updated to {stop_loss}")

                if (side == "buy" and current_price <= stop_loss) or (side == "sell" and current_price >= stop_loss):
                    exit_side = "sell" if side == "buy" else "buy"
                    order = exchange.create_market_order(config["symbol"], exit_side, quantity)
                    if order is None:
                        logging.error("run_bot: create_market_order returned None")
                        time.sleep(60)
                        continue
                    logging.info(f"Exited {side} at {current_price} due to stop-loss, P/L: {profit_loss * current_leverage:.2%}")
                    cumulative_fees += order.get("fee", (quantity * current_price / current_leverage) * 0.0004)
                    in_position = False
                    state.update({"in_position": False, "cumulative_fees": cumulative_fees, "current_leverage": current_leverage})
                    save_state(state)

            time.sleep(60)
        except ccxt.RateLimitExceeded as e:
            logging.error(f"Rate limit hit: {e}. Retrying with backoff...")
            time.sleep(30)
        except Exception as e:
            logging.error(f"Bot error: {e}, Traceback: {str(e)}")
            time.sleep(10)

if __name__ == "__main__":
    final_balance, trades = backtest(days=30)
    print(f"Backtest result: Final balance = {final_balance} USDT, Total trades = {len(trades)}")
    run_bot(dry_run=False)