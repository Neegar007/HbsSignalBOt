import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os

# ==============================
# CONFIG
# ==============================


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")


EXCHANGE = ccxt.bybit()
MIN_VOLUME = 20_000_000  # only scan coins with at least $20M turnover

# ==============================
# UTILS
# ==============================
def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print("Telegram error:", e)


def fetch_ohlcv(symbol, timeframe="1h", limit=200):
    try:
        data = EXCHANGE.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df
    except Exception as e:
        print(f"Error fetching {symbol} {timeframe}: {e}")
        return None


# ==============================
# BOS + CHoCH Detection
# ==============================
def detect_bos_and_choch(df, swing_len=3):
    """
    Swing-high/swing-low BOS + ChoCh detection.
    BOS counts reset after each ChoCh.
    """
    df = df.copy()
    df["swing_high"] = df["high"][(df["high"] > df["high"].shift(1)) & (df["high"] > df["high"].shift(-1))]
    df["swing_low"] = df["low"][(df["low"] < df["low"].shift(1)) & (df["low"] < df["low"].shift(-1))]

    bos_count = 0
    choch_detected = False
    last_trend = None
    signals = []

    for i in range(2, len(df)):
        if pd.notna(df["swing_high"].iloc[i]):
            if last_trend == "up":
                signals.append(("ChoCh", df["time"].iloc[i]))
                bos_count = 0
                choch_detected = True
            last_trend = "down"
            bos_count += 1
        elif pd.notna(df["swing_low"].iloc[i]):
            if last_trend == "down":
                signals.append(("ChoCh", df["time"].iloc[i]))
                bos_count = 0
                choch_detected = True
            last_trend = "up"
            bos_count += 1

    return bos_count, choch_detected, signals


# ==============================
# Pattern Detection (1H only)
# ==============================
def detect_double_top(df, tolerance=0.003):
    highs = df["high"].values
    for i in range(2, len(highs)-2):
        if abs(highs[i] - highs[i+2]) / highs[i] < tolerance:
            neckline = min(df["low"].iloc[i:i+3])
            if df["close"].iloc[-1] < neckline:
                return True, "DOUBLE TOP"
    return False, None


def detect_double_bottom(df, tolerance=0.003):
    lows = df["low"].values
    for i in range(2, len(lows)-2):
        if abs(lows[i] - lows[i+2]) / lows[i] < tolerance:
            neckline = max(df["high"].iloc[i:i+3])
            if df["close"].iloc[-1] > neckline:
                return True, "DOUBLE BOTTOM"
    return False, None


def detect_head_shoulders(df):
    highs = df["high"].values
    for i in range(3, len(highs)-3):
        L, H, R = highs[i-2], highs[i], highs[i+2]
        if H > L and H > R and abs(L - R)/H < 0.05:  # shoulders close height
            neckline = min(df["low"].iloc[i-2:i+3])
            if df["close"].iloc[-1] < neckline:
                return True, "HEAD & SHOULDERS"
    return False, None


def detect_inverse_head_shoulders(df):
    lows = df["low"].values
    for i in range(3, len(lows)-3):
        L, H, R = lows[i-2], lows[i], lows[i+2]
        if H < L and H < R and abs(L - R)/H < 0.05:  # shoulders close depth
            neckline = max(df["high"].iloc[i-2:i+3])
            if df["close"].iloc[-1] > neckline:
                return True, "INVERSE HEAD & SHOULDERS"
    return False, None


def detect_patterns(df):
    detectors = [detect_double_top, detect_double_bottom, detect_head_shoulders, detect_inverse_head_shoulders]
    for detector in detectors:
        found, pattern = detector(df)
        if found:
            return pattern
    return None


# ==============================
# Main Analyzer
# ==============================
def analyze_symbol(symbol):
    try:
        df_daily = fetch_ohlcv(symbol, "1d", 200)
        df_1h = fetch_ohlcv(symbol, "1h", 200)
        df_5m = fetch_ohlcv(symbol, "5m", 200)

        if df_daily is None or df_1h is None or df_5m is None:
            return

        # BOS/ChoCh detection
        bos_daily, choch_daily, _ = detect_bos_and_choch(df_daily)
        bos_1h, choch_1h, _ = detect_bos_and_choch(df_1h)
        bos_5m, choch_5m, _ = detect_bos_and_choch(df_5m)

        # === Trend Following ===
        if bos_1h <= 1 and bos_5m <= 1:
            msg = f"🔔 TREND FOLLOWING\n{symbol}\nDaily + 1H aligned. BOS(1H)={bos_1h}, BOS(5m)={bos_5m}"
            send_telegram_message(msg)

        # === Counter Trend ===
        if bos_1h <= 1 and bos_5m <= 1 and choch_daily:
            msg = f"🔔 COUNTER-TREND\n{symbol}\nDaily vs 1H opposite. BOS(1H)={bos_1h}, BOS(5m)={bos_5m}"
            send_telegram_message(msg)

        # === Reversal Warning ===
        if bos_1h > 4:
            msg = f"⚠️ POSSIBLE REVERSAL\n{symbol}\nMore than 4 BOS on 1H → Trend may reverse!"
            send_telegram_message(msg)

        # === Chart Pattern Detection (1H) ===
        pattern = detect_patterns(df_1h)
        if pattern:
            msg = f"🔔 PATTERN ALERT\n{symbol}\nPattern: {pattern}\nTimeframe: 1H"
            send_telegram_message(msg)

    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")


# ==============================
# Runner
# ==============================
def run_bot():
    markets = EXCHANGE.load_markets()
    usdt_pairs = [s for s in markets if s.endswith("/USDT")]

    for symbol in usdt_pairs:
        try:
            ticker = EXCHANGE.fetch_ticker(symbol)
            if ticker["quoteVolume"] and ticker["quoteVolume"] >= MIN_VOLUME:
                analyze_symbol(symbol)
        except Exception as e:
            print("Error filtering:", e)

# Add this function where you send Telegram alerts
#Telegram message that shows that the code fires

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Error sending message: {e}")

if __name__ == "__main__":
    try:
        # run all your scanning functions here
        run_bot()
    finally:
        # Send a heartbeat every time the cron job runs
        send_telegram_message("✅ Bot scan completed and is alive.")