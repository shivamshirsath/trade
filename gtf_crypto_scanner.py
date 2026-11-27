import os
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# ==========================================
# 0. KEEP-ALIVE WEB SERVER (FOR RENDER)
# ==========================================
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"GTF Sniper Scanner is running 24/7!")
    def log_message(self, format, *args):
        pass  # Silence HTTP ping logs

def start_keep_alive():
    port = int(os.environ.get("PORT", 10000))
    HTTPServer(("0.0.0.0", port), HealthHandler).serve_forever()

# ==========================================
# 1. USER CONFIGURATION
# ==========================================
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]

CAPITAL_INR        = 5000.0
USDT_INR_RATE      = 85.0
RISK_PCT           = 1.5
RR_RATIO           = 2.0
SL_BUFFER_PCT      = 0.10
MAX_STOP_PCT       = 2.5
ALLOW_SHORTS       = True

TIMEFRAME          = "30m"
SCAN_INTERVAL_SEC  = 60

COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
    "DOTUSDT", "UNIUSDT", "ATOMUSDT", "NEARUSDT", "ARBUSDT",
    "OPUSDT", "SUIUSDT", "APTUSDT", "FILUSDT", "BCHUSDT"
]

# ==========================================
# 2. TELEGRAM & BINANCE DATA HELPERS
# ==========================================
def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            print(f"[Telegram Error] {r.text}")
    except Exception as e:
        print(f"[Telegram Exception] {e}")

def fetch_candles(symbol: str, interval: str = "30m", limit: int = 250) -> pd.DataFrame:
    url = "https://data-api.binance.vision/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df.iloc[:-1].reset_index(drop=True)

# ==========================================
# 3. GTF SNIPER ENGINE
# ==========================================
def analyze_coin(symbol: str):
    df = fetch_candles(symbol, TIMEFRAME, limit=250)
    if len(df) < 210:
        return None

    df["ema20"]  = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"]  = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    c_range = df["high"] - df["low"]
    c_body  = (df["close"] - df["open"]).abs()
    df["body_ratio"]  = np.where(c_range > 0, c_body / c_range, 0.0)
    df["is_boring"]   = df["body_ratio"] <= 0.50
    df["is_bull_exc"] = (df["body_ratio"] > 0.50) & (df["close"] > df["open"])
    df["is_bear_exc"] = (df["body_ratio"] > 0.50) & (df["close"] < df["open"])

    dz_list, sz_list = [], []
    n = len(df)

    for i in range(15, n):
        row = df.iloc[i]
        atr = row["atr14"]
        if pd.isna(atr):
            continue

        if i < n - 1:
            for z in dz_list[:]:
                if row["close"] < z["dist"]:
                    dz_list.remove(z)
                elif i > z["idx"] + 2 and row["low"] <= z["prox"]:
                    z["fresh"] = False
            for z in sz_list[:]:
                if row["close"] > z["dist"]:
                    sz_list.remove(z)
                elif i > z["idx"] + 2 and row["high"] >= z["prox"]:
                    z["fresh"] = False

            base_cnt = 0
            for b in range(1, 4):
                if df.iloc[i - b]["is_boring"]:
                    base_cnt += 1
                else:
                    break

            if 1 <= base_cnt <= 3:
                leg_in = df.iloc[i - base_cnt - 1]
                if row["is_bull_exc"] and (row["close"] - row["open"]) > atr * 1.0 and (leg_in["is_bull_exc"] or leg_in["is_bear_exc"]):
                    is_dbr = bool(leg_in["is_bear_exc"])
                    closing_ok = (row["close"] > leg_in["high"]) if is_dbr else (row["close"] > df.iloc[i - 1]["high"])
                    if closing_ok:
                        bases = df.iloc[i - base_cnt : i]
                        prox = max(bases[["open", "close"]].max(axis=1))
                        dist = min(bases["low"].min(), row["low"])
                        if is_dbr:
                            dist = min(dist, leg_in["low"])
                        if prox > dist:
                            dz_list.append({"idx": i, "prox": prox, "dist": dist, "fresh": True, "pat": "DBR" if is_dbr else "RBR"})
                            if len(dz_list) > 12:
                                dz_list.pop(0)

                if row["is_bear_exc"] and (row["open"] - row["close"]) > atr * 1.0 and (leg_in["is_bull_exc"] or leg_in["is_bear_exc"]):
                    is_rbd = bool(leg_in["is_bull_exc"])
                    closing_ok = (row["close"] < leg_in["low"]) if is_rbd else (row["close"] < df.iloc[i - 1]["low"])
                    if closing_ok:
                        bases = df.iloc[i - base_cnt : i]
                        prox = min(bases[["open", "close"]].min(axis=1))
                        dist = max(bases["high"].max(), row["high"])
                        if is_rbd:
                            dist = max(dist, leg_in["high"])
                        if dist > prox:
                            sz_list.append({"idx": i, "prox": prox, "dist": dist, "fresh": True, "pat": "RBD" if is_rbd else "DBD"})
                            if len(sz_list) > 12:
                                sz_list.pop(0)

    last = df.iloc[-1]
    trend_up   = (last["ema20"] > last["ema50"]) and (last["close"] > last["ema50"]) and (last["close"] > last["ema200"])
    trend_down = (last["ema20"] < last["ema50"]) and (last["close"] < last["ema50"]) and (last["close"] < last["ema200"])

    if trend_up:
        for z in reversed(dz_list):
            if z["fresh"] and last["low"] <= z["prox"] and last["close"] > last["open"] and last["close"] >= z["prox"] and last["close"] > z["dist"]:
                entry = last["close"]
                sl = z["dist"] * (1.0 - SL_BUFFER_PCT / 100.0)
                rpu = entry - sl
                tp = entry + (rpu * RR_RATIO)
                stop_pct = (rpu / entry) * 100.0
                blocked = any(sz["prox"] > entry and sz["prox"] < tp for sz in sz_list)
                if rpu > 0 and stop_pct <= MAX_STOP_PCT and not blocked:
                    return {"symbol": symbol, "side": "BUY (LONG)", "pat": z["pat"], "entry": entry, "sl": sl, "tp": tp, "stop_pct": stop_pct, "close_time": int(last["close_time"])}

    if ALLOW_SHORTS and trend_down:
        for z in reversed(sz_list):
            if z["fresh"] and last["high"] >= z["prox"] and last["close"] < last["open"] and last["close"] <= z["prox"] and last["close"] < z["dist"]:
                entry = last["close"]
                sl = z["dist"] * (1.0 + SL_BUFFER_PCT / 100.0)
                rpu = sl - entry
                tp = entry - (rpu * RR_RATIO)
                stop_pct = (rpu / entry) * 100.0
                blocked = any(dz["prox"] < entry and dz["prox"] > tp for dz in dz_list)
                if rpu > 0 and stop_pct <= MAX_STOP_PCT and not blocked:
                    return {"symbol": symbol, "side": "SELL (SHORT)", "pat": z["pat"], "entry": entry, "sl": sl, "tp": tp, "stop_pct": stop_pct, "close_time": int(last["close_time"])}

    return None

# ==========================================
# 4. MAIN 24/7 LOOP
# ==========================================
def main():
    threading.Thread(target=start_keep_alive, daemon=True).start()

    sent_alerts = set()
    risk_inr  = CAPITAL_INR * (RISK_PCT / 100.0)
    risk_usdt = risk_inr / USDT_INR_RATE

    start_msg = (
        f"🚀 <b>GTF Sniper Scanner Started (24/7 Cloud)</b>\n"
        f"Coins: {len(COINS)} | Timeframe: <b>{TIMEFRAME}</b>\n"
        f"Capital: ₹{CAPITAL_INR:.0f} | Risk/Trade: ₹{risk_inr:.0f} ({RISK_PCT}%)\n"
        f"Mode: {'Futures (Long & Short)' if ALLOW_SHORTS else 'Spot (BUY Only)'}"
    )
    send_telegram(start_msg)

    while True:
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now_str} UTC] Scanning {len(COINS)} coins...", flush=True)

        for sym in COINS:
            try:
                sig = analyze_coin(sym)
                if sig:
                    alert_key = (sig["symbol"], sig["close_time"])
                    if alert_key not in sent_alerts:
                        sent_alerts.add(alert_key)
                        rpu = abs(sig["entry"] - sig["sl"])
                        qty = risk_usdt / rpu
                        pos_usdt = qty * sig["entry"]
                        emoji = "🟢" if "BUY" in sig["side"] else "🔴"

                        msg = (
                            f"{emoji} <b>GTF SNIPER ALERT — {sig['symbol']} ({TIMEFRAME})</b>\n"
                            f"<b>Action:</b> {sig['side']} ({sig['pat']} 7/7)\n"
                            f"<b>Entry:</b> <code>{sig['entry']:.5g}</code>\n"
                            f"<b>Stop Loss:</b> <code>{sig['sl']:.5g}</code> ({sig['stop_pct']:.2f}%)\n"
                            f"<b>Take Profit (2:1):</b> <code>{sig['tp']:.5g}</code>\n"
                            f"<b>Position Size:</b> {qty:.4f} (~${pos_usdt:.2f})\n"
                            f"<b>Risk:</b> ₹{risk_inr:.0f} | <b>Target:</b> +₹{risk_inr * RR_RATIO:.0f}"
                        )
                        print(f"  -> ALERT SENT: {sig['symbol']} {sig['side']} @ {sig['entry']}", flush=True)
                        send_telegram(msg)
            except Exception as e:
                print(f"  [Skip {sym}] {e}", flush=True)

            time.sleep(0.3)

        time.sleep(SCAN_INTERVAL_SEC)

if __name__ == "__main__":
    main()