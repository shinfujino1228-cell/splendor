#!/usr/bin/env python3
"""
Flask server for stock data retrieval using yfinance.
Provides REST API for pattern-finder.html.

Usage:
    python server.py

Endpoints:
    GET /api/health                              - Health check
    GET /api/stock?ticker=AAPL&period=5y        - OHLCV data
    GET /api/tickers                             - Preset ticker list
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf

app = Flask(__name__)
CORS(app)

VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
VALID_INTERVALS = {"1d", "1wk", "1mo"}


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "version": "1.0.0"})


@app.route("/api/stock")
def get_stock():
    """
    Fetch historical OHLCV data for a ticker.

    Query params:
        ticker   : Stock symbol, e.g. AAPL, 7203.T, BTC-USD  (default: AAPL)
        period   : 1mo | 3mo | 6mo | 1y | 2y | 5y | 10y | ytd | max  (default: 5y)
        interval : 1d | 1wk | 1mo  (default: 1d)
    """
    ticker = request.args.get("ticker", "AAPL").upper().strip()
    period = request.args.get("period", "5y")
    interval = request.args.get("interval", "1d")

    if period not in VALID_PERIODS:
        return jsonify({"error": f"Invalid period '{period}'. Choose from: {sorted(VALID_PERIODS)}"}), 400
    if interval not in VALID_INTERVALS:
        return jsonify({"error": f"Invalid interval '{interval}'. Choose from: {sorted(VALID_INTERVALS)}"}), 400

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)

        if hist.empty:
            return jsonify({"error": f"No data found for ticker '{ticker}'"}), 404

        # Drop timezone info from index (makes JSON serialization simpler)
        if hist.index.tzinfo is not None:
            hist.index = hist.index.tz_localize(None)

        return jsonify({
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "count": len(hist),
            "dates":  hist.index.strftime("%Y-%m-%d").tolist(),
            "open":   [round(float(v), 4) for v in hist["Open"]],
            "high":   [round(float(v), 4) for v in hist["High"]],
            "low":    [round(float(v), 4) for v in hist["Low"]],
            "close":  [round(float(v), 4) for v in hist["Close"]],
            "volume": [int(v) for v in hist["Volume"]],
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/tickers")
def suggest_tickers():
    """Return preset tickers for the UI dropdown."""
    tickers = [
        {"symbol": "AAPL",    "name": "Apple Inc."},
        {"symbol": "MSFT",    "name": "Microsoft Corp."},
        {"symbol": "GOOGL",   "name": "Alphabet Inc."},
        {"symbol": "AMZN",    "name": "Amazon.com Inc."},
        {"symbol": "TSLA",    "name": "Tesla Inc."},
        {"symbol": "NVDA",    "name": "NVIDIA Corp."},
        {"symbol": "SPY",     "name": "S&P 500 ETF"},
        {"symbol": "QQQ",     "name": "Nasdaq-100 ETF"},
        {"symbol": "7203.T",  "name": "Toyota Motor (JP)"},
        {"symbol": "9984.T",  "name": "SoftBank Group (JP)"},
        {"symbol": "6758.T",  "name": "Sony Group (JP)"},
        {"symbol": "BTC-USD", "name": "Bitcoin / USD"},
        {"symbol": "GLD",     "name": "Gold ETF"},
    ]
    return jsonify(tickers)


if __name__ == "__main__":
    print("=" * 50)
    print(" Stock Pattern Finder — Flask Server")
    print("=" * 50)
    print(" API endpoints:")
    print("   GET /api/health")
    print("   GET /api/stock?ticker=AAPL&period=5y&interval=1d")
    print("   GET /api/tickers")
    print()
    print(" Server: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000, host="0.0.0.0")
