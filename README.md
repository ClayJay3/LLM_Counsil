# 🏛️ AI Trading Council

An experimental **multi-agent AI market analysis system** that simulates a structured investment committee debate using real market data, technical indicators, fundamentals, and news sentiment.

Instead of a single model making a trade call, this project **forces disagreement**, **fact-checks claims**, and **renders a final verdict** through a judge-style agent.

> ⚠️ **Educational / research project only. Not financial advice.**

---

## 🔍 What This Project Does

For any stock ticker:

1. **Fetches market data**

   * Historical prices (Yahoo Finance)
   * Technical indicators (RSI, MACD, SMAs)
   * Fundamentals (P/E, Market Cap)
   * Recent news headlines (normalized across providers)

2. **Runs a structured debate**

   * **Fundamental Analyst** – valuation and business health
   * **Technical Analyst** – indicators and price action
   * **Sentiment Analyst** – news and volume
   * **Skeptic** – attacks weak logic and assumptions
   * **Fact Checker** – deterministic, non-LLM validation
   * **Counsel (Judge)** – final BUY / SELL / HOLD decision

3. **Produces a final verdict**

   * Decision: `BUY`, `SELL`, or `HOLD`
   * Confidence score (0–100)
   * One-sentence reasoning
   * Full trace of analyst arguments and rebuttals

4. **Optionally backtests decisions**

   * Weekly step-forward “time travel” backtesting
   * Logs wins, losses, and neutral outcomes

---

## 🧠 Why This Exists

Most AI trading scripts:

* Trust a **single LLM**
* Lack **deterministic validation**
* Collapse under malformed JSON or API drift

This project instead:

* Forces **agent disagreement**
* Separates **logic from opinion**
* Treats LLM output as **untrusted input**
* Normalizes inconsistent APIs defensively

The goal is **decision quality**, not prediction hype.

---

## 🧱 Architecture Overview

```
MarketDataEngine
 ├─ Yahoo Finance (prices, fundamentals, news)
 ├─ Technical indicators (pandas_ta)
 └─ News normalization & sanitization

Research Agents (LLMs)
 ├─ Fundamental Analyst
 ├─ Technical Analyst
 └─ Sentiment Analyst

Skeptic Agent
 └─ Attacks analyst logic only

FactChecker (Non-LLM)
 └─ RSI / SMA / valuation sanity checks

Counsel Agent (Judge)
 └─ Produces final structured verdict
```

All LLM responses are:

* Parsed defensively
* JSON-sanitized
* Normalized to canonical keys

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/ai-trading-council.git
cd ai-trading-council
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**

* `yfinance`
* `pandas`
* `pandas_ta`
* `requests`
* `termcolor`
* `tabulate`
* `tqdm`

---

### 3️⃣ Configure LLM Access

The project uses an **Ollama-compatible API**.

Set your API key (if required):

```bash
export OLLAMA_API_KEY="your_key_here"
```

Optional debug mode:

```bash
export DEBUG=1
```

---

### 4️⃣ Run the Program

```bash
python council.py
```

---

## 📊 Usage Modes

### Live Debate (Current Market Data)

```
1. Live Council Debate
```

* Pulls current price data
* Runs full verbose agent debate
* Prints final verdict

---

### Time-Travel Backtesting

```
2. Time Travel Backtest
```

* Simulates historical weekly decisions
* Logs outcomes step-by-step
* Outputs win/loss statistics

---

### Mode Switching

```
3. Change Mode
```

| Mode     | Behavior                        |
| -------- | ------------------------------- |
| INVESTOR | Low volatility, long-term focus |
| DEGEN    | Momentum, short-term risk focus |

---

## 📰 News Handling (Important)

Yahoo Finance news data is **inconsistent** across tickers and regions.

This project:

* Normalizes headlines from multiple possible fields
* Handles missing or malformed titles safely
* Preserves raw news objects for debugging

This prevents `None` / `null` headline issues and LLM hallucinations.

---

## 🧪 Backtesting Philosophy

* Weekly horizon
* No look-ahead bias
* HOLD decisions are excluded from win-rate stats
* Designed for **logic validation**, not alpha claims

---

## ⚠️ Limitations

* Not a trading bot
* No order execution
* No portfolio management
* LLMs can still hallucinate (by design, they are challenged)

---

## 📌 Future Improvements

* Dedicated news APIs (Finnhub, NewsAPI)
* Multi-asset portfolio reasoning
* Risk-weighted position sizing
* Persistent result storage
* Web dashboard

---

## 📜 Disclaimer

This project is for **educational and research purposes only**.

It does **not** constitute financial advice, investment recommendations, or trading signals.
Use at your own risk.