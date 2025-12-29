import os
import time
import json
import random
import requests
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from termcolor import colored
from tabulate import tabulate
from tqdm import tqdm

# ==========================================
# CONFIGURATION & GLOBAL SETTINGS
# ==========================================

class Config:
    # API Configuration
    OLLAMA_API_URL = "https://ollama.craysoftware.com/api/generate"
    DEFAULT_MODEL = "gemma2:27b" # Or 'mistral'
    
    # Modes
    MODE = "INVESTOR" # Options: "INVESTOR", "DEGEN"
    
    # Mode Settings
    SETTINGS = {
        "INVESTOR": {
            "tickers": ["AAPL", "MSFT", "GOOGL", "JPM", "V"],
            "temp": 0.1,
            "period": "2y",
            "focus": "Fundamental growth, safety, and long-term stability. Prioritize low volatility."
        },
        "DEGEN": {
            "tickers": ["TSLA", "AMD", "COIN", "NVDA", "GME"],
            "temp": 0.8,
            "period": "6mo",
            "focus": "Short term volatility, momentum, and options play potential. Prioritize high RSI/MACD readings."
        }
    }

    @staticmethod
    def get_setting(key):
        return Config.SETTINGS[Config.MODE][key]

# ==========================================
# DATA ENGINE
# ==========================================

class MarketDataEngine:
    def __init__(self, ticker: str, end_date: Optional[str] = None):
        self.ticker = ticker
        self.end_date = end_date
        self.data = None
        self.info = None
        self.news = []              # normalized list of {"raw":..., "title":...}
        self.news_headlines = []    # convenience list of titles

    def fetch_data(self):
        ticker_obj = yf.Ticker(self.ticker)
        
        period_str = Config.get_setting("period")
        
        if self.end_date:
            end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
            # include the end day by adding one day to 'end' (yfinance end is exclusive)
            start_dt = end_dt - timedelta(days=400)
            self.data = ticker_obj.history(start=start_dt.strftime("%Y-%m-%d"), end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"))
        else:
            self.data = ticker_obj.history(period=period_str)
            
        # info and raw news
        try:
            self.info = ticker_obj.info
        except Exception:
            self.info = {}

        try:
            raw_news = getattr(ticker_obj, "news", None)
            if raw_news:
                raw_news = list(raw_news)  # ensure list-like
            else:
                raw_news = []
        except Exception:
            raw_news = []

        # Normalize raw news into consistent structure with a reliable title extraction
        def _extract_title(item):
            if not item:
                return None
            if isinstance(item, str):
                return item.strip()[:300]
            if not isinstance(item, dict):
                return None

            # Try common top-level keys
            for k in ('title', 'headline', 'name', 'headlineTitle'):
                t = item.get(k)
                if t:
                    return str(t).strip()

            # Try nested content dictionaries (yfinance often places title under ['content']['title'])
            content = item.get('content') or item.get('article') or item.get('data') or item.get('payload')
            if isinstance(content, dict):
                for k in ('title', 'headline', 'summary', 'description'):
                    t = content.get(k)
                    if t:
                        return str(t).strip()

            # Some providers put a nested 'summary' or 'snippet'
            for k in ('summary', 'description', 'snippet'):
                t = item.get(k)
                if t:
                    return str(t).strip()[:300]

            # Links or ids as fallback
            if item.get('link'):
                return str(item.get('link'))[:300]
            if item.get('id'):
                return str(item.get('id'))[:300]

            # Try to find a title-like string anywhere in the dict values (last resort)
            for v in item.values():
                if isinstance(v, str) and len(v.split()) > 3 and len(v) < 300:
                    return v.strip()
                if isinstance(v, dict):
                    for vv in v.values():
                        if isinstance(vv, str) and len(vv.split()) > 3 and len(vv) < 300:
                            return vv.strip()
            return None

        norm = []
        for raw in (raw_news or [])[:10]:
            try:
                title = _extract_title(raw)
            except Exception:
                title = None
            norm.append({"raw": raw, "title": title})

        # Keep top 5 normalized items
        self.news = norm[:5]
        # Build friendly headlines list (with fallback label if missing)
        self.news_headlines = [n['title'] if n['title'] else f"(no title — id:{(n['raw'].get('id') if isinstance(n['raw'], dict) else 'unknown')})" for n in self.news]

        if self.data is None or self.data.empty:
            raise ValueError(f"No data found for {self.ticker}")

        self._add_technicals()

    def _add_technicals(self):
        # RSI
        self.data['RSI'] = ta.rsi(self.data['Close'], length=14)
        
        # MACD - pandas_ta returns multiple column names; handle gracefully
        try:
            macd_df = ta.macd(self.data['Close'])
        except Exception:
            macd_df = None

        if macd_df is not None and not macd_df.empty:
            # merge preserving existing names
            self.data = pd.concat([self.data, macd_df], axis=1)
        else:
            # add placeholder columns with deterministic names
            self.data['MACD_12_26_9'] = 0.0

        # SMA
        self.data['SMA_50'] = ta.sma(self.data['Close'], length=50)
        self.data['SMA_200'] = ta.sma(self.data['Close'], length=200)

        # Drop rows missing critical indicators (but keep at least 2 rows)
        self.data.dropna(subset=['Close', 'RSI'], inplace=True)
        if len(self.data) < 2:
            raise ValueError("Data too short after technical calculation cleanup.")

    def get_latest_metrics(self) -> Dict:
        if len(self.data) < 2:
             raise ValueError("Data frame has fewer than two rows after cleaning.")
             
        latest = self.data.iloc[-1]
        prev = self.data.iloc[-2]

        # try to find a MACD column name from common names
        macd_col = None
        for col in ['MACD_12_26_9', 'MACD', 'macd']:
            if col in self.data.columns:
                macd_col = col
                break

        macd_val = None
        if macd_col:
            macd_val = latest.get(macd_col, 0.0)
        else:
            # try to inspect pandas_ta macd-like columns
            for c in self.data.columns:
                if c.lower().startswith('macd'):
                    macd_val = latest.get(c, 0.0)
                    break
            if macd_val is None:
                macd_val = 0.0

        metrics = {
            "current_date": latest.name.strftime("%Y-%m-%d"),
            "current_price": round(latest['Close'], 2),
            "price_change_pct": round(((latest['Close'] - prev['Close']) / prev['Close']) * 100, 2),
            "volume": int(latest['Volume']),
            "rsi": round(latest['RSI'], 2) if pd.notna(latest['RSI']) else 50,
            "macd": round(float(macd_val), 3) if macd_val is not None else 0.0,
            "sma_50": round(latest['SMA_50'], 2) if pd.notna(latest['SMA_50']) else 0,
            "pe_ratio": self.info.get('trailingPE', 'N/A'),
            "market_cap": self.info.get('marketCap', 'N/A'),
            "news_headlines": self.news_headlines,
            "raw_news": [n['raw'] for n in self.news]  # raw news objects for debugging
        }
        return metrics

# ==========================================
# AGENT ARCHITECTURE
# ==========================================

class OllamaAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.model = Config.DEFAULT_MODEL
        self.api_url = Config.OLLAMA_API_URL
        # read API key from env if present
        self.api_key = os.getenv("OLLAMA_API_KEY", None)

    def query(self, prompt: str, system_prompt: str = "") -> Dict:
        """
        Returns a dict with keys:
          - ok: bool
          - status_code: int or None
          - response_text: str (raw)
          - response_json: dict (if json)
          - error: str (if any)
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": Config.get_setting("temp")
            }
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Extra debug output to console (controlled by environment DEBUG)
        debug = os.getenv("DEBUG", "0") == "1"
        if debug:
            print(colored(f"[DEBUG] {self.name} -> {self.api_url}", "cyan"))
            print(colored(f"[DEBUG] Payload: {json.dumps(payload)[:1000]}", "cyan"))

        try:
            resp = requests.post(self.api_url, json=payload, headers=headers, timeout=90)
            status = resp.status_code
            text = resp.text
            resp_json = None
            try:
                resp_json = resp.json()
            except Exception:
                resp_json = None

            if debug:
                print(colored(f"[DEBUG] {self.name} HTTP {status}", "cyan"))
                print(colored(f"[DEBUG] {self.name} Body: {text[:2000]}", "cyan"))

            if not resp.ok:
                return {
                    "ok": False,
                    "status_code": status,
                    "response_text": text,
                    "response_json": resp_json,
                    "error": f"HTTP {status}"
                }

            # typical Ollama returns {"response": "..."} but preserve entire JSON
            return {
                "ok": True,
                "status_code": status,
                "response_text": text,
                "response_json": resp_json,
                "error": None
            }
        except requests.RequestException as e:
            return {
                "ok": False,
                "status_code": None,
                "response_text": None,
                "response_json": None,
                "error": str(e)
            }

class Researcher(OllamaAgent):
    def analyze(self, ticker: str, data: Dict) -> Dict:
        system_prompt = f"""You are a {self.role} for a top-tier hedge fund. 
        Your Mode is: {Config.MODE}. {Config.get_setting('focus')}
        Analyze the provided data strictly from your perspective. Be concise, professional, and decisive.
        If you are the Sentiment Analyst, focus heavily on the news headlines and volume.
        If you are the Fundamental Analyst, focus heavily on PE ratio and Market Cap.
        If you are the Technical Analyst, focus heavily on RSI, MACD, and SMAs."""
        
        user_prompt = f"""Ticker: {ticker} (Data as of {data['current_date']})
        Data: {json.dumps(data, indent=2)}
        
        Provide your specific analysis and a momentary sentiment (BULLISH, BEARISH, or NEUTRAL). Output a concise paragraph."""
        
        raw = self.query(user_prompt, system_prompt)
        # extract best-effort text
        text = ""
        if raw.get("response_json") and isinstance(raw["response_json"], dict):
            # common key 'response' (Ollama) or 'output'
            text = raw["response_json"].get("response") or raw["response_json"].get("output") or str(raw["response_json"])
        elif raw.get("response_text"):
            text = raw["response_text"]
        else:
            text = f"API_FAIL: {raw.get('error')}"
        
        return {
            "agent": self.name,
            "ok": raw.get("ok", False),
            "http_status": raw.get("status_code"),
            "raw": raw,
            "text": text.strip()
        }

class Skeptic(OllamaAgent):
    def rebut(self, theses: Dict[str, Dict]) -> Dict:
        system_prompt = "You are The Skeptic. Your job is to destroy arguments. Find logical fallacies, emotional biases, or over-reliance on weak data in the following reports. Do not introduce new data."
        
        user_prompt = "Here are the analyst reports:\n\n"
        for agent, report in theses.items():
            user_prompt += f"--- {agent} ---\n{report.get('text', str(report))}\n\n"
            
        user_prompt += "Provide a ruthless rebuttal identifying the biggest flaws and risks. Be concise."
        raw = self.query(user_prompt, system_prompt)
        text = ""
        if raw.get("response_json"):
            text = raw["response_json"].get("response") or str(raw["response_json"])
        elif raw.get("response_text"):
            text = raw["response_text"]
        else:
            text = f"API_FAIL: {raw.get('error')}"
        return {"ok": raw.get("ok", False), "http_status": raw.get("status_code"), "raw": raw, "text": text.strip()}

class Counsel(OllamaAgent):
    def deliberate(self, theses: Dict[str, str], rebuttal: str, fact_check: str) -> Dict:
        system_prompt = """You are The Counsel (Judge). You define the final trading decision.
        Weigh the hard Fact Check data most heavily, followed by the Skeptic's risks. Synthesize the debate.
        You must output JSON ONLY in this exact format, with concise content:
        {
            "decision": "BUY" | "SELL" | "HOLD",
            "confidence": <integer 0-100>,
            "reasoning": "<1 sentence summary of the final verdict>"
        }"""
        
        user_prompt = f"""Review the Council's findings.
        
        HARD FACT CHECK (Deterministic Data):
        {fact_check}
        
        ANALYST ARGUMENTS:
        {json.dumps(theses, indent=2)}
        
        SKEPTIC'S WARNINGS:
        {rebuttal}
        
        Output valid JSON only."""
        
        raw = self.query(user_prompt, system_prompt)

        # Extract text from raw response
        text = ""
        if raw.get("response_json") and isinstance(raw["response_json"], dict):
            text = raw["response_json"].get("response") or raw["response_json"].get("output") or json.dumps(raw["response_json"])
        elif raw.get("response_text"):
            text = raw["response_text"]
        else:
            text = f"API_FAIL: {raw.get('error')}"

        # Clean code fences and whitespace
        clean = text.strip().replace("```json", "").replace("```", "").strip()

        # Try JSON parse
        parsed = None
        try:
            parsed = json.loads(clean)
        except Exception:
            # If the LLM prefixed lines or returned free text with JSON embedded,
            # try to find the first JSON object in the text using a naive approach.
            import re
            m = re.search(r"\{.*\}", clean, flags=re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None

        # If parsed is a dict, normalize keys. Otherwise build fallback dict.
        result = {}
        if isinstance(parsed, dict):
            # direct copy
            result.update(parsed)

            # --- handle nested 'analysis' style response ---
            analysis = parsed.get("analysis") if isinstance(parsed.get("analysis"), dict) else None
            if analysis:
                # map market_sentiment -> decision
                ms = analysis.get("market_sentiment") or analysis.get("sentiment") or analysis.get("sentiment_label")
                if ms:
                    msl = str(ms).strip().upper()
                    if "BUY" in msl or "BULL" in msl or "LONG" in msl:
                        result["decision"] = "BUY"
                    elif "SELL" in msl or "SHORT" in msl or "BEAR" in msl:
                        result["decision"] = "SELL"
                    elif "NEUTRAL" in msl or "HOLD" in msl:
                        result["decision"] = "HOLD"
                    else:
                        # fallback to hold but preserve original label as reasoning
                        result.setdefault("decision", "HOLD")
                        result.setdefault("reasoning", str(ms))

                # confidence_score can be float 0..1 or 0..100
                cs = analysis.get("confidence_score") or analysis.get("confidence")
                if cs is not None:
                    try:
                        cf = float(cs)
                        if 0 <= cf <= 1:
                            cf_int = int(round(cf * 100))
                        else:
                            cf_int = int(round(cf))
                        result["confidence"] = max(0, min(100, cf_int))
                    except Exception:
                        result.setdefault("confidence", 0)

                # Try to synthesize reasoning from key_factors and risk_factors
                reasoning_parts = []
                kf = analysis.get("key_factors") if isinstance(analysis.get("key_factors"), list) else None
                if kf:
                    # take top 2 factor explanations
                    taken = 0
                    for f in kf:
                        if isinstance(f, dict):
                            expl = f.get("explanation") or f.get("impact") or f.get("factor")
                            if expl:
                                reasoning_parts.append(str(expl))
                                taken += 1
                        if taken >= 2:
                            break
                rf = analysis.get("risk_factors") if isinstance(analysis.get("risk_factors"), list) else None
                if rf and len(reasoning_parts) < 2:
                    taken = 0
                    for r in rf:
                        if isinstance(r, dict):
                            expl = r.get("explanation") or r.get("risk")
                            if expl:
                                reasoning_parts.append(str(expl))
                                taken += 1
                        if taken >= 2:
                            break
                if reasoning_parts:
                    result.setdefault("reasoning", " ".join(reasoning_parts)[:800])

            # --- map other common alt keys ---
            if "sentiment" in parsed and "decision" not in result:
                s = str(parsed.get("sentiment", "")).upper()
                if "BUY" in s or "BULL" in s:
                    result["decision"] = "BUY"
                elif "SELL" in s or "BEAR" in s:
                    result["decision"] = "SELL"
                elif "NEUTRAL" in s or "HOLD" in s:
                    result["decision"] = "HOLD"

            if "verdict" in parsed and "decision" not in result:
                v = str(parsed.get("verdict", ""))
                low = v.lower()
                if any(k in low for k in ["buy", "long"]):
                    result["decision"] = "BUY"
                elif any(k in low for k in ["sell", "short"]):
                    result["decision"] = "SELL"
                elif "hold" in low or "neutral" in low:
                    result["decision"] = "HOLD"
                else:
                    result.setdefault("decision", "HOLD")
                    result.setdefault("reasoning", v)

            # normalize confidence if present as float 0..1
            if "confidence" in result:
                try:
                    conf_val = float(result["confidence"])
                    if 0 <= conf_val <= 1:
                        result["confidence"] = int(round(conf_val * 100))
                    else:
                        result["confidence"] = int(round(conf_val))
                except Exception:
                    result["confidence"] = int(result.get("confidence", 0))

            # Ensure canonical keys exist
            result.setdefault("decision", result.get("decision", "HOLD"))
            try:
                result.setdefault("confidence", int(result.get("confidence", 0)))
            except Exception:
                result["confidence"] = 0
            result.setdefault("reasoning", result.get("reasoning", "") or "No reasoning provided by Counsel.")
            result["ok"] = raw.get("ok", False)
            result["raw"] = raw
            return result

        # If parsing failed: fallback heuristics
        low = clean.lower()
        if "buy" in low and "sell" not in low:
            decision_guess = "BUY"
        elif "sell" in low and "buy" not in low:
            decision_guess = "SELL"
        elif "hold" in low or "neutral" in low:
            decision_guess = "HOLD"
        else:
            decision_guess = "HOLD"

        # Try to extract numeric confidence 0-100 or 0-1
        import re
        conf = 0
        mnums = re.findall(r"\b([0-9]{1,3}(?:\.[0-9]+)?)\b", clean)
        for m in mnums:
            try:
                mv = float(m)
                # if small float <=1 consider as fraction
                if 0 <= mv <= 1:
                    mv = mv * 100
                if 0 <= mv <= 100:
                    conf = max(conf, int(round(mv)))
            except:
                continue

        # Create a compact fallback reasoning (truncate)
        fallback_reason = clean.replace("\n", " ").strip()[:800] or "No reasoning provided by Counsel."

        return {
            "decision": decision_guess,
            "confidence": conf,
            "reasoning": fallback_reason,
            "ok": raw.get("ok", False),
            "raw": raw
        }

# ==========================================
# LOGIC & EXECUTION
# ==========================================

class FactChecker:
    """Non-LLM Logic to provide ground truth."""
    @staticmethod
    def verify(data: Dict) -> str:
        checks = []
        
        # Check Trend (SMA)
        if data['sma_50'] > 0 and data['current_price'] > data['sma_50']:
            checks.append("Price is ABOVE 50-day SMA (Bullish Trend Confirmed).")
        elif data['sma_50'] > 0:
            checks.append("Price is BELOW 50-day SMA (Bearish Trend Warning).")
        
        # Check RSI
        if data['rsi'] >= 70:
            checks.append(f"RSI is {data['rsi']} (EXTREMELY Overbought - High Risk).")
        elif data['rsi'] <= 30:
            checks.append(f"RSI is {data['rsi']} (EXTREMELY Oversold - Potential Bounce).")
        
        # Check PE Ratio (Fundamental Check)
        if isinstance(data['pe_ratio'], (int, float)) and data['pe_ratio'] > 50:
            checks.append(f"P/E Ratio is {data['pe_ratio']} (High Valuation Risk).")
            
        return "\n".join(checks)

class CouncilSession:
    def __init__(self, ticker: str, date: Optional[str] = None):
        self.ticker = ticker
        self.date = date
        self.data_engine = MarketDataEngine(ticker, date)
        
        # Initialize Agents
        self.fundamental = Researcher("Fundamental", "Fundamental Equity Analyst")
        self.technical = Researcher("Technical", "Chartered Market Technician")
        self.sentiment = Researcher("Sentiment", "Market Sentiment & News Analyst")
        self.skeptic = Skeptic("Skeptic", "Risk Manager")
        self.counsel = Counsel("Counsel", "Head Portfolio Manager")

    def run(self, verbose=True):
        if verbose: print(colored(f"\n--- 🔔 CONVENING COUNCIL FOR {self.ticker} [{Config.MODE}] (Date: {self.date if self.date else 'Current'}) ---", "cyan", attrs=['bold']))
        try:
            self.data_engine.fetch_data()
            metrics = self.data_engine.get_latest_metrics()
        except Exception as e:
            if verbose: print(colored(f"Data Fetch ERROR: {str(e)}", 'red'))
            return {"decision": "ERROR", "reasoning": str(e), 'date': self.date}

        if verbose:
            print(f"Metrics | Price: {metrics['current_price']} | RSI: {metrics['rsi']} | MACD: {metrics['macd']}")
            print(f"News Headlines: {metrics['news_headlines']}")
            if metrics.get('raw_news'):
                print(colored(f"Raw news items: {len(metrics.get('raw_news', []))}", "yellow"))

        # 2. Researchers Work
        theses = {}
        if verbose: print(colored("  >> 1. Researchers are analyzing...", "yellow"))
        for agent_obj in [self.fundamental, self.technical, self.sentiment]:
            res = agent_obj.analyze(self.ticker, metrics)
            theses[agent_obj.name] = res
            if verbose:
                print(f"     - {agent_obj.name}: ok={res.get('ok')} status={res.get('http_status')} text_preview: {res.get('text','')[:140]}")

        # 3. Skeptic Attacks
        if verbose: print(colored("  >> 2. The Skeptic is rebutting...", "red"))
        rebut = self.skeptic.rebut(theses)
        if verbose: print(f"     - Skeptic: ok={rebut.get('ok')} status={rebut.get('http_status')} preview: {str(rebut.get('text',''))[:140]}")

        # 4. Fact Checker Validates
        facts = FactChecker.verify(metrics)
        if verbose: print(colored("  >> 3. Fact Checker Findings:", "green"), facts.replace('\n', ' | '))

        # 5. The Counsel Decides
        if verbose: print(colored("  >> 4. The Counsel is deliberating...", "blue"))
        counsel_res_raw = self.counsel.deliberate(theses, rebut.get('text', ''), facts)

        # ------------------------------
        # Normalize counsel output to canonical keys (defensive)
        # ------------------------------
        normalized_decision = None
        normalized_confidence = 0
        normalized_reasoning = ""

        if isinstance(counsel_res_raw, dict):
            # If Counsel already returned canonical keys (we favor those)
            normalized_decision = str(counsel_res_raw.get('decision') or "").upper()
            if not normalized_decision or normalized_decision not in ("BUY", "SELL", "HOLD"):
                # try other common fields
                for alt in ('sentiment', 'verdict', 'market_sentiment'):
                    v = counsel_res_raw.get(alt)
                    if v:
                        vl = str(v).lower()
                        if "buy" in vl or "bull" in vl or "long" in vl:
                            normalized_decision = "BUY"; break
                        if "sell" in vl or "bear" in vl or "short" in vl:
                            normalized_decision = "SELL"; break
                        if "hold" in vl or "neutral" in vl:
                            normalized_decision = "HOLD"; break
                # fallback
                if not normalized_decision:
                    normalized_decision = "HOLD"

            # confidence: normalize floats 0..1 to 0..100
            conf_val = counsel_res_raw.get('confidence') or counsel_res_raw.get('confidence_score') or counsel_res_raw.get('cert') or 0
            try:
                cf = float(conf_val)
                if 0 <= cf <= 1:
                    normalized_confidence = int(round(cf * 100))
                else:
                    normalized_confidence = int(round(cf))
            except Exception:
                normalized_confidence = 0

            # reasoning: prefer exact field, then try nested analysis or raw text
            normalized_reasoning = (
                str(counsel_res_raw.get('reasoning') or
                    counsel_res_raw.get('explanation') or
                    counsel_res_raw.get('rationale') or
                    "")
            ).strip()

            if not normalized_reasoning:
                # look inside nested 'analysis' if present
                analysis = counsel_res_raw.get('analysis') if isinstance(counsel_res_raw.get('analysis'), dict) else None
                if analysis:
                    # try to join short key factor excerpts
                    parts = []
                    kf = analysis.get('key_factors') if isinstance(analysis.get('key_factors'), list) else None
                    if kf:
                        for f in kf[:2]:
                            if isinstance(f, dict):
                                parts.append(str(f.get('explanation') or f.get('factor') or f.get('impact') or ""))
                    rf = analysis.get('risk_factors') if isinstance(analysis.get('risk_factors'), list) else None
                    if rf and len(parts) < 2:
                        for r in rf[:2]:
                            if isinstance(r, dict):
                                parts.append(str(r.get('explanation') or r.get('risk') or ""))
                    if parts:
                        normalized_reasoning = " ".join([p for p in parts if p])[:800]

            if not normalized_reasoning:
                # fall back to counsel raw response text (if available)
                if isinstance(counsel_res_raw.get('raw'), dict):
                    normalized_reasoning = str(counsel_res_raw['raw'].get('response_text') or counsel_res_raw['raw'].get('response_json') or "")[:800]
                else:
                    normalized_reasoning = str(counsel_res_raw)[:800]

        else:
            # not a dict — attempt to parse text
            blob = str(counsel_res_raw)
            if "buy" in blob.lower(): normalized_decision = "BUY"
            elif "sell" in blob.lower(): normalized_decision = "SELL"
            elif "hold" in blob.lower() or "neutral" in blob.lower(): normalized_decision = "HOLD"
            else: normalized_decision = "HOLD"
            normalized_reasoning = blob[:800]
            normalized_confidence = 0

        # safety defaults
        if normalized_decision is None:
            normalized_decision = "HOLD"
        if normalized_confidence is None:
            normalized_confidence = 0
        if not normalized_reasoning:
            normalized_reasoning = "No reasoning provided by Counsel."

        result = {
            "decision": normalized_decision,
            "confidence": normalized_confidence,
            "reasoning": normalized_reasoning,
            "research": {k: v for k, v in theses.items()},
            "rebuttal": rebut,
            "counsel_raw": counsel_res_raw,
            "date": metrics['current_date'],
            "fact_check": facts,
            "api_health": {
                "fundamental_ok": theses['Fundamental'].get('ok'),
                "technical_ok": theses['Technical'].get('ok'),
                "sentiment_ok": theses['Sentiment'].get('ok'),
                "skeptic_ok": rebut.get('ok'),
                "counsel_ok": (counsel_res_raw.get('ok') if isinstance(counsel_res_raw, dict) else True)
            }
        }

        if verbose:
            print("\n" + "="*60)
            print(colored(f"FINAL VERDICT ({metrics['current_date']}): {result['decision']} (Confidence: {result['confidence']}%)", "white", "on_blue", attrs=['bold']))
            print(f"Reasoning: {result['reasoning']}")
            print("="*60 + "\n")

        return result


# ==========================================
# TIME TRAVEL BACKTESTER
# ==========================================

class TimeTravelBacktester:
    def __init__(self, ticker: str, start_date: str, weeks: int = 4):
        self.ticker = ticker
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.weeks = weeks
        self.results = []
        # Pre-fetch the full history once and clean it
        self.full_hist = yf.Ticker(self.ticker).history(period="5y").dropna(subset=['Close'])
        # ensure index is DatetimeIndex
        if not isinstance(self.full_hist.index, pd.DatetimeIndex):
            self.full_hist.index = pd.to_datetime(self.full_hist.index)

    def run(self):
        print(colored(f"\n⏳ STARTING TIME TRAVEL BACKTEST: {self.ticker} ({self.weeks} steps)", "magenta", attrs=['bold']))
        
        decisive_trades = 0

        for i in tqdm(range(self.weeks)):
            # Step forward 7 days at a time
            current_sim_date = self.start_date + timedelta(days=i*7)
            
            if current_sim_date >= datetime.now():
                break
                
            date_str = current_sim_date.strftime("%Y-%m-%d")
            
            # --- Entry Date Calculation (Robust, pandas-version-safe) ---
            try:
                target_ts = pd.Timestamp(date_str)
                eligible_dates = self.full_hist.index[self.full_hist.index <= target_ts]
                if len(eligible_dates) == 0:
                    # no market data yet for this date
                    continue

                entry_date = eligible_dates[-1]
                entry_price = self.full_hist.loc[entry_date]['Close']
            except Exception as e:
                # log and skip this step
                if os.getenv("DEBUG", "0") == "1":
                    print(colored(f"Entry lookup error for {date_str}: {e}", 'red'))
                continue

            # Run the council session (verbose=False for speed)
            session = CouncilSession(self.ticker, date=entry_date.strftime("%Y-%m-%d"))
            decision = session.run(verbose=False)
            
            if decision.get('decision') == 'ERROR':
                print(colored(f"[{decision.get('date', date_str)}] - Error in debate: {decision['reasoning']}", 'red'))
                continue

            # --- Exit Date/Outcome Calculation (Robust) ---
            try:
                future_ts = entry_date + timedelta(days=7)
                future_dates = self.full_hist.index[self.full_hist.index >= future_ts]
                if len(future_dates) == 0:
                    # no future price available within the window
                    continue

                exit_date = future_dates[0]
                exit_price = self.full_hist.loc[exit_date]['Close']

                pct_change = ((exit_price - entry_price) / entry_price) * 100

                outcome_color = 'white'
                outcome = "LOSS"
                if decision['decision'] == "BUY" and pct_change > 0:
                    outcome = "WIN"; outcome_color = 'green'
                elif decision['decision'] == "SELL" and pct_change < 0:
                    outcome = "WIN"; outcome_color = 'green'
                elif decision['decision'] == "HOLD":
                    outcome = "NEUTRAL"; outcome_color = 'yellow'
                else:
                    outcome_color = 'red'
                
                if outcome != "NEUTRAL":
                    decisive_trades += 1

                # NEW: Print a log line for every step
                log_line = (
                    f"[{entry_date.strftime('%Y-%m-%d')}] | Dec: {colored(decision['decision'], 'cyan')}/{decision['confidence']}% "
                    f"| Rtn: {round(pct_change, 2):>6}% "
                    f"| {colored(outcome, outcome_color, attrs=['bold']):<11} "
                    f"| Reason: {decision.get('reasoning','')}"
                )
                print(log_line)

                self.results.append({
                    "Date": entry_date.strftime("%Y-%m-%d"),
                    "Decision": decision['decision'],
                    "Confidence": decision['confidence'],
                    "Return%": round(pct_change, 2),
                    "Outcome": outcome,
                    "Reasoning": decision.get('reasoning', ''),
                })
            except Exception as e:
                if os.getenv("DEBUG", "0") == "1":
                    print(colored(f"Exit lookup error for entry {entry_date}: {e}", 'red'))
                continue

        self._print_report(decisive_trades)

    def _print_report(self, total_decisive):
        if not self.results:
            print(colored("\nNo decisive trades were executed or no valid data for the period.", "red"))
            return

        df = pd.DataFrame(self.results)
        decisive_df = df[df['Outcome'].isin(['WIN', 'LOSS'])]
        wins = decisive_df[decisive_df['Outcome'] == 'WIN'].shape[0]
        losses = decisive_df[decisive_df['Outcome'] == 'LOSS'].shape[0]
        total = wins + losses 
        
        print(colored("\n--- FINAL BACKTEST PERFORMANCE REPORT ---", "cyan", attrs=['bold']))
        
        # Table of results (omitting full reasoning for report cleanliness)
        summary_df = df[['Date', 'Decision', 'Confidence', 'Return%', 'Outcome']]
        print(tabulate(summary_df, headers="keys", tablefmt="grid"))
        
        if total > 0:
            win_rate = round((wins/total)*100, 1)
            print(colored(f"\nTotal Debates (Decisive Trades): {total}", 'white'))
            print(colored(f"Council's Win Rate (Excluding HOLDs): {win_rate}%", "yellow", attrs=['bold']))
        else:
            print(colored("No decisive trades (BUY/SELL) were executed in this period to calculate a Win Rate.", 'yellow'))

# ==========================================
# MAIN ENTRY POINT
# ==========================================

def main():
    print(colored("🚀 启动 AI TRADING COUNCIL...", "green", attrs=['bold']))
    
    while True:
        print("\n" + "~"*40)
        print("1. Live Council Debate (Current Data - Full Verbosity)")
        print("2. Time Travel Backtest (Historical Data - Step-by-Step Log)")
        print(f"3. Change Mode (Currently: {colored(Config.MODE, 'magenta', attrs=['bold'])})")
        print("4. Exit")
        print("~"*40)
        
        choice = input("Select option: ")
        
        if choice == "1":
            ticker = input("Enter Ticker (e.g., NVDA): ").upper()
            session = CouncilSession(ticker)
            session.run(verbose=True)
            
        elif choice == "2":
            ticker = input("Enter Ticker: ").upper()
            # Use a conservative date in the past for robust testing
            start = input("Start Date (YYYY-MM-DD, e.g., 2024-01-01): ")
            weeks = int(input("How many weeks to test? (e.g., 10): "))
            tester = TimeTravelBacktester(ticker, start, weeks)
            tester.run()
            
        elif choice == "3":
            if Config.MODE == "INVESTOR":
                Config.MODE = "DEGEN"
            else:
                Config.MODE = "INVESTOR"
            print(f"Mode switched to {colored(Config.MODE, 'magenta', attrs=['bold'])}. Temperature, Tickers, and Focus updated.")
            
        elif choice == "4":
            print(colored("Exiting Council. Good luck with your trades!", "green"))
            break
        else:
            print(colored("Invalid option. Please choose 1, 2, 3, or 4.", 'red'))

if __name__ == "__main__":
    main()
