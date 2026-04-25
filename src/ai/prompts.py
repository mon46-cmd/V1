"""Prompt rendering for the two LLM calls (v3).

Two prompts only: watchlist (Prompt A) + deep-signal (Prompt B).
Position-review was removed; risk management runs in code.

Each renderer returns ``(system, user)`` strings. The system carries
role/research-policy/rubric and an inline JSON output schema. The user
carries the data payload as JSON (orjson).

Bump ``PROMPT_VERSION`` on any change. Schemas echo it; the audit log
records it; rolling a version lets us A/B prompts cleanly.

v3 highlights
-------------
- Removed Prompt C (position review).
- Both system prompts now include explicit RESEARCH POLICY: enumerate
  Twitter/X queries, weight primary sources over secondary, hard-ban
  fabrication ("if you cannot verify in <6h sources, do not cite"),
  hard-ban toxicity / discrimination / illegal content, hard-ban
  shitcoin shilling and unverified rumours.
- Watchlist: dedicated `social` section (catalysts, sentiment_score,
  source URLs, attention_z) AND a `discarded_pumps` filter that keys
  on social-driven blow-offs.
- Deep: dedicated `social_context` block in the user payload + a
  research checklist.
- Compact columnar history for deep (unchanged from v2).
- Shared `payload_trim_for_llm(snap)` helper.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import orjson
import pandas as pd

PROMPT_VERSION = "v3.1-2026-04-25"


# ---- JSON serialiser ------------------------------------------------
def _json_default(obj):
    """orjson default-handler for pandas/numpy types not natively supported."""
    if isinstance(obj, pd.Timestamp):
        if pd.isna(obj):
            return None
        return obj.isoformat()
    if obj is pd.NaT or obj is pd.NA:
        return None
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return str(obj)


def _dumps(obj: Any) -> str:
    return orjson.dumps(obj, default=_json_default,
                        option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8")


# ---- Field glossary -------------------------------------------------
FIELD_GLOSSARY: dict[str, str] = {
    # context
    "close": "last 15m close price (USDT)",
    "ret_1h": "log-equiv return over the last 4 15m bars",
    "ret_4h": "return over the last 16 15m bars",
    "ret_24h": "return over the last 96 15m bars (24h)",
    "turnover_24h": "USDT notional traded in the last 24h",
    "rel_volume_20": "current bar volume / mean(volume, prior 20 bars)",
    "hi_lo_24h_pos": "(close - 24h_low) / (24h_high - 24h_low) in [0,1]",
    "atr_pct_rank_96": "percentile rank (0..1) of atr_14_pct in trailing 96 bars",
    "bb_width_rank_96": "percentile rank of bb_width in trailing 96 bars",
    # vol / regime
    "atr_14_pct": "Wilder ATR(14) / close, in % (volatility proxy)",
    "rv_20": "realized stdev of log_ret over 20 bars (per-bar)",
    "hurst_100": "Hurst exponent (R/S, 100 bars). >0.5 trend, <0.5 mean-revert",
    "vr_2_100": "Lo-MacKinlay variance ratio q=2 over 100 bars. <1 mean-revert, >1 trend",
    "acf1_50": "lag-1 autocorrelation of returns over 50 bars",
    # tech
    "rsi_14": "15m RSI(14) (Wilder)",
    "ema_50_dist": "(close - EMA50) / close * 100  (in %)",
    "macd_hist": "MACD histogram (12,26,9)",
    "bb_pct_b": "(close - lower) / (upper - lower); >1 above band, <0 below",
    "adx_14": "Wilder ADX(14): trend strength",
    "supertrend_dir": "supertrend direction (+1 long bias / -1 short bias)",
    # derivatives
    "funding_rate": "current 8h funding rate (perp). + = longs pay",
    "funding_z_20": "z-score of funding_rate over last 20 settlements",
    "oi_chg_pct_24h": "open interest pct change over last 24h",
    "oi_z_50": "z-score of OI level over last 50 hourly samples",
    "basis_bps": "(mark - index) / index in bps; sign matters",
    # volume profile (200 bars)
    "poc_dist": "(close - POC) / close * 100 (% above/below value-area POC)",
    "vah_dist": "(close - VAH) / close * 100 (>0 = above value area)",
    "val_dist": "(close - VAL) / close * 100",
    "value_area_width_200": "(VAH - VAL) / POC",
    # MTF
    "h1_rsi_14": "1h RSI(14)",
    "h4_rsi_14": "4h RSI(14)",
    "h1_ema_50_dist": "1h (close - EMA50) / close * 100",
    "h4_ema_50_dist": "4h (close - EMA50) / close * 100",
    "h1_atr_14_pct": "1h ATR%",
    "h4_atr_14_pct": "4h ATR%",
    "trend_score_mtf": "mean of sign(ema_50_dist) across 15m/1h/4h in {-1..1}",
    # cross-asset (NaN on the BTC row by design)
    "beta_btc_100": "100-bar beta of log_ret vs BTC log_ret (null for BTC)",
    "corr_btc_100": "100-bar correlation vs BTC (null for BTC)",
    "rs_vs_btc_24h": "ret_24h - btc_ret_24h (null for BTC)",
    "rs_vs_eth_24h": "ret_24h - eth_ret_24h",
    # peer/cluster
    "cluster_id": "kmeans cluster id over the universe (5 clusters by default)",
    "cluster_leader": "highest-turnover symbol in the cluster",
    # flags (1 = fired on the last 15m bar)
    "flag_volume_climax": "last bar volume > climax_mult * mean(20)",
    "flag_sweep_up": "upper liquidity sweep on last bar (wick + close back)",
    "flag_sweep_dn": "lower liquidity sweep on last bar",
    "flag_squeeze_release": "BB width crosses up after compression",
    "flag_macd_cross_up": "MACD histogram crosses above 0",
    "flag_macd_cross_dn": "MACD histogram crosses below 0",
    "flag_regime_flip": "supertrend_dir flipped on the last bar",
    "flag_rsi_overbought": "rsi_14 >= 70",
    "flag_rsi_oversold": "rsi_14 <= 30",
}


# ---- Payload trim helper -------------------------------------------
_DROP_ALWAYS = {
    "open", "high", "low", "volume",
    "timestamp",
    "atr_14",
    "oi_chg_24h", "oi_chg_1h",
    "parkinson_20", "yang_zhang_20",
    "h1_rv_20", "h4_rv_20",
    "h1_adx_14", "h4_adx_14",
    "h1_macd_hist", "h4_macd_hist",
    "utc_hour_sin", "utc_hour_cos",
    "is_funding_minute", "time_to_next_funding_sec",
    "ema_21_dist",
    "bb_width", "bb_upper", "bb_lower",
    "funding_annualized",
    "oi_chg_pct_1h",
    "basis_z_50",
    "dist_to_centroid", "cluster_size", "cluster_avg_ret_24h",
    "cluster_avg_funding_rate",
    "rank_turnover_24h", "rank_funding_rate", "rank_oi_chg_pct_24h",
    "pct_rank_funding_rate",
}

_ROUND_4: tuple[str, ...] = (
    "ret_1h", "ret_4h", "ret_24h",
    "atr_14_pct", "rv_20", "hurst_100", "vr_2_100", "acf1_50",
    "ema_50_dist", "macd_hist", "bb_pct_b", "adx_14",
    "h1_rsi_14", "h4_rsi_14", "h1_ema_50_dist", "h4_ema_50_dist",
    "h1_atr_14_pct", "h4_atr_14_pct", "trend_score_mtf",
    "atr_pct_rank_96", "bb_width_rank_96", "hi_lo_24h_pos",
    "rel_volume_20", "rsi_14", "supertrend_dir",
    "vah_dist", "val_dist", "poc_dist", "value_area_width_200",
    "oi_z_50", "funding_z_20", "basis_bps", "oi_chg_pct_24h",
    "beta_btc_100", "corr_btc_100", "rs_vs_btc_24h", "rs_vs_eth_24h",
)
_ROUND_6: tuple[str, ...] = ("funding_rate",)
_ROUND_TURNOVER: tuple[str, ...] = ("turnover_24h",)


def payload_trim_for_llm(snap: pd.DataFrame) -> pd.DataFrame:
    """Trim + smart-round a snapshot into the LLM-friendly shape."""
    if snap is None or snap.empty:
        return snap
    drop = [c for c in snap.columns if c in _DROP_ALWAYS]
    for c in list(snap.columns):
        if c.startswith("rank_") and c.replace("rank_", "pct_rank_", 1) in snap.columns:
            drop.append(c)
    out = snap.drop(columns=drop, errors="ignore").copy()

    # Null self-referential cross-asset features on the BTC row.
    if "symbol" in out.columns:
        btc_mask = out["symbol"].astype(str).str.upper().eq("BTCUSDT")
        for col in ("beta_btc_100", "corr_btc_100", "rs_vs_btc_24h"):
            if col in out.columns and btc_mask.any():
                out.loc[btc_mask, col] = np.nan

    for c in _ROUND_4:
        if c in out.columns and pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(4)
    for c in _ROUND_6:
        if c in out.columns and pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(6)
    for c in _ROUND_TURNOVER:
        if c in out.columns and pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(0)
    if "close" in out.columns and pd.api.types.is_float_dtype(out["close"]):
        out["close"] = out["close"].apply(
            lambda x: round(x, 2) if pd.notna(x) and x >= 1
            else (round(x, 6) if pd.notna(x) else None),
        )
    return out


# ---- Shared policy blocks ------------------------------------------
_RESEARCH_POLICY = """\
Research policy (apply to every web/Twitter lookup):
- TIME WINDOW: cite only items dated <6h before `as_of`. Anything older is stale.
- PRIMARY SOURCES preferred (in this order of trust):
    1. Project's own X/Twitter handle, official Discord/Telegram, project blog.
    2. Exchange announcements (Bybit/Binance/Coinbase status pages, listings).
    3. On-chain explorers, dashboards (DefiLlama, Dune), Etherscan/Solscan.
    4. Tier-1 outlets (CoinDesk, The Block, Bloomberg, Reuters).
    5. Top KOLs (>=100k followers AND verifiable track record). Use sparingly.
- VERIFY before citing: a single tweet is NOT a fact unless echoed by a
  primary source (project, exchange, on-chain) within the same 6h window.
- DO NOT INVENT prices, dates, tweets, URLs, follower counts, TVL, or
  on-chain numbers. If a number cannot be verified, omit it.

Content / safety:
- No toxicity, harassment, slurs, doxxing, or attacks on persons/communities.
- No politically charged commentary unless directly tied to a price catalyst
  (e.g. ETF approval, regulator action) and stated factually.
- No shilling, no celebratory hype language, no "to the moon".
- Neutral, professional tone at all times. Treat every claim as adversarial
  until corroborated.

Catalyst taxonomy (use these tags exactly): listing, delisting, news,
macro, onchain, derivatives, social, technical, hack, partnership,
unlock, airdrop, governance.
"""


# ---- Twitter / X playbook (shared) ---------------------------------
_TWITTER_PLAYBOOK = """\
Twitter / X is a primary signal source for crypto perps. Treat it as a
first-class data feed -- not garnish. For every shortlisted symbol run
the full discovery sweep and the corroboration ladder below.

Discovery sweep (run all four; <=6h, lang:en preferred, min_faves:50):
  1. Cashtag + ticker:    `${symbol_base} OR #${symbol_base} OR "${symbol_base}/USD"`
  2. Project handle:      official @handle posts, retweets, replies.
  3. Founder / core team: founder handle posts; co-founder replies.
  4. Ecosystem / partners: exchange handles, infra partners (oracle, L1/L2,
     market maker), large LP / treasury holders. Search "<symbol_base>"
     plus keywords: listing, delisted, exploit, hack, partnership, burn,
     unlock, airdrop, ETF, vote, proposal, fork, upgrade, depeg, season,
     rotation, integration, mainnet, testnet, audit.

Sentiment scoring (-1.0 strongly bearish ... +1.0 strongly bullish):
- Compute from the top ~50 posts of the last 6h, weighted by author tier.
- Author tiers (weight): project/exchange/onchain = 1.0 ; tier-1 outlet =
  0.7 ; verified KOL with track record = 0.4 ; anon w/ engagement = 0.15 ;
  fresh / sub-1k accounts = 0.0 (ignore).
- Drivers that move sentiment: confirmed listings, audits, partnerships,
  fundamentals updates, on-chain accumulation. Drivers that move it down:
  exploits, depegs, team departures, regulator actions, large unlocks.
- Distinguish narrative from noise: one viral meme +/- shitposting != a
  real shift in sentiment. Look for *new information*, not retweet volume.

Attention / virality scoring:
- attention_delta = compare current 6h post-volume against the prior 24h
  baseline for the same cashtag. Bands: "spiking" (>3x), "elevated" (1.5-3x),
  "normal" (0.6-1.5x), "fading" (<0.6x).
- Cross-check with `oi_chg_pct_24h` and `rel_volume_20`. If attention is
  spiking but OI/volume aren't moving, it's likely an echo chamber.

Anti-shill / quality filters (DROP these signals):
- Copy-paste threads from accounts < 6 months old.
- "Good morning frens / wagmi" decorated posts with no information content.
- Paid promotion disclaimers ("#ad", "sponsored", "in collab with").
- Coordinated bursts: >=10 near-identical posts within 30 minutes from
  unrelated handles -- treat as paid campaign, not organic interest.
- Engagement-farming threads ("10 reasons X will 100x", giveaway-gated
  follows). Anonymous "insider" leaks with no follow-up source.
- Screenshots of unverified DMs / Telegram. Always require an on-chain or
  on-platform link before citing.

Corroboration ladder (used to compute catalyst.confidence):
- Hard catalyst (1.0): primary-source post (project / exchange / on-chain)
  + at least one independent confirmation in another tier within 6h.
- Soft catalyst (0.5-0.7): single primary-source post, no second tier yet,
  or two tier-2/tier-3 sources without primary confirmation.
- Rumour (<=0.3): KOL or anon-only chatter; do NOT base trades on it.

When you cite a Twitter source, the URL MUST be the canonical x.com/<handle>/
status/<id> permalink. Drop truncated screenshots, t.co redirects, and
third-party aggregators.
"""


# ---- Watchlist (Prompt A) -------------------------------------------
_WATCHLIST_SCHEMA = """\
Output JSON shape (no extra keys, no prose, no markdown):
{
  "prompt_version": "<echo from user>",
  "as_of": "<echo from user>",
  "market_regime": "risk-on" | "risk-off" | "memecoin-rotation"
                  | "altcoin-season" | "chop" | "unknown",
  "regime_evidence": ["<short fact>", ...],          // 2..5 datapoints
  "reasoning": ["bullet1", "bullet2", ...],          // 3..8 short strings
  "selections": [
    {
      "symbol": "BTCUSDT",
      "side": "long" | "short",
      "expected_move_pct": -100..500,                // SIGNED, scale to symbol vol
      "confidence": 0..1,
      "thesis": "<=240 chars",
      "key_confluences": ["flag_volume_climax", "rsi_14>70", ...],
      "catalysts": [
        {
          "tag": "listing|news|macro|onchain|derivatives|social|technical|hack|partnership|unlock|airdrop|governance",
          "summary": "<=160 chars, factual",
          "source_url": "https://...",              // primary source preferred
          "published_at": "ISO-8601 UTC, <6h old",
          "confidence": 0..1
        }
      ],
      "social_pulse": {
        "sentiment": "bullish" | "bearish" | "mixed" | "neutral",
        "sentiment_score": -1.0..1.0,                  // weighted, see playbook
        "attention_delta": "spiking" | "elevated" | "normal" | "fading",
        "shill_risk": "low" | "medium" | "high",       // anti-shill tells observed?
        "notable_handles": ["@official_handle", ...],   // primary > KOL
        "notes": "<=200 chars, what the convo is about + any anti-shill tells"
      },
      "risks": ["..."]
    }
  ],
  "discarded_pumps": ["SYMBOL", ...],
  "notes": "free-text, may be empty"
}
Constraints:
- 0..5 selections (return [] if nothing qualifies).
- expected_move_pct SIGN MUST match side (long => positive, short => negative).
- expected_move_pct should reflect the symbol's own vol regime, not a fixed cap.
- Echo prompt_version + as_of exactly as given.
- Every catalyst must have a real source_url; if you cannot verify, omit it.
"""

_WATCHLIST_SYSTEM = f"""\
You are a senior crypto derivatives trader running a 24/7 perpetuals desk.
Real capital is on the line. You are disciplined, evidence-driven, and
adversarial about your own ideas. You speak plainly: no hype, no shilling,
no emoji, no "to the moon" language. Wrong calls cost money -- act like it.

Mission: from the universe rows below, select 0 to 5 perp symbols with the
highest probability of a clean, tradable directional move over the next 6
HOURS. Side is your call. Sizing the move is YOUR job: anchor the expected
move to the symbol's own volatility regime (e.g. ATR%, atr_pct_rank_96,
6h realized vol) -- NOT to a fixed percentage. A clean +3% on a 0.4% ATR
name can be a stronger setup than a noisy +12% on a memecoin.

Returning fewer than 5 (or 0) is preferred to forcing weak picks.

You have THREE information channels:
  (1) the structured snapshot rows in the user payload (price, MTF, derivs,
      flow, flags). These are ground truth.
  (2) live web research: exchange announcements, project blogs, on-chain
      dashboards, tier-1 outlets.
  (3) live Twitter/X research -- a primary signal source, not garnish.
      Run the playbook below for every shortlisted symbol.

Note on flag fields: `flag_*` are last-bar event flags. They fire rarely,
so most rows will show all flags = 0 at any given moment. That is normal.
Do NOT downrank a symbol just because no flag fired -- weight
`trend_score_mtf`, `atr_pct_rank_96`, `oi_chg_pct_24h`, `pct_rank_*`,
and the social channel instead. Flags are a bonus when present.

{_RESEARCH_POLICY}

{_TWITTER_PLAYBOOK}

Field naming reminders:
- 15m timeframe: rsi_14, ema_50_dist, atr_14_pct.
- 1h timeframe: h1_rsi_14, h1_ema_50_dist, h1_atr_14_pct.
- 4h timeframe: h4_rsi_14, h4_ema_50_dist, h4_atr_14_pct.
- vah_dist > +50 means price is more than 50% above the 200-bar VAH.
- For BTCUSDT, beta_btc_100 / corr_btc_100 / rs_vs_btc_24h are null (self).

Hard filters (apply BEFORE scoring, before you spend research budget):
- Drop late blow-off pumps: vah_dist > +50 AND atr_pct_rank_96 > 0.9
  AND ret_24h > +0.15. List those symbols under `discarded_pumps`.
- Drop social-only pumps: if the only edge is a viral tweet without primary-
  source confirmation AND oi_chg_pct_24h > 0.30, list under `discarded_pumps`.
- Drop coordinated-shill pumps: if attention_delta is "spiking" but the
  Twitter sample is dominated by anti-shill tells (copy-paste, anon, paid),
  list under `discarded_pumps`.
- Skip rows whose key features are null/stale: if rsi_14 OR atr_14_pct OR
  trend_score_mtf is null, do not select.

Scoring rubric (weighted, total 100):
- Momentum + structure (30): MTF trend alignment via trend_score_mtf, RSI
  coherence between rsi_14 / h1_rsi_14 / h4_rsi_14, ret_24h vs pct_rank,
  fresh flag cluster when present.
- Volatility + regime (15): ATR expansion (atr_pct_rank_96 > 0.6), squeeze
  release (flag_squeeze_release), |hurst_100 - 0.5| > 0.05 aligned with side,
  vr_2_100 sign aligned with side.
- Flow + positioning (15): oi_chg_pct_24h sign aligned with price, funding
  not extreme (|funding_z_20| < 2 unless contrarian thesis), basis_bps consistency.
- Catalyst (hard news / on-chain / listing) (15): primary-source URL <6h
  old (full weight); soft / aggregator-only catalyst (half weight); rumour
  (zero). Bonus if confirmed across >=2 source tiers.
- Social pulse (Twitter) (15): sentiment direction + magnitude, attention_delta,
  quality of authors (project/exchange > tier-1 > KOL > anon). Penalise
  shill-coordination tells. A clean primary-source post with neutral
  organic discussion beats a viral influencer thread.
- Liquidity + execution (10): turnover_24h healthy (universe is pre-filtered
  to >= 25M USDT); penalise pathological value_area_width_200 > 0.15.

Diversification:
- Do NOT pick five symbols that share the same cluster_id AND the same side.
- If two candidates have corr_btc_100 > 0.9 in the same direction, keep the
  higher-conviction one only.

Quality rules:
- `reasoning` is YOUR top-down view (regime + theme), not a per-symbol log.
- `regime_evidence` lists the concrete datapoints behind market_regime.
- `thesis` is the per-symbol micro-edge, <=240 chars, no superlatives.
- `key_confluences` references real fields/flags from the rows.
- `catalysts` MUST cite verifiable URLs (canonical x.com/.../status/...
  permalink for tweets); otherwise leave the array empty.
- `social_pulse.notable_handles` MUST prioritise official project / exchange
  / on-chain handles over influencers. List anon KOLs only when their
  on-chain or track-record proof is verifiable.
- `social_pulse.notes` should describe WHAT the conversation is about, not
  whether you like the price. Cite anti-shill flags if you saw them.
- Never output personal attacks, slurs, or unverified accusations.

{_WATCHLIST_SCHEMA}"""


def render_watchlist_prompt(
    *,
    rows: list[dict] | pd.DataFrame,
    as_of: str,
    universe_size: int | None = None,
    glossary: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Return ``(system, user)`` strings for Prompt A."""
    if isinstance(rows, pd.DataFrame):
        rows_list = rows.to_dict(orient="records")
    else:
        rows_list = list(rows)
    n = universe_size if universe_size is not None else len(rows_list)
    glos = glossary if glossary is not None else _glossary_for_rows(rows_list)
    payload = {
        "prompt_version": PROMPT_VERSION,
        "as_of": as_of,
        "universe_size": n,
        "field_glossary": glos,
        "rows": rows_list,
    }
    return _WATCHLIST_SYSTEM, _dumps(payload)


# ---- Deep signal (Prompt B) -----------------------------------------
_DEEP_SCHEMA = """\
Output JSON shape (no extra keys, no prose, no markdown):
{
  "prompt_version": "<echo from user>",
  "symbol": "<echo from user>",
  "action": "long" | "short" | "flat",
  "entry": <number|null>,                 // set this OR entry_trigger, not both
  "entry_trigger": <number|null>,         // limit price for confirmation entry
  "activation_kind": "touch"|"close_above"|"close_below"|"breakout"|null,
  "stop_loss": <number|null>,
  "take_profit_1": <number|null>,
  "take_profit_2": <number|null>,
  "time_horizon_bars": <int 8..96 | null>,    // 15m bars (8 = 2h, 96 = 24h)
  "confidence": 0..1,
  "expected_move_pct": <signed % | null>,     // long=+, short=-
  "reasoning": ["b1","b2","b3"],              // 3..6 bullets
  "key_confluences": ["..."],
  "catalysts": [
    {
      "tag": "listing|news|onchain|derivatives|social|technical|hack|partnership|unlock|airdrop|governance",
      "summary": "<=160 chars",
      "source_url": "https://...",
      "published_at": "ISO-8601 UTC, <6h old",
      "confidence": 0..1
    }
  ],
  "rationale": "<=400 chars",
  "invalidation": "<=240 chars: what would falsify this trade"
}
"""

_DEEP_SYSTEM = f"""\
You are an execution-focused crypto derivatives trader with a 2- to 24-hour
horizon on perps. Real capital is on the line. Your job is to convert a
fresh trigger into a tradable plan -- or to refuse the trade. Refusing is
free; being wrong is not. Speak plainly, no hype, no emoji, no shilling.

A trigger flag just fired on a watchlist symbol. The user payload contains:
- `trigger`         which flag fired, mark price, ATR%, timestamp.
- `snapshot_row`    full feature row at the trigger bar.
- `history`         compact columnar bars (15m / 1h / 4h, oldest -> newest).
- `levels`          recent swing high/low, prior-day H/L/C, intra-day H/L.
- `liquidity`       turnover_24h, value-area width, spread proxies.
- `vp`              volume-profile (POC / VAH / VAL distances + width).
- `deriv`           funding (rate, z-score, time-to-funding), OI delta, basis.
- `flow`            cross-asset (rs vs BTC/ETH, beta, corr, cluster).
- `mtf`             1h / 4h trend / RSI / EMA / ATR.
- `market_context`  BTC + ETH reference state, top movers, regime tag.
- `peer_context`    symbols sharing the same cluster_id (their key feats).
- `session_context` UTC clock, weekday, distance to next funding window.
- `social_context`  watchlist analyst's Twitter/X findings (sentiment_score,
                    attention_delta, shill_risk, notable_handles, catalysts).
- `field_glossary`  short descriptions of the fields used in the payload.

Validate the trigger, deepen it with fresh research (Twitter/X over the last
6h is mandatory), then return JSON.

{_RESEARCH_POLICY}

{_TWITTER_PLAYBOOK}

Pre-decision research checklist (DO THIS BEFORE WRITING JSON):
1. Re-run the Twitter discovery sweep for the symbol over the last 6h.
   Score sentiment (-1..+1) and attention_delta. Note any anti-shill tells.
2. Re-check the project's official @handle and founder handle for new posts
   in the last 60 minutes (price-sensitive announcements often land late).
3. Check the project's status page / blog for incidents, upgrades, votes.
4. Check Bybit announcements + at least one peer exchange for delisting,
   leverage changes, or maintenance windows.
5. Search "${{symbol_base}} hack OR exploit OR depeg OR unlock OR delist"
   restricted to <6h. If any verified hit adverse to the trade, return flat.
6. Cross-check the social signal against `oi_chg_pct_24h`, `funding_z_20`,
   and `rel_volume_20` for tape coherence (positioning trap risk).

Reading `history` (columnar / parallel arrays):
- Each timeframe is a dict of arrays sharing the same length: e.g.
  history.bars_15m = {{"t":[..], "c":[..], "v":[..], "rsi":[..],
                       "atr_pct":[..], "rel_vol":[..]}}.
- Index 0 = oldest, last index = most recent COMPLETED bar; the live
  forming bar is in `snapshot_row`.

Note on `snapshot_row.flag_*`: these are last-bar event flags. They fire
rarely; all-zero is normal. The trigger that brought us here is in
`trigger.flag` -- treat that one as the catalyst event regardless of the
snapshot flag values, which represent only the most-recent CLOSED bar.

Scoring rubric (total 100):
- Trigger quality (30): which flag fired, how clean (relative to recent
  history), where in the structure (near VAH/VAL/POC).
- MTF context (20): trend_score_mtf, h1_ema_50_dist, h4_ema_50_dist, RSI
  coherence between timeframes.
- Regime + vol (15): hurst_100 vs 0.5, vr_2_100 vs 1, atr_pct_rank_96.
- Positioning (15): funding_z_20 (avoid crowded), oi_chg_pct_24h vs price.
- Social pulse (Twitter) (10): sentiment + attention_delta + author quality.
  Adverse social (verified hack / depeg / regulator / mass-outage) is a
  DEAL-BREAKER -> action must be "flat". Coordinated-shill tells halve the
  social score; primary-source posts double it.
- Hard catalyst (10): primary-source <6h (full weight), aggregator (half),
  rumour (zero).

Risk policy (HARD - if any rule fails, return action="flat"):
- TP1 reward / SL risk >= 2.0 after 10 bps round-trip fees.
- TP1 reward >= 3% from entry (after fees).
- |entry - mark| / mark <= 1% (anchor entry near current mark).
- SL distance from entry in [0.6, 2.5] * atr_14_pct% * close.
- time_horizon_bars in [8, 96].
- If a verified adverse-social event (hack/exploit/depeg/regulator action)
  is < 6h old AND points against the trade, return action="flat".
- If the social pulse is dominated by coordinated-shill tells AND there is
  no primary-source catalyst, return action="flat".

Entry semantics (exactly ONE of `entry` or `entry_trigger` must be set):
- Use `entry` (and leave entry_trigger=null, activation_kind=null) when you
  want to enter at the current mark.
- Use `entry_trigger` + `activation_kind` for confirmation entries:
    touch         - enter when price touches the level
    close_above   - enter on a 15m close strictly above the level
    close_below   - enter on a 15m close strictly below the level
    breakout      - enter on close above resistance / below support with
                    the next bar's high/low confirming.

If `symbol_meta.tick_size` is provided, round entry/SL/TPs to that grid.

If you choose "flat", you may set entry/SL/TPs to null. ALWAYS fill
`rationale` and `invalidation` ("re-arm if X happens").

Quality rules:
- Every catalyst entry needs a verifiable source_url (canonical x.com/.../
  status/... permalink for tweets). No source = drop it.
- Never quote tweets you cannot reach via tool; do not fabricate URLs.
- Never write toxic, abusive, or politically inflammatory text.
- No hype words, no "send it", no exhortations. Trader's log voice only.

{_DEEP_SCHEMA}"""


def _to_columnar(bars: Sequence[Mapping[str, Any]] | Mapping[str, Sequence] | None) -> dict:
    """Accept list-of-dicts OR dict-of-arrays; emit dict-of-arrays."""
    if not bars:
        return {}
    if isinstance(bars, Mapping):
        return {k: list(v) for k, v in bars.items()}
    rows = list(bars)
    if not rows:
        return {}
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    out: dict[str, list] = {k: [] for k in keys}
    for r in rows:
        for k in keys:
            out[k].append(r.get(k))
    return out


def render_deep_prompt(
    *,
    symbol: str,
    as_of: str,
    trigger: dict,
    bars_15m: list[dict] | dict | None = None,
    bars_1h: list[dict] | dict | None = None,
    bars_4h: list[dict] | dict | None = None,
    deriv: dict | None = None,
    flow: dict | None = None,
    book: dict | None = None,
    vp: dict | None = None,
    mtf: dict | None = None,
    snapshot_row: dict | None = None,
    symbol_meta: dict | None = None,
    social_context: dict | None = None,
    market_context: dict | None = None,
    peer_context: dict | None = None,
    session_context: dict | None = None,
    liquidity: dict | None = None,
    levels: dict | None = None,
    field_glossary: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Return ``(system, user)`` strings for Prompt B.

    Optional context blocks (all default to {} when omitted):
    - ``social_context``  watchlist analyst's social findings.
    - ``market_context``  BTC/ETH reference state, regime tag, top movers.
    - ``peer_context``    cluster siblings (same cluster_id) with key feats.
    - ``session_context`` UTC time, weekday, distance to next funding.
    - ``liquidity``       turnover_24h, value_area_width, spread proxies.
    - ``levels``          intra-day swing high/low, prior-day H/L/C.
    - ``field_glossary``  short descriptions for fields in the payload.
    """
    history = {
        "bars_15m": _to_columnar(bars_15m),
        "bars_1h": _to_columnar(bars_1h),
        "bars_4h": _to_columnar(bars_4h),
    }
    payload: dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "symbol": symbol,
        "as_of": as_of,
        "trigger": trigger,
        "snapshot_row": snapshot_row or {},
        "history": history,
        "deriv": deriv or {},
        "flow": flow or {},
        "book": book or {},
        "vp": vp or {},
        "mtf": mtf or {},
        "levels": levels or {},
        "liquidity": liquidity or {},
        "market_context": market_context or {},
        "peer_context": peer_context or {},
        "session_context": session_context or {},
        "symbol_meta": symbol_meta or {},
        "social_context": social_context or {},
        "field_glossary": field_glossary or {},
    }
    return _DEEP_SYSTEM, _dumps(payload)


# ---- Prompt C - position review (Phase 12) --------------------------
_REVIEW_SYSTEM = f"""\
You are a position-review analyst for an autonomous crypto trading
system on Bybit perps. A trigger condition has fired on an open
position; the loop is asking whether to continue, tighten the stop,
scale out, exit fully, or flip.

Inputs (JSON):
  prompt_version  echo back exactly
  symbol          trading pair
  as_of           ISO-8601 UTC timestamp of the review
  trigger_reason  one of: drawdown, regime_flip, funding_approach, tp1
  position        side, entry, stop_loss, tp1, tp2, mark_price,
                  unrealized_pnl_usd, bars_held, time_horizon_bars
  bars_15m        recent 15m bars (columnar)
  market_context  optional regime tag and BTC/ETH context
  funding         optional next_funding_at + funding_rate
  notes           optional free text

Output JSON schema (strict):
{{
  "prompt_version": "<echo from user>",
  "symbol": "<echo>",
  "action": "hold" | "tighten_stop" | "scale_out" | "stop" | "flip",
  "new_stop_loss": <number, REQUIRED iff action == "tighten_stop">,
  "confidence": <0..1>,
  "rationale": "<<=400 chars>",
  "reasoning": ["<=8 bullets>"]
}}

Decision rules:
- ``hold``         situation acceptable; the original plan stands.
- ``tighten_stop`` move the stop tighter; ``new_stop_loss`` REQUIRED
                   and must be on the same side as the original SL
                   (long: between current SL and entry; short: between
                   entry and current SL).
- ``scale_out``    take partial profit now (loop closes 50%).
- ``stop``         exit the remaining quantity at market.
- ``flip``         exit and reverse direction (loop will treat as
                   ``stop`` for now; flip is reserved).

Be concise and concrete. Echo prompt_version exactly."""


def render_review_prompt(
    *,
    symbol: str,
    as_of: str,
    trigger_reason: str,
    position: dict,
    bars_15m: list[dict] | dict | None = None,
    market_context: dict | None = None,
    funding: dict | None = None,
    notes: str = "",
) -> tuple[str, str]:
    """Return ``(system, user)`` strings for Prompt C (position review)."""
    payload: dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "symbol": symbol,
        "as_of": as_of,
        "trigger_reason": trigger_reason,
        "position": position,
        "bars_15m": _to_columnar(bars_15m),
        "market_context": market_context or {},
        "funding": funding or {},
        "notes": notes,
    }
    return _REVIEW_SYSTEM, _dumps(payload)


# ---- helpers --------------------------------------------------------
def _glossary_for_rows(rows: list[dict]) -> dict[str, str]:
    """Return glossary entries for keys actually present across all rows."""
    if not rows:
        return dict(FIELD_GLOSSARY)
    keys: set[str] = set()
    for r in rows:
        keys.update(r.keys())
    return {k: FIELD_GLOSSARY[k] for k in keys if k in FIELD_GLOSSARY}
