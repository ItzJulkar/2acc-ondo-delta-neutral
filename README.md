# 2acc Ondo Delta Neutral

Dual-account **delta-neutral** limit-order bot for [Ondo Perps](https://ondoperps.xyz).

Two separate Ondo accounts open **equal-and-opposite** positions on **XAU** / **XAG** perps, then close together after a **1% underlying price move**. If one side fills and the other does not, the bot **cancels → reconciles → reprices** the lagging limit every **3 seconds** until sizes match.

> **Not financial advice.** This is experimental trading software. You can lose money (fees, funding, partial fills, latency, liquidation). Use at your own risk.

---

## Order map

| Label | Account | Role |
|-------|---------|------|
| **A** | Account 1 | Long **entry** limit |
| **B** | Account 1 | **Close** long (reduce-only sell limit) |
| **C** | Account 2 | Short **entry** limit |
| **D** | Account 2 | **Close** short (reduce-only buy limit) |

- **A + C** share the same size (and same entry price when possible).
- **B + D** share the same close price.
- Built-in exchange TP/SL is **not** used (those fire as market orders). The bot places its own reduce-only limits.

---

## Strategy (high level)

```
1. Size = min(acc1, acc2 available margin) × 40% × leverage / price
2. Place A + B + C + D TOGETHER (limit / maker, NO exchange TP/SL):
   - A = Acc1 long entry @ P
   - C = Acc2 short entry @ P (same size)
   - B = Acc1 close sell @ P×(1+1%)
   - D = Acc2 close buy @ same price as B
3. When price moves ~1%, B and D fill → both flat ≈ net zero (minus fees)
4. 5th order ONLY if needed:
   - Acc1 fully flat (A+B done) but Acc2 still short (D not filled)
     → cancel D → place E at current order-book price to close the short
   - Symmetric if Acc2 flat but Acc1 still long
5. When both flat → next cycle
```

**1% = underlying price move**, not ROI. At 20×, a 1% price move is roughly ~20% position ROI on each side (opposite signs → combined ≈ flat minus fees/funding/slippage).

---

## Defaults

| Setting | Default |
|---------|---------|
| Markets | `XAU-USD.P`, `XAG-USD.P` |
| Leverage | 20× |
| Margin usage | 40% of available (per account, sized to the weaker one) |
| Close target | 1% price move either way |
| Reprice interval | 3 seconds |
| Order mode | `fast_limit` (aggressive book limit; may take if it crosses) |
| Liq buffer check | warn if estimated distance &lt; 7% |

### Order modes

- **`fast_limit`** (default): limit at best ask (buys) / best bid (sells) for faster fill. Can be taker if it crosses.
- **`strict_maker`**: `postOnly=true`. Exchange rejects if it would take (`post_only_has_match`); bot reprices next loop. **No fill guarantee.**

### Liquidation note

At 20×, isolated-style theoretical buffer is often **under 7%**. The bot checks exchange `liquidationPrice` after fill when available. If the 7% rule blocks every cycle, either:

- lower `risk.min_liq_distance_pct`, or  
- keep extra unused margin (cross-style cushion), or  
- lower leverage in `config.yaml`.

---

## Requirements

- Python **3.10+**
- Two Ondo Perps accounts with API keys  
  Docs: [API Key Authentication](https://docs.ondoperps.xyz/api-reference/api_key_authentication)
- Key safety:
  - **Trade** permission only  
  - **Withdrawal OFF**  
  - **IP whitelist ON**  
  - Never commit `.env` or paste secrets into GitHub / chat

---

## Quick start (friends)

```bash
# 1. Clone
git clone https://github.com/ItzJulkar/2acc-ondo-delta-neutral.git
cd 2acc-ondo-delta-neutral

# 2. Virtualenv (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

# 3. Install
pip install -r requirements.txt

# 4. Config
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux
copy config.example.yaml config.yaml

# 5. Edit .env — put BOTH accounts' keys
#    ONDO_KEY_ID_1 / ONDO_API_SECRET_1
#    ONDO_KEY_ID_2 / ONDO_API_SECRET_2

# 6. Smoke test (balances + markets)
python main.py --check

# 7. Optional dry run (no real orders; simulated fills)
python main.py --dry-run

# 8. Live
python main.py
```

Logs: `logs/bot.log`

Stop: `Ctrl+C` (bot tries to cancel its open orders tagged with `order_prefix`).

---

## Config

Primary file: `config.yaml` (see `config.example.yaml`).

Important keys:

```yaml
markets: ["XAU-USD.P", "XAG-USD.P"]
leverage: 20
sizing:
  margin_usage_pct: 40.0
strategy:
  close_price_pct: 1.0
  close_direction: "either"   # either | up | down
  order_mode: "fast_limit"    # fast_limit | strict_maker
  reprice_sec: 3.0
risk:
  min_liq_distance_pct: 7.0
bot:
  dry_run: false
  order_prefix: "dn2_"
  max_cycles: 0               # 0 = infinite
```

Environment (`.env`):

```env
ONDO_KEY_ID_1=ondoKeyId_...
ONDO_API_SECRET_1=ondoApiSecret_...
ONDO_KEY_ID_2=ondoKeyId_...
ONDO_API_SECRET_2=ondoApiSecret_...
# ONDO_BASE_URL=https://api.ondoperps.xyz
# DRY_RUN=true
```

---

## Project layout

```
2acc Ondo delta neutral/
├── main.py                 # entrypoint
├── config.yaml             # local settings (no secrets)
├── config.example.yaml
├── .env.example            # secret template only
├── requirements.txt
├── bot/
│   ├── client.py           # Ondo REST + HMAC auth (dual instances)
│   ├── config.py
│   ├── engine.py           # A/C → B/D state machine + reprice
│   ├── risk.py             # equal sizing + liq checks
│   ├── models.py
│   └── main.py
├── scripts/check_connection.py
└── logs/
```

---

## Risk & limitations

- Maker-only **and** instant fill cannot both be guaranteed.
- Cancel races: an order can fill while cancel is in flight — bot waits for terminal status before replacing.
- Partial fills: only residual (imbalance) size is re-ordered.
- Funding, fees, and reprice slippage mean combined PnL is **not** exactly zero.
- Two accounts posting opposite limits at the same price may hit **self-match / STP** rules on some venues; monitor logs for `order_matches_against_self` / `selfMatchPrevention`.
- **Never** share API secrets. Rotate keys if exposed.

---

## API reference

- Auth: https://docs.ondoperps.xyz/api-reference/api_key_authentication  
- Create order: https://docs.ondoperps.xyz/api-reference/orders/create-order  
- Markets: `XAU-USD.P`, `XAG-USD.P` (20× max) — https://docs.ondoperps.xyz/markets  

---

## License

MIT — use freely; no warranty.
