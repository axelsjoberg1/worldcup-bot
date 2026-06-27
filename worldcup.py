"""World Cup 2026 match predictor using Polymarket prediction markets.

For each match shows:
  - Most likely exact final score (from Exact Score markets)
  - Win/Draw/Loss probabilities
  - Total-goals Over/Under lines (also folded into the Poisson model so the
    predicted score distribution matches the market's implied goal count)

Run: python worldcup.py
     python worldcup.py --group A
"""

import json
import math
import sys
import re
import argparse
import requests
import urllib3

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

GAMMA_URL  = "https://gamma-api.polymarket.com"
WC_TAG_ID  = 102232


def fetch_wc_events() -> list[dict]:
    events: list[dict] = []
    offset = 0
    while True:
        try:
            resp = requests.get(
                f"{GAMMA_URL}/events",
                params={
                    "tag_id": WC_TAG_ID,
                    "limit": 100,
                    "offset": offset,
                    "active": True,
                    "closed": False,
                },
                timeout=20,
                verify=False,
            )
            resp.raise_for_status()
            batch = resp.json()
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            break

        if not isinstance(batch, list) or not batch:
            break
        events.extend(batch)
        if len(batch) < 100:
            break
        offset += 100

    return events


def _named_price(market: dict, target: str) -> float:
    oc = market.get("outcomes", "[]")
    pr = market.get("outcomePrices", "[]")
    if isinstance(oc, str): oc = json.loads(oc)
    if isinstance(pr, str): pr = json.loads(pr)
    for i, name in enumerate(oc):
        if name.strip().lower() == target and i < len(pr):
            return float(pr[i])
    return 0.0


def yes_price(market: dict) -> float:
    return _named_price(market, "yes")


def over_price(market: dict) -> float:
    return _named_price(market, "over")


def canonical_title(title: str) -> str:
    """Strip suffixes like '- Exact Score', '- Halftime Result', '- More Markets' and normalize team name variants."""
    t = re.sub(r"\s*-\s*(exact score|halftime result|more markets)\s*$", "", title, flags=re.IGNORECASE).strip()
    t = re.sub(r"\bBosnia-Herzegovina\b", "Bosnia and Herzegovina", t)
    t = re.sub(r"\s+", " ", t)
    return t


def group_events(events: list[dict]) -> dict[str, dict]:
    """
    Returns dict keyed by canonical match title, value = {
        'title': str,
        'start_date': str | None,
        'result_liquidity': float,
        'score_liquidity': float,
        'winner_markets': [market, ...],
        'score_markets': [market, ...],
    }
    """
    grouped: dict[str, dict] = {}
    seen_market_ids: set[str] = set()

    for ev in events:
        raw_title = ev.get("title", "")
        # Must look like "Team A vs. Team B": "vs" must appear within first 30 chars
        if not re.match(r"^.{1,30}\bvs\.?\s", raw_title):
            continue
        # Exclude prop/side markets by colon or known prefix.  Of the suffixed
        # events we keep only "- Exact Score" (per-score prices) and
        # "- More Markets" (the goal Over/Under lines).
        if ":" in raw_title or re.match(r"^world cup\b", raw_title, re.IGNORECASE):
            continue
        suffix_match = re.search(r"\s+-\s+(.+)$", raw_title, re.IGNORECASE)
        if suffix_match:
            suffix = suffix_match.group(1).lower()
            if "exact score" not in suffix and "more markets" not in suffix:
                continue

        canon = canonical_title(raw_title)
        if canon not in grouped:
            grouped[canon] = {
                "title": canon,
                "start_date": None,
                "result_liquidity": 0.0,
                "score_liquidity": 0.0,
                "winner_markets": [],
                "score_markets": [],
                "goals_markets": [],
            }

        entry = grouped[canon]
        sd = ev.get("endDate")
        if sd and (entry["start_date"] is None or sd < entry["start_date"]):
            entry["start_date"] = sd
        liq = float(ev.get("liquidity") or 0)

        is_exact_score_event  = "exact score" in raw_title.lower()
        is_more_markets_event = "more markets" in raw_title.lower()
        is_side_event         = is_more_markets_event or "halftime result" in raw_title.lower()

        if not is_exact_score_event and not is_side_event:
            entry["result_liquidity"] += liq

        for m in ev.get("markets", []):
            mid = m.get("id", "")
            if mid and mid in seen_market_ids:
                continue
            if mid:
                seen_market_ids.add(mid)
            label = m.get("groupItemTitle", "")
            if is_exact_score_event or label.lower().startswith("exact score:"):
                entry["score_markets"].append(m)
                entry["score_liquidity"] += float(m.get("volumeNum") or 0)
            elif is_more_markets_event:
                entry["goals_markets"].append(m)
            elif not is_side_event:
                entry["winner_markets"].append(m)

    return grouped


def get_winner_probs(winner_markets: list[dict]) -> list[tuple[str, float]]:
    """Return exactly [home, draw, away] probabilities from the 3 binary win/draw/win markets."""
    rows: list[tuple[str, float]] = []
    for m in winner_markets:
        label = m.get("groupItemTitle", "")
        if not label:
            continue
        # Only include team win or draw markets, skip prop markets (BTTS, handicap, etc.)
        is_draw = label.lower().startswith("draw")
        is_team = not is_draw and not any(
            kw in label.lower()
            for kw in ["both teams", "over", "under", "score", "goal", "card", "corner", "clean sheet"]
        )
        if not (is_draw or is_team):
            continue
        p = yes_price(m)
        if p > 0:
            display = "Draw" if is_draw else label
            rows.append((display, p))
    return sorted(rows, key=lambda x: x[1], reverse=True)


def _norm_team(s: str) -> str:
    s = re.sub(r"\bBosnia-Herzegovina\b", "Bosnia and Herzegovina", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def _drop_nonmonotonic_over(items: list[tuple[float, float]], tol: float = 0.05) -> list[tuple[float, float]]:
    """Filter Over/Under lines (sorted ascending by line) so the Over price never
    increases with the line. Untraded high goal lines default to 0.50 on
    Polymarket; left in, they would distort the Poisson fit upward. Over(line)
    must be non-increasing, so any line jumping back above the running minimum
    (beyond a small noise tolerance) is dropped."""
    kept: list[tuple[float, float]] = []
    ceil: float | None = None
    for line, p in items:
        if ceil is None or p <= ceil + tol:
            kept.append((line, p))
            ceil = p if ceil is None else min(ceil, p)
    return kept


def parse_goals_markets(
    goals_markets: list[dict], home_team: str = "", away_team: str = ""
) -> dict:
    """Extract full-match Over/Under goal lines, per-team O/U lines and Both
    Teams To Score from the '- More Markets' event.

    Returns {
        'total': [(line, p_over), ...],   # full match, line = 0.5, 1.5, ...
        'home':  [(line, p_over), ...],   # home team total goals O/U
        'away':  [(line, p_over), ...],   # away team total goals O/U
        'btts':  p_yes | None,
        'liquidity': float,               # combined volume of the goal markets used
    }
    Half-time / per-half markets are ignored (we model the full 90 minutes).
    """
    home_n, away_n = _norm_team(home_team), _norm_team(away_team)
    total: dict[float, float] = {}
    home:  dict[float, float] = {}
    away:  dict[float, float] = {}
    btts: float | None = None
    liquidity = 0.0

    for m in goals_markets:
        label = m.get("groupItemTitle", "").strip()
        low = label.lower()
        if "half" in low:                      # skip 1st/2nd-half markets
            continue
        vol = float(m.get("volumeNum") or 0)

        if low == "both teams to score":
            p = yes_price(m)
            if 0 < p < 1:
                btts = p
                liquidity += vol
            continue

        # Full-match total: e.g. "O/U 2.5"
        mt = re.fullmatch(r"o/u\s+(\d+(?:\.\d+)?)", low)
        if mt:
            p = over_price(m)
            if 0 < p < 1:
                total[float(mt.group(1))] = p
                liquidity += vol
            continue

        # Per-team total: e.g. "France O/U 1.5"
        mteam = re.fullmatch(r"(.+?)\s+o/u\s+(\d+(?:\.\d+)?)", low)
        if mteam:
            p = over_price(m)
            if not (0 < p < 1):
                continue
            who, line = mteam.group(1).strip(), float(mteam.group(2))
            if who == home_n:
                home[line] = p
                liquidity += vol
            elif who == away_n:
                away[line] = p
                liquidity += vol

    return {
        "total": _drop_nonmonotonic_over(sorted(total.items())),
        "home":  _drop_nonmonotonic_over(sorted(home.items())),
        "away":  _drop_nonmonotonic_over(sorted(away.items())),
        "btts":  btts,
        "liquidity": liquidity,
    }


def _parse_score(s: str) -> tuple[int, int] | None:
    m = re.match(r"^(\d+)-(\d+)$", s.strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


def _score_points(gh: int, ga: int, ah: int, aa: int) -> int:
    def outcome(h, a): return 0 if h == a else (1 if h > a else -1)
    correct = outcome(gh, ga) == outcome(ah, aa)
    hm, am = gh == ah, ga == aa
    if hm and am:        return 6
    if correct and (hm or am): return 4
    if correct:          return 3
    if hm or am:         return 1
    return 0


# The 9 key scores that must all have >= $100 volume to trust the market entirely.
# These cover ~85 % of all WC results; rare scores like 3-3 are excluded from the gate.
_KEY_SCORES = {"0-0", "1-0", "0-1", "1-1", "2-0", "0-2", "2-1", "1-2", "2-2"}
_MARKET_MIN_VOL = 10


def _poisson_match_probs(la: float, lb: float) -> tuple[float, float, float]:
    """P(A wins), P(draw), P(B wins) under independent Poisson scoring (up to 7 goals each)."""
    ea, eb = math.exp(-la), math.exp(-lb)
    pa, pb = [ea], [eb]
    for k in range(1, 8):
        pa.append(pa[-1] * la / k)
        pb.append(pb[-1] * lb / k)
    pw = pd = pl = 0.0
    for h, pha in enumerate(pa):
        for a, pba in enumerate(pb):
            p = pha * pba
            if h > a:    pw += p
            elif h == a: pd += p
            else:        pl += p
    return pw, pd, pl


# Relative weight of each market family when fitting (λ_A, λ_B). The result
# (win/draw/loss) market fixes the *difference* between the teams; the goal
# Over/Under markets fix the *totals*. They are complementary, so combining
# them pins down both parameters far better than the result market alone.
_W_RESULT = 1.0   # per win/draw/loss term  (3 terms)
_W_TOTAL  = 1.0   # per full-match O/U line
_W_TEAM   = 0.7   # per per-team O/U line (partly redundant with the totals)
_W_BTTS   = 0.5   # both-teams-to-score


def _poisson_cdf(k: int, lam: float) -> float:
    """P(N <= k) for N ~ Poisson(lam), k an integer >= 0."""
    if lam <= 0:
        return 1.0
    term = math.exp(-lam)
    cdf = term
    for i in range(1, k + 1):
        term *= lam / i
        cdf += term
    return cdf


def _poisson_over(line: float, lam: float) -> float:
    """P(N > line) for N ~ Poisson(lam); `line` is an x.5 goal line."""
    return 1.0 - _poisson_cdf(int(math.floor(line)), lam)


def _goals_error(la: float, lb: float, goals: dict | None) -> float:
    """Weighted squared error of (λ_A, λ_B) against the goal Over/Under markets."""
    if not goals:
        return 0.0
    err = 0.0
    lt = la + lb
    for line, p in goals.get("total", []):
        err += _W_TOTAL * (_poisson_over(line, lt) - p) ** 2
    for line, p in goals.get("home", []):
        err += _W_TEAM * (_poisson_over(line, la) - p) ** 2
    for line, p in goals.get("away", []):
        err += _W_TEAM * (_poisson_over(line, lb) - p) ** 2
    if goals.get("btts") is not None:
        model = (1 - math.exp(-la)) * (1 - math.exp(-lb))
        err += _W_BTTS * (model - goals["btts"]) ** 2
    return err


def _fit_poisson(
    p_win: float, p_draw: float, p_loss: float, goals: dict | None = None
) -> tuple[float, float]:
    """Grid-search for (λ_A, λ_B) reproducing the result market and, when
    available, the goal Over/Under markets."""
    t = p_win + p_draw + p_loss
    if t <= 0:
        return 1.25, 1.25
    pw, pd, pl = p_win / t, p_draw / t, p_loss / t

    def err_at(la: float, lb: float) -> float:
        cw, cd, cl = _poisson_match_probs(la, lb)
        e = _W_RESULT * ((cw - pw) ** 2 + (cd - pd) ** 2 + (cl - pl) ** 2)
        return e + _goals_error(la, lb, goals)

    best_err = float("inf")
    best_la = best_lb = 1.25

    # Coarse pass: 0.1–4.0 in steps of 0.1
    for i in range(1, 41):
        la = i * 0.1
        for j in range(1, 41):
            lb = j * 0.1
            err = err_at(la, lb)
            if err < best_err:
                best_err, best_la, best_lb = err, la, lb

    # Fine pass: ±0.1 around best in steps of 0.01
    for di in range(-10, 11):
        for dj in range(-10, 11):
            la = max(0.05, round(best_la + di * 0.01, 2))
            lb = max(0.05, round(best_lb + dj * 0.01, 2))
            err = err_at(la, lb)
            if err < best_err:
                best_err, best_la, best_lb = err, la, lb

    return best_la, best_lb


def _score_dist_from_lambda(la: float, lb: float) -> list[tuple[int, int, float]]:
    """Independent-Poisson score distribution for the given goal rates."""
    ea, eb = math.exp(-la), math.exp(-lb)
    pa, pb = [ea], [eb]
    for k in range(1, 8):
        pa.append(pa[-1] * la / k)
        pb.append(pb[-1] * lb / k)
    return [
        (h, a, pa[h] * pb[a])
        for h in range(8)
        for a in range(8)
        if pa[h] * pb[a] > 1e-6
    ]


def _rank_scores(
    raw: list[tuple[int, int, str, float]],
    result_probs: dict[str, float],
    top_n: int,
) -> list[tuple[str, float, float]]:
    """Calibrate raw (h, a, score, weight) tuples to the result-market outcome
    proportions, normalize to probabilities, compute the expected points (EV)
    of guessing each score, and return the top_n scores ranked by EV."""
    if not raw:
        return []

    # Calibrate: rescale each outcome bucket to match result market proportions
    raw_by: dict[str, float] = {"home": 0.0, "draw": 0.0, "away": 0.0}
    for h, a, _, p in raw:
        raw_by["home" if h > a else ("draw" if h == a else "away")] += p

    calibrated: list[tuple[int, int, str, float]] = []
    for h, a, score_str, p in raw:
        outcome = "home" if h > a else ("draw" if h == a else "away")
        scale = result_probs.get(outcome, 0.0) / raw_by[outcome] if raw_by[outcome] > 0 else 1.0
        calibrated.append((h, a, score_str, p * scale))

    total = sum(p for _, _, _, p in calibrated)
    if total <= 0:
        return []

    score_probs = [(h, a, p / total) for h, a, _, p in calibrated]

    result = []
    for h, a, score_str, p in calibrated:
        ev = sum(prob * _score_points(h, a, ah, aa) for ah, aa, prob in score_probs)
        result.append((score_str, p / total, ev))

    return sorted(result, key=lambda x: x[2], reverse=True)[:top_n]


def _goals_display(goals: dict, la: float | None, lb: float | None) -> dict:
    """Pair each market Over/Under probability with the fitted-model probability.
    The model column is None when no Poisson fit was possible."""
    def rows(items, lam):
        return [
            (line, p, (_poisson_over(line, lam) if lam is not None else None))
            for line, p in items
        ]

    lt = (la + lb) if (la is not None and lb is not None) else None
    btts = None
    if goals.get("btts") is not None:
        model = ((1 - math.exp(-la)) * (1 - math.exp(-lb))
                 if (la is not None and lb is not None) else None)
        btts = (goals["btts"], model)

    return {
        "total":     rows(goals.get("total", []), lt),
        "home":      rows(goals.get("home", []), la),
        "away":      rows(goals.get("away", []), lb),
        "btts":      btts,
        "lambda":    ((la, lb) if la is not None else None),
        "liquidity": goals.get("liquidity", 0.0),
    }


def get_score_predictions(
    score_markets: list[dict],
    home_team: str = "",
    winner_rows: list | None = None,
    top_n: int = 5,
    goals_markets: list[dict] | None = None,
    away_team: str = "",
) -> dict:
    """Compute BOTH the market-based and Poisson-based exact-score rankings.

    Returns a dict:
        {
          "market":      [(score, prob, ev), ...],  # empty if no usable prices
          "poisson":     [(score, prob, ev), ...],  # always available
          "use_market":  bool,   # True when every key score is liquid enough
          "min_key_vol": float,  # thinnest key-score volume (liquidity gauge)
          "goals":       {...},  # Over/Under goal lines: market vs model
        }

    Both rankings are calibrated to the same result-market win/draw/loss
    proportions, so the only difference is where the per-score shape comes from:
    live Polymarket prices vs. a fitted independent-Poisson model.  The Poisson
    model is fitted to the result market *and* the goal Over/Under markets
    (full-match totals, per-team totals, both-teams-to-score), which are far
    more liquid than the exact-score markets and pin down the goal totals.
    World Cup games are on neutral ground, so home/away labels are convention.
    """
    goals = parse_goals_markets(goals_markets or [], home_team, away_team)

    # Collect volumes and valid prices for all score markets
    score_volumes: dict[str, float] = {}
    score_prices: dict[str, float] = {}
    for m in score_markets:
        label = m.get("groupItemTitle", "")
        if "any other" in label.lower():
            continue
        m_score = re.search(r"\b(\d+)\s*-\s*(\d+)\b", label)
        if not m_score:
            continue
        score_str = f"{m_score.group(1)}-{m_score.group(2)}"
        vol = float(m.get("volumeNum") or 0)
        score_volumes[score_str] = max(score_volumes.get(score_str, 0), vol)
        p = yes_price(m)
        if 0 < p < 0.30:
            score_prices[score_str] = p

    # All key scores must clear the volume gate before we trust the market
    use_market = all(score_volumes.get(s, 0) >= _MARKET_MIN_VOL for s in _KEY_SCORES)
    min_key_vol = min((score_volumes.get(s, 0) for s in _KEY_SCORES), default=0.0)

    empty = {
        "market": [], "poisson": [], "use_market": False,
        "min_key_vol": min_key_vol, "goals": _goals_display(goals, None, None),
    }
    if not winner_rows:
        return empty

    ht_lower = home_team.lower()
    rp: dict[str, float] = {}
    for name, p in winner_rows:
        nl = name.lower()
        if nl == "draw":
            rp["draw"] = p
        elif nl == ht_lower:
            rp["home"] = p
        else:
            rp["away"] = p

    if not rp:
        return empty

    rt = sum(rp.values())
    result_probs = {k: v / rt for k, v in rp.items()}

    # Fit (λ_home, λ_away) to the result market AND the goal Over/Under markets
    la, lb = _fit_poisson(
        result_probs.get("home", 1 / 3),
        result_probs.get("draw", 1 / 3),
        result_probs.get("away", 1 / 3),
        goals,
    )

    # Market-based ranking (only if any prices survived the filter)
    market_raw: list[tuple[int, int, str, float]] = []
    for score_str, p in score_prices.items():
        parsed = _parse_score(score_str)
        if parsed:
            market_raw.append((parsed[0], parsed[1], score_str, p))
    market_rows = _rank_scores(market_raw, result_probs, top_n)

    # Poisson-based ranking (always available)
    poisson_raw = [(h, a, f"{h}-{a}", p) for h, a, p in _score_dist_from_lambda(la, lb)]
    poisson_rows = _rank_scores(poisson_raw, result_probs, top_n)

    return {
        "market": market_rows,
        "poisson": poisson_rows,
        "use_market": use_market,
        "min_key_vol": min_key_vol,
        "goals": _goals_display(goals, la, lb),
    }


def get_combined_scores(
    score_markets: list[dict],
    home_team: str = "",
    winner_rows: list | None = None,
    top_n: int = 5,
) -> tuple[list[tuple[str, float, float]], bool]:
    """Backward-compatible wrapper: returns (scores, used_market_data) for the
    single recommended method — market when liquid, otherwise Poisson."""
    pred = get_score_predictions(score_markets, home_team, winner_rows, top_n)
    if pred["use_market"] and pred["market"]:
        return pred["market"], True
    return pred["poisson"], False


def _split_teams(title: str) -> tuple[str, str]:
    parts = re.split(r"\s+vs\.?\s+", title, maxsplit=1)
    home = parts[0].strip()
    away = parts[1].strip() if len(parts) > 1 else ""
    return home, away


def print_match(entry: dict) -> None:
    title            = entry["title"]
    result_liq       = entry.get("result_liquidity", 0.0)
    score_liq        = entry.get("score_liquidity", 0.0)
    winner_rows      = get_winner_probs(entry["winner_markets"])
    home_team, away_team = _split_teams(title)
    pred             = get_score_predictions(
        entry["score_markets"], home_team, winner_rows,
        goals_markets=entry.get("goals_markets"), away_team=away_team,
    )

    market_rows  = pred["market"]
    poisson_rows = pred["poisson"]
    liquid       = pred["use_market"]

    start = entry.get("start_date", "")[:10] if entry.get("start_date") else ""
    print(f"\n  {title}" + (f"  [{start}]" if start else ""))

    liq_flag = "[OK  market liquid]" if liquid else "[!]  LOW liquidity -> rely on Poisson"
    print(f"  Liquidity:  result ${result_liq:,.0f}  |  exact score ${score_liq:,.0f}   {liq_flag}")

    def fmt_rows(rows):
        return "  |  ".join(f"{s} EV{ev:.2f} ({p*100:.1f}%)" for s, p, ev in rows)

    # Recommendation: market when it's liquid, otherwise the Poisson model
    if liquid and market_rows:
        rec_rows, rec_name = market_rows, "MARKET "
    else:
        rec_rows, rec_name = poisson_rows, "POISSON"

    if rec_rows:
        top_score, top_p, top_ev = rec_rows[0]
        print(f"  Recommend [{rec_name}]:  {top_score}  (EV {top_ev:.2f}p, {top_p*100:.1f}%)")
    else:
        print(f"  Recommend:  (no prediction available)")

    if market_rows:
        print(f"    Market:   {fmt_rows(market_rows)}")
    else:
        print(f"    Market:   (thin / no usable prices)")
    if poisson_rows:
        print(f"    Poisson:  {fmt_rows(poisson_rows)}")

    if winner_rows:
        parts = "  |  ".join(
            f"{name} {p*100:.0f}%" + (" <--" if i == 0 else "")
            for i, (name, p) in enumerate(winner_rows)
        )
        print(f"  Result:            {parts}")

    goals = pred.get("goals", {})
    total = goals.get("total", [])
    if total:
        cells = "  ".join(f"O{line:g} {mkt*100:.0f}%" for line, mkt, _ in total)
        print(f"  Total goals (mkt): {cells}")
        if goals.get("btts"):
            print(f"  Both teams score:  {goals['btts'][0]*100:.0f}%")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", "-g", help="Filter by group letter (A-L)")
    parser.add_argument("--debug", action="store_true", help="Show raw exact-score event titles and matching info")
    args = parser.parse_args()

    print("=" * 62)
    print("   POLYMARKET -- FIFA WORLD CUP 2026 PREDICTIONS")
    print("=" * 62)
    print("Fetching data...")

    events = fetch_wc_events()
    print(f"  -> {len(events)} markets fetched\n")

    if args.debug:
        print("[DEBUG] Exact Score events found:")
        for ev in events:
            t = ev.get("title", "")
            if "exact score" in t.lower() and " vs" in t.lower():
                print(f"  raw:    {t!r}")
                print(f"  canon:  {canonical_title(t)!r}")
        print()
        print("[DEBUG] First event keys:", list(events[0].keys()) if events else "none")
        print("[DEBUG] First event volume/liquidity fields:", {k: events[0].get(k) for k in events[0] if "liquid" in k.lower() or "volume" in k.lower()} if events else {})
        print()

    # If group filter requested, pre-filter events by tag
    if args.group:
        grp = args.group.upper()
        events = [
            e for e in events
            if any(
                t.get("label", "").upper() == f"GROUP {grp}"
                for t in e.get("tags", [])
            )
        ]

    grouped = group_events(events)

    matches = sorted(grouped.values(), key=lambda x: x.get("start_date") or "9999")

    if not matches:
        print("  No match markets found.")
        return

    print(f"  {len(matches)} matches\n")
    print("-" * 62)

    for entry in matches:
        print_match(entry)

    print("\n" + "=" * 62)
    print("  Prices = Polymarket crowd-consensus probabilities.")
    print("=" * 62)


if __name__ == "__main__":
    main()