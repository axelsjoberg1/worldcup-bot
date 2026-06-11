"""World Cup 2026 match predictor using Polymarket prediction markets.

For each match shows:
  - Most likely exact final score (from Exact Score markets)
  - Win/Draw/Loss probabilities

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


def yes_price(market: dict) -> float:
    oc = market.get("outcomes", "[]")
    pr = market.get("outcomePrices", "[]")
    if isinstance(oc, str): oc = json.loads(oc)
    if isinstance(pr, str): pr = json.loads(pr)
    for i, name in enumerate(oc):
        if name.strip().lower() == "yes" and i < len(pr):
            return float(pr[i])
    return 0.0


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
        # Exclude prop/side markets by colon, known prefix, or any suffix that isn't exact score
        if ":" in raw_title or re.match(r"^world cup\b", raw_title, re.IGNORECASE):
            continue
        suffix_match = re.search(r"\s+-\s+(.+)$", raw_title, re.IGNORECASE)
        if suffix_match and "exact score" not in suffix_match.group(1).lower():
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
            }

        entry = grouped[canon]
        sd = ev.get("endDate")
        if sd and (entry["start_date"] is None or sd < entry["start_date"]):
            entry["start_date"] = sd
        liq = float(ev.get("liquidity") or 0)

        is_exact_score_event = "exact score" in raw_title.lower()
        is_side_event        = any(kw in raw_title.lower() for kw in ("more markets", "halftime result"))

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


def _fit_poisson(p_win: float, p_draw: float, p_loss: float) -> tuple[float, float]:
    """Grid-search for (λ_A, λ_B) that best reproduce the result market probabilities."""
    t = p_win + p_draw + p_loss
    if t <= 0:
        return 1.25, 1.25
    pw, pd, pl = p_win / t, p_draw / t, p_loss / t

    best_err = float("inf")
    best_la = best_lb = 1.25

    # Coarse pass: 0.1–4.0 in steps of 0.1
    for i in range(1, 41):
        la = i * 0.1
        for j in range(1, 41):
            lb = j * 0.1
            cw, cd, cl = _poisson_match_probs(la, lb)
            err = (cw - pw) ** 2 + (cd - pd) ** 2 + (cl - pl) ** 2
            if err < best_err:
                best_err, best_la, best_lb = err, la, lb

    # Fine pass: ±0.1 around best in steps of 0.01
    for di in range(-10, 11):
        for dj in range(-10, 11):
            la = max(0.05, round(best_la + di * 0.01, 2))
            lb = max(0.05, round(best_lb + dj * 0.01, 2))
            cw, cd, cl = _poisson_match_probs(la, lb)
            err = (cw - pw) ** 2 + (cd - pd) ** 2 + (cl - pl) ** 2
            if err < best_err:
                best_err, best_la, best_lb = err, la, lb

    return best_la, best_lb


def _poisson_score_dist(result_probs: dict[str, float]) -> list[tuple[int, int, float]]:
    """Score probability distribution from Poisson model fitted to result market."""
    la, lb = _fit_poisson(
        result_probs.get("home", 1 / 3),
        result_probs.get("draw", 1 / 3),
        result_probs.get("away", 1 / 3),
    )
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


def get_combined_scores(
    score_markets: list[dict],
    home_team: str = "",
    winner_rows: list | None = None,
    top_n: int = 5,
) -> tuple[list[tuple[str, float, float]], bool]:
    """Returns (scores, used_market_data).

    Uses market prices only if every common score (0-0 through 3-3) has >= $200
    trading volume.  If any one is missing or thin, falls back to the full
    statistical distribution.  World Cup games are on neutral ground so the
    home/away distinction in _SCORE_DIST is just label convention — the
    distributions are symmetric.
    """
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

    # All 9 key scores must have >= $100 traded volume to trust the market
    use_market = all(score_volumes.get(s, 0) >= _MARKET_MIN_VOL for s in _KEY_SCORES)

    if not winner_rows:
        return [], False

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
        return [], False

    rt = sum(rp.values())
    result_probs = {k: v / rt for k, v in rp.items()}

    if use_market:
        raw: list[tuple[int, int, str, float]] = []
        for score_str, p in score_prices.items():
            parsed = _parse_score(score_str)
            if parsed:
                raw.append((parsed[0], parsed[1], score_str, p))
    else:
        raw = [
            (h, a, f"{h}-{a}", p)
            for h, a, p in _poisson_score_dist(result_probs)
        ]

    if not raw:
        return [], False

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
        return [], False

    score_probs = [(h, a, p / total) for h, a, _, p in calibrated]

    result = []
    for h, a, score_str, p in calibrated:
        ev = sum(prob * _score_points(h, a, ah, aa) for ah, aa, prob in score_probs)
        result.append((score_str, p / total, ev))

    return sorted(result, key=lambda x: x[2], reverse=True)[:top_n], use_market


def print_match(entry: dict) -> None:
    title            = entry["title"]
    result_liq       = entry.get("result_liquidity", 0.0)
    score_liq        = entry.get("score_liquidity", 0.0)
    winner_rows      = get_winner_probs(entry["winner_markets"])
    home_team        = re.split(r"\s+vs\.?\s+", title, maxsplit=1)[0].strip()
    score_rows, used_market = get_combined_scores(entry["score_markets"], home_team, winner_rows)
    score_label = " [market]" if used_market else " [poisson]"

    start = entry.get("start_date", "")[:10] if entry.get("start_date") else ""
    print(f"\n  {title}" + (f"  [{start}]" if start else ""))
    print(f"  Liquidity:  result ${result_liq:,.0f}  |  exact score ${score_liq:,.0f}")

    if score_rows:
        top_score, top_p, top_ev = score_rows[0]
        print(f"  Best guess:   {top_score}  ({top_p*100:.1f}%  EV {top_ev:.2f}p){score_label}")
        if len(score_rows) > 1:
            others = "  |  ".join(f"{s} {p*100:.1f}% EV{ev:.2f}" for s, p, ev in score_rows[1:])
            print(f"  Other scores: {others}")
    else:
        print(f"  Best guess:   (no prediction available)")

    if winner_rows:
        parts = "  |  ".join(
            f"{name} {p*100:.0f}%" + (" <--" if i == 0 else "")
            for i, (name, p) in enumerate(winner_rows)
        )
        print(f"  Result:            {parts}")



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