"""
worldcup_top3.py — pick the EV-optimal WC 2026 podium (gold/silver/bronze).

Separate one-off bet, locked before the knockouts start:
    10p correct gold (champion) · 8p correct silver (runner-up) · 6p correct bronze (third)

We read Polymarket's tournament futures markets and turn them into finishing-
position probabilities — no bracket simulation needed:

    P(gold)   = P(win the World Cup)
    P(silver) = P(reach final) - P(win)              # reached the final, lost it
    P(bronze) = 0.5 * (P(reach SF) - P(reach final)) # lost the SF, ~50% win the 3rd-place game

Then we choose three DISTINCT teams maximising
    EV = 10*P(gold) + 8*P(silver) + 6*P(bronze)

Run: python worldcup_top3.py
     python worldcup_top3.py --top 12     # how many candidates to list per medal
"""

import sys
import re
import argparse
from itertools import permutations

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from worldcup import fetch_wc_events, yes_price

POINTS = {"gold": 10, "silver": 8, "bronze": 6}

# Title fragments identifying each tournament-level market (case-insensitive).
# Order matters: 'reach final' must not match the plain winner market.
_MARKETS = {
    "win":   r"world cup winner\b",
    "final": r"nation to reach final",
    "semi":  r"nation to reach semifinal",
}
_SKIP_OUTCOME = ("other", "field", "any other", "no team")


def _norm(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\bcongo dr\b", "dr congo", s)
    s = re.sub(r"\bunited states\b", "usa", s)
    # Winner market spells it 'Bosnia-Herzegovina'; reach-final/SF use 'Bosnia and
    # Herzegovina'. Collapse both to one key so the team isn't split across markets.
    s = re.sub(r"\bbosnia[-\s]herzegovina\b", "bosnia and herzegovina", s)
    return re.sub(r"\s+", " ", s)


def extract_market(events: list[dict], pattern: str) -> dict[str, float]:
    """Return {team: yes_price} for the futures market whose title matches pattern.
    Picks the market with the most priced team outcomes (the real one)."""
    best: dict[str, float] = {}
    for ev in events:
        title = ev.get("title", "")
        if " vs" in title.lower():
            continue
        if not re.search(pattern, title, re.I):
            continue
        if re.search(r"\bgroup\b|continent|under-|u-?\d", title, re.I):
            continue
        probs: dict[str, float] = {}
        for m in ev.get("markets", []):
            team = m.get("groupItemTitle", "").strip()
            if not team or any(k in team.lower() for k in _SKIP_OUTCOME):
                continue
            p = yes_price(m)
            if 0 < p <= 1:
                probs[team] = p
        if len(probs) > len(best):
            best = probs
    return best


def finishing_probs(events: list[dict]) -> tuple[dict, dict, dict, dict]:
    """Return (gold, silver, bronze, display_name) keyed by normalised team."""
    win   = extract_market(events, _MARKETS["win"])
    final = extract_market(events, _MARKETS["final"])
    semi  = extract_market(events, _MARKETS["semi"])

    if not win or not final or not semi:
        missing = [k for k, v in (("winner", win), ("reach-final", final),
                                  ("reach-SF", semi)) if not v]
        raise SystemExit(f"Missing market(s): {', '.join(missing)} — cannot compute podium.")

    # Map normalised team -> a display name (prefer the winner market's spelling)
    display: dict[str, str] = {}
    for src in (win, final, semi):
        for name in src:
            display.setdefault(_norm(name), name)

    win_n   = {_norm(k): v for k, v in win.items()}
    final_n = {_norm(k): v for k, v in final.items()}
    semi_n  = {_norm(k): v for k, v in semi.items()}

    gold, silver, bronze = {}, {}, {}
    for t in display:
        g = win_n.get(t, 0.0)
        f = final_n.get(t, 0.0)
        s = semi_n.get(t, 0.0)
        gold[t]   = g
        silver[t] = max(0.0, f - g)              # reached final, lost it
        bronze[t] = max(0.0, 0.5 * (s - f))      # lost SF, ~50% win bronze game
    return gold, silver, bronze, display


def best_podium(gold: dict, silver: dict, bronze: dict) -> tuple[tuple[str, str, str], float]:
    """Exact EV-optimal distinct (gold, silver, bronze) assignment.

    Only the top handful of teams per medal can appear in the optimum, so we
    brute-force over those candidates (fast and exact)."""
    cg = sorted(gold,   key=lambda t: -gold[t])[:8]
    cs = sorted(silver, key=lambda t: -silver[t])[:8]
    cb = sorted(bronze, key=lambda t: -bronze[t])[:8]
    pool = list(dict.fromkeys(cg + cs + cb))

    best_combo, best_ev = None, -1.0
    for g, s, b in permutations(pool, 3):
        ev = POINTS["gold"] * gold[g] + POINTS["silver"] * silver[s] + POINTS["bronze"] * bronze[b]
        if ev > best_ev:
            best_ev, best_combo = ev, (g, s, b)
    return best_combo, best_ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10, help="Candidates to list per medal")
    args = ap.parse_args()

    print("=" * 60)
    print("   FIFA WC 2026 — EV-OPTIMAL PODIUM (10/8/6)")
    print("=" * 60)
    print("Fetching Polymarket futures markets...", flush=True)
    events = fetch_wc_events()
    gold, silver, bronze, name = finishing_probs(events)

    def table(label: str, probs: dict, pts: int):
        print(f"\n  {label}  ({pts}p)  — top {args.top} by P:")
        for t in sorted(probs, key=lambda x: -probs[x])[: args.top]:
            print(f"     {name[t]:<26} {probs[t]*100:5.1f}%   EV {pts*probs[t]:.2f}")

    table("GOLD  (champion)", gold, POINTS["gold"])
    table("SILVER (runner-up)", silver, POINTS["silver"])
    table("BRONZE (third)", bronze, POINTS["bronze"])

    (g, s, b), ev = best_podium(gold, silver, bronze)
    print("\n" + "=" * 60)
    print("  EV-OPTIMAL PICK (three distinct teams):")
    print(f"     GOLD   : {name[g]:<24} {gold[g]*100:5.1f}%   -> {POINTS['gold']*gold[g]:.2f}p")
    print(f"     SILVER : {name[s]:<24} {silver[s]*100:5.1f}%   -> {POINTS['silver']*silver[s]:.2f}p")
    print(f"     BRONZE : {name[b]:<24} {bronze[b]*100:5.1f}%   -> {POINTS['bronze']*bronze[b]:.2f}p")
    print(f"     ----> total expected {ev:.2f}p")
    print("=" * 60)


if __name__ == "__main__":
    main()
