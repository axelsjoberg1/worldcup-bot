"""
worldcup_outright.py — Screen Polymarket for WC 2026 outright position markets.

Shows:
  1st place: "World Cup Winner" market (direct)
  2nd place: "Nation to Reach Final" minus "World Cup Winner" (derived)
  3rd place: not available on Polymarket — shown as N/A

Run: python worldcup_outright.py
     python worldcup_outright.py --top 15
"""

import sys
import re
import argparse
from collections import defaultdict

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from worldcup import fetch_wc_events, yes_price


def fetch_market_odds(events: list[dict], title_pattern: str) -> dict[str, float]:
    """
    Find the event matching title_pattern and return {team: yes_price}.
    """
    for ev in events:
        title = ev.get("title", "")
        if re.search(title_pattern, title, re.IGNORECASE):
            odds: dict[str, float] = {}
            for m in ev.get("markets", []):
                team = m.get("groupItemTitle", "")
                p = yes_price(m)
                if team and p > 0 and "field" not in team.lower() and "other" not in team.lower():
                    odds[team] = p
            if odds:
                return odds
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=12, help="Teams to show (default 12)")
    args = parser.parse_args()

    print("=" * 62)
    print("   POLYMARKET — WC 2026 OUTRIGHT POSITION ODDS")
    print("=" * 62)
    print("Fetching data...", flush=True)

    events = fetch_wc_events()

    winner_odds  = fetch_market_odds(events, r"^world cup winner")
    finalist_odds = fetch_market_odds(events, r"nation to reach final")

    if not winner_odds:
        print("[ERROR] Could not find 'World Cup Winner' market.")
        return

    all_teams = sorted(
        set(winner_odds) | set(finalist_odds),
        key=lambda t: -winner_odds.get(t, 0)
    )

    print(f"  {len(winner_odds)} teams in Winner market")
    print(f"  {len(finalist_odds)} teams in Finalist market")
    print()

    # Derived 2nd place: P(reach final) - P(win)
    # Derived 3rd place: not available

    col_w = 28
    print(f"  {'Team':<{col_w}} {'1st (win)':>10}  {'2nd (derived)':>14}  {'3rd':>6}")
    print("  " + "─" * 62)

    rows = []
    for team in all_teams:
        p1   = winner_odds.get(team, 0.0)
        pfin = finalist_odds.get(team, 0.0)
        p2   = max(0.0, pfin - p1)       # finalist but not winner
        rows.append((team, p1, p2, pfin))

    rows.sort(key=lambda x: -x[1])

    for team, p1, p2, pfin in rows[: args.top]:
        p2_str  = f"{p2*100:6.1f}%"  if p2 > 0 else "    —  "
        fin_str = f"(finalist {pfin*100:.1f}%)" if pfin > 0 else ""
        print(f"  {team:<{col_w}} {p1*100:8.1f}%  {p2_str:>14}  {'N/A':>6}  {fin_str}")

    print()
    print("  Notes:")
    print("  • 1st  = Polymarket 'World Cup Winner' market (direct)")
    print("  • 2nd  = 'Nation to Reach Final' minus 'Winner' (derived)")
    print("  • 3rd  = No Polymarket market exists for this position")
    print()

    # Best guess
    print("  BEST GUESS (Polymarket consensus, no overlap):")
    used: set[str] = set()

    best_1st = max(winner_odds, key=winner_odds.get)
    used.add(best_1st)
    print(f"    1st: {best_1st}  ({winner_odds[best_1st]*100:.1f}%)")

    p2_map = {t: max(0, finalist_odds.get(t, 0) - winner_odds.get(t, 0))
              for t in finalist_odds if t not in used}
    if p2_map:
        best_2nd = max(p2_map, key=p2_map.get)
        used.add(best_2nd)
        print(f"    2nd: {best_2nd}  ({p2_map[best_2nd]*100:.1f}% derived)")
    else:
        print("    2nd: N/A")

    print(f"    3rd: N/A  (no Polymarket market — use worldcup_standings.py)")
    print("=" * 62)


if __name__ == "__main__":
    main()
