"""
worldcup_bracket.py — WC 2026 bracket predictor.

Outputs the most likely teams at each knockout stage plus Golden Boot.

Run: python worldcup_bracket.py [--sims 50000]
"""

import sys
import random
import argparse
from collections import defaultdict

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from worldcup import fetch_wc_events, yes_price
from worldcup_standings import (
    extract_match_probs, extract_team_strengths, detect_groups,
    _strength_fallback, sim_group, sim_ko_match,
)

STAGES = ["R32", "R16", "QF", "SF", "Final", "Winner"]
STAGE_LABELS = {
    "R32":    "32 lag → 16-delsfinal",
    "R16":    "16 lag → 8-delsfinal",
    "QF":     "8 lag  → Kvartsfinal",
    "SF":     "4 lag  → Semifinal",
    "Final":  "2 lag  → Final",
    "Winner": "Vinnare",
}
STAGE_SHOW = {"R32": 32, "R16": 16, "QF": 8, "SF": 4, "Final": 2, "Winner": 1}


def fetch_golden_boot(events: list[dict]) -> dict[str, float]:
    for ev in events:
        if "golden boot" in ev.get("title", "").lower():
            odds: dict[str, float] = {}
            for m in ev.get("markets", []):
                name = m.get("groupItemTitle", "")
                p = yes_price(m)
                if name and p > 0:
                    odds[name] = p
            if odds:
                return odds
    return {}


def simulate_bracket(groups, match_probs, strengths, n_sims: int) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {s: defaultdict(int) for s in STAGES}

    for _ in range(n_sims):
        # ── Group stage ───────────────────────────────────────────────────
        firsts, seconds, thirds_pts = [], [], []
        for g in groups:
            standing, pts = sim_group(g, match_probs, strengths)
            if len(standing) >= 1: firsts.append(standing[0])
            if len(standing) >= 2: seconds.append(standing[1])
            if len(standing) >= 3: thirds_pts.append((standing[2], pts[standing[2]]))

        thirds_pts.sort(key=lambda x: -x[1])
        advancing = firsts + seconds + [t for t, _ in thirds_pts[:8]]

        for t in advancing:
            counts["R32"][t] += 1

        random.shuffle(advancing)
        bracket = advancing[:]

        # ── R32 → R16 ─────────────────────────────────────────────────────
        bracket = [sim_ko_match(bracket[i], bracket[i + 1], strengths)
                   for i in range(0, len(bracket) - 1, 2)]
        for t in bracket: counts["R16"][t] += 1

        # ── R16 → QF ──────────────────────────────────────────────────────
        bracket = [sim_ko_match(bracket[i], bracket[i + 1], strengths)
                   for i in range(0, len(bracket) - 1, 2)]
        for t in bracket: counts["QF"][t] += 1

        # ── QF → SF ───────────────────────────────────────────────────────
        bracket = [sim_ko_match(bracket[i], bracket[i + 1], strengths)
                   for i in range(0, len(bracket) - 1, 2)]
        for t in bracket: counts["SF"][t] += 1

        # ── SF → Final ────────────────────────────────────────────────────
        bracket = [sim_ko_match(bracket[i], bracket[i + 1], strengths)
                   for i in range(0, len(bracket) - 1, 2)]
        for t in bracket: counts["Final"][t] += 1

        # ── Final ─────────────────────────────────────────────────────────
        if len(bracket) >= 2:
            champion = sim_ko_match(bracket[0], bracket[1], strengths)
            counts["Winner"][champion] += 1

    return counts


def build_results(counts: dict, n_sims: int) -> dict:
    """Convert raw counts to sorted list of (team, probability) per stage."""
    results = {}
    for stage in STAGES:
        n_show = STAGE_SHOW[stage]
        ranked = sorted(counts[stage].items(), key=lambda x: -x[1])[:n_show]
        results[stage] = [(team, cnt / n_sims) for team, cnt in ranked]
    return results


def print_results(results: dict, golden_boot: dict, n_sims: int) -> None:
    print("\n" + "=" * 55)
    print("   FIFA WC 2026 — PREDICTED BRACKET")
    print("=" * 55)

    for stage in STAGES:
        label = STAGE_LABELS[stage]
        entries = results[stage]
        print(f"\n  {'─'*51}")
        print(f"  {label.upper()}")
        print(f"  {'─'*51}")
        for team, p in entries:
            bar = "█" * round(p * 20)
            print(f"  {team:<28}  {p*100:5.1f}%  {bar}")

    if golden_boot:
        print(f"\n  {'─'*51}")
        print(f"  GOLDEN BOOT (FLEST MÅL)")
        print(f"  {'─'*51}")
        top = sorted(golden_boot.items(), key=lambda x: -x[1])[:10]
        for name, p in top:
            bar = "█" * round(p * 20 / max(v for _, v in top))
            print(f"  {name:<28}  {p*100:5.1f}%  {bar}")

    print("\n" + "=" * 55)
    print(f"  Based on {n_sims:,} Monte Carlo simulations.")
    print("=" * 55)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=50_000)
    args = parser.parse_args()

    print("Fetching Polymarket data...", flush=True)
    events      = fetch_wc_events()
    match_probs = extract_match_probs(events)
    strengths   = extract_team_strengths(events) or _strength_fallback(match_probs)
    groups      = detect_groups(match_probs)
    golden_boot = fetch_golden_boot(events)

    print(f"  {len(match_probs)} matches  |  {len(groups)} groups  |  {len(golden_boot)} players in Golden Boot market")
    print(f"Running {args.sims:,} simulations...", flush=True)

    counts  = simulate_bracket(groups, match_probs, strengths, args.sims)
    results = build_results(counts, args.sims)

    print_results(results, golden_boot, args.sims)


if __name__ == "__main__":
    main()
