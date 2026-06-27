"""
worldcup_standings.py — Monte Carlo WC 2026 standings predictor.

Simulates the full tournament using Polymarket group-stage match probabilities
and Polymarket WC Winner outright odds for knockout strength estimation.

Run: python worldcup_standings.py
     python worldcup_standings.py --sims 100000
     python worldcup_standings.py --top 15
"""

import sys
import re
import random
import argparse
from collections import defaultdict
from itertools import combinations

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from worldcup import fetch_wc_events, yes_price

N_SIMS = 50_000


# ── Data extraction ────────────────────────────────────────────────────────

def _is_group_match_event(raw: str) -> bool:
    if ":" in raw:                              return False
    if re.match(r"^world cup\b", raw, re.I):   return False
    for kw in ("exact score", "halftime", "more markets", "h2h"):
        if kw in raw.lower():                   return False
    return "vs" in raw.lower()


def _parse_teams(raw: str) -> tuple[str, str] | None:
    m = re.match(r"^(.+?)\s+vs\.?\s+(.+?)(?:\s*-.*)?$", raw.strip(), re.I)
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def extract_match_probs(events: list[dict]) -> dict:
    """Returns {(home, away): (p_home, p_draw, p_away)} — normalised."""
    probs = {}
    for ev in events:
        raw = ev.get("title", "")
        if not _is_group_match_event(raw):
            continue
        parsed = _parse_teams(raw)
        if not parsed:
            continue
        home, away = parsed

        ph = pd = pa = 0.0
        for m in ev.get("markets", []):
            label = m.get("groupItemTitle", "")
            if any(kw in label.lower() for kw in
                   ["both teams", "over", "under", "score", "goal",
                    "card", "corner", "clean sheet", "exact"]):
                continue
            p = yes_price(m)
            if label.lower().startswith("draw"):
                pd = p
            elif label == home:
                ph = p
            elif label == away:
                pa = p

        total = ph + pd + pa
        if total > 0.2:
            probs[(home, away)] = (ph / total, pd / total, pa / total)

    return probs


def extract_team_strengths(events: list[dict]) -> dict:
    """
    WC Winner outright market → {team: probability}.
    Used as relative knockout strength via Bradley-Terry: P(A>B) = sA/(sA+sB).
    """
    for ev in events:
        title = ev.get("title", "")
        if not re.search(r"world cup.*winner|winner.*world cup", title, re.I):
            continue
        strengths: dict[str, float] = {}
        for m in ev.get("markets", []):
            team = m.get("groupItemTitle", "")
            p = yes_price(m)
            if team and p > 0 and "field" not in team.lower() and "other" not in team.lower():
                strengths[team] = p
        if len(strengths) > 10:
            return strengths
    return {}


def _strength_fallback(match_probs: dict) -> dict:
    """Estimate team strength from group-stage win probabilities when no outright market exists."""
    s: dict[str, float] = defaultdict(float)
    for (home, away), (ph, pd, pa) in match_probs.items():
        s[home] += ph
        s[away] += pa
    return dict(s)


# ── Group detection ────────────────────────────────────────────────────────

def detect_groups(match_probs: dict) -> list[list[str]]:
    """
    Build adjacency graph from matches.
    Each WC group has 4 teams playing 6 matches (C(4,2)).
    Allows 1 missing match per group.
    """
    adj: dict[str, set] = defaultdict(set)
    for (h, a) in match_probs:
        adj[h].add(a)
        adj[a].add(h)

    assigned: set[str] = set()
    groups: list[list[str]] = []

    for team in sorted(adj):
        if team in assigned:
            continue
        pool = sorted(adj[team] - assigned)

        best_group: list[str] = []
        best_score = -1

        for n in range(min(3, len(pool)), 0, -1):
            for combo in combinations(pool, n):
                candidate = [team] + list(combo)
                score = sum(1 for a, b in combinations(candidate, 2) if b in adj[a])
                if score > best_score:
                    best_score = score
                    best_group = candidate
            if best_score >= 5:   # found a near-complete K4 (≥5 of 6 edges)
                break

        if not best_group:
            best_group = [team]

        groups.append(best_group)
        assigned.update(best_group)

    return groups


# ── Match simulation ───────────────────────────────────────────────────────

def _get_strengths(team: str, strengths: dict) -> float:
    if team in strengths:
        return strengths[team]
    # Fuzzy: try partial match (handles "IR Iran" vs "Iran" etc.)
    tl = team.lower()
    for k, v in strengths.items():
        if k.lower() in tl or tl in k.lower():
            return v
    return 1 / 48


def sim_group_match(home: str, away: str, match_probs: dict, strengths: dict) -> str:
    """Returns 'home', 'draw', or 'away'."""
    if (home, away) in match_probs:
        ph, pd, pa = match_probs[(home, away)]
    elif (away, home) in match_probs:
        pa, pd, ph = match_probs[(away, home)]
    else:
        sh = _get_strengths(home, strengths)
        sa = _get_strengths(away, strengths)
        t  = sh + sa
        ph = sh / t * 0.65
        pa = sa / t * 0.65
        pd = 1 - ph - pa

    r = random.random()
    if r < ph:        return "home"
    if r < ph + pd:   return "draw"
    return "away"


def sim_ko_match(a: str, b: str, strengths: dict) -> str:
    """Knockout match (no draws). Uses Bradley-Terry."""
    sa = _get_strengths(a, strengths)
    sb = _get_strengths(b, strengths)
    return a if random.random() < sa / (sa + sb) else b


# ── Tournament simulation ──────────────────────────────────────────────────

def sim_group(group: list[str], match_probs: dict, strengths: dict) -> tuple[list[str], dict]:
    pts: dict[str, int]   = defaultdict(int)
    gd:  dict[str, float] = defaultdict(float)

    for home, away in combinations(group, 2):
        r = sim_group_match(home, away, match_probs, strengths)
        if r == "home":
            pts[home] += 3; gd[home] += 1; gd[away] -= 1
        elif r == "away":
            pts[away] += 3; gd[away] += 1; gd[home] -= 1
        else:
            pts[home] += 1; pts[away] += 1

    standing = sorted(group, key=lambda t: (pts[t], gd[t], random.random()), reverse=True)
    return standing, dict(pts)


ROUND_NAMES = ["R32", "R16", "QF", "SF"]

def sim_tournament(
    groups: list[list[str]], match_probs: dict, strengths: dict
) -> tuple[str, str, str, list[list[tuple[str, str]]]]:
    """
    Returns (champion, runner_up, third, rounds_matchups).
    rounds_matchups[i] = list of (teamA, teamB) pairs played in round i.
    """
    firsts:     list[str]              = []
    seconds:    list[str]              = []
    thirds_pts: list[tuple[str, int]]  = []

    for g in groups:
        standing, pts = sim_group(g, match_probs, strengths)
        if len(standing) >= 1: firsts.append(standing[0])
        if len(standing) >= 2: seconds.append(standing[1])
        if len(standing) >= 3: thirds_pts.append((standing[2], pts[standing[2]]))

    thirds_pts.sort(key=lambda x: x[1], reverse=True)
    best_thirds = [t for t, _ in thirds_pts[:8]]

    advancing = firsts + seconds + best_thirds
    random.shuffle(advancing)

    bracket       = advancing[:]
    rounds_matchups: list[list[tuple[str, str]]] = []

    # Knockout rounds down to 4 (R32 → R16 → QF)
    while len(bracket) > 4:
        nxt      = []
        matchups = []
        for i in range(0, len(bracket) - 1, 2):
            matchups.append((bracket[i], bracket[i + 1]))
            nxt.append(sim_ko_match(bracket[i], bracket[i + 1], strengths))
        if len(bracket) % 2:
            nxt.append(bracket[-1])
        rounds_matchups.append(matchups)
        bracket = nxt

    while len(bracket) < 4:
        bracket.append(bracket[-1])

    # Semifinals
    sf_winners, sf_losers, sf_matchups = [], [], []
    for i in range(0, 4, 2):
        sf_matchups.append((bracket[i], bracket[i + 1]))
        w = sim_ko_match(bracket[i], bracket[i + 1], strengths)
        l = bracket[i + 1] if w == bracket[i] else bracket[i]
        sf_winners.append(w)
        sf_losers.append(l)
    rounds_matchups.append(sf_matchups)

    champion  = sim_ko_match(sf_winners[0], sf_winners[1], strengths)
    runner_up = sf_winners[1] if champion == sf_winners[0] else sf_winners[0]
    third     = sim_ko_match(sf_losers[0], sf_losers[1], strengths)

    rounds_matchups.append([(sf_winners[0], sf_winners[1])])   # Final
    rounds_matchups.append([(sf_losers[0],  sf_losers[1])])    # 3rd place

    return champion, runner_up, third, rounds_matchups


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=N_SIMS,  help="Number of simulations (default 50000)")
    parser.add_argument("--top",  type=int, default=12,      help="Teams to show per column (default 12)")
    args = parser.parse_args()

    print("=" * 60)
    print("   FIFA WC 2026 — MONTE CARLO STANDINGS PREDICTOR")
    print("=" * 60)
    print("Fetching Polymarket data...", flush=True)

    events      = fetch_wc_events()
    match_probs = extract_match_probs(events)
    strengths   = extract_team_strengths(events)
    groups      = detect_groups(match_probs)

    if not strengths:
        print("  [!] No WC Winner market found — using match-win fallback for strength")
        strengths = _strength_fallback(match_probs)

    print(f"  {len(match_probs)} group-stage matches")
    print(f"  {len(strengths)} teams in knockout strength table")
    print(f"  {len(groups)} groups detected:\n")
    for i, g in enumerate(groups, 1):
        print(f"    Group {i:>2}: {', '.join(g)}")

    print(f"\nRunning {args.sims:,} simulations...", flush=True)

    counts: dict[str, dict[str, int]] = {
        "1st": defaultdict(int),
        "2nd": defaultdict(int),
        "3rd": defaultdict(int),
    }
    # matchup_counts[round_idx][(teamA, teamB)] = times this matchup occurred.
    # Grown on demand: round count varies with how many teams advance.
    matchup_counts: list[dict[tuple[str, str], int]] = []

    for _ in range(args.sims):
        c, r, t, rounds = sim_tournament(groups, match_probs, strengths)
        counts["1st"][c] += 1
        counts["2nd"][r] += 1
        counts["3rd"][t] += 1
        for ri, matchups in enumerate(rounds):
            while ri >= len(matchup_counts):
                matchup_counts.append(defaultdict(int))
            for a, b in matchups:
                key = tuple(sorted([a, b]))
                matchup_counts[ri][key] += 1

    n = args.sims
    all_teams = set(counts["1st"]) | set(counts["2nd"]) | set(counts["3rd"])
    rows = []
    for team in all_teams:
        p1 = counts["1st"][team] / n
        p2 = counts["2nd"][team] / n
        p3 = counts["3rd"][team] / n
        rows.append((team, p1, p2, p3, p1 + p2 + p3))
    rows.sort(key=lambda x: -x[4])

    print(f"\n{'Team':<30} {'1st':>7}  {'2nd':>7}  {'3rd':>7}  {'Top-3':>8}")
    print("─" * 60)
    for team, p1, p2, p3, pt in rows[: args.top]:
        print(f"  {team:<28} {p1*100:6.1f}%  {p2*100:6.1f}%  {p3*100:6.1f}%  {pt*100:7.1f}%")

    print("\n" + "=" * 60)
    print("  BEST GUESS  (no team used twice):")
    used: set[str] = set()
    for pos in ("1st", "2nd", "3rd"):
        for team, cnt in sorted(counts[pos].items(), key=lambda x: -x[1]):
            if team not in used:
                print(f"    {pos}: {team}  ({cnt/n*100:.1f}%)")
                used.add(team)
                break
    print("=" * 60)

    # ── Most likely bracket ────────────────────────────────────────────────
    round_labels = ["R32", "R16", "QF", "SF", "Final", "3rd place"]
    top_per_round = [8, 5, 4, 2, 1, 1]

    print("\n" + "=" * 60)
    print("  MOST LIKELY KNOCKOUT MATCHUPS (by round)")
    print("=" * 60)
    for ri, mc in enumerate(matchup_counts):
        if not mc:
            continue
        label  = round_labels[ri]  if ri < len(round_labels)  else f"Round {ri + 1}"
        n_show = top_per_round[ri] if ri < len(top_per_round) else 4
        top = sorted(mc.items(), key=lambda x: -x[1])[:n_show]
        print(f"\n  {label}:")
        for (a, b), cnt in top:
            print(f"    {a:<26} vs  {b:<26}  {cnt/n*100:5.1f}%")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
