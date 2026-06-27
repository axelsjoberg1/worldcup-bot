"""
worldcup_knockout.py — Polymarket WC 2026 KNOCKOUT-stage predictor.

From the Round of 32 onward (28 Jun) the scoring changes. Everything below is
scored on the 90-MINUTE (regulation) result, exactly like the group stage, PLUS
a bonus for correctly picking the team that ADVANCES:

  Exact-score tip:  6 exact · 4 right outcome + one team's goals · 3 right
                    outcome · 1 wrong outcome + one team's goals · 0 else
  PLUS  +3 if you also pick the right team to advance.
      -> max 9 (perfect exact score + right team through)
      -> the "1X2" level is 3 (right outcome) + 3 (advance) = 6

Because the 6/4/3/1 scheme already pays 3p for the right outcome plus partial
credit, an exact-score tip weakly dominates a pure 1X2 tip in expected points,
so we always recommend an exact score (the 1X2 EV is shown for comparison).

Advance mechanic (from the user's game):
  * If your tipped score has a winner, THAT team is automatically your advance
    pick.
  * If you tip a DRAW you must separately choose who advances; we pick the team
    more likely to go through.

So the total expected points of a tipped score depend on the outcome it implies:
  home-win score :  scoreEV + 3 * P(home advances)
  away-win score :  scoreEV + 3 * P(away advances)
  draw score     :  scoreEV + 3 * max(P(home adv), P(away adv))

We rank candidate scores by this TOTAL EV (not score-EV alone), which correctly
penalises tipping an underdog to win.

P(advance) includes extra time + penalties. We use a Polymarket "to advance" /
"to qualify" market when one exists, otherwise we derive it from the 90-minute
result market:
      P(home adv) = P(home win) + P(draw) * P(home win) / (P(home win)+P(away win))
i.e. a tie after 90' is resolved in proportion to the teams' relative win odds.

Always prints a console report AND writes a PDF (worldcup_knockout.pdf).

Run: python worldcup_knockout.py                     # report + PDF
     python worldcup_knockout.py --out report.pdf    # custom PDF path
     python worldcup_knockout.py --from 2026-06-01   # include earlier matches (testing)
"""

import sys
import re
import argparse
from datetime import datetime

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from worldcup import (
    fetch_wc_events,
    group_events,
    get_winner_probs,
    get_score_predictions,
)

# Knockout rounds start here for WC 2026 (Round of 32). Matches before this are
# group stage and are ignored by this tool.
DEFAULT_FROM = "2026-06-28"

# Show this many ranked scores per method in the report.
TOP_N_DISPLAY = 5
# Pull this many scores from the engine so the best TOTAL-EV score is never
# truncated before we add the advance bonus (poisson has <=64 scores).
TOP_N_FETCH = 100


# ── Result / advance probabilities ───────────────────────────────────────────

def _team_similarity(a: str, b: str) -> float:
    """Jaccard token overlap — robust to label variants like 'Bosnia-Herzegovina'
    vs 'Bosnia and Herzegovina' or accents."""
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def result_probs(winner_rows: list, home_team: str, away_team: str = "") -> dict:
    """Map [(name, p), ...] (incl. 'Draw') to normalised {home, draw, away}.

    The two non-draw rows are assigned to home/away by name similarity rather
    than exact match, so a knockout team whose Polymarket label differs slightly
    from the match title can never be silently swapped (which would attribute the
    advance probability to the wrong side)."""
    draw = 0.0
    cand: list[tuple[str, float]] = []
    for name, p in winner_rows:
        if name.strip().lower() == "draw":
            draw = p
        else:
            cand.append((name, p))

    home = away = 0.0
    if cand:
        cand.sort(key=lambda np: _team_similarity(np[0], home_team), reverse=True)
        home = cand[0][1]
        rest = cand[1:]
        if rest:
            if away_team:
                away = max(rest, key=lambda np: _team_similarity(np[0], away_team))[1]
            else:
                away = rest[0][1]

    total = home + draw + away
    if total <= 0:
        return {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}
    return {"home": home / total, "draw": draw / total, "away": away / total}


def derive_advance_prob(rp: dict) -> float:
    """P(home advances) from the 90-min result market: regulation win plus the
    share of draws the home team would win in extra time / penalties, split in
    proportion to the two sides' regulation win odds."""
    pw, pd, pl = rp["home"], rp["draw"], rp["away"]
    decisive = pw + pl
    et_share = pw / decisive if decisive > 0 else 0.5
    return pw + pd * et_share


def extract_advance_prob(entry: dict, home: str, away: str) -> float | None:
    """Best-effort: read a direct 'to advance' / 'to qualify' market for the
    home team if Polymarket lists one for this match.

    The knockout matchups are not on Polymarket until the group stage ends, so
    the exact market shape is unconfirmed. We scan every market attached to the
    match entry for a Yes-priced 'advance/qualify' market naming the home team,
    and return None (falling back to derive_advance_prob) when nothing matches.
    Revisit once real R32 markets exist to lock the label format.
    """
    from worldcup import yes_price  # local import; only needed here

    hn, an = home.lower(), away.lower()
    home_p = away_p = None
    for m in entry.get("winner_markets", []):
        label = m.get("groupItemTitle", "").lower()
        if not any(k in label for k in ("advance", "qualif", "to win the tie", "progress")):
            continue
        p = yes_price(m)
        if not (0 < p < 1):
            continue
        if hn in label:
            home_p = p
        elif an in label:
            away_p = p
    if home_p is not None and away_p is not None:
        total = home_p + away_p
        return home_p / total if total > 0 else home_p
    if home_p is not None:
        return home_p
    if away_p is not None:
        return 1 - away_p
    return None


# ── Per-match knockout pick ───────────────────────────────────────────────────

def _split_teams(title: str) -> tuple[str, str]:
    parts = re.split(r"\s+vs\.?\s+", title, maxsplit=1)
    home = parts[0].strip()
    away = parts[1].strip() if len(parts) > 1 else ""
    return home, away


def _augment_rows(rows, p_home_adv, home, away):
    """Turn (score, prob, scoreEV) rows into rows with the +3 advance bonus and
    the implied advancing team, ranked by TOTAL expected points."""
    out = []
    for score, prob, score_ev in rows:
        m = re.match(r"^(\d+)-(\d+)$", score)
        if not m:
            continue
        h, a = int(m.group(1)), int(m.group(2))
        if h > a:
            bonus, adv = p_home_adv, home
        elif h < a:
            bonus, adv = 1 - p_home_adv, away
        else:  # draw -> we choose the more likely team to advance
            if p_home_adv >= 0.5:
                bonus, adv = p_home_adv, home
            else:
                bonus, adv = 1 - p_home_adv, away
        out.append({
            "score": score, "prob": prob,
            "score_ev": score_ev, "adv_bonus": 3 * bonus,
            "total_ev": score_ev + 3 * bonus,
            "advance": adv,
        })
    return sorted(out, key=lambda r: -r["total_ev"])


def knockout_pick(entry: dict) -> dict:
    """Full knockout analysis for one match entry from group_events()."""
    title = entry["title"]
    home, away = _split_teams(title)
    winner_rows = get_winner_probs(entry["winner_markets"])
    rp = result_probs(winner_rows, home, away) if winner_rows else None

    pred = get_score_predictions(
        entry["score_markets"], home, winner_rows,
        top_n=TOP_N_FETCH, goals_markets=entry.get("goals_markets"), away_team=away,
    )

    # P(home advances): direct market if present, else derived from 90-min result
    p_home_adv = None
    src = "derived"
    if rp:
        direct = extract_advance_prob(entry, home, away)
        if direct is not None:
            p_home_adv, src = direct, "market"
        else:
            p_home_adv = derive_advance_prob(rp)

    liquid = pred["use_market"]
    market_rows  = _augment_rows(pred["market"],  p_home_adv, home, away) if p_home_adv is not None else []
    poisson_rows = _augment_rows(pred["poisson"], p_home_adv, home, away) if p_home_adv is not None else []

    rec_rows = market_rows if (liquid and market_rows) else poisson_rows
    rec_name = "market" if (liquid and market_rows) else "poisson"

    # Best pure-1X2 tip (for comparison): 3p * P(outcome) + 3p advance bonus
    one_x_two = None
    if rp:
        opts = [
            ("home", home, 3 * rp["home"] + 3 * p_home_adv),
            ("draw", "Draw", 3 * rp["draw"] + 3 * max(p_home_adv, 1 - p_home_adv)),
            ("away", away,  3 * rp["away"] + 3 * (1 - p_home_adv)),
        ]
        one_x_two = max(opts, key=lambda x: x[2])

    return {
        "title": title, "home": home, "away": away,
        "start": (entry.get("start_date") or "")[:10],
        "winner_rows": winner_rows, "result_probs": rp,
        "p_home_adv": p_home_adv, "adv_source": src,
        "liquid": liquid,
        "market_rows": market_rows, "poisson_rows": poisson_rows,
        "rec_rows": rec_rows, "rec_name": rec_name,
        "one_x_two": one_x_two,
        "goals": pred.get("goals", {}),
        "result_liq": entry.get("result_liquidity", 0.0),
        "score_liq": entry.get("score_liquidity", 0.0),
    }


# ── Match selection ───────────────────────────────────────────────────────────

def knockout_matches(events: list[dict], from_date: str) -> list[dict]:
    grouped = group_events(events)
    out = []
    for entry in grouped.values():
        start = (entry.get("start_date") or "")[:10]
        # Keep only matches on/after the cutoff. Group matches that have already
        # finished drop off the active feed anyway.
        if start and start < from_date:
            continue
        out.append(entry)
    return sorted(out, key=lambda x: x.get("start_date") or "9999")


# ── Console report ────────────────────────────────────────────────────────────

def print_ko_match(p: dict) -> None:
    title = p["title"] + (f"  [{p['start']}]" if p["start"] else "")
    print(f"\n  {title}")

    liq_flag = "[OK market liquid]" if p["liquid"] else "[!] LOW liquidity -> Poisson"
    print(f"  Liquidity: result ${p['result_liq']:,.0f} | score ${p['score_liq']:,.0f}   {liq_flag}")

    if p["p_home_adv"] is None:
        print("  (no result market — cannot model)")
        return

    pa = p["p_home_adv"]
    print(f"  Advance ({p['adv_source']}): {p['home']} {pa*100:.0f}%  |  {p['away']} {(1-pa)*100:.0f}%")

    if p["rec_rows"]:
        r = p["rec_rows"][0]
        print(f"  >> TIP [{p['rec_name']}]: {r['score']}  ->  {r['advance']} advances")
        print(f"     expected {r['total_ev']:.2f}p  (score {r['score_ev']:.2f} + advance {r['adv_bonus']:.2f})  "
              f"[{r['prob']*100:.1f}% exact]")
    if p["one_x_two"]:
        _, name, ev = p["one_x_two"]
        print(f"     1X2 alt: {name} -> {ev:.2f}p (dominated by exact)")

    def fmt(rows):
        return "  |  ".join(
            f"{r['score']} {r['total_ev']:.2f}p ({r['prob']*100:.1f}%)" for r in rows[:TOP_N_DISPLAY]
        )
    if p["market_rows"]:
        print(f"     Market:  {fmt(p['market_rows'])}")
    else:
        print(f"     Market:  (thin / no usable prices)")
    if p["poisson_rows"]:
        print(f"     Poisson: {fmt(p['poisson_rows'])}")

    if p["winner_rows"]:
        parts = "  |  ".join(
            f"{n} {pr*100:.0f}%" + (" <--" if i == 0 else "")
            for i, (n, pr) in enumerate(p["winner_rows"])
        )
        print(f"  90-min result: {parts}")

    total = p["goals"].get("total", [])
    if total:
        cells = "  ".join(f"O{line:g} {mkt*100:.0f}%" for line, mkt, _ in total)
        print(f"  Total goals (mkt): {cells}")


# ── PDF report ────────────────────────────────────────────────────────────────

def generate_pdf(picks: list[dict], output_path: str) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable,
    )
    from worldcup_pdf import (
        build_styles, fmt_liq,
        GREEN, LIGHT_GREEN, GREEN_BORDER, LIGHT_ORANGE, ORANGE_BORDER, MID_GREY,
    )

    styles = build_styles()

    def card(p: dict) -> Table:
        left = [Paragraph(
            f'{p["title"]}'
            + (f'  <font size="8" color="#777777">{p["start"]}</font>' if p["start"] else ""),
            styles["match_title"])]

        if p["p_home_adv"] is None or not p["rec_rows"]:
            left.append(Paragraph("No result market — cannot model", styles["no_score"]))
            t = Table([[left]], colWidths=[176 * mm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_ORANGE),
                ("BOX", (0, 0), (-1, -1), 0.5, ORANGE_BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]))
            return t

        r = p["rec_rows"][0]
        rec_color = "#1a7a4a" if (p["liquid"] and p["rec_name"] == "market") else "#b35c00"
        left.append(Paragraph(
            f'<font size="7">Tip <font color="{rec_color}">— {p["rec_name"]} '
            f'· {"market liquid" if p["liquid"] else "low liq, Poisson"}</font></font>',
            styles["label"]))
        left.append(Paragraph(
            f'{r["score"]}  <font color="#1a7a4a">&#8594; {r["advance"]} advances</font>',
            styles["score_best"]))
        left.append(Paragraph(
            f'<font color="#c8a400">EV {r["total_ev"]:.2f}p</font> '
            f'<font size="8" color="#777777">(score {r["score_ev"]:.2f} + advance {r["adv_bonus"]:.2f} · '
            f'{r["prob"]*100:.1f}% exact)</font>',
            styles["score_ev"]))

        def fmt_scores(rows):
            return "  ·  ".join(
                f'{x["score"]} <font color="#c8a400">{x["total_ev"]:.2f}p</font> ({x["prob"]*100:.1f}%)'
                for x in rows[:TOP_N_DISPLAY]
            )

        left.append(Spacer(1, 2))
        if p["market_rows"]:
            left.append(Paragraph(
                f"<font color='#1a7a4a'><b>Market:</b></font>&nbsp; {fmt_scores(p['market_rows'])}",
                styles["score_others"]))
        else:
            left.append(Paragraph(
                "<font color='#1a7a4a'><b>Market:</b></font>&nbsp; "
                "<font color='#999999'>thin / no usable prices</font>", styles["score_others"]))
        if p["poisson_rows"]:
            left.append(Paragraph(
                f"<font color='#b35c00'><b>Poisson:</b></font>&nbsp; {fmt_scores(p['poisson_rows'])}",
                styles["score_others"]))

        goals = p["goals"].get("total", [])
        if goals:
            cells = "  ·  ".join(f"O{line:g} {mkt*100:.0f}%" for line, mkt, _ in goals)
            left.append(Spacer(1, 2))
            left.append(Paragraph(
                f"<font color='#444444'><b>Goals:</b></font>&nbsp; {cells}", styles["score_others"]))

        # Right cell: advance probabilities + 90-min result + liquidity
        pa = p["p_home_adv"]
        right = [Paragraph(
            f'<b>Advance</b> <font size="7" color="#777777">({p["adv_source"]})</font><br/>'
            f'<font color="#1a7a4a"><b>{p["home"]} {pa*100:.0f}%</b></font> · '
            f'{p["away"]} {(1-pa)*100:.0f}%',
            styles["result"])]
        if p["winner_rows"]:
            parts = []
            for i, (n, pr) in enumerate(p["winner_rows"]):
                parts.append((f'<b>{n} {pr*100:.0f}%</b>' if i == 0 else f'{n} {pr*100:.0f}%'))
            right.append(Spacer(1, 2))
            right.append(Paragraph("90': " + "  |  ".join(parts), styles["result"]))
        right.append(Spacer(1, 2))
        right.append(Paragraph(
            f'Result {fmt_liq(p["result_liq"])} · Score {fmt_liq(p["score_liq"])}', styles["liq"]))

        bg = LIGHT_GREEN if p["liquid"] else LIGHT_ORANGE
        border = GREEN_BORDER if p["liquid"] else ORANGE_BORDER
        t = Table([[left, right]], colWidths=[120 * mm, 50 * mm])
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, -1), bg),
            ("ROUNDEDCORNERS", [4]),
            ("BOX", (0, 0), (-1, -1), 0.5, border),
            ("LEFTPADDING", (0, 0), (0, -1), 8), ("RIGHTPADDING", (1, 0), (1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        return t

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=12 * mm, rightMargin=12 * mm, topMargin=12 * mm, bottomMargin=12 * mm,
    )
    story = []
    header = Table([[Paragraph("FIFA WORLD CUP 2026 — KNOCKOUTS", styles["title"])]],
                   colWidths=[176 * mm])
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), GREEN), ("ROUNDEDCORNERS", [6]),
        ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(header)
    story.append(Paragraph(
        f"Polymarket predictions · {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
        "tip = 90-min exact score + team to advance",
        styles["subtitle"]))
    story.append(Spacer(1, 6 * mm))

    for p in picks:
        story.append(card(p))
        story.append(Spacer(1, 3 * mm))

    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "Scored on the 90-minute result. Exact-score tip: 6 exact · 4 right outcome + one team · "
        "3 right outcome · 1 wrong outcome + one team. <b>+3 for the right team to advance</b> "
        "(automatic when your score has a winner; pick the favourite when you tip a draw) → max 9. "
        "EV shown is total expected points (score + advance). "
        "Advance % includes extra time/penalties, from a 'to advance' market when present, else "
        "derived from the 90-min result market.",
        styles["footer"]))
    doc.build(story)
    print(f"PDF saved to: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_date", default=DEFAULT_FROM,
                    help=f"Only matches on/after this date (YYYY-MM-DD, default {DEFAULT_FROM})")
    ap.add_argument("--out", default="worldcup_knockout.pdf", help="PDF output path")
    args = ap.parse_args()

    print("=" * 62)
    print("   POLYMARKET — FIFA WORLD CUP 2026 KNOCKOUT PREDICTIONS")
    print("=" * 62)
    print("Fetching data...")
    events = fetch_wc_events()
    print(f"  -> {len(events)} markets fetched")

    matches = knockout_matches(events, args.from_date)
    print(f"  -> {len(matches)} knockout matches on/after {args.from_date}\n")
    if not matches:
        print("  No knockout matches yet — they appear once the group stage ends (28 Jun).")
        print("  To test the pipeline on current matches: --from 2026-06-01")
        return

    print("-" * 62)
    picks = [knockout_pick(e) for e in matches]
    for p in picks:
        print_ko_match(p)

    print("\n" + "=" * 62)
    generate_pdf(picks, args.out)
    print("=" * 62)


if __name__ == "__main__":
    main()
