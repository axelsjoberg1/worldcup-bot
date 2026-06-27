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
import json
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
from worldcup_top3 import finishing_probs, best_podium

# Knockout rounds start here for WC 2026 (Round of 32). Matches before this are
# group stage and are ignored by this tool.
DEFAULT_FROM = "2026-06-28"

# Show this many ranked scores per method in the report.
TOP_N_DISPLAY = 5
# Pull this many scores from the engine so the best TOTAL-EV score is never
# truncated before we add the advance bonus (poisson has <=64 scores).
TOP_N_FETCH = 100
# Minimum traded volume for a 'Team to Advance' market to be trusted; below this
# it is treated as untraded and we fall back to the derived advance probability.
ADV_MIN_VOL = 50


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


def extract_advance_prob(entry: dict, home: str, away: str) -> tuple[float, float] | None:
    """Read Polymarket's direct 'Team to Advance' market for this knockout tie.

    Confirmed structure (e.g. South Africa vs. Canada, R32): it is a single
    2-way market whose two outcomes are the team names, living inside the
    '- More Markets' event — which group_events() collects into goals_markets.
    Prices already include extra time / penalties (e.g. SA 0.28 / Canada 0.73).

    Returns (P(home advances), traded volume), normalised for the over-round, or
    None when the market is absent / untraded / can't be mapped to the two teams
    — in which case the caller falls back to derive_advance_prob().
    """
    pools = list(entry.get("goals_markets", [])) + list(entry.get("winner_markets", []))
    for m in pools:
        label = m.get("groupItemTitle", "").strip().lower()
        if "advance" not in label and "qualif" not in label:
            continue
        vol = float(m.get("volumeNum") or 0)
        if vol < ADV_MIN_VOL:                       # untraded -> derive instead
            continue
        oc = m.get("outcomes", "[]")
        pr = m.get("outcomePrices", "[]")
        if isinstance(oc, str): oc = json.loads(oc)
        if isinstance(pr, str): pr = json.loads(pr)
        if len(oc) != 2 or len(pr) != 2:
            continue
        try:
            prices = [float(x) for x in pr]
        except (TypeError, ValueError):
            continue
        if not all(0 < x < 1 for x in prices):     # untraded / default market
            continue
        sims = [_team_similarity(name, home) for name in oc]
        if max(sims) <= 0:                          # can't map outcomes -> teams
            continue
        hi = sims.index(max(sims))
        p_home, p_away = prices[hi], prices[1 - hi]
        total = p_home + p_away
        if total > 0:
            return p_home / total, vol
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

    # Advance probability per view:
    #   Market view  = traded 'Team to Advance' market, derived as fallback.
    #   Poisson view = always the derived estimate.
    adv_derived = derive_advance_prob(rp) if rp else None
    adv_market, adv_liq = None, 0.0
    if rp:
        res = extract_advance_prob(entry, home, away)
        if res is not None:
            adv_market, adv_liq = res
    market_adv  = adv_market if adv_market is not None else adv_derived
    market_src  = "market" if adv_market is not None else "derived"
    poisson_adv = adv_derived

    liquid = pred["use_market"]
    market_rows  = _augment_rows(pred["market"],  market_adv,  home, away) if market_adv  is not None else []
    poisson_rows = _augment_rows(pred["poisson"], poisson_adv, home, away) if poisson_adv is not None else []

    # The recommendation follows its view's advance source: market when the
    # exact-score market is liquid, otherwise Poisson (with derived advance).
    if liquid and market_rows:
        rec_rows, rec_name, p_home_adv, src = market_rows, "market", market_adv, market_src
    else:
        rec_rows, rec_name, p_home_adv, src = poisson_rows, "poisson", poisson_adv, "derived"

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
        "adv_market": adv_market, "adv_derived": adv_derived,
        "adv_liq": adv_liq, "market_adv_source": market_src,
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
    pick = p["rec_rows"][0]["advance"] if p["rec_rows"] else (p["home"] if pa >= 0.5 else p["away"])
    hm = " <--" if pick == p["home"] else ""
    am = " <--" if pick == p["away"] else ""
    adv_liq_txt = f", ${p['adv_liq']:,.0f} vol" if p["adv_market"] is not None else ""
    print(f"  Advance ({p['adv_source']}{adv_liq_txt}): {p['home']} {pa*100:.0f}%{hm}  |  {p['away']} {(1-pa)*100:.0f}%{am}")

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
        print(f"     Market  (incl. {p['market_adv_source']} adv):  {fmt(p['market_rows'])}")
    else:
        print(f"     Market  (incl. adv):  (thin / no usable prices)")
    if p["poisson_rows"]:
        print(f"     Poisson (incl. derived adv):  {fmt(p['poisson_rows'])}")

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


def explain_match(p: dict) -> None:
    """Print the full reasoning chain for one match, step by step."""
    print("\n" + "=" * 66)
    print(f"  {p['title']}" + (f"   [{p['start']}]" if p["start"] else ""))
    print("=" * 66)

    if p["p_home_adv"] is None:
        print("  No result market — cannot model this match.")
        return

    home, away = p["home"], p["away"]

    # STEP 1 — result market (1X2, 90 min)
    print(f"\n  STEP 1 — Result market (1X2, 90 min)   [Result ${p['result_liq']:,.0f}]")
    for n, pr in (p["winner_rows"] or []):
        print(f"     {n:<26} {pr*100:5.1f}%")

    # STEP 2 — Poisson model fitted to 1X2 + goal O/U
    print(f"\n  STEP 2 — Poisson fit to the 1X2 + goal Over/Under markets")
    lam = p["goals"].get("lambda")
    if lam:
        la, lb = lam
        print(f"     lambda({home}) = {la:.2f}    lambda({away}) = {lb:.2f}    (expected goals {la+lb:.2f})")
    g = p["goals"].get("total", [])
    if g:
        print("     goal lines used: " + "  ".join(f"O{l:g} {mkt*100:.0f}%" for l, mkt, _ in g))
    print("     Poisson score distribution (top, before advance bonus):")
    for r in sorted(p["poisson_rows"], key=lambda x: -x["prob"])[:8]:
        print(f"       {r['score']:<5} {r['prob']*100:5.1f}%    score-EV {r['score_ev']:.2f}")

    # STEP 3 — Team to Advance (market vs derived)
    print(f"\n  STEP 3 — Team to Advance  (incl. extra time / penalties)")
    rp, ad, am = p["result_probs"], p["adv_derived"], p["adv_market"]
    if ad is not None:
        print(f"     derived:  {home} {ad*100:.0f}%  |  {away} {(1-ad)*100:.0f}%"
              f"   [{rp['home']:.2f} + {rp['draw']:.2f}*{rp['home']:.2f}/({rp['home']:.2f}+{rp['away']:.2f})]")
    if am is not None:
        print(f"     market:   {home} {am*100:.0f}%  |  {away} {(1-am)*100:.0f}%   (traded ${p['adv_liq']:,.0f} vol)")
    else:
        print(f"     market:   not traded / too thin  ->  falling back to derived")
    pa = p["p_home_adv"]
    print(f"     USING {p['adv_source'].upper()}:  {home} {pa*100:.0f}%  |  {away} {(1-pa)*100:.0f}%")

    # STEP 4 — total EV ranking
    print(f"\n  STEP 4 — Total EV = score-EV + 3 * P(advance for that outcome)")
    print(f"     score shape: {p['rec_name'].upper()} "
          f"({'exact-score market liquid' if p['liquid'] else 'exact-score market thin -> Poisson'})")
    for r in p["rec_rows"][:TOP_N_DISPLAY]:
        print(f"       {r['score']:<5} total {r['total_ev']:.2f} = score {r['score_ev']:.2f} "
              f"+ adv {r['adv_bonus']:.2f}   -> {r['advance']} advances")
    if p["rec_rows"]:
        r = p["rec_rows"][0]
        print(f"\n  >> TIP: {r['score']}  ->  {r['advance']} advances   ({r['total_ev']:.2f}p expected)")


# ── PDF report ────────────────────────────────────────────────────────────────

def generate_pdf(picks: list[dict], output_path: str, top3: tuple | None = None) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable,
    )
    from worldcup_pdf import (
        build_styles, fmt_liq, GOLD,
        GREEN, LIGHT_GREEN, GREEN_BORDER, LIGHT_ORANGE, ORANGE_BORDER, MID_GREY,
    )

    styles = build_styles()

    def top3_card() -> Table:
        (combo, ev), gold, silver, bronze, tname = top3
        medals = [
            ("GOLD",   combo[0], gold,   10, "#c8a400"),
            ("SILVER", combo[1], silver,  8, "#8a8d91"),
            ("BRONZE", combo[2], bronze,  6, "#b35c00"),
        ]
        cells = [Paragraph(
            f'<b>VM TOPP 3</b>&nbsp;&nbsp;<font size="8" color="#777777">'
            f'EV-optimal podium · scored 10 / 8 / 6 · expected {ev:.2f}p</font>',
            styles["match_title"])]
        for label, t, probs, pts, col in medals:
            cells.append(Paragraph(
                f'<font color="{col}"><b>{label}</b></font>&nbsp;&nbsp; '
                f'<b>{tname[t]}</b>&nbsp; '
                f'<font size="8" color="#777777">{probs[t]*100:.1f}% · '
                f'{pts} × {probs[t]*100:.0f}% = {pts*probs[t]:.2f}p</font>',
                styles["score_others"]))
        tbl = Table([[c] for c in cells], colWidths=[176 * mm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fbf7e6")),
            ("ROUNDEDCORNERS", [4]),
            ("BOX", (0, 0), (-1, -1), 0.6, GOLD),
            ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return tbl

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
                f"<font color='#1a7a4a'><b>Market (incl. {p['market_adv_source']} adv):</b></font>&nbsp; "
                f"{fmt_scores(p['market_rows'])}", styles["score_others"]))
        else:
            left.append(Paragraph(
                "<font color='#1a7a4a'><b>Market (incl. adv):</b></font>&nbsp; "
                "<font color='#999999'>thin / no usable prices</font>", styles["score_others"]))
        if p["poisson_rows"]:
            left.append(Paragraph(
                f"<font color='#b35c00'><b>Poisson (incl. derived adv):</b></font>&nbsp; "
                f"{fmt_scores(p['poisson_rows'])}", styles["score_others"]))

        goals = p["goals"].get("total", [])
        if goals:
            cells = "  ·  ".join(f"O{line:g} {mkt*100:.0f}%" for line, mkt, _ in goals)
            left.append(Spacer(1, 2))
            left.append(Paragraph(
                f"<font color='#444444'><b>Goals:</b></font>&nbsp; {cells}", styles["score_others"]))

        # Right cell: advance probabilities + 90-min result + liquidity.
        # Highlight the team the TIP backs to advance (not always the home side).
        pa = p["p_home_adv"]
        home_cell = f'{p["home"]} {pa*100:.0f}%'
        away_cell = f'{p["away"]} {(1-pa)*100:.0f}%'
        if r["advance"] == p["home"]:
            home_cell = f'<font color="#1a7a4a"><b>{home_cell}</b></font>'
        else:
            away_cell = f'<font color="#1a7a4a"><b>{away_cell}</b></font>'
        right = [Paragraph(
            f'<b>Advance</b> <font size="7" color="#777777">({p["adv_source"]})</font><br/>'
            f'{home_cell} · {away_cell}',
            styles["result"])]
        if p["winner_rows"]:
            parts = []
            for i, (n, pr) in enumerate(p["winner_rows"]):
                parts.append((f'<b>{n} {pr*100:.0f}%</b>' if i == 0 else f'{n} {pr*100:.0f}%'))
            right.append(Spacer(1, 2))
            right.append(Paragraph("90': " + "  |  ".join(parts), styles["result"]))
        right.append(Spacer(1, 2))
        liq_line = f'Result {fmt_liq(p["result_liq"])} · Score {fmt_liq(p["score_liq"])}'
        if p["adv_market"] is not None:
            liq_line += f' · Advance {fmt_liq(p["adv_liq"])}'
        right.append(Paragraph(liq_line, styles["liq"]))

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
    story.append(Spacer(1, 4 * mm))

    if top3 is not None:
        story.append(top3_card())
        story.append(Spacer(1, 4 * mm))

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
    ap.add_argument("--explain", nargs="?", const="", default=None,
                    help="Print the full reasoning chain per match "
                         "(optionally a team substring to filter, e.g. --explain canada)")
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

    # Podium (top-3) for the PDF header + console summary; skipped if the
    # tournament futures markets aren't available.
    top3 = None
    try:
        gold, silver, bronze, tname = finishing_probs(events)
        combo, ev = best_podium(gold, silver, bronze)
        top3 = ((combo, ev), gold, silver, bronze, tname)
        labels = [("GOLD  ", 10, gold), ("SILVER", 8, silver), ("BRONZE", 6, bronze)]
        print("  VM TOP 3 (10/8/6) — EV-optimal podium:")
        for (lab, pts, pr), t in zip(labels, combo):
            print(f"     {lab}: {tname[t]:<22} {pr[t]*100:5.1f}%  -> {pts*pr[t]:.2f}p")
        print(f"     expected total {ev:.2f}p")
    except SystemExit:
        print("  (top-3 podium skipped — tournament futures markets not found)")

    print("-" * 62)
    picks = [knockout_pick(e) for e in matches]
    if args.explain is not None:
        flt = args.explain.lower()
        for p in picks:
            if flt and flt not in p["title"].lower():
                continue
            explain_match(p)
    else:
        for p in picks:
            print_ko_match(p)

    print("\n" + "=" * 62)
    generate_pdf(picks, args.out, top3)
    print("=" * 62)


if __name__ == "__main__":
    main()
