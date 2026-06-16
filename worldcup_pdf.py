"""Generate a PDF report from Polymarket World Cup predictions.

Run: python worldcup_pdf.py
     python worldcup_pdf.py --group A
     python worldcup_pdf.py --out my_report.pdf
"""

import argparse
import re
import sys
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from worldcup import (
    fetch_wc_events,
    group_events,
    get_winner_probs,
    get_score_predictions,
    canonical_title,
)


# ── Colours ──────────────────────────────────────────────────────────────────
GREEN       = colors.HexColor("#1a7a4a")
LIGHT_GREEN = colors.HexColor("#e8f5ee")
GREEN_BORDER= colors.HexColor("#c8ddd0")
ORANGE      = colors.HexColor("#b35c00")
LIGHT_ORANGE= colors.HexColor("#fdf1e3")
ORANGE_BORDER=colors.HexColor("#eccfa6")
DARK_GREY   = colors.HexColor("#333333")
MID_GREY    = colors.HexColor("#777777")
LIGHT_GREY  = colors.HexColor("#f4f4f4")
GOLD        = colors.HexColor("#c8a400")
WHITE       = colors.white


def build_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", fontSize=20, textColor=WHITE,
                                 fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=2),
        "subtitle": ParagraphStyle("subtitle", fontSize=9, textColor=colors.HexColor("#ccddcc"),
                                    fontName="Helvetica", alignment=TA_CENTER),
        "match_title": ParagraphStyle("match_title", fontSize=11, textColor=DARK_GREY,
                                       fontName="Helvetica-Bold", spaceAfter=1),
        "label": ParagraphStyle("label", fontSize=7.5, textColor=MID_GREY,
                                 fontName="Helvetica", spaceAfter=1),
        "score_best": ParagraphStyle("score_best", fontSize=12, textColor=GREEN,
                                      fontName="Helvetica-Bold"),
        "score_ev": ParagraphStyle("score_ev", fontSize=8, textColor=GOLD,
                                    fontName="Helvetica-Bold"),
        "score_others": ParagraphStyle("score_others", fontSize=8, textColor=DARK_GREY,
                                        fontName="Helvetica"),
        "result": ParagraphStyle("result", fontSize=8.5, textColor=DARK_GREY,
                                  fontName="Helvetica"),
        "liq": ParagraphStyle("liq", fontSize=7.5, textColor=MID_GREY,
                               fontName="Helvetica", alignment=TA_RIGHT),
        "no_score": ParagraphStyle("no_score", fontSize=8, textColor=MID_GREY,
                                    fontName="Helvetica-Oblique"),
        "footer": ParagraphStyle("footer", fontSize=7, textColor=MID_GREY,
                                  fontName="Helvetica", alignment=TA_CENTER),
    }


def fmt_liq(v: float) -> str:
    if v >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"${v/1_000:.0f}k"
    return f"${v:.0f}"


def build_match_table(entry: dict, styles: dict) -> Table:
    title       = entry["title"]
    result_liq  = entry.get("result_liquidity", 0.0)
    score_liq   = entry.get("score_liquidity", 0.0)
    winner_rows = get_winner_probs(entry["winner_markets"])
    parts_t     = re.split(r"\s+vs\.?\s+", title, maxsplit=1)
    home_team   = parts_t[0].strip()
    away_team   = parts_t[1].strip() if len(parts_t) > 1 else ""
    pred        = get_score_predictions(
        entry["score_markets"], home_team, winner_rows,
        goals_markets=entry.get("goals_markets"), away_team=away_team,
    )

    market_rows  = pred["market"]
    poisson_rows = pred["poisson"]
    liquid       = pred["use_market"]

    # Recommendation: trust the market when it's liquid, otherwise the model
    if liquid and market_rows:
        rec_rows, rec_name, rec_color = market_rows, "market data", "#1a7a4a"
    else:
        rec_rows, rec_name, rec_color = poisson_rows, "Poisson model", "#b35c00"

    def fmt_scores(rows):
        return "  ·  ".join(
            f"{s} <font color='#c8a400'>EV{ev:.2f}</font> ({p*100:.1f}%)"
            for s, p, ev in rows
        )

    # ── Left cell: match title + recommendation + both methods ───────────────
    start = entry.get("start_date", "")[:10] if entry.get("start_date") else ""
    title_str = f'{title}  <font size="8" color="#777777">{start}</font>' if start else title
    left = [Paragraph(title_str, styles["match_title"])]

    if rec_rows:
        top_score, top_p, top_ev = rec_rows[0]
        left.append(Paragraph(
            f'<font size="7">Recommended <font color="{rec_color}">— {rec_name}</font></font>',
            styles["label"]
        ))
        left.append(Paragraph(
            f'{top_score}  <font color="#c8a400">EV {top_ev:.2f}p</font>  '
            f'<font size="8" color="#777777">({top_p*100:.1f}%)</font>',
            styles["score_best"]
        ))
    else:
        left.append(Paragraph("No Exact Score market", styles["no_score"]))

    left.append(Spacer(1, 2))
    if market_rows:
        left.append(Paragraph(
            f"<font color='#1a7a4a'><b>Market:</b></font>&nbsp; {fmt_scores(market_rows)}",
            styles["score_others"]
        ))
    else:
        left.append(Paragraph(
            "<font color='#1a7a4a'><b>Market:</b></font>&nbsp; "
            "<font color='#999999'>thin / no usable prices</font>",
            styles["score_others"]
        ))
    if poisson_rows:
        left.append(Paragraph(
            f"<font color='#b35c00'><b>Poisson:</b></font>&nbsp; {fmt_scores(poisson_rows)}",
            styles["score_others"]
        ))

    # ── Total-goals Over/Under (market price, with fitted-model check) ────────
    goals = pred.get("goals", {})
    total = goals.get("total", [])
    if total:
        cells = "  ·  ".join(
            f"O{line:g} {mkt*100:.0f}%"
            + (f"<font color='#999999'>/{model*100:.0f}</font>" if model is not None else "")
            for line, mkt, model in total
        )
        btts = goals.get("btts")
        btts_str = f"&nbsp;&nbsp;|&nbsp;&nbsp;BTTS {btts[0]*100:.0f}%" if btts else ""
        left.append(Spacer(1, 2))
        left.append(Paragraph(
            f"<font color='#444444'><b>Goals:</b></font>&nbsp; {cells}{btts_str}",
            styles["score_others"]
        ))

    # ── Right cell: result probabilities + liquidity flag ────────────────────
    right = []
    if winner_rows:
        parts = []
        for i, (name, p) in enumerate(winner_rows):
            pct = f"{p*100:.0f}%"
            if i == 0:
                parts.append(f'<b>{name} {pct}</b>')
            else:
                parts.append(f'{name} {pct}')
        right.append(Paragraph("  |  ".join(parts), styles["result"]))

    right.append(Spacer(1, 2))
    if liquid:
        right.append(Paragraph(
            "<font color='#1a7a4a'><b>market liquid</b></font>", styles["liq"]))
    else:
        right.append(Paragraph(
            "<font color='#b35c00'><b>LOW LIQUIDITY — use Poisson</b></font>", styles["liq"]))
    right.append(Paragraph(
        f'Result {fmt_liq(result_liq)}  ·  Score {fmt_liq(score_liq)}',
        styles["liq"]
    ))

    bg     = LIGHT_GREEN if liquid else LIGHT_ORANGE
    border = GREEN_BORDER if liquid else ORANGE_BORDER

    col_widths = [115*mm, 55*mm]
    t = Table([[left, right]], colWidths=col_widths)
    t.setStyle(TableStyle([
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND",  (0, 0), (-1, -1), bg),
        ("ROUNDEDCORNERS", [4]),
        ("BOX",         (0, 0), (-1, -1), 0.5, border),
        ("LEFTPADDING",  (0, 0), (0, -1), 8),
        ("RIGHTPADDING", (1, 0), (1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 7),
    ]))
    return t


def generate_pdf(output_path: str, group_filter: str | None = None) -> None:
    print("Fetching data from Polymarket...")
    events = fetch_wc_events()
    print(f"  -> {len(events)} markets fetched")

    if group_filter:
        grp = group_filter.upper()
        events = [
            e for e in events
            if any(t.get("label", "").upper() == f"GROUP {grp}" for t in e.get("tags", []))
        ]

    grouped = group_events(events)
    matches = sorted(grouped.values(), key=lambda x: x.get("start_date") or "9999", reverse=False)
    print(f"  -> {len(matches)} matches\n")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=12*mm, rightMargin=12*mm,
        topMargin=12*mm, bottomMargin=12*mm,
    )

    styles = build_styles()
    story  = []

    # ── Header ────────────────────────────────────────────────────────────────
    header_data = [[
        Paragraph("FIFA WORLD CUP 2026", styles["title"]),
    ]]
    header_table = Table(header_data, colWidths=[176*mm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), GREEN),
        ("ROUNDEDCORNERS", [6]),
        ("TOPPADDING",   (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    story.append(header_table)

    subtitle = f"Polymarket predictions · {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    if group_filter:
        subtitle += f"  ·  Group {group_filter.upper()}"
    story.append(Paragraph(subtitle, styles["subtitle"]))
    story.append(Spacer(1, 6*mm))

    # ── Match cards ───────────────────────────────────────────────────────────
    for entry in matches:
        story.append(build_match_table(entry, styles))
        story.append(Spacer(1, 3*mm))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "Each card shows both methods: <font color='#1a7a4a'><b>Market</b></font> = live Polymarket exact-score prices · "
        "<font color='#b35c00'><b>Poisson</b></font> = model fitted to the win/draw/loss market. "
        "Green cards are liquid (trust the market); <font color='#b35c00'>orange = low liquidity, rely on Poisson</font>. "
        "<font color='#444444'><b>Goals</b></font> = market Over/Under goal lines (BTTS = both teams to score); the Poisson model "
        "is fitted to these totals too, so its score shape matches the implied goal count. "
        "EV uses game rules: 6p exact · 4p right outcome + one team correct · 3p right outcome · 1p wrong outcome + one team correct.",
        styles["footer"]
    ))

    doc.build(story)
    print(f"PDF saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", "-g", help="Filter by group letter (A-L)")
    parser.add_argument("--out",   "-o", default="worldcup_predictions.pdf",
                        help="Output PDF filename (default: worldcup_predictions.pdf)")
    args = parser.parse_args()

    generate_pdf(args.out, group_filter=args.group)


if __name__ == "__main__":
    main()
