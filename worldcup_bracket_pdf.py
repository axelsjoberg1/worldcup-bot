"""
worldcup_bracket_pdf.py — PDF version of the WC 2026 bracket predictor.

Run: python worldcup_bracket_pdf.py [--sims 50000] [--out bracket.pdf]
"""

import sys
import argparse

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph,
    Spacer, HRFlowable, KeepTogether,
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from worldcup import fetch_wc_events
from worldcup_standings import (
    extract_match_probs, extract_team_strengths, detect_groups, _strength_fallback,
)
from worldcup_bracket import simulate_bracket, build_results, fetch_golden_boot, STAGES, STAGE_LABELS, STAGE_SHOW

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN       = colors.HexColor("#1a7a4a")
LIGHT_GREEN = colors.HexColor("#e8f5ee")
DARK_GREY   = colors.HexColor("#333333")
MID_GREY    = colors.HexColor("#777777")
GOLD        = colors.HexColor("#c8a400")
WHITE       = colors.white

STAGE_COLORS = {
    "R32":    colors.HexColor("#e8f5ee"),
    "R16":    colors.HexColor("#d4edda"),
    "QF":     colors.HexColor("#b8dfc4"),
    "SF":     colors.HexColor("#8ecba6"),
    "Final":  colors.HexColor("#5db07e"),
    "Winner": colors.HexColor("#1a7a4a"),
}
STAGE_TEXT_COLORS = {s: DARK_GREY for s in STAGES}
STAGE_TEXT_COLORS["Winner"] = WHITE


def build_styles():
    return {
        "header_title": ParagraphStyle("ht", fontSize=20, textColor=WHITE,
                                        fontName="Helvetica-Bold", alignment=TA_CENTER),
        "header_sub":   ParagraphStyle("hs", fontSize=9,  textColor=colors.HexColor("#ccddcc"),
                                        fontName="Helvetica", alignment=TA_CENTER),
        "stage_title":  ParagraphStyle("st", fontSize=10, textColor=DARK_GREY,
                                        fontName="Helvetica-Bold", spaceAfter=2),
        "team":         ParagraphStyle("te", fontSize=9,  textColor=DARK_GREY,
                                        fontName="Helvetica"),
        "team_winner":  ParagraphStyle("tw", fontSize=11, textColor=WHITE,
                                        fontName="Helvetica-Bold"),
        "pct":          ParagraphStyle("pc", fontSize=9,  textColor=MID_GREY,
                                        fontName="Helvetica"),
        "pct_winner":   ParagraphStyle("pw", fontSize=11, textColor=GOLD,
                                        fontName="Helvetica-Bold"),
        "footer":       ParagraphStyle("fo", fontSize=7,  textColor=MID_GREY,
                                        fontName="Helvetica", alignment=TA_CENTER),
        "gb_title":     ParagraphStyle("gb", fontSize=10, textColor=DARK_GREY,
                                        fontName="Helvetica-Bold", spaceAfter=2),
        "gb_player":    ParagraphStyle("gp", fontSize=9,  textColor=DARK_GREY,
                                        fontName="Helvetica"),
    }


def prob_bar(p: float, max_p: float, width: int = 40) -> str:
    filled = round(p / max_p * width) if max_p > 0 else 0
    return "█" * filled + "░" * (width - filled)


def build_stage_block(stage: str, entries: list, styles: dict, page_width: float) -> KeepTogether:
    is_winner = stage == "Winner"
    bg        = STAGE_COLORS[stage]
    label     = STAGE_LABELS[stage]
    max_p     = entries[0][1] if entries else 1

    title_style = styles["stage_title"]
    team_style  = styles["team_winner"] if is_winner else styles["team"]
    pct_style   = styles["pct_winner"]  if is_winner else styles["pct"]

    n_cols   = 1 if is_winner or stage == "Final" else (2 if stage in ("SF",) else 4)
    col_w    = page_width / n_cols

    rows_per_col = -(-len(entries) // n_cols)  # ceiling division
    columns: list[list] = [[] for _ in range(n_cols)]

    for i, (team, p) in enumerate(entries):
        col_idx = i // rows_per_col
        if col_idx >= n_cols:
            col_idx = n_cols - 1
        bar = prob_bar(p, max_p, width=16)
        cell = [
            Paragraph(team, team_style),
            Paragraph(f"{p*100:.1f}%  {bar}", pct_style),
            Spacer(1, 1),
        ]
        columns[col_idx].append(cell)

    # Flatten columns into a single table row per column item
    all_cells: list[list] = []
    col_widths = [col_w] * n_cols

    if n_cols == 1:
        for cell in columns[0]:
            all_cells.append(cell)
        inner = [[item] for item in all_cells]
        t = Table(inner, colWidths=[page_width])
    else:
        max_rows = max(len(c) for c in columns)
        for r in range(max_rows):
            row = []
            for c in range(n_cols):
                if r < len(columns[c]):
                    cell_content = columns[c][r]
                    row.append(cell_content)
                else:
                    row.append(Spacer(1, 1))
            all_cells.append(row)
        t = Table(all_cells, colWidths=col_widths)

    t.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND",   (0, 0), (-1, -1), bg),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
    ]))

    title_row = Table(
        [[Paragraph(f"▶  {label}", title_style)]],
        colWidths=[page_width],
    )
    title_row.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), bg),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
        ("BOX",          (0, 0), (-1, -1), 0.5, colors.HexColor("#aaaaaa")),
    ]))

    return KeepTogether([title_row, t, Spacer(1, 3 * mm)])


def build_golden_boot_block(golden_boot: dict, styles: dict, page_width: float) -> KeepTogether:
    top = sorted(golden_boot.items(), key=lambda x: -x[1])[:10]
    if not top:
        return KeepTogether([Spacer(1, 1)])

    max_p = top[0][1]
    n_cols = 2
    col_w = page_width / n_cols
    mid = -(-len(top) // n_cols)

    left_col  = top[:mid]
    right_col = top[mid:]

    max_rows = max(len(left_col), len(right_col))
    rows = []
    for r in range(max_rows):
        row = []
        for col in (left_col, right_col):
            if r < len(col):
                name, p = col[r]
                bar = prob_bar(p, max_p, width=14)
                row.append([
                    Paragraph(name, styles["gb_player"]),
                    Paragraph(f"{p*100:.1f}%  {bar}", styles["pct"]),
                    Spacer(1, 1),
                ])
            else:
                row.append(Spacer(1, 1))
        rows.append(row)

    t = Table(rows, colWidths=[col_w, col_w])
    t.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND",   (0, 0), (-1, -1), colors.HexColor("#fff8e1")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
    ]))

    title_row = Table(
        [[Paragraph("⚽  GOLDEN BOOT — FLEST MÅL", styles["gb_title"])]],
        colWidths=[page_width],
    )
    title_row.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), GOLD),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("BOX",          (0, 0), (-1, -1), 0.5, colors.HexColor("#aaaaaa")),
    ]))

    return KeepTogether([title_row, t, Spacer(1, 3 * mm)])


def generate_pdf(output_path: str, n_sims: int) -> None:
    print("Fetching Polymarket data...", flush=True)
    events      = fetch_wc_events()
    match_probs = extract_match_probs(events)
    strengths   = extract_team_strengths(events) or _strength_fallback(match_probs)
    groups      = detect_groups(match_probs)
    golden_boot = fetch_golden_boot(events)

    print(f"Running {n_sims:,} simulations...", flush=True)
    counts  = simulate_bracket(groups, match_probs, strengths, n_sims)
    results = build_results(counts, n_sims)

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=12*mm, rightMargin=12*mm,
        topMargin=12*mm, bottomMargin=12*mm,
    )
    page_width = A4[0] - 24*mm
    styles = build_styles()
    story  = []

    # ── Header ────────────────────────────────────────────────────────────────
    header = Table(
        [[Paragraph("FIFA WORLD CUP 2026", styles["header_title"])]],
        colWidths=[page_width],
    )
    header.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), GREEN),
        ("TOPPADDING",   (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
    ]))
    story.append(header)
    story.append(Paragraph(
        f"Predicted bracket · {n_sims:,} Monte Carlo simulations · Polymarket odds",
        styles["header_sub"],
    ))
    story.append(Spacer(1, 5*mm))

    # ── Stage blocks ──────────────────────────────────────────────────────────
    for stage in STAGES:
        story.append(build_stage_block(stage, results[stage], styles, page_width))

    # ── Golden Boot ───────────────────────────────────────────────────────────
    if golden_boot:
        story.append(build_golden_boot_block(golden_boot, styles, page_width))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "Probabilities from Polymarket prediction markets. Knockout strength via Bradley-Terry model.",
        styles["footer"],
    ))

    doc.build(story)
    print(f"PDF saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=50_000)
    parser.add_argument("--out",  default="worldcup_bracket.pdf")
    args = parser.parse_args()
    generate_pdf(args.out, args.sims)


if __name__ == "__main__":
    main()
