#!/usr/bin/env python
"""Generate corrections.pdf: published.pdf marked up with the correction changes.

Every textual change is a strikethrough over the published "Old" text, with the comment
holding ONLY the corrected ("New") text (e.g. "B = -0.45"). Figure replacements have no
text to anchor to, so they get a margin sticky-note naming the replacement file.

The corrected text shows up in the comment/markup pane of any PDF reader (Preview,
Acrobat).

This module is normally driven by corrections/run_corrections.sh, but can be run alone:
    python make_corrections_pdf.py            # reads input/, writes output/

The two regression sections (Exp1 strategy regressions p. 913, Exp1 processing-pattern
p. 915) take their corrected B-values from regression_corrections.py, the same records
the verifier checks against the standalone R scripts. Every change is also written to a
plain-text log (CHANGE_LOG) so the run is auditable without opening the PDF.

Notes on matching:
- Page numbers below are 0-based PDF page indices, not journal page numbers.
- published.pdf uses Unicode minus (U+2212 "-"), "chi2" for chi-square, "alpha", "kappa".
- Where the published value (e.g. "32.9" or "114.9") repeats and a literal search is
  ambiguous, the change targets an explicit word-index span in page reading order
  instead. Word indices were read off published.pdf and are asserted at runtime.
"""

import os

import fitz  # PyMuPDF

import regression_corrections as rc

HERE = os.path.dirname(os.path.abspath(__file__))

SRC = os.path.join(HERE, "input", "published.pdf")
OUT = os.path.join(HERE, "output", "corrections.pdf")
LOG = os.path.join(HERE, "output", "corrections.log")

RED = (0.85, 0.1, 0.1)

# Every change appends a (page_index, kind, old, new) record here, written to LOG.
CHANGE_LOG = []


def _c(key):
    """Look up the corrected comment string for a regression record by its R-output key."""
    for r in rc.ALL:
        if r["key"] == key:
            return r["comment"]
    raise KeyError(f"no regression correction with key {key!r}")


def reading_order_words(page):
    """Words sorted in reading order: list of (x0, y0, x1, y1, text, ...)."""
    w = page.get_text("words")
    w.sort(key=lambda x: (x[5], x[6], x[7]))
    return w


def strike_text(page, text, comment, expect=1, kind="text"):
    """Strike every rect of `text` (a single logical span, possibly line-wrapped)."""
    rects = page.search_for(text)
    assert len(rects) == expect, f"{text!r}: expected {expect} hit(s), got {len(rects)}"
    a = page.add_strikeout_annot(rects)
    a.set_colors(stroke=RED)
    a.set_info(content=comment)
    a.update()
    CHANGE_LOG.append((page.number, kind, text, comment))


def strike_parts(page, parts, comment, expect, kind="text"):
    """Strike one logical span given as consecutive `parts` under a single annotation
    and comment. Use when a soft-hyphen line break (e.g. "low-\\ndispersion") stops
    `search_for` from matching the whole span at once. `expect` is the total rect
    count across all parts (a part may wrap a line and yield several rects)."""
    rects = []
    for part in parts:
        rects.extend(page.search_for(part))
    assert len(rects) == expect, f"{parts!r}: expected {expect} rect(s), got {len(rects)}"
    a = page.add_strikeout_annot(rects)
    a.set_colors(stroke=RED)
    a.set_info(content=comment)
    a.update()
    CHANGE_LOG.append((page.number, kind, " ".join(parts), comment))


def strike_word_span(page, start, end, expect_words, comment, kind="text"):
    """Strike the word-index span [start, end) (reading order). `expect_words` is the
    list of word strings that span must contain, asserted to guard against drift."""
    words = reading_order_words(page)
    span = words[start:end]
    got = [w[4] for w in span]
    assert got == expect_words, f"word span {start}:{end} drifted: {got} != {expect_words}"
    a = page.add_strikeout_annot([fitz.Rect(w[:4]) for w in span])
    a.set_colors(stroke=RED)
    a.set_info(content=comment)
    a.update()
    CHANGE_LOG.append((page.number, kind, " ".join(got), comment))


def figure_note(page, comment):
    """Margin sticky-note near the top of `page` (figure replacements have no anchor)."""
    a = page.add_text_annot(fitz.Point(20, 60), comment)
    a.set_colors(stroke=RED)
    a.update()
    CHANGE_LOG.append((page.number, "figure", "(figure replacement)", comment))


def build(doc):
    # ===== 1. Changes to figures (margin notes; no text anchor) =====
    # Figure -> (PDF page index, replacement filename produced by make_new_figs.py).
    figures = [
        (10, "Figure 5: replace with fig5_nr_clicks__processing_pattern.png"),
        (11, "Figure 6: replace with fig6_payoff_gross_relative.png"),
        (34, "Figure E2: replace with figE2_click_var_outcome__click_var_gamble.png"),
        (40, "Figure F1: replace with figF1_payoff_gross_relative_exclude.png"),
    ]
    for pidx, note in figures:
        figure_note(doc[pidx], note)

    # ===== 2. Changes to statistics (regression B-values; PROVEN by the R scripts) =====
    # Experiment 1 strategy regressions (journal p. 913, PDF idx 8). Corrected values
    # come from regression_corrections.STRATEGY.
    p913 = doc[8]
    # Note: published.pdf uses Unicode minus (U+2212 "−") in these values, so the
    # search strings below must too, even though our corrected comments use ASCII "-".
    strike_text(p913, "B = −2.3", _c("SAT_TTB ~ sigma"), kind="regression")
    strike_text(p913, "B = −1.3", _c("TTB_SAT ~ sigma"), kind="regression")
    strike_text(p913, "B = −5.5", _c("TTB ~ alpha"), kind="regression")
    # The −1.4 clause is replaced by new prose, not just a value. Strike the whole
    # replaced clause (keeping "However, while the resource-rational model"); the
    # comment holds only the corrected text. Split at the "low-dispersion" hyphen break
    # so each part matches.
    strike_parts(
        p913,
        ["most often uses targeted search in low",
         "dispersion environments, participants often resorted to choosing randomly "
         "instead (B = −1.4, p < .001)"],
        _c("RandOther ~ alpha"),
        expect=3,  # part 1 (1 rect) + part 2 wrapping a line (2 rects)
        kind="regression",
    )
    strike_text(p913, "B = −0.8", _c("TTB_SAT ~ cost"), kind="regression")
    strike_text(p913, "B = −3.9", _c("TTB ~ cost"), kind="regression")
    strike_text(p913, "B = −3.3", _c("SAT_TTB ~ cost"), kind="regression")

    # Experiment 1 processing-pattern regression (journal p. 915, PDF idx 10).
    # The dispersion clause moves after the cost effect and flips sign; strike the
    # whole changed span so the comment holds only the corrected text. Corrected value
    # comes from regression_corrections.PROCESSING.
    strike_text(
        doc[10],
        "and as dispersion increased (B = 0.084, p < .001), and they used more "
        "attribute-based processing",
        _c("processing_pattern ~ alpha"),
        expect=3,  # span wraps across three lines
        kind="regression",
    )

    # ===== 3. Chi-square and kappa copy-paste errors (NOT re-derived by this workflow) =====
    # Experiment 2 chi-square HD conditions (journal p. 919, PDF idx 14).
    # Each HD value reused the LD file; strike the value+p+d and comment the corrected
    # numbers. Targeted/SAT 'value, p, d' = 7 words; exhaustive too.
    p919 = doc[14]
    # targeted search
    strike_word_span(p919, 224, 231,
                     ["32.9,", "p", "<", ".001,", "d", "=", "0.25;"],
                     "56.8, p < .001, d = 0.34", kind="chi2")  # HD-LC
    strike_word_span(p919, 235, 242,
                     ["32.9,", "p", "<", ".001,", "d", "=", "0.25."],
                     "41.6, p < .001, d = 0.30", kind="chi2")  # HD-HC
    # SAT-TTB
    strike_word_span(p919, 288, 295,
                     ["12.4,", "p", "<", ".001,", "d", "=", "−0.16;"],
                     "5.9, p = .015, d = -0.11", kind="chi2")  # HD-LC
    strike_word_span(p919, 299, 306,
                     ["12.4,", "p", "<", ".001,", "d", "=", "−0.16."],
                     "31.9, p < .001, d = -0.26", kind="chi2")  # HD-HC
    # the LD-HC "p < .8" should read "p = .8"
    strike_word_span(p919, 278, 281, ["p", "<", ".8,"], "p = .8", kind="chi2")
    # exhaustive search
    strike_word_span(p919, 401, 408,
                     ["114.9,", "p", "<", ".001,", "d", "=", "0.53;"],
                     "0.5, p = .5, d = 0.00", kind="chi2")  # HD-LC
    strike_word_span(p919, 412, 419,
                     ["114.9,", "p", "<", ".001,", "d", "=", "0.53."],
                     "12.2, p < .001, d = 0.17", kind="chi2")  # HD-HC

    # Figure B1 participant kappa (journal p. 931, PDF idx 26): second "kappa = 0.572, ..."
    strike_word_span(
        doc[26], 185, 193,
        ["κ", "=", "0.572,", "95%", "CI", "[0.571,", "0.572]", "for"],
        "κ = 0.535, 95% CI [0.529, 0.541] for participants",
        kind="kappa",
    )

    # ===== 4. Other text changes from inverting the meaning of alpha =====
    # Experiment 1 dispersion description (journal p. 912, PDF idx 7).
    strike_text(
        doc[7],
        "one outcome being much more likely than others for low dispersion and all outcomes being roughly equally likely for high dispersion",
        "all outcomes being roughly equally likely for low dispersion and one outcome much more likely than others for high dispersion",
        expect=3,  # wraps across lines
    )

    # Experiment 1 decision-quality sentence (journal p. 916, PDF idx 11):
    # "decreases with the dispersion" -> "increases".
    strike_word_span(doc[11], 145, 146, ["decreases"], "increases")

    # Appendix captions misdescribing alpha-1 (single changed word each).
    strike_text(doc[29], "homogeneity", "peakiness")  # Figure D1 (p. 934)
    strike_text(doc[30], "homogeneity", "peakiness")  # Figure D2 (p. 935)
    strike_text(doc[34], "uniformity", "dispersion")  # Figure E1 (p. 939)
    strike_text(doc[41], "uniformity", "dispersion")  # Figure F2 (p. 946)

    # ===== 5. Miscellanea =====
    # Appendix E effect-size typo (journal p. 938, PDF idx 33).
    strike_word_span(doc[33], 726, 727, ["46,"], "0.46,")

    # Experiment 2 discussion cross-reference (journal p. 922, PDF idx 17).
    strike_word_span(doc[17], 191, 193, ["Experiment", "2."], "Experiment 1")

    # Figure E5 caption typo (journal p. 943, PDF idx 38).
    strike_word_span(doc[38], 61, 62, ["stakes"], "dispersion")
    strike_word_span(doc[38], 68, 69, ["stakes"], "dispersion")


def write_log(path):
    """Write the plain-text change log: one block per annotation, grouped by kind."""
    lines = [
        "corrections.pdf change log",
        f"source: {os.path.relpath(SRC, HERE)}",
        f"annotations: {len(CHANGE_LOG)}",
        "",
        "kind legend: regression = proven against the standalone R scripts;",
        "             chi2/kappa = copy-paste/file-reference fixes (not re-derived here);",
        "             figure = bitmap replacement; text = wording fix.",
        "=" * 78,
    ]
    for pidx, kind, old, new in CHANGE_LOG:
        lines += [
            "",
            f"[p{pidx} | {kind}]",
            f"  old: {old}",
            f"  new: {new}",
        ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    doc = fitz.open(SRC)
    build(doc)
    doc.save(OUT)
    write_log(LOG)
    print(f"wrote {OUT} ({doc.page_count} pages, {len(CHANGE_LOG)} annotations)")
    print(f"wrote {LOG}")


if __name__ == "__main__":
    main()
