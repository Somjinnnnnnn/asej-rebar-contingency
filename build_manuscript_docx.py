"""Build the submission Word files in the layout of the companion ASEJ manuscript.

Text source: the Word export of the Claude Doc draft (outputs/manuscript/
Manuscript_rebar_ASEJ_draft.docx); the highlights and declarations added to the draft after
that export are kept below as DECLARATIONS and HIGHLIGHTS. Tables 1-7 come from the
manuscript_tables.py and revision_tables.py CSVs, figures and captions from manuscript_figures.py and
captions_draft.md. Each table and figure is placed after the paragraph that first cites it; the
supplementary file holds Note S1, Tables S1-S20 and Figs. S1-S3.

Layout (as in the companion manuscript): A4, 2.5 cm margins, continuous line numbers, Times New
Roman 12 pt, single-spaced (BODY_LINE) justified body text with a 0.75 cm first-line indent, bold headings,
"Table N." captions above tables and "Fig. N." captions below figures (11 pt), three-line
tables, native Word equations.

Writes outputs/manuscript/Manuscript_rebar_ASEJ.docx and Highlights_rebar_ASEJ.docx.

Usage:
    python build_manuscript_docx.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import docx
import pandas as pd
from docx.enum.section import WD_ORIENT  # noqa: F401  (kept for landscape variants)
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt

MS = Path(__file__).resolve().parent / 'outputs' / 'manuscript'
SOURCE = MS / 'Manuscript_rebar_ASEJ_draft.docx'
TEXT_WIDTH_CM = 16.0
FIG_WIDTH_CM = 14.0
FIG_WIDTHS_CM = {1: 11.0, 3: 12.0}  # narrower widths keep these figures on the page with the text that cites them
# items placed this many text blocks after their first citation instead of right after it, so that the page
# above a large figure or table fills with text (set from the page layout of rev10; Fig. 5 from rev15)
DEFER = {('Fig', 1): 3, ('Fig', 2): 2, ('Fig', 4): 2, ('Fig', 5): 4, ('Table', 5): 1, ('Fig', 6): 1, ('Table', 6): 1}

TITLE = ('Probabilistic Forecasting of Reinforcing Steel Prices for Expenditure-Based Price-Escalation Allowances: '
         'Evidence from Thailand')
AUTHORS = [('Somjintana Kanangkaew', '1'), ('Dikai Pang', '2'), ('Pongammard Kanangkaew', '2'),
           ('Noppadon Jokkaw', '3,*')]
AFFILIATIONS = [('1', 'Department of Civil Engineering, Faculty of Engineering, Chiang Mai University, Thailand'),
                ('2', 'Department of Civil Engineering, Faculty of Engineering, Khon Kaen University, Thailand'),
                ('3', 'Department of Civil Engineering, Faculty of Engineering, Chulalongkorn University, Thailand')]
CORRESPONDING = '* Corresponding author: Noppadon Jokkaw (noppadon.j@chula.ac.th)'

HIGHLIGHTS = [  # as in the Highlights section of the Claude Doc
    'Thai rebar list prices changed in 90% of months, unlike administered concrete prices',
    'No evaluated model showed an established advantage over a random walk at 3–12 months',
    'A one-month Machine Learning gain of 8% was not significant after Holm adjustment',
    'Fixed 5% and 10% allowances covered only 70–85% of 3- to 12-month outcomes',
    'Model P95 allowances protected only 78–81% of past 24-month schedules',
]
# The declarations (authors' contribution, funding, data availability, AI use) are part of the text source.

# print layout per table: column widths in cm (after the section column is removed), font size, a column
# whose values become full-width section rows, cells split before " (" onto a second line, and horizon
# sub-headers shortened to the number of months
L_ROBUST = dict(widths=[4.4, 2.9, 2.9, 2.9, 2.9], size=8, two_line=True, section='Run')
L_COVERAGE = dict(widths=[5.6] + [1.3] * 8, size=8)
L_CONFORMAL = dict(widths=[3.4] + [1.05] * 12, size=8, section='Model', short_h=True)
L_BANDS = dict(widths=[4.9, 1.2] + [1.3] * 4 + [1.45] * 3, size=8, section='Band')
L_VALIDATION = dict(widths=[4.0, 1.7] + [1.1] * 9, size=8, section='Terms', short_h=True)
# tables of the paper, numbered in order of first citation, and of the supplementary material
TABLES = {
    1: ('table_02_code_junctions', dict(widths=[1.6, 1.6, 1.9, 2.4, 4.1, 2.0, 2.0], size=9)),
    2: ('table_01_data_audit', dict(widths=[1.48, 1.6, 2.6, 1.32, 1.35, 1.55, 1.6, 2.25, 2.25], size=8)),
    3: ('table_rev_crps', dict(widths=[4.4, 2.9, 2.9, 2.9, 2.9], size=8, two_line=True)),
    4: ('table_rev_primary_tests', dict(widths=[6.4, 3.0, 3.0, 3.6], size=9)),
    5: ('table_rev_procurement', dict(widths=[2.6] + [1.9] * 7, size=8, section='Schedule')),
    6: ('table_rev_protection', dict(widths=[5.8, 3.2, 3.5, 3.5], size=9)),
}
# rev7 table number -> current number. Rev8 moved rev7 Tables 3, 6, 7, 11 and 15 to Supplementary Tables S16-S20
# and dropped rev7 Tables 8 and 12 (reported in full in S7 and S9); rev10 moved rev7 Table 10 (model risk) to S21;
# rev12 returned rev7 Table 1 (code junctions) to the main text and replaced rev7 Table 9 (bands) by S8
MOVED_TABLES = {3: 'S16', 6: 'S17', 7: 'S18', 11: 'S19', 15: 'S20', 10: 'S21'}
REV7_TABLES = {1: 1, 2: 2, 4: 3, 5: 4, 9: 'S8', 13: 5, 14: 6, 8: 'S7', 12: 'S9', **MOVED_TABLES}
REV7_FIGS = {1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 9: 6, 5: 'S1', 7: 'S2', 8: 'S3'}
SUPP_TABLES = {
    'S1': ('table_S01_code_chain', dict(widths=[3.0, 4.5, 0.9, 1.3, 1.3, 1.0, 1.4, 1.4, 1.2], size=8, section='Series')),
    'S2': ('table_S02_release_timing', dict(widths=[3.4, 1.8, 2.2, 3.4, 1.6, 3.6], size=8)),
    'S3': ('table_S03_dm_adjusted', dict(widths=[6.4, 2.4, 2.4, 2.4, 2.4], size=8, section='Horizon')),
    'S4': ('table_S04_nonoverlap', dict(widths=[1.6, 1.8, 2.4, 5.2, 5.0], size=8, section='Model')),
    'S5': ('table_S05_crps_grid', dict(widths=[2.0, 3.5, 3.5, 3.5, 3.5], size=9, section='Model')),
    'S6': ('table_S06_coverage_full', L_COVERAGE),
    'S7': ('table_S07_conformal_full', L_CONFORMAL),
    'S8': ('table_S08_bands_full', L_BANDS),
    'S9': ('table_S09_validation_full', L_VALIDATION),
    'S10': ('table_S10_validation_block_length', dict(widths=[3.2] * 5, size=9)),
    'S11': ('table_S11_dependence', dict(widths=[2.0, 4.6, 4.6, 4.8], size=9, section='Resampling')),
    'S12': ('table_S12_threshold', dict(widths=[2.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.8, 2.0], size=9)),
    'S13': ('table_S13_boot_backtest', dict(widths=[1.8, 2.4, 4.4, 1.8, 2.4, 2.4], size=9, section='Mean block')),
    'S14': ('table_S14_allowance_start', dict(widths=[1.7, 1.8, 1.9, 2.5, 1.9, 1.8, 1.6, 1.9], size=9)),
    'S15': ('table_S15_protection_blocks', dict(widths=[4.4, 1.8, 2.5, 2.5, 2.5, 2.5], size=9)),
    'S16': ('table_03_process_diagnostics', dict(widths=[1.9, 1.4, 1.8, 2.4, 2.0, 1.5, 1.7, 1.6, 1.6], size=8)),
    'S17': ('table_06_robustness', L_ROBUST),
    'S18': ('table_rev_uncertainty', dict(widths=[3.0, 6.5, 6.5], size=9, section='Model')),
    'S19': ('table_rev_sensitivity', dict(widths=[5.6, 2.2, 2.2, 2.2, 2.2, 1.6], size=8, two_line=True)),
    'S20': ('table_rev_profiles', dict(widths=[3.4, 2.1, 2.3, 2.3, 2.9, 3.0], size=8, section='Model')),
    'S21': ('table_09_model_risk_long_horizon', dict(widths=[4.4, 2.9, 2.9, 2.9, 2.9], size=9)),
    'S22': ('table_S22_literature', dict(widths=[2.4, 2.3, 2.5, 2.2, 2.2, 2.4, 2.0], size=8)),
    'S23': ('table_S23_protection_procedures', dict(widths=[4.4, 3.0, 2.0, 1.9, 2.3, 2.0], size=8, section='Model')),
}
# supplementary notes printed before the tables: (heading, [paragraphs])
SUPP_NOTES = [
    ('Note S1. Estimation-uncertainty histories', [
        'The estimation uncertainty in Table 5 of the manuscript and in Table S19 is obtained by refitting the models '
        'to alternative histories of the price. The 116 monthly changes x_1, …, x_T of the log real SD40 12 mm price, '
        'from February 2017 to September 2026, are taken as observed and are not demeaned before resampling. An index '
        'sequence of length T is then drawn by the stationary bootstrap with a mean block of 12 months: the first index '
        'is drawn uniformly, and each later index either continues the previous block, wrapping at T, or, with '
        'probability 1/12, starts a new block at a uniformly drawn position. The resampled changes form one alternative '
        'history, and 200 such histories are drawn with a fixed random seed.',
        'Each model is next refitted to each history exactly as to the observed data. The block bootstrap removes the '
        'mean of the resampled changes, which sets the real drift to zero, and resamples them with a mean block of 12 '
        'months, whereas the random walk uses their standard deviation with zero drift. Each refit simulates 2,000 paths '
        'of the real price ratio P(t)/P_0 as the exponential of the cumulative simulated changes, so that no price '
        'level has to be reconstructed. For the 20-year bands, the P95 of the real price ratio at month 240 is read from '
        'each refit, as reported in Table S19. For the procurement example, the ratios at the purchase months are '
        'multiplied by the 2% inflation factor, the expenditure ratio E is formed with the schedule-A quantities, and '
        'its P95 and the P95 of the ratio at month 24 are read from each refit, as reported in Table 5 of the '
        'manuscript. The 5th and 95th percentiles over the 200 refits form the reported 90% intervals.',
        'The process simulation thus resamples the changes of one history to generate future paths, whereas the '
        'resampling of histories generates the alternative data sets to which the models are refitted. The code is in '
        'revision_analyses.py, part sensitivity, and revision2_analyses.py, part allowance.']),
    ('Note S2. Model specifications', [
        'Every model is refitted at each origin to the monthly log changes up to the origin and reduced to the 19 '
        'quantiles τ = 0.05, …, 0.95 of the h-month log price ratio. The simulation models use 2,000 paths per origin '
        'in the backtest, 5,000 in the procurement evaluation and 10,000 for the long-horizon bands. Every random stream '
        'is derived from the base seed 20261022 together with the stage, model, series and origin, so that each '
        'result is reproducible on its own.',
        'Among the closed-form and empirical models, the random walk uses the sample standard deviation σ of the changes '
        'with zero drift, so that the h-month quantile is √h σ z_τ, GBM uses the sample mean and standard deviation of '
        'the changes, and the empirical model takes the quantiles of the overlapping historical h-month changes.',
        'The GARCH(1,1), AR(1) and AR(1)-GARCH(1,1) models are fitted by Gaussian quasi-maximum likelihood to 100 × the '
        'log changes with the arch package, with a constant mean for GARCH(1,1) and an AR(1) mean otherwise. GARCH(1,1) '
        'is simulated with normal errors, whereas the two AR(1) models are simulated by filtered historical simulation, '
        'in which standardized residuals are resampled with replacement and passed through the fitted mean and variance '
        'recursions. A fit counts as converged when the optimizer reports success, and all 75 backtest fits of each '
        'model converged.',
        'Because TPSO prices are reported to the nearest 0.01 THB, the Merton jump-diffusion and Markov switching models '
        'treat each change as an interval, and their likelihood is the probability of that interval, the grouped-data '
        'likelihood. The Merton density, Eq. (1), sums over at most K = 3 jumps per month; its parameters are bounded '
        '(drift ±0.05, diffusion standard deviation 10⁻⁵ to 0.2, jump intensity 10⁻⁴ to 1, jump mean ±0.5, jump '
        'standard deviation 10⁻⁵ to 0.5) and estimated by L-BFGS-B from three starting values, keeping the highest '
        'likelihood. The two-regime Markov model is estimated through the Hamilton filter with bounded regime means '
        '(±0.5), standard deviations (10⁻⁵ to 0.5) and logit transition probabilities (±15), from four starting '
        'values of the persistence of the second regime. A failed or non-finite Markov fit would fall back to the '
        'data-calibrated GBM, and a Merton fit without reported optimizer success is flagged; all 75 backtest fits of '
        'both models reported success, with no fallback and no standard deviation at its lower bound.',
        'The stationary bootstrap resamples demeaned changes in blocks of geometric length with a mean of 12 months, and '
        'the mean-reverting model is an AR(1) in the log price level with the bias correction φ + (1 + 3φ)/T, capped at '
        '0.999, and resampled residuals.',
        'For the quantile regressions, linear quantile regression uses standardized features and an L1 penalty α, '
        'fitted at each quantile with the HiGHS solver. XGBoost uses the multi-quantile pinball objective with the histogram '
        'tree method; the fixed settings are a depth of 2, 200 trees, a learning rate of 0.05, row and column '
        'subsampling of 0.8, a minimum child weight of 3 and an L2 penalty of 1, and the features are not scaled. '
        'Predicted quantiles are sorted to remove crossing. The features are r_s and r_{s−1}, the 3- and 6-month '
        'cumulative changes, the 1- and 3-month log changes of iron ore (lag 0) and of the US scrap index (lag 1), the '
        '1-month log changes of Brent, coal and Bangkok diesel (lag 0), and the 3-month log changes of the US steel mill '
        'index (lag 1) and of THB/USD (lag 0).',
        'For tuning, the training rows at every origin and horizon are split into three expanding folds. A validation '
        'block of max(4, ⌊n/6⌋) rows starts at row a, and its training rows are those whose targets are realized by '
        'then (s + h ≤ s_a), so that at least h months separate the last training row from the validation block and '
        'no target realized after s_a enters the fit; a fold needs at least 18 training rows. The criterion is the mean pinball loss at '
        'τ ∈ {0.05, 0.25, 0.5, 0.75, 0.95}. The linear penalty is chosen from α ∈ {10⁻⁴, 3 × 10⁻⁴, 10⁻³, 3 × 10⁻³, '
        '10⁻², 3 × 10⁻², 10⁻¹}, and XGBoost from a depth of 1 or 2 and 50, 150 or 300 trees with a minimum child '
        'weight of 5. When no fold is available, as at the earliest 12-month origins, the fixed settings are used.']),
]
FIG_FILES = {1: 'fig_01_research_framework', 2: 'fig_02_price_history', 3: 'fig_03_backtest_skill',
             4: 'fig_04_coverage', 5: 'fig_06_fan_bands', 6: 'fig_09_procurement'}
SUPP_FIGS = {'S1': 'fig_05_conformal', 'S2': 'fig_07_model_risk', 'S3': 'fig_08_validation', 'S4': 'fig_10_timeline'}
# columns "<group>, <sub>" become a two-row header with the group merged over its sub-columns
GROUP_PATTERNS = [re.compile(r'^(?P<g>.+), (?P<s>h = \d+)$'), re.compile(r'^(?P<g>.+), (?P<s>\d+ m)$'),
                  re.compile(r'^(?P<g>as fitted|zero real drift), (?P<s>\d+ yr): P5 / P50 / P95 \(%\)$'),
                  re.compile(r'^(?P<g>Change from today \(%\)|Price level), (?P<s>P\d\d)$'),
                  re.compile(r'^(?P<g>.+), (?P<s>\d+ yr)$')]
SUPERSCRIPT = str.maketrans('0123456789−', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')
EQUATION = '<equation>'  # marker for the last element written
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ''  # optional file-name suffix, e.g. when a file is open in Word
BODY_LINE = 240  # body line spacing in 240ths of a line: single (rev9 on, to fit 20 pages); 480 = double, as to rev8


# --------------------------------------------------------------------------- paragraphs

def set_spacing(p, line=BODY_LINE, before=0, after=0, first_indent=None, left=None, hanging=None):
    pf = p.paragraph_format
    pPr = p._p.get_or_add_pPr()
    sp = pPr.find(qn('w:spacing'))
    if sp is None:
        sp = parse_xml(f'<w:spacing {nsdecls("w")}/>')
        pPr.append(sp)
    sp.set(qn('w:line'), str(line)); sp.set(qn('w:lineRule'), 'auto')
    sp.set(qn('w:before'), str(before)); sp.set(qn('w:after'), str(after))
    if first_indent is not None:
        pf.first_line_indent = Cm(first_indent)
    if left is not None:
        pf.left_indent = Cm(left)
    if hanging is not None:
        pf.first_line_indent = Cm(-hanging)


_SUBSUP = re.compile(r'([_^])(\{[^}]*\}|\([^)]*\)|.)')


def add_text(p, text, bold=None, italic=None, size=None, math_marks=True):
    """Runs with _x/_{..} as subscripts and ^x/^(..)/^{..} as superscripts."""
    pos = 0
    pieces = []
    if math_marks:
        for m in _SUBSUP.finditer(text):
            if m.start() > pos:
                pieces.append((text[pos:m.start()], None))
            body = m.group(2)
            if body.startswith('{') and body.endswith('}'):
                body = body[1:-1]
            pieces.append((body, 'sub' if m.group(1) == '_' else 'sup'))
            pos = m.end()
    pieces.append((text[pos:], None))
    for t, kind in pieces:
        if not t:
            continue
        r = p.add_run(t)
        r.bold, r.italic = bold, italic
        if size:
            r.font.size = Pt(size)
        if kind == 'sub':
            r.font.subscript = True
        elif kind == 'sup':
            r.font.superscript = True


def body_para(doc, runs, first_indent=0.75, math_marks=True, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.alignment = align
    set_spacing(p, first_indent=first_indent)
    for t, b, i in runs:
        add_text(p, t, bold=b, italic=i, math_marks=math_marks)
    return p


def heading(doc, text, level):
    p = doc.add_paragraph()
    set_spacing(p, before=240 if level == 1 else 120)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.bold = True
    if level == 2:
        r.italic = False
    return p


# --------------------------------------------------------------------------- equations (OMML)

def _r(t, plain=False):
    sty = '<m:rPr><m:sty m:val="p"/></m:rPr>' if plain else ''
    t = t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    return (f'<m:r>{sty}<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr>'
            f'<m:t xml:space="preserve">{t}</m:t></m:r>')


def _sub(b, s): return f'<m:sSub><m:e>{b}</m:e><m:sub>{s}</m:sub></m:sSub>'
def _sup(b, s): return f'<m:sSup><m:e>{b}</m:e><m:sup>{s}</m:sup></m:sSup>'
def _subsup(b, s, u): return f'<m:sSubSup><m:e>{b}</m:e><m:sub>{s}</m:sub><m:sup>{u}</m:sup></m:sSubSup>'
def _frac(n, d): return f'<m:f><m:num>{n}</m:num><m:den>{d}</m:den></m:f>'
def _acc(b, c='̃'): return f'<m:acc><m:accPr><m:chr m:val="{c}"/></m:accPr><m:e>{b}</m:e></m:acc>'


def _nary(lo, hi, body, c='∑'):
    return (f'<m:nary><m:naryPr><m:chr m:val="{c}"/><m:limLoc m:val="undOvr"/></m:naryPr>'
            f'<m:sub>{lo}</m:sub><m:sup>{hi}</m:sup><m:e>{body}</m:e></m:nary>')


def _d(body, beg='(', end=')'):
    return f'<m:d><m:dPr><m:begChr m:val="{beg}"/><m:endChr m:val="{end}"/></m:dPr><m:e>{body}</m:e></m:d>'


def equation_xml(tex: str) -> str:
    q_tau_j = _sub(_r('q'), _sub(_r('τ'), _r('j')))
    if tex.startswith('f(r)'):
        m = (_r('f') + _d(_r('r')) + _r('=') +
             _nary(_r('k=0'), _r('K'), _frac(_sup(_r('e'), _r('−λ')) + _sup(_r('λ'), _r('k')), _r('k!')) + _r('φ') +
                   _d(_r('r; μ+k') + _sub(_r('μ'), _r('J')) + _r(', ') + _sup(_r('σ'), _r('2')) +
                      _r('+k') + _subsup(_r('σ'), _r('J'), _r('2')))))
    elif tex.startswith('\\mathrm{CRPS}'):
        m = (_r('CRPS', plain=True) + _d(_r('F, R')) + _r('≈') + _frac(_r('2'), _r('19')) +
             _nary(_r('j=1'), _r('19'), _d(_r('𝟙') + _d(_r('R≤') + q_tau_j, '{', '}') + _r('−') +
                                           _sub(_r('τ'), _r('j'))) + _d(q_tau_j + _r('−R'))))
    elif tex.startswith('\\tilde'):
        m = (_subsup(_acc(_r('p')), _r('t'), _r('(i)')) + _r('=') + _subsup(_r('p'), _r('t'), _r('(i)')) +
             _r('−m t, m=') + _frac(_r('1'), _r('HN')) +
             _nary(_r('i=1'), _r('N'), _subsup(_r('p'), _r('H'), _r('(i)'))))
    else:
        raise ValueError(f'no OMML template for equation: {tex[:40]}')
    return (f'<w:p {nsdecls("w", "m")}><w:pPr><w:spacing w:before="120" w:after="120" w:line="240" '
            f'w:lineRule="auto"/><w:jc w:val="center"/></w:pPr><m:oMathPara><m:oMath>{m}</m:oMath>'
            f'</m:oMathPara></w:p>')


def add_equation(doc, tex: str, number: int):
    """Display equation in a borderless three-cell row, the number at the right margin."""
    t = doc.add_table(rows=1, cols=3)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    t._tbl.tblPr.append(parse_xml(f'<w:tblLayout {nsdecls("w")} w:type="fixed"/>'))
    widths = [1.5, TEXT_WIDTH_CM - 3.0, 1.5]
    for gc, w in zip(t._tbl.tblGrid.findall(qn('w:gridCol')), widths):
        gc.set(qn('w:w'), str(round(w * 567)))
    for cell, w in zip(t.rows[0].cells, widths):
        cell.width = Cm(w)
        cell._tc.get_or_add_tcPr().append(parse_xml(f'<w:vAlign {nsdecls("w")} w:val="center"/>'))
    mid = t.cell(0, 1)
    mid._tc.replace(mid.paragraphs[0]._p, parse_xml(equation_xml(tex)))
    num = t.cell(0, 2).paragraphs[0]
    num.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_spacing(num, line=240)
    num.add_run(f'({number})')


# --------------------------------------------------------------------------- tables

def _borders(cell, top=None, bottom=None):
    tcPr = cell._tc.get_or_add_tcPr()
    b = tcPr.find(qn('w:tcBorders'))
    if b is None:
        b = parse_xml(f'<w:tcBorders {nsdecls("w")}/>')
        tcPr.append(b)
    for side, val in (('top', top), ('bottom', bottom)):
        if val:
            b.append(parse_xml(f'<w:{side} {nsdecls("w")} w:val="single" w:sz="{val}" w:space="0" w:color="000000"/>'))


def _cell_text(cell, text, bold=False, size=8, align=WD_ALIGN_PARAGRAPH.LEFT, italic=None, math_marks=False):
    cell.text = ''
    p = cell.paragraphs[0]
    p.alignment = align
    set_spacing(p, line=240)
    # paragraph mark at the cell font size, so that empty cells do not make the row taller
    p._p.get_or_add_pPr().append(parse_xml(f'<w:rPr {nsdecls("w")}><w:sz w:val="{round(size * 2)}"/>'
                                           f'<w:szCs w:val="{round(size * 2)}"/></w:rPr>'))
    for k, line in enumerate(str(text).split('\n')):
        if k:
            p.add_run().add_break()
        add_text(p, line, bold=bold, italic=italic, size=size, math_marks=math_marks)


def _header_repeat(row):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))


def _widths(df: pd.DataFrame, header_labels: list[str], size: float) -> list[float]:
    """Column widths in cm from the text length (about 0.16 cm per character at 8 pt plus cell padding)."""
    per_char = 0.16 * size / 8
    w = []
    for c, lab in zip(df.columns, header_labels):
        lens = df[c].astype(str).str.len()
        chars = max(min(lens.max(), 40), max(len(x) for x in lab.split()) * 1.1, 3)
        w.append(chars * per_char + 0.4)
    tot = sum(w)
    return [TEXT_WIDTH_CM * x / tot for x in w]


def add_table(doc, df: pd.DataFrame, size=None, widths=None, section=None, two_line=False, short_h=False,
              math_marks=False):
    df = df.fillna('')
    sections = None
    if section:
        sections = df[section].tolist()
        df = df.drop(columns=[section])
    if two_line:
        df = df.apply(lambda col: col.map(lambda v: str(v).replace(' (', '\n(', 1)) if col.name != df.columns[0] else col)
    ncol = len(df.columns)
    size = size or (8 if ncol >= 8 else 9)
    groups = []
    for c in df.columns:
        g = s = None
        for pat in GROUP_PATTERNS:
            m = pat.match(c)
            if m:
                g, s = m.group('g'), m.group('s')
                if short_h:
                    s = re.sub(r'^h = (\d+)$|^(\d+) m$', lambda mm: mm.group(1) or mm.group(2), s)
                break
        groups.append((g, s))
    two_rows = any(g for g, _ in groups)
    nhead = 2 if two_rows else 1
    body = []  # ('section', text) or ('row', values)
    for i, (_, row) in enumerate(df.iterrows()):
        if sections is not None and (i == 0 or sections[i] != sections[i - 1]):
            body.append(('section', sections[i]))
        body.append(('row', list(row)))
    t = doc.add_table(rows=nhead + len(body), cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr
    tblPr.append(parse_xml(f'<w:tblLayout {nsdecls("w")} w:type="fixed"/>'))
    labels = [s if g else c for c, (g, s) in zip(df.columns, groups)]
    widths = widths or _widths(df, labels, size)
    assert len(widths) == ncol, (len(widths), ncol, list(df.columns))
    for gc, w in zip(t._tbl.tblGrid.findall(qn('w:gridCol')), widths):
        gc.set(qn('w:w'), str(round(w * 567)))
    last_plain = max((j for j, (g, _) in enumerate(groups) if g is None), default=0)
    for j, (c, (g, s)) in enumerate(zip(df.columns, groups)):
        if two_rows:
            if g:
                _cell_text(t.cell(0, j), g[0].upper() + g[1:], bold=True, size=size, align=WD_ALIGN_PARAGRAPH.CENTER)
                _cell_text(t.cell(1, j), s, bold=True, size=size, align=WD_ALIGN_PARAGRAPH.CENTER)
            else:
                _cell_text(t.cell(0, j), c, bold=True, size=size)
                _cell_text(t.cell(1, j), 'h (months)' if short_h and j == last_plain else '', size=size,
                           italic=None, align=WD_ALIGN_PARAGRAPH.RIGHT)
        else:
            _cell_text(t.cell(0, j), c, bold=True, size=size,
                       align=WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER)
    for i, (kind, vals) in enumerate(body):
        r = nhead + i
        if kind == 'section':
            first = t.cell(r, 0).merge(t.cell(r, ncol - 1))
            _cell_text(first, vals, size=size, italic=True)
            set_spacing(first.paragraphs[0], line=240, before=60)
            continue
        for j, v in enumerate(vals):
            _cell_text(t.cell(r, j), v, size=size, math_marks=math_marks,
                       align=WD_ALIGN_PARAGRAPH.LEFT if j == 0 or not _numeric(v) else WD_ALIGN_PARAGRAPH.CENTER)
    for row in t.rows:
        row._tr.get_or_add_trPr().append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        for j, cell in enumerate(row.cells):
            cell.width = Cm(widths[j])
    # merge group headers and ungrouped header cells
    if two_rows:
        j = 0
        while j < ncol:
            g = groups[j][0]
            if g is None:
                if not (short_h and j == last_plain):
                    t.cell(0, j).merge(t.cell(1, j))
                j += 1
                continue
            k = j
            while k + 1 < ncol and groups[k + 1][0] == g:
                k += 1
            if k > j:
                first = t.cell(0, j)
                for kk in range(j + 1, k + 1):
                    t.cell(0, kk).text = ''
                first.merge(t.cell(0, k))
                _cell_text(first, g[0].upper() + g[1:], bold=True, size=size, align=WD_ALIGN_PARAGRAPH.CENTER)
            j = k + 1
    for j in range(ncol):
        _borders(t.cell(0, j), top=8)
        _borders(t.cell(nhead - 1, j), bottom=6)
        _borders(t.cell(len(t.rows) - 1, j), bottom=8)
        if two_rows and groups[j][0]:
            _borders(t.cell(0, j), bottom=4)
    for r in range(nhead):
        _header_repeat(t.rows[r])
    return t


def _numeric(v) -> bool:
    v = str(v)
    return (v == '–' or bool(re.match(r'^\d+(, \d+)+$', v))
            or bool(re.match(r'^[\s+\-−–]?[\d.,]+%?( × 10[⁰-⁹⁻¹²³]+)?(\s*\(.*\))?$', v)))


def _print_value(v: str) -> str:
    """Typographic minus signs, e-notation as powers of ten, and no negative zero."""
    v = re.sub(r'(?<![\w.])-(?=\d)', '−', str(v))
    v = re.sub(r'(?<![\w.])−0(?![\d.])', '0', v)
    return re.sub(r'(\d(?:\.\d+)?)e([−-]?)0*(\d+)',
                  lambda m: f'{m.group(1)} × 10' + (('−' if m.group(2) else '') + m.group(3)).translate(SUPERSCRIPT), v)


def table_frame(name: str) -> pd.DataFrame:
    """Print version of the CSV `name` (in outputs/manuscript)."""
    df = pd.read_csv(MS / f'{name}.csv', encoding='utf-8-sig', dtype=str).fillna('')
    # coverage shares with two decimals
    for c in df.columns:
        if c.startswith(('P5–P95', '≤ P95')):
            df[c] = df[c].map(lambda x: f'{float(x):.2f}' if x != '' else '')
    if name == 'table_01_data_audit':
        df['Grade'] = df['Grade'].str.replace('Ready-mixed concrete 180 ksc (reference)', 'Concrete, 180 ksc',
                                              regex=False)
        df['Period'] = df['Period'].str.replace(' to ', '–', regex=False)
        df = df.rename(columns={'Diameters (mm)': 'Diameter (mm)'})
    if name == 'table_S06_coverage_full':
        df['Model'] = (df['Model'].str.replace('2020-06..2022-12', '2020-06 to 2022-12', regex=False)
                       .str.replace('2023-01..', '2023-01 on', regex=False)
                       .str.replace(' contingency', ' allowance', regex=False))
    if name in ('table_S07_conformal_full', 'table_rev_conformal_compact'):
        for c in df.columns:
            if c.startswith('CRPS'):
                df[c] = df[c].map(lambda x: f'{float(x):.2f}')
    if name == 'table_S08_bands_full':  # band as section rows, levels with thousands separators, unit implied by the basis
        df['Band'] = (df['Band'].str.replace('headline: Block bootstrap (demeaned)',
                                             'Headline band: block bootstrap of demeaned real changes', regex=False)
                      .str.replace('comparator: Random walk (real, driftless)',
                                   'Comparator: driftless real random walk', regex=False))
        for c in ('P50', 'P80', 'P95'):
            df[c] = df[c].map(lambda x: f'{float(x):,.0f}')
        df['Basis'] = df['Basis'].str.replace(r'(\d) %', r'\1%', regex=True)
        df = df.drop(columns=['Unit']).rename(
            columns={**{f'P{q} (%)': f'Change from today (%), P{q}' for q in (50, 80, 90, 95)},
                     **{f'P{q}': f'Price level, P{q}' for q in (50, 80, 95)}})
    if name == 'table_S09_validation_full':
        df['Terms'] = df['Terms'].map(lambda x: 'Real terms (deflated by US CPI)' if x.startswith('real')
                                      else 'Nominal terms' if x == 'nominal' else x)
    return df.apply(lambda col: col.map(_print_value))


# --------------------------------------------------------------------------- captions, figures

def read_captions() -> dict:
    caps = {}
    for line in (MS / 'captions_draft.md').read_text(encoding='utf-8').splitlines():
        m = re.match(r'^\*\*(Fig\.|Table) (S?\d+[a-z]?)\.\*\* (.+)$', line.strip())
        if m:
            caps[('Fig' if m.group(1) == 'Fig.' else 'Table', m.group(2))] = m.group(3)
    return caps


def caption(doc, kind, n, text):
    p = doc.add_paragraph()
    set_spacing(p, line=240, before=120 if kind == 'Table' else 60, after=60 if kind == 'Table' else 200)
    if kind == 'Table':
        p.paragraph_format.keep_with_next = True
    r = p.add_run(f'{"Fig." if kind == "Fig" else "Table"} {n}. ')
    r.bold = True; r.font.size = Pt(11)
    add_text(p, text, size=11, math_marks=False)
    return p


def add_figure(doc, n, caps, files=FIG_FILES):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_spacing(p, line=240, before=120)
    p.paragraph_format.keep_with_next = True
    width = FIG_WIDTHS_CM.get(n, FIG_WIDTH_CM) if files is FIG_FILES else FIG_WIDTH_CM
    p.add_run().add_picture(str(MS / f'{files[n]}.png'), width=Cm(width))
    caption(doc, 'Fig', n, caps[('Fig', str(n))])


# --------------------------------------------------------------------------- document

def new_document():
    d = docx.Document()
    st = d.styles['Normal']
    st.font.name = 'Times New Roman'
    st.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    st.font.size = Pt(12)
    s = d.sections[0]
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Cm(2.5)
    s._sectPr.append(parse_xml(f'<w:lnNumType {nsdecls("w")} w:countBy="1" w:distance="283" w:restart="continuous"/>'))
    fp = s.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fp._p.append(parse_xml(f'<w:r {nsdecls("w")}><w:fldChar w:fldCharType="begin"/></w:r>'))
    fp._p.append(parse_xml(f'<w:r {nsdecls("w")}><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'))
    fp._p.append(parse_xml(f'<w:r {nsdecls("w")}><w:fldChar w:fldCharType="end"/></w:r>'))
    d.core_properties.title = TITLE
    d.core_properties.author = ', '.join(a for a, _ in AUTHORS)
    return d


def title_page(d):
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; set_spacing(p, line=360, after=240)
    r = p.add_run(TITLE); r.bold = True; r.font.size = Pt(14)
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; set_spacing(p, line=360, after=120)
    for i, (name, sup) in enumerate(AUTHORS):
        sep = '' if i == 0 else (', and ' if i == len(AUTHORS) - 1 else ', ')
        p.add_run(sep + name)
        rs = p.add_run(sup); rs.font.superscript = True
    for num, aff in AFFILIATIONS:
        p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; set_spacing(p, line=240)
        rs = p.add_run(num); rs.font.superscript = True; rs.font.size = Pt(11)
        r = p.add_run(' ' + aff); r.font.size = Pt(11)
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; set_spacing(p, line=240, after=240)
    r = p.add_run(CORRESPONDING); r.font.size = Pt(11)


def numbering_formats(src) -> dict:
    """numId -> numFmt of level 0 ('decimal', 'bullet', ...) in the source document."""
    out = {}
    try:
        numbering = src.part.numbering_part.element
    except Exception:  # noqa: BLE001
        return out
    abstract = {}
    for an in numbering.findall(qn('w:abstractNum')):
        lvl = an.find(qn('w:lvl'))
        fmt = lvl.find(qn('w:numFmt')).get(qn('w:val')) if lvl is not None and lvl.find(qn('w:numFmt')) is not None else 'bullet'
        abstract[an.get(qn('w:abstractNumId'))] = fmt
    for num in numbering.findall(qn('w:num')):
        out[num.get(qn('w:numId'))] = abstract.get(num.find(qn('w:abstractNumId')).get(qn('w:val')), 'bullet')
    return out


def build():
    src = docx.Document(str(SOURCE))
    fmts = numbering_formats(src)
    caps = read_captions()
    d = new_document()
    title_page(d)

    placed_t, placed_f = set(), set()
    pending: list[tuple[str, int]] = []
    last_text = ''
    list_counter = {}
    in_refs = False
    src_tables = iter(src.tables)
    started = False
    skipping = False
    eq_count = 0

    def flush():
        nonlocal pending
        keep = []
        for kind, n in pending:
            if DEFER.get((kind, n), 0) > 0:  # placed after more text so that the page above it fills
                DEFER[(kind, n)] -= 1
                keep.append((kind, n))
                continue
            if kind == 'Table':
                caption(d, 'Table', n, caps[('Table', str(n))])
                name, layout = TABLES[n]
                add_table(d, table_frame(name), **layout)
                spacer = d.add_paragraph(); set_spacing(spacer, line=240, after=120)
            else:
                add_figure(d, n, caps)
        pending = keep

    def note_citations(text):
        for m in re.finditer(r'\b(Table|Fig\.) (\d+)', text):
            kind = 'Table' if m.group(1) == 'Table' else 'Fig'
            n = int(m.group(2))
            seen = placed_t if kind == 'Table' else placed_f
            if n not in seen and (n in TABLES if kind == 'Table' else n in FIG_FILES):
                seen.add(n)
                pending.append((kind, n))

    for el in src.element.body.iterchildren():
        tag = el.tag.split('}')[1]
        if tag == 'p':
            p = docx.text.paragraph.Paragraph(el, src)
            text = p.text
            style = p.style.name
            if not started:
                if style.startswith('Heading') and text.strip() == 'Abstract':
                    started = True
                else:
                    continue
            if not text.strip():
                continue
            if style.startswith('Heading'):
                skipping = text.strip() == 'Highlights'  # submitted as a separate file
                if skipping:
                    continue
            elif skipping:
                continue
            if style.startswith('Heading'):
                if text.strip() == 'References':
                    flush()
                    in_refs = True
                elif pending and not last_text.rstrip().endswith(':'):
                    flush()
                heading(d, text.strip(), 1 if style == 'Heading 2' else 2)
                last_text = text
                continue
            if style == 'Code':
                eq_count += 1
                add_equation(d, text.strip(), eq_count)
                last_text = EQUATION
                continue
            runs = [(r.text, r.bold, r.italic) for r in p.runs]
            numpr = el.find('.//' + qn('w:numPr'))
            if numpr is not None:
                nid = numpr.find(qn('w:numId')).get(qn('w:val'))
                if fmts.get(nid, 'bullet') == 'bullet':
                    q = d.add_paragraph(style='List Bullet')
                    q.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                    set_spacing(q)
                else:
                    list_counter[nid] = list_counter.get(nid, 0) + 1
                    q = d.add_paragraph()
                    q.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                    set_spacing(q, left=1.0, hanging=0.5)
                    q.add_run(f'{list_counter[nid]}.\t')
                for t, b, i in runs:
                    add_text(q, t, bold=b, italic=i)
                note_citations(text)
                last_text = text
                continue
            if in_refs:
                q = d.add_paragraph()
                q.alignment = WD_ALIGN_PARAGRAPH.LEFT  # long DOIs and URLs stretch justified lines
                set_spacing(q, left=0.85, hanging=0.85)
                add_text(q, text, math_marks=False)
                continue
            if pending and not last_text.rstrip().endswith(':'):
                flush()
            no_indent = (text.startswith('Keywords:') or last_text.strip() == 'Abstract'
                         or (last_text is EQUATION and text[:1].islower()))  # "where ..." continues the sentence
            body_para(d, runs, first_indent=0 if no_indent else 0.75)
            note_citations(text)
            last_text = text
        elif tag == 'tbl':
            if not started:
                continue
            t = next(src_tables)
            rows = [[c.text for c in r.cells] for r in t.rows]
            frame = pd.DataFrame(rows[1:], columns=rows[0]).apply(lambda col: col.map(_print_value))
            add_table(d, frame, size=9, math_marks=True)
            spacer = d.add_paragraph(); set_spacing(spacer, line=240, after=120)
            last_text = ' '.join(rows[-1])
    flush()
    missing_t = sorted(set(TABLES) - placed_t)
    missing_f = sorted(set(FIG_FILES) - placed_f)
    out = MS / f'Manuscript_rebar_ASEJ{SUFFIX}.docx'
    d.save(out)

    s = new_document()
    p = s.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; set_spacing(p, line=360, after=120)
    r = p.add_run('Supplementary material'); r.bold = True; r.font.size = Pt(14)
    p = s.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; set_spacing(p, line=360, after=240)
    r = p.add_run(TITLE); r.italic = True
    for note_title, paras in SUPP_NOTES:
        p = s.add_paragraph(); set_spacing(p, line=360, after=120)
        p.add_run(note_title).bold = True
        for text in paras:
            q = s.add_paragraph(); q.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            set_spacing(q, line=360, first_indent=0.75)
            add_text(q, text, math_marks='.py' not in text)  # _x subscripts, except in script file names
        spacer = s.add_paragraph(); set_spacing(spacer, line=240, after=240)
    for key, (name, layout) in SUPP_TABLES.items():
        caption(s, 'Table', key, caps[('Table', key)])
        add_table(s, table_frame(name), **layout)
        spacer = s.add_paragraph(); set_spacing(spacer, line=240, after=240)
    for key in SUPP_FIGS:
        add_figure(s, key, caps, files=SUPP_FIGS)
    s.save(MS / f'Supplementary_rebar_ASEJ{SUFFIX}.docx')

    h = new_document()
    p = h.add_paragraph(); set_spacing(p, line=360, after=120)
    r = p.add_run('Highlights'); r.bold = True
    for item in HIGHLIGHTS:
        q = h.add_paragraph(style='List Bullet'); set_spacing(q, line=360)
        q.add_run(item)
    h.save(MS / f'Highlights_rebar_ASEJ{SUFFIX}.docx')
    print(f'wrote {out.name}, Supplementary_rebar_ASEJ.docx and Highlights_rebar_ASEJ.docx; tables placed {sorted(placed_t)}, '
          f'figures placed {sorted(placed_f)}; not cited: tables {missing_t}, figures {missing_f}')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    build()
