"""Report assembler: tool outputs + figures -> a markdown and HTML report.

The last Phase 1 tool, and the one that ties the others together. It takes the
tidy outputs of the analysis tools -- ``differential_abundance``,
``alpha_diversity``, ``beta_diversity`` and ``over_representation_analysis`` --
generates a figure for each, and assembles everything into two artifacts:

- ``report.md``   : markdown (diffs well, renders on GitHub), figures linked.
- ``report.html`` : a single self-contained file with figures embedded as
                    base64, openable in any browser with no other files.

Both are rendered from one shared list of sections, so the two formats never
drift apart. Every input is optional: the assembler renders whatever subset of
tools was actually run, which is exactly the flexibility the agent needs when it
decides on the fly which analyses a question calls for.

The return value is a tidy ``Report`` dataclass (paths + the rendered strings),
following the project's one-predictable-return-value convention.
"""

from __future__ import annotations

import base64
import datetime as _dt
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: render to files, never to a screen.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from microbiome_agent.analysis.diversity import BetaDiversityResult  # noqa: E402

_SIG_COLOR = "#c0392b"
_NULL_COLOR = "#bdc3c7"
_Q_FLOOR = 1e-300  # avoid log10(0) when a q-value underflows to exactly 0.


# --------------------------------------------------------------------------- #
# Return type
# --------------------------------------------------------------------------- #
@dataclass
class Report:
    """A rendered analysis report.

    Attributes
    ----------
    title:
        The report title.
    markdown, html:
        The rendered report as strings (also written to disk).
    markdown_path, html_path:
        Where the two files were written.
    figures:
        Mapping of figure key (e.g. ``"volcano"``) -> PNG path on disk.
    """

    title: str
    markdown: str
    html: str
    markdown_path: Path
    html_path: Path
    figures: dict[str, Path] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #
def _fmt_num(value: object) -> str:
    """Compact, readable formatting for a table cell."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if pd.isna(v):
            return ""
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.2e}"
        return f"{v:.4g}"
    return str(value)


def _fmt_p(p: float) -> str:
    return "< 0.001" if p < 1e-3 else f"{p:.3g}"


def _section_table(df: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, int]:
    """Return the first ``top_n`` rows of ``df`` plus the full row count."""
    return df.head(top_n).copy(), len(df)


# --------------------------------------------------------------------------- #
# Figures (each saves a PNG and returns its path)
# --------------------------------------------------------------------------- #
def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _volcano_plot(differential: pd.DataFrame, path: Path, fdr_alpha: float) -> Path:
    q = differential["q_value"].to_numpy(dtype=float).clip(min=_Q_FLOOR)
    neglogq = -np.log10(q)
    lfc = differential["log2_fold_change"].to_numpy(dtype=float)
    sig = differential["significant"].to_numpy(dtype=bool)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(lfc[~sig], neglogq[~sig], s=18, c=_NULL_COLOR,
               label="not significant", edgecolors="none")
    ax.scatter(lfc[sig], neglogq[sig], s=22, c=_SIG_COLOR,
               label=f"q < {fdr_alpha}", edgecolors="none")
    ax.axhline(-np.log10(fdr_alpha), ls="--", lw=0.8, c="0.4")
    ax.axvline(0, ls="-", lw=0.6, c="0.7")

    # Annotate the few strongest hits by effect size among the significant.
    if sig.any():
        sig_idx = np.where(sig)[0]
        strongest = sig_idx[np.argsort(-np.abs(lfc[sig_idx]))[:5]]
        for i in strongest:
            ax.annotate(str(differential.iloc[i]["feature"]),
                        (lfc[i], neglogq[i]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("log2 fold change")
    ax.set_ylabel("-log10(q-value)")
    ax.set_title("Differential abundance (volcano)")
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, path)


def _alpha_boxplot(alpha: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if "group" in alpha.columns:
        labels = sorted(alpha["group"].unique())
        data = [alpha.loc[alpha["group"] == g, "shannon"].to_numpy() for g in labels]
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        rng = np.random.default_rng(0)
        for i, vals in enumerate(data, start=1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals,
                       s=16, c=_SIG_COLOR, alpha=0.6, edgecolors="none")
        ax.set_xlabel("group")
    else:
        ax.boxplot([alpha["shannon"].to_numpy()], tick_labels=["all samples"],
                   showfliers=False)
    ax.set_ylabel("Shannon index (bits)")
    ax.set_title("Alpha diversity")
    return _save(fig, path)


def _pcoa_plot(beta: BetaDiversityResult, groups: pd.Series | None, path: Path) -> Path:
    from skbio import DistanceMatrix
    from skbio.stats.ordination import pcoa

    d = beta.distances
    dm = DistanceMatrix(d.to_numpy(dtype=float), ids=list(d.index))
    with warnings.catch_warnings():
        # Bray-Curtis is non-metric -> small negative eigenvalues are expected.
        warnings.simplefilter("ignore", RuntimeWarning)
        ordination = pcoa(dm)
    coords = ordination.samples
    prop = ordination.proportion_explained

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if groups is not None:
        g = groups.loc[coords.index]
        labels = sorted(g.unique())
        cmap = plt.get_cmap("tab10")
        for i, lab in enumerate(labels):
            sel = (g == lab).to_numpy()
            ax.scatter(coords["PC1"].to_numpy()[sel], coords["PC2"].to_numpy()[sel],
                       s=40, color=cmap(i), label=str(lab), edgecolors="none")
        ax.legend(fontsize=8, frameon=False, title="group")
    else:
        ax.scatter(coords["PC1"], coords["PC2"], s=40, c=_SIG_COLOR,
                   edgecolors="none")

    ax.set_xlabel(f"PC1 ({prop['PC1'] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({prop['PC2'] * 100:.1f}%)")
    ax.set_title(f"Beta diversity ({beta.metric}, PCoA)")
    return _save(fig, path)


def _enrichment_bar(enrichment: pd.DataFrame, path: Path, top_n: int) -> Path:
    top = enrichment.head(top_n).iloc[::-1]  # most significant at top of the chart
    colors = [_SIG_COLOR if s else _NULL_COLOR for s in top["significant"]]
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.4 * len(top) + 1)))
    ax.barh(range(len(top)), top["fold_enrichment"].to_numpy(dtype=float), color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["set"].astype(str), fontsize=8)
    ax.axvline(1.0, ls="--", lw=0.8, c="0.4")  # fold = 1 -> no enrichment
    ax.set_xlabel("fold enrichment")
    ax.set_title("Functional enrichment (top sets)")
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Section building (format-agnostic)
# --------------------------------------------------------------------------- #
def _build_sections(
    *,
    differential: pd.DataFrame | None,
    alpha: pd.DataFrame | None,
    beta: BetaDiversityResult | None,
    enrichment: pd.DataFrame | None,
    groups: pd.Series | None,
    fig_dir: Path,
    figures: dict[str, Path],
    top_n: int,
    fdr_alpha: float,
) -> tuple[list[dict], list[str]]:
    """Return (sections, summary_lines). Each section: title/prose/table/figure."""
    sections: list[dict] = []
    summary: list[str] = []

    if differential is not None:
        n_sig = int(differential["significant"].sum())
        n_total = len(differential)
        top = differential.iloc[0]
        direction = ("higher in " + str(top["group_b"])
                     if top["log2_fold_change"] > 0
                     else "higher in " + str(top["group_a"]))
        summary.append(
            f"Differential abundance: {n_sig} of {n_total} features significant "
            f"at q < {fdr_alpha}; strongest is {top['feature']} "
            f"({direction}, q = {_fmt_p(float(top['q_value']))}).")
        figures["volcano"] = _volcano_plot(
            differential, fig_dir / "volcano.png", fdr_alpha)
        tbl, total = _section_table(
            differential[["feature", "log2_fold_change", "p_value",
                          "q_value", "significant"]], top_n)
        sections.append({
            "title": "Differential abundance",
            "prose": (f"{n_sig} of {n_total} features are significant at "
                      f"q < {fdr_alpha}. "
                      + (f"Showing the top {top_n} by adjusted p-value."
                         if total > top_n else "")),
            "table": tbl,
            "figure": figures["volcano"],
            "figure_alt": "Volcano plot",
        })

    if alpha is not None:
        if "group" in alpha.columns:
            means = alpha.groupby("group")["shannon"].mean()
            desc = "; ".join(f"{g}: {m:.2f}" for g, m in means.items())
            summary.append(f"Alpha diversity (mean Shannon by group): {desc}.")
            prose = f"Mean Shannon index by group -- {desc} (bits)."
        else:
            summary.append(
                f"Alpha diversity: mean Shannon {alpha['shannon'].mean():.2f} bits "
                f"across {len(alpha)} samples.")
            prose = f"Shannon index across {len(alpha)} samples."
        figures["alpha"] = _alpha_boxplot(alpha, fig_dir / "alpha_diversity.png")
        sections.append({
            "title": "Alpha diversity",
            "prose": prose,
            "table": None,
            "figure": figures["alpha"],
            "figure_alt": "Alpha diversity boxplot",
        })

    if beta is not None:
        row = beta.permanova.iloc[0]
        p = float(row["p_value"])
        verdict = ("groups differ significantly"
                   if p < fdr_alpha else "no significant group separation")
        summary.append(
            f"Beta diversity ({beta.metric}): PERMANOVA {verdict} "
            f"(pseudo-F = {float(row['statistic']):.3g}, p = {_fmt_p(p)}, "
            f"{int(row['num_permutations'])} permutations).")
        figures["pcoa"] = _pcoa_plot(beta, groups, fig_dir / "pcoa.png")
        sections.append({
            "title": "Beta diversity",
            "prose": (f"PERMANOVA on {beta.metric} distances: "
                      f"pseudo-F = {float(row['statistic']):.3g}, "
                      f"p = {_fmt_p(p)} ({int(row['num_permutations'])} "
                      f"permutations). {verdict.capitalize()}."),
            "table": None,
            "figure": figures["pcoa"],
            "figure_alt": "PCoA ordination",
        })

    if enrichment is not None:
        n_sig = int(enrichment["significant"].sum())
        summary.append(
            f"Functional enrichment: {n_sig} of {len(enrichment)} sets "
            f"over-represented at q < {fdr_alpha}"
            + (f"; top is {enrichment.iloc[0]['set']}." if n_sig else "."))
        figures["enrichment"] = _enrichment_bar(
            enrichment, fig_dir / "enrichment.png", top_n)
        tbl, total = _section_table(
            enrichment[["set", "set_size", "n_hits_in_set", "fold_enrichment",
                        "q_value", "significant"]], top_n)
        sections.append({
            "title": "Functional enrichment",
            "prose": (f"{n_sig} of {len(enrichment)} sets over-represented at "
                      f"q < {fdr_alpha}. "
                      + (f"Showing the top {top_n} by adjusted p-value."
                         if total > top_n else "")),
            "table": tbl,
            "figure": figures["enrichment"],
            "figure_alt": "Enrichment bar chart",
        })

    return sections, summary


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [
        "| " + " | ".join(_fmt_num(r[c]) for c in cols) + " |"
        for _, r in df.iterrows()
    ]
    return "\n".join([head, sep, *rows])


def _to_markdown(title: str, stamp: str, summary: list[str],
                 sections: list[dict]) -> str:
    parts = [f"# {title}", "", f"_Generated by microbiome-agent · {stamp}_", ""]
    if summary:
        parts += ["## Summary", ""]
        parts += [f"- {line}" for line in summary]
        parts += [""]
    for s in sections:
        parts += [f"## {s['title']}", "", s["prose"], ""]
        if s["figure"] is not None:
            parts += [f"![{s['figure_alt']}](figures/{Path(s['figure']).name})", ""]
        if s["table"] is not None:
            parts += [_md_table(s["table"]), ""]
    return "\n".join(parts).rstrip() + "\n"


def _img_data_uri(path: Path) -> str:
    b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


_HTML_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
max-width:900px;margin:2rem auto;padding:0 1rem;color:#222;line-height:1.5}
h1{border-bottom:2px solid #c0392b;padding-bottom:.3rem}
h2{margin-top:2rem;color:#333}
table{border-collapse:collapse;margin:1rem 0;font-size:.9rem}
th,td{border:1px solid #ddd;padding:.35rem .6rem;text-align:right}
th{background:#f5f5f5}
td:first-child,th:first-child{text-align:left}
img{max-width:100%;height:auto;margin:1rem 0}
.stamp{color:#888;font-size:.85rem}
ul{background:#fafafa;border-left:3px solid #c0392b;padding:.6rem 1.4rem}
"""


def _to_html(title: str, stamp: str, summary: list[str],
             sections: list[dict]) -> str:
    out = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
           f"<title>{title}</title><style>{_HTML_CSS}</style></head><body>",
           f"<h1>{title}</h1>",
           f"<p class='stamp'>Generated by microbiome-agent · {stamp}</p>"]
    if summary:
        out.append("<h2>Summary</h2><ul>")
        out += [f"<li>{line}</li>" for line in summary]
        out.append("</ul>")
    for s in sections:
        out += [f"<h2>{s['title']}</h2>", f"<p>{s['prose']}</p>"]
        if s["figure"] is not None:
            out.append(
                f"<img alt='{s['figure_alt']}' src='{_img_data_uri(s['figure'])}'>")
        if s["table"] is not None:
            out.append(s["table"].to_html(
                index=False, border=0, escape=True,
                float_format=lambda v: _fmt_num(v)))
    out.append("</body></html>")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _require_columns(df: pd.DataFrame, cols: set[str], name: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"`{name}` must be a pandas DataFrame.")
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"`{name}` is missing columns: {sorted(missing)}.")


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def assemble_report(
    *,
    title: str,
    output_dir: str | Path,
    differential: pd.DataFrame | None = None,
    alpha: pd.DataFrame | None = None,
    beta: BetaDiversityResult | None = None,
    enrichment: pd.DataFrame | None = None,
    groups: pd.Series | None = None,
    top_n: int = 20,
    fdr_alpha: float = 0.05,
) -> Report:
    """Assemble tool outputs and figures into a markdown + HTML report.

    Parameters
    ----------
    title:
        Report title, used as the top heading and the HTML ``<title>``.
    output_dir:
        Directory to write ``report.md``, ``report.html`` and ``figures/``.
        Created if it does not exist.
    differential:
        Output of ``differential_abundance`` (or ``None`` to omit the section).
    alpha:
        Output of ``alpha_diversity``. If it carries a ``group`` column the
        boxplot is split by group.
    beta:
        A ``BetaDiversityResult`` from ``beta_diversity``.
    enrichment:
        Output of ``over_representation_analysis``.
    groups:
        Optional grouping vector used to colour the PCoA ordination. Pass the
        same Series you handed to ``beta_diversity``.
    top_n:
        Row cap for the differential and enrichment tables, and bar cap for the
        enrichment figure.
    fdr_alpha:
        Significance threshold quoted in the narrative and drawn on the volcano.

    Returns
    -------
    Report
        Rendered strings plus the paths written to disk.

    Raises
    ------
    ValueError
        If no tool output is supplied, or if a supplied output is missing the
        columns its section needs.
    """
    if all(x is None for x in (differential, alpha, beta, enrichment)):
        raise ValueError(
            "Nothing to report: supply at least one of differential, alpha, "
            "beta, enrichment.")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("`title` must be a non-empty string.")

    if differential is not None:
        _require_columns(
            differential,
            {"feature", "group_a", "group_b", "log2_fold_change",
             "p_value", "q_value", "significant"},
            "differential")
    if alpha is not None:
        _require_columns(alpha, {"sample", "shannon"}, "alpha")
    if enrichment is not None:
        _require_columns(
            enrichment,
            {"set", "set_size", "n_hits_in_set", "fold_enrichment",
             "q_value", "significant"},
            "enrichment")
    if beta is not None and not isinstance(beta, BetaDiversityResult):
        raise ValueError("`beta` must be a BetaDiversityResult.")

    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    figures: dict[str, Path] = {}
    sections, summary = _build_sections(
        differential=differential, alpha=alpha, beta=beta,
        enrichment=enrichment, groups=groups, fig_dir=fig_dir,
        figures=figures, top_n=top_n, fdr_alpha=fdr_alpha)

    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    markdown = _to_markdown(title, stamp, summary, sections)
    html = _to_html(title, stamp, summary, sections)

    md_path = output_dir / "report.md"
    html_path = output_dir / "report.html"
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")

    return Report(
        title=title, markdown=markdown, html=html,
        markdown_path=md_path, html_path=html_path, figures=figures)
