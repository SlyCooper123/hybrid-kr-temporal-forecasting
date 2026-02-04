#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_knowledge_graph_from_rules.py
Author: Anonimous (RuleFit pipeline) + helper script

Purpose
-------
Given a CSV of RuleFit rules with columns such as:
  - rule  (string with conditions joined by '&')
  - coef / coefficient / importance  (real-valued effect)
  - support (optional, in [0,1])
this script builds several knowledge-graph style visualisations:

1) Detailed rule-level graph (conditions + rules + outcomes).
2) Cleaner "paper" version of the same graph.
3) Aggregated variable-level graphs:
   - variables -> {Consumption↑, Consumption↓}
   - variables -> Consumption↑ only
   - variables -> Consumption↓ only

Usage (example)
---------------
python make_knowledge_graph_from_rules.py \
    --rules-csv outputs_knowledge/rulefit_rules.csv \
    --out-prefix knowledge_graph_rules \
    --min-support 0.02 \
    --top 40
"""

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# --------------------------------------------------------------------
# Helpers to detect columns and parse rules
# --------------------------------------------------------------------
def detect_coef_column(df: pd.DataFrame) -> str | None:
    for cand in ["coef", "coefficient", "importance"]:
        if cand in df.columns:
            return cand
        for c in df.columns:
            if c.lower() == cand:
                return c
    return None


def detect_support_column(df: pd.DataFrame) -> str | None:
    for cand in ["support", "supp"]:
        if cand in df.columns:
            return cand
        for c in df.columns:
            if c.lower() == cand:
                return c
    return None


def detect_rule_column(df: pd.DataFrame) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in ["rule", "rules"]:
        if cand in cols_lower:
            return cols_lower[cand]
    raise SystemExit("Could not find a 'rule' column in the CSV.")


def split_conditions(rule_str: str) -> list[str]:
    parts = [p.strip() for p in rule_str.split("&")]
    return [p for p in parts if p]


def get_variable_name(condition: str) -> str:
    # e.g. "tmean_c <= 13.55" -> "tmean_c"
    m = re.match(r"\s*([A-Za-z0-9_]+)", condition)
    if m:
        return m.group(1)
    return condition.strip()


def pretty_condition(condition: str, ndigits: int = 2) -> str:
    """
    Round numeric thresholds in the condition string to ndigits.
    Example:
      "tmean_c <= 13.550000190734863" -> "tmean_c <= 13.55"
    """

    def repl(m):
        op = m.group(1)
        num = float(m.group(2))
        return f"{op} {round(num, ndigits)}"

    # operators followed by number
    pattern = r"(<=|>=|<|>)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    return re.sub(pattern, repl, condition)


# --------------------------------------------------------------------
# Build detailed rule-level graph
# --------------------------------------------------------------------
def build_rule_level_graph(df: pd.DataFrame, coef_col: str, support_col: str | None):
    G = nx.DiGraph()

    # outcome nodes
    up_node = "Consumption ↑"
    down_node = "Consumption ↓"
    G.add_node(up_node, kind="outcome_up", label=up_node)
    G.add_node(down_node, kind="outcome_down", label=down_node)

    for idx, row in df.iterrows():
        rule_str = str(row["rule_str"])
        conditions = split_conditions(rule_str)
        coef = float(row[coef_col])
        support = float(row[support_col]) if support_col is not None else 1.0

        if math.isnan(coef) or coef == 0.0:
            continue  # skip neutral/undefined rules

        rule_node = f"R{idx+1}"
        G.add_node(
            rule_node,
            kind="rule",
            coef=coef,
            support=support,
            label=f"R{idx+1}",
        )

        # condition nodes
        for cond in conditions:
            var_name = get_variable_name(cond)
            cond_node = cond  # key = full condition string
            if cond_node not in G:
                G.add_node(
                    cond_node,
                    kind="condition",
                    var=var_name,
                    label=pretty_condition(cond, ndigits=2),
                )
            G.add_edge(cond_node, rule_node, kind="cond_to_rule")

        # outcome edge
        if coef > 0:
            G.add_edge(
                rule_node,
                up_node,
                kind="rule_to_outcome",
                sign="increase",
                weight=abs(coef) * support,
            )
        elif coef < 0:
            G.add_edge(
                rule_node,
                down_node,
                kind="rule_to_outcome",
                sign="decrease",
                weight=abs(coef) * support,
            )

    return G, up_node, down_node


# --------------------------------------------------------------------
# Draw detailed graph (with legend)
# --------------------------------------------------------------------
def draw_detailed_graph(G, up_node, down_node, out_path: Path):
    pos = nx.spring_layout(G, seed=0, k=1.0 / np.sqrt(max(1, len(G.nodes()))))

    # node groups
    cond_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "condition"]
    rule_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "rule"]
    up_nodes = [up_node]
    down_nodes = [down_node]

    plt.figure(figsize=(14, 8))

    # conditions (blue)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=cond_nodes,
        node_color="#1f78b4",
        node_size=400,
        alpha=0.9,
    )

    # rules (orange)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=rule_nodes,
        node_color="#ff8c00",
        node_size=260,
        alpha=0.9,
    )

    # outcomes (green / red)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=up_nodes,
        node_color="#33a02c",
        node_size=900,
        alpha=0.95,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=down_nodes,
        node_color="#e31a1c",
        node_size=900,
        alpha=0.95,
    )

    # edges cond -> rule (light gray)
    cond_edges = [
        (u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "cond_to_rule"
    ]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=cond_edges,
        edge_color="#d3d3d3",
        width=0.8,
        alpha=0.6,
        arrows=False,
    )

    # edges rule -> outcome (sign-dependent)
    rule_edges_inc = []
    rule_edges_dec = []
    for u, v, d in G.edges(data=True):
        if d.get("kind") == "rule_to_outcome":
            if d.get("sign") == "increase":
                rule_edges_inc.append((u, v))
            else:
                rule_edges_dec.append((u, v))

    if rule_edges_inc:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=rule_edges_inc,
            edge_color="#33a02c",
            width=2.0,
            alpha=0.9,
            arrows=True,
            arrowstyle="->",
            arrowsize=12,
        )
    if rule_edges_dec:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=rule_edges_dec,
            edge_color="#e31a1c",
            width=2.0,
            alpha=0.9,
            arrows=True,
            arrowstyle="->",
            arrowsize=12,
        )

    # labels (all nodes, using pretty labels)
    labels = {n: d.get("label", n) for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.title("Knowledge graph from RuleFit rules", fontsize=16)

    # Legend (English explanation of colors)
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Condition (variable + threshold)",
            markerfacecolor="#1f78b4",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Rule node (AND of conditions)",
            markerfacecolor="#ff8c00",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Outcome: Consumption ↑ (increase)",
            markerfacecolor="#33a02c",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Outcome: Consumption ↓ (decrease)",
            markerfacecolor="#e31a1c",
            markersize=10,
        ),
    ]
    plt.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[ok] Detailed graph saved to {out_path}")


# --------------------------------------------------------------------
# Draw "paper" version (cleaner)
# --------------------------------------------------------------------
def draw_paper_graph(G, up_node, down_node, out_path: Path):
    pos = nx.spring_layout(G, seed=0, k=1.0 / np.sqrt(max(1, len(G.nodes()))))

    cond_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "condition"]
    rule_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "rule"]
    up_nodes = [up_node]
    down_nodes = [down_node]

    plt.figure(figsize=(10, 6))

    nx.draw_networkx_nodes(
        G, pos, nodelist=cond_nodes, node_color="#1f78b4", node_size=280, alpha=0.9
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=rule_nodes, node_color="#ff8c00", node_size=180, alpha=0.7
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=up_nodes, node_color="#33a02c", node_size=700, alpha=0.95
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=down_nodes, node_color="#e31a1c", node_size=700, alpha=0.95
    )

    cond_edges = [
        (u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "cond_to_rule"
    ]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=cond_edges,
        edge_color="#d3d3d3",
        width=0.6,
        alpha=0.3,
        arrows=False,
    )

    rule_edges_inc = []
    rule_edges_dec = []
    for u, v, d in G.edges(data=True):
        if d.get("kind") == "rule_to_outcome":
            if d.get("sign") == "increase":
                rule_edges_inc.append((u, v))
            else:
                rule_edges_dec.append((u, v))

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=rule_edges_inc,
        edge_color="#33a02c",
        width=2.0,
        alpha=0.9,
        arrows=True,
        arrowstyle="->",
        arrowsize=10,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=rule_edges_dec,
        edge_color="#e31a1c",
        width=2.0,
        alpha=0.9,
        arrows=True,
        arrowstyle="->",
        arrowsize=10,
    )

    # labels: only conditions + outcomes, using pretty labels
    labels = {}
    for n, d in G.nodes(data=True):
        if d.get("kind") == "condition" or "outcome" in d.get("kind", ""):
            labels[n] = d.get("label", n)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)

    plt.title("Rule-based knowledge graph (clean view)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[ok] Paper-style graph saved to {out_path}")


# --------------------------------------------------------------------
# Aggregated variable-level graphs
# --------------------------------------------------------------------
def build_variable_level_weights(
    df: pd.DataFrame, coef_col: str, support_col: str | None
):
    var_inc = {}
    var_dec = {}

    for _, row in df.iterrows():
        rule_str = str(row["rule_str"])
        coef = float(row[coef_col])
        if math.isnan(coef) or coef == 0.0:
            continue
        support = float(row[support_col]) if support_col is not None else 1.0
        conditions = split_conditions(rule_str)
        vars_in_rule = {get_variable_name(c) for c in conditions}

        for v in vars_in_rule:
            if coef > 0:
                var_inc[v] = var_inc.get(v, 0.0) + coef * support
            elif coef < 0:
                var_dec[v] = var_dec.get(v, 0.0) + abs(coef) * support

    return var_inc, var_dec


def draw_variable_graph(var_inc, var_dec, mode: str, out_path: Path):
    """
    mode: 'both' | 'increase' | 'decrease'
    """
    G = nx.DiGraph()
    # nodes
    for v in sorted(set(var_inc.keys()) | set(var_dec.keys())):
        G.add_node(v, kind="var", label=v)

    if mode in ("both", "increase"):
        up_node = "Consumption ↑"
        G.add_node(up_node, kind="outcome_up", label=up_node)
    else:
        up_node = None

    if mode in ("both", "decrease"):
        down_node = "Consumption ↓"
        G.add_node(down_node, kind="outcome_down", label=down_node)
    else:
        down_node = None

    max_w = 0.0
    for v, w in var_inc.items():
        if up_node is not None and w > 0:
            G.add_edge(v, up_node, sign="increase", weight=w)
            max_w = max(max_w, w)
    for v, w in var_dec.items():
        if down_node is not None and w > 0:
            G.add_edge(v, down_node, sign="decrease", weight=w)
            max_w = max(max_w, w)

    if len(G.nodes) == 0:
        print(f"[warn] No edges to draw for mode={mode}. Skipping.")
        return

    pos = nx.spring_layout(G, seed=1, k=1.0 / np.sqrt(max(1, len(G.nodes()))))

    plt.figure(figsize=(8, 5))

    var_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "var"]
    up_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "outcome_up"]
    down_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "outcome_down"]

    nx.draw_networkx_nodes(
        G, pos, nodelist=var_nodes, node_color="#1f78b4", node_size=500, alpha=0.9
    )
    if up_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=up_nodes, node_color="#33a02c", node_size=900, alpha=0.95
        )
    if down_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=down_nodes, node_color="#e31a1c", node_size=900, alpha=0.95
        )

    # edges
    inc_edges = []
    inc_widths = []
    dec_edges = []
    dec_widths = []

    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1.0)
        width = 1.0 + 6.0 * (w / max_w) if max_w > 0 else 1.0
        if d.get("sign") == "increase":
            inc_edges.append((u, v))
            inc_widths.append(width)
        else:
            dec_edges.append((u, v))
            dec_widths.append(width)

    if inc_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=inc_edges,
            width=inc_widths,
            edge_color="#33a02c",
            alpha=0.9,
            arrows=True,
            arrowstyle="->",
            arrowsize=12,
        )
    if dec_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=dec_edges,
            width=dec_widths,
            edge_color="#e31a1c",
            alpha=0.9,
            arrows=True,
            arrowstyle="->",
            arrowsize=12,
        )

    labels = {n: d.get("label", n) for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    if mode == "both":
        title = "Variable-level knowledge graph (Consumption ↑ / ↓)"
    elif mode == "increase":
        title = "Variable-level knowledge graph (Consumption ↑ only)"
    else:
        title = "Variable-level knowledge graph (Consumption ↓ only)"

    plt.title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[ok] Variable-level graph ({mode}) saved to {out_path}")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Build knowledge graph visualisations from RuleFit rules CSV."
    )
    ap.add_argument("--rules-csv", required=True, help="Path to rules CSV")
    ap.add_argument(
        "--out-prefix",
        default="knowledge_graph_rules",
        help="Prefix for output image files",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=0,
        help="Use only top-N rules (0 = use all, after filtering)",
    )
    ap.add_argument(
        "--min-support",
        type=float,
        default=0.0,
        help="Filter rules by minimum support (if available)",
    )

    args = ap.parse_args()

    path = Path(args.rules_csv)
    if not path.exists():
        raise SystemExit(f"Rules CSV not found: {path}")

    df = pd.read_csv(path)

    rule_col = detect_rule_column(df)
    coef_col = detect_coef_column(df)
    if coef_col is None:
        raise SystemExit("Could not detect coefficient column.")

    support_col = detect_support_column(df)

    df = df.copy()
    df["rule_str"] = df[rule_col].astype(str)

    if support_col is not None and args.min_support > 0:
        df = df[df[support_col] >= args.min_support].copy()

    if args.top and args.top > 0:
        df = df.head(args.top)

    if df.empty:
        raise SystemExit("No rules remaining after filtering.")

    # 1) Detailed rule-level graph
    G, up_node, down_node = build_rule_level_graph(df, coef_col, support_col)
    draw_detailed_graph(G, up_node, down_node, Path(f"{args.out_prefix}_detailed.png"))

    # 2) Paper-style graph
    draw_paper_graph(G, up_node, down_node, Path(f"{args.out_prefix}_paper.png"))

    # 3) Variable-level graphs
    var_inc, var_dec = build_variable_level_weights(df, coef_col, support_col)
    draw_variable_graph(
        var_inc, var_dec, mode="both", out_path=Path(f"{args.out_prefix}_vars_both.png")
    )
    draw_variable_graph(
        var_inc,
        var_dec,
        mode="increase",
        out_path=Path(f"{args.out_prefix}_vars_increase.png"),
    )
    draw_variable_graph(
        var_inc,
        var_dec,
        mode="decrease",
        out_path=Path(f"{args.out_prefix}_vars_decrease.png"),
    )


if __name__ == "__main__":
    main()
