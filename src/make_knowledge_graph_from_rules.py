#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import re
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ==========================================
# CONFIGURAÇÃO DE CAMINHOS
# ==========================================
CSV_PATH = r"C:\Users\pcata\00_Thesis\outputs_knowledge\rules.csv"
OUTPUT_PREFIX = "knowledge_graph_v3_clear"
TOP_RULES = 20  # Recomendo 15 ou 20 para o seu professor conseguir ler tudo bem

# --------------------------------------------------------------------
# Funções de Suporte
# --------------------------------------------------------------------
def detect_column(df, options):
    for opt in options:
        if opt in df.columns: return opt
        for c in df.columns:
            if c.lower() == opt: return c
    return None

def pretty_condition(condition: str, ndigits: int = 2) -> str:
    def repl(m):
        op, num = m.group(1), float(m.group(2))
        return f"{op} {round(num, ndigits)}"
    pattern = r"(<=|>=|<|>)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    return re.sub(pattern, repl, condition)

# --------------------------------------------------------------------
# Construção do Grafo
# --------------------------------------------------------------------
def build_graph(df, coef_col, support_col):
    G = nx.DiGraph()
    up_node, down_node = "Consumption Increase", "Consumption Decrease"
    G.add_node(up_node, kind="outcome_up", label=up_node)
    G.add_node(down_node, kind="outcome_down", label=down_node)

    for idx, row in df.iterrows():
        rule_str = str(row["rule"])
        conditions = [p.strip() for p in rule_str.split("&") if p.strip()]
        coef = float(row[coef_col])
        support = float(row[support_col]) if support_col else 1.0

        if math.isnan(coef) or coef == 0.0: continue

        rule_node = f"R{idx+1}"
        G.add_node(rule_node, kind="rule", label=rule_node)

        for cond in conditions:
            if cond not in G:
                G.add_node(cond, kind="condition", label=pretty_condition(cond))
            G.add_edge(cond, rule_node, kind="link")

        weight_val = abs(coef) * support
        target = up_node if coef > 0 else down_node
        G.add_edge(rule_node, target, kind="outcome", sign="inc" if coef > 0 else "dec", weight=weight_val)

    return G, up_node, down_node

def draw_graph(G, up_node, down_node, filename):
    # k=5.0 é uma força de repulsão muito alta (o normal é 0.1)
    # iterations=100 dá tempo ao algoritmo para encontrar a melhor posição
    pos = nx.spring_layout(G, k=5.0/np.sqrt(len(G.nodes())), seed=42, iterations=100)
    
    plt.figure(figsize=(24, 14)) # Imagem maior para evitar amontoados
    
    conds = [n for n, d in G.nodes(data=True) if d.get("kind") == "condition"]
    rules = [n for n, d in G.nodes(data=True) if d.get("kind") == "rule"]

    # 1. Desenhar nós
    nx.draw_networkx_nodes(G, pos, nodelist=conds, node_color="#1f78b4", node_size=600, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=rules, node_color="#ff8c00", node_size=350, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, nodelist=[up_node], node_color="#33a02c", node_size=2000)
    nx.draw_networkx_nodes(G, pos, nodelist=[down_node], node_color="#e31a1c", node_size=2000)

    # 2. Desenhar Arestas
    edge_weights = [d.get('weight', 0) for u, v, d in G.edges(data=True) if d.get('kind') == 'outcome']
    max_w = max(edge_weights) if edge_weights else 1

    for u, v, d in G.edges(data=True):
        if d.get("kind") == "outcome":
            color = "#33a02c" if d.get("sign") == "inc" else "#e31a1c"
            width = 1.5 + 8.5 * (d.get('weight', 0) / max_w)
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color, 
                                   width=width, alpha=0.5, arrowsize=30)
        else:
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color="#cccccc", 
                                   width=1.0, alpha=0.3, arrows=False)

    # 3. Labels (Nomes) com caixa branca opaca para não misturar com as linhas
    labels = {n: d.get("label", n) for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=2))

    # --- LEGENDA EM INGLÊS ---
    legend_elements = [
        Line2D([0], [0], color='gray', lw=1.5, label='Low Importance'),
        Line2D([0], [0], color='gray', lw=8.5, label='High Importance'),
        Line2D([0], [0], marker='o', color='w', label='Condition', markerfacecolor='#1f78b4', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Rule', markerfacecolor='#ff8c00', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Increase', markerfacecolor='#33a02c', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Decrease', markerfacecolor='#e31a1c', markersize=12),
    ]
    plt.legend(handles=legend_elements, loc='lower left', title="Legend", fontsize=12, frameon=True, borderpad=1.5)

    plt.title("Knowledge Graph: Impact of Climatic Rules on Energy Consumption", fontsize=20, pad=30)
    plt.axis('off')
    
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.show()

# --------------------------------------------------------------------
# Execução
# --------------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print("Ficheiro não encontrado!")
    else:
        df = pd.read_csv(CSV_PATH)
        rule_col = detect_column(df, ["rule", "rules"])
        coef_col = detect_column(df, ["coef", "coefficient"])
        supp_col = detect_column(df, ["support", "supp"])
        
        df["rule_str"] = df[rule_col].astype(str)
        df['abs_coef'] = df[coef_col].abs()
        df_plot = df.sort_values(by='abs_coef', ascending=False).head(TOP_RULES)

        G, up, down = build_graph(df_plot, coef_col, supp_col)
        draw_graph(G, up, down, OUTPUT_PREFIX)