"""Génère le schéma visuel du HeteroTransactionGNN — style réseau de neurones."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Chemins ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "docs" / "figures" / "soutenance" / "hetero_gnn_neural_network.png"

# ── Palette ────────────────────────────────────────────────────────────────────
BG       = "#0d1117"
C_ACCT   = "#4fc3f7"   # bleu clair — Account
C_TX     = "#ff7043"   # orange-rouge — Transaction
C_BANK   = "#66bb6a"   # vert — Bank
C_PROJ   = "#90caf9"   # bleu pâle — couches de projection
C_CONV   = "#ffe082"   # jaune doux — HeteroConv
C_SKIP   = "#ce93d8"   # violet — skip connection
C_MLP    = "#80cbc4"   # turquoise — MLP head
C_OUT    = "#f48fb1"   # rose — sortie
C_NORM   = "#b0bec5"   # gris — LayerNorm / Dropout
C_LINE   = "#ffffff"

# ── Helpers ────────────────────────────────────────────────────────────────────
def circle(ax, x, y, r=0.28, color=C_LINE, alpha=1.0, zorder=4, lw=1.5, fill=None):
    fc = fill if fill else color
    c = plt.Circle((x, y), r, color=fc, alpha=alpha, zorder=zorder,
                   linewidth=lw, edgecolor=color)
    ax.add_patch(c)
    return c


def label(ax, x, y, text, fs=8.5, color=C_LINE, ha="center", va="center",
          bold=False, zorder=6, alpha=1.0):
    w = "bold" if bold else "normal"
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va=va,
            fontweight=w, zorder=zorder, alpha=alpha)


def arrow(ax, x0, y0, x1, y1, color=C_LINE, alpha=0.25, lw=0.7, zorder=3):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-", color=color, lw=lw, alpha=alpha),
                zorder=zorder)


def thick_arrow(ax, x0, y0, x1, y1, color=C_LINE, alpha=0.7, lw=1.5, zorder=4,
                label_txt="", label_color=C_LINE, label_fs=7.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>",
                                color=color, lw=lw, alpha=alpha,
                                mutation_scale=10),
                zorder=zorder)
    if label_txt:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + 0.18
        label(ax, mx, my, label_txt, fs=label_fs, color=label_color, alpha=alpha + 0.2)


def box(ax, x, y, w, h, color, alpha=0.12, lw=1.5, zorder=2, label_txt="",
        label_color=None, label_fs=8, label_bold=False):
    rect = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                          boxstyle="round,pad=0.05",
                          edgecolor=color, facecolor=color,
                          alpha=alpha, linewidth=lw, zorder=zorder)
    ax.add_patch(rect)
    if label_txt:
        lc = label_color or color
        label(ax, x, y, label_txt, fs=label_fs, color=lc, bold=label_bold, zorder=zorder + 1)


def layer_header(ax, x, y_top, title, subtitle="", color=C_LINE):
    label(ax, x, y_top + 0.55, title, fs=8.5, color=color, bold=True)
    if subtitle:
        label(ax, x, y_top + 0.25, subtitle, fs=7.2, color=color, alpha=0.75)


def draw_neurons(ax, x, ys, color, r=0.24, alpha_fill=0.85, n_visible=None,
                 total=None, label_inside=None):
    """Dessine une colonne de neurones, avec '…' si trop nombreux."""
    shown = ys[:n_visible] if n_visible else ys
    for i, y in enumerate(shown):
        fill = color
        c = plt.Circle((x, y), r, facecolor=fill, edgecolor=C_LINE,
                        linewidth=1.2, alpha=alpha_fill, zorder=5)
        ax.add_patch(c)
        if label_inside and i == 0:
            label(ax, x, y, label_inside, fs=7, color=BG, bold=True, zorder=6)
    if total and total > len(shown):
        ax.text(x, shown[-1] - 0.55, f"⋮\n({total})", fontsize=7.5,
                color=color, ha="center", va="top", alpha=0.8, zorder=5)


# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(24, 14))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 24)
ax.set_ylim(0, 14)
ax.axis("off")

# ── Titre ──────────────────────────────────────────────────────────────────────
ax.text(12, 13.5, "HeteroTransactionGNN — Architecture du réseau de neurones",
        fontsize=14, color=C_LINE, ha="center", va="top", fontweight="bold",
        zorder=10)
ax.text(12, 13.0,
        "Account [5] · Transaction [27] · Bank [2]  →  Projection [64]  →  2× HeteroConv  →  Skip  →  MLP  →  Score fraude [1]",
        fontsize=8.5, color=C_LINE, ha="center", va="top", alpha=0.65, zorder=10)

# ════════════════════════════════════════════════════════════════════
# COLONNE 1 — Entrées (x ≈ 1.5)
# ════════════════════════════════════════════════════════════════════
X_IN = 1.7

# Account (5 features)
acct_ys = np.linspace(10.8, 9.0, 5)
box(ax, X_IN, 9.9, 1.4, 2.4, C_ACCT, alpha=0.08, lw=1)
draw_neurons(ax, X_IN, acct_ys, C_ACCT)
layer_header(ax, X_IN, 11.1, "Account", "[5 features]", C_ACCT)
feat_acct = ["bias", "log_out_deg", "log_in_deg", "log_tw", "log_sb"]
for y, nm in zip(acct_ys, feat_acct):
    label(ax, X_IN + 0.95, y, nm, fs=6.5, color=C_ACCT, ha="left", alpha=0.75)

# Transaction (27 features — on en montre 8 + ...)
tx_full_ys = np.linspace(7.6, 4.0, 8)
box(ax, X_IN, 5.8, 1.4, 4.2, C_TX, alpha=0.08, lw=1)
draw_neurons(ax, X_IN, tx_full_ys, C_TX, n_visible=8, total=27)
layer_header(ax, X_IN, 7.9, "Transaction", "[27 features]", C_TX)
tx_labels = ["amount_z", "USD", "EUR", "Yuan", "BTC", "Wire", "Cash", "ts_1 …"]
for y, nm in zip(tx_full_ys, tx_labels):
    label(ax, X_IN + 0.95, y, nm, fs=6.5, color=C_TX, ha="left", alpha=0.75)

# Bank (2 features)
bank_ys = [2.4, 1.6]
box(ax, X_IN, 2.0, 1.4, 1.3, C_BANK, alpha=0.08, lw=1)
draw_neurons(ax, X_IN, bank_ys, C_BANK)
layer_header(ax, X_IN, 2.8, "Bank", "[2 features]", C_BANK)
for y, nm in zip(bank_ys, ["bankCode_", "bankCode_Other"]):
    label(ax, X_IN + 0.95, y, nm, fs=6.5, color=C_BANK, ha="left", alpha=0.75)

# ════════════════════════════════════════════════════════════════════
# COLONNE 2 — Projection Linéaire (x ≈ 5)
# ════════════════════════════════════════════════════════════════════
X_PROJ = 5.5

proj_acct_ys = np.linspace(10.8, 9.0, 6)
proj_tx_ys   = np.linspace(7.5, 4.2, 6)
proj_bank_ys = [2.4, 1.6]

box(ax, X_PROJ, 9.9, 1.4, 2.4, C_ACCT, alpha=0.08, lw=1)
draw_neurons(ax, X_PROJ, proj_acct_ys, C_PROJ, n_visible=6, total=64)
layer_header(ax, X_PROJ, 11.1, "Linear(5→64)", "+ ReLU", C_ACCT)

box(ax, X_PROJ, 5.85, 1.4, 4.2, C_TX, alpha=0.08, lw=1)
draw_neurons(ax, X_PROJ, proj_tx_ys, C_PROJ, n_visible=6, total=64)
layer_header(ax, X_PROJ, 7.9, "Linear(27→64)", "+ ReLU", C_TX)

box(ax, X_PROJ, 2.0, 1.4, 1.3, C_BANK, alpha=0.08, lw=1)
draw_neurons(ax, X_PROJ, proj_bank_ys, C_PROJ, n_visible=2, total=64)
layer_header(ax, X_PROJ, 2.8, "Linear(2→64)", "+ ReLU", C_BANK)

# Annotations poids
for y in [9.9, 5.85, 2.0]:
    label(ax, (X_IN + X_PROJ) / 2, y, "W  ·  x  +  b", fs=6.5,
          color="#aaaaaa", alpha=0.8)

# Connexions entrée → projection (quelques lignes représentatives)
for ay in acct_ys:
    for py in proj_acct_ys:
        arrow(ax, X_IN + 0.26, ay, X_PROJ - 0.26, py, color=C_ACCT, alpha=0.08)
for ay in tx_full_ys:
    for py in proj_tx_ys:
        arrow(ax, X_IN + 0.26, ay, X_PROJ - 0.26, py, color=C_TX, alpha=0.07)
for ay in bank_ys:
    for py in proj_bank_ys:
        arrow(ax, X_IN + 0.26, ay, X_PROJ - 0.26, py, color=C_BANK, alpha=0.15)

# ════════════════════════════════════════════════════════════════════
# COLONNE 3 — HeteroConv Couche 1 (x ≈ 9)
# ════════════════════════════════════════════════════════════════════
X_HC1 = 9.5

hc1_acct_ys = np.linspace(10.8, 9.0, 6)
hc1_tx_ys   = np.linspace(7.5, 4.2, 6)
hc1_bank_ys = [2.4, 1.6]

box(ax, X_HC1, 9.9, 1.4, 2.4, C_ACCT, alpha=0.06, lw=1)
draw_neurons(ax, X_HC1, hc1_acct_ys, C_CONV, n_visible=6, total=64)
layer_header(ax, X_HC1, 11.1, "HeteroConv — L1", "Account [64]", C_CONV)

box(ax, X_HC1, 5.85, 1.4, 4.2, C_TX, alpha=0.06, lw=1)
draw_neurons(ax, X_HC1, hc1_tx_ys, C_CONV, n_visible=6, total=64)
layer_header(ax, X_HC1, 7.9, "HeteroConv — L1", "Transaction [64]", C_CONV)

box(ax, X_HC1, 2.0, 1.4, 1.3, C_BANK, alpha=0.06, lw=1)
draw_neurons(ax, X_HC1, hc1_bank_ys, C_CONV, n_visible=2, total=64)
layer_header(ax, X_HC1, 2.8, "HeteroConv — L1", "Bank [64]", C_CONV)

# Relations RDF annotées entre proj et HC1
relations = [
    ("sends_via", 10.2, C_ACCT,  "Account→Tx"),
    ("deposits_to", 7.5, C_TX,   "Tx→Account"),
    ("hosted_by", 5.2, C_BANK,   "Account→Bank"),
    ("transacted_with", 3.5, C_ACCT, "Acct↔Acct ★"),
    ("same_bank_as", 1.2, C_BANK,    "Acct↔Acct ★"),
]
for rel_name, y_mid, col, desc in relations:
    x0, x1 = X_PROJ + 0.26, X_HC1 - 0.26
    ax.annotate("", xy=(x1, y_mid), xytext=(x0, y_mid),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2,
                                alpha=0.55, mutation_scale=8), zorder=4)
    label(ax, (x0 + x1) / 2, y_mid + 0.25,
          f"SAGEConv_{rel_name[:6]}…", fs=6, color=col, alpha=0.85)
    label(ax, (x0 + x1) / 2, y_mid - 0.18,
          f"W_{rel_name[:3]} · msg", fs=5.8, color="#aaaaaa", alpha=0.7)

# LayerNorm + Dropout label
box(ax, X_HC1, 12.65, 3.0, 0.5, C_NORM, alpha=0.15, lw=1,
    label_txt="LayerNorm + ReLU + Dropout(0.3)",
    label_color=C_NORM, label_fs=7)

# ════════════════════════════════════════════════════════════════════
# COLONNE 4 — HeteroConv Couche 2 (x ≈ 13)
# ════════════════════════════════════════════════════════════════════
X_HC2 = 13.5

hc2_acct_ys = np.linspace(10.8, 9.0, 6)
hc2_tx_ys   = np.linspace(7.5, 4.2, 6)
hc2_bank_ys = [2.4, 1.6]

box(ax, X_HC2, 9.9, 1.4, 2.4, C_ACCT, alpha=0.06, lw=1)
draw_neurons(ax, X_HC2, hc2_acct_ys, C_CONV, n_visible=6, total=64)
layer_header(ax, X_HC2, 11.1, "HeteroConv — L2", "Account [64]", C_CONV)

box(ax, X_HC2, 5.85, 1.4, 4.2, C_TX, alpha=0.06, lw=1)
draw_neurons(ax, X_HC2, hc2_tx_ys, C_CONV, n_visible=6, total=64)
layer_header(ax, X_HC2, 7.9, "HeteroConv — L2", "Transaction [64]", C_CONV)

box(ax, X_HC2, 2.0, 1.4, 1.3, C_BANK, alpha=0.06, lw=1)
draw_neurons(ax, X_HC2, hc2_bank_ys, C_CONV, n_visible=2, total=64)
layer_header(ax, X_HC2, 2.8, "HeteroConv — L2", "Bank [64]", C_CONV)

# Connexions HC1 → HC2
for ay in hc1_acct_ys:
    for by in hc2_acct_ys:
        arrow(ax, X_HC1 + 0.26, ay, X_HC2 - 0.26, by, color=C_CONV, alpha=0.06)
for ay in hc1_tx_ys:
    for by in hc2_tx_ys:
        arrow(ax, X_HC1 + 0.26, ay, X_HC2 - 0.26, by, color=C_CONV, alpha=0.05)
for ay in hc1_bank_ys:
    for by in hc2_bank_ys:
        arrow(ax, X_HC1 + 0.26, ay, X_HC2 - 0.26, by, color=C_CONV, alpha=0.12)

for rel_name, y_mid, col, desc in relations:
    x0, x1 = X_HC1 + 0.26, X_HC2 - 0.26
    ax.annotate("", xy=(x1, y_mid), xytext=(x0, y_mid),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2,
                                alpha=0.45, mutation_scale=8), zorder=4)
    label(ax, (x0 + x1) / 2, y_mid + 0.25,
          f"SAGEConv_{rel_name[:6]}…", fs=6, color=col, alpha=0.75)

box(ax, X_HC2, 12.65, 3.0, 0.5, C_NORM, alpha=0.15, lw=1,
    label_txt="LayerNorm + ReLU + Dropout(0.3)",
    label_color=C_NORM, label_fs=7)

# ════════════════════════════════════════════════════════════════════
# COLONNE 5 — Skip Connection (x ≈ 16.5)
# ════════════════════════════════════════════════════════════════════
X_SK = 17.0

# On n'utilise QUE les noeuds Transaction pour la suite
tx_emb_ys = np.linspace(8.5, 4.5, 8)
raw_tx_ys  = np.linspace(3.6, 1.2, 6)

box(ax, X_SK, 6.5, 1.5, 5.2, C_TX, alpha=0.06, lw=1)
draw_neurons(ax, X_SK, tx_emb_ys, C_CONV, n_visible=8, total=64)
label(ax, X_SK, 9.2, "emb_GNN [64]", fs=8, color=C_CONV, bold=True)
label(ax, X_SK, 8.85, "(sortie L2 transaction)", fs=6.8, color=C_CONV, alpha=0.7)

draw_neurons(ax, X_SK, raw_tx_ys, C_TX, n_visible=6, total=27)
label(ax, X_SK, 4.1, "x_raw_tx [27]", fs=8, color=C_TX, bold=True)
label(ax, X_SK, 3.75, "(features brutes — skip)", fs=6.8, color=C_TX, alpha=0.7)

# Flèche skip depuis input transaction
ax.annotate("", xy=(X_SK - 0.26, 2.4), xytext=(X_IN + 0.26, 5.5),
            arrowprops=dict(arrowstyle="-|>", color=C_TX, lw=1.3,
                            alpha=0.4, connectionstyle="arc3,rad=-0.3",
                            mutation_scale=9), zorder=4)
label(ax, (X_IN + X_SK) / 2 - 0.5, 1.0,
      "skip connection\n(features brutes préservées)", fs=7, color=C_TX, alpha=0.8)

# Flèche emb GNN depuis HC2
for y in hc2_tx_ys[:3]:
    arrow(ax, X_HC2 + 0.26, y, X_SK - 0.26, tx_emb_ys[0] + 0.5, color=C_CONV, alpha=0.18)

# concat block
box(ax, X_SK, 0.65, 1.8, 0.55, C_SKIP, alpha=0.2, lw=1.5,
    label_txt="concat [ 64 || 27 ] = 91", label_color=C_SKIP, label_fs=7.5, label_bold=True)

# ════════════════════════════════════════════════════════════════════
# COLONNE 6 — MLP Head (x ≈ 20)
# ════════════════════════════════════════════════════════════════════
X_MLP = 20.0

# Positions des sub-couches MLP, bien espacées verticalement
mlp_91_ys = np.linspace(10.5, 4.5, 9)
mlp_64_ys = np.linspace(10.0, 5.0, 7)
mlp_32_ys = np.linspace(9.5, 5.5, 5)
mlp_1_ys  = [7.5]

X91 = X_MLP - 0.9
X64 = X_MLP + 0.2
X32 = X_MLP + 1.2
X1  = X_MLP + 2.3

# ── [91] entrée MLP ────────────────────────────────────────────────
draw_neurons(ax, X91, mlp_91_ys, C_SKIP, n_visible=9, total=91, r=0.19)
label(ax, X91, 11.3, "concat [91]", fs=8, color=C_SKIP, bold=True)
label(ax, X91, 11.0, "emb(64) + raw(27)", fs=6.5, color=C_SKIP, alpha=0.75)

# ── [64] couche 1 ──────────────────────────────────────────────────
draw_neurons(ax, X64, mlp_64_ys, C_MLP, n_visible=7, total=64, r=0.19)
label(ax, X64, 11.3, "Linear(91→64)", fs=7.5, color=C_MLP, bold=True)
label(ax, X64, 11.0, "+ ReLU + Dropout", fs=6.5, color="#aaaaaa", alpha=0.8)

# ── [32] couche 2 ──────────────────────────────────────────────────
draw_neurons(ax, X32, mlp_32_ys, C_MLP, n_visible=5, total=32, r=0.19)
label(ax, X32, 11.3, "Linear(64→32)", fs=7.5, color=C_MLP, bold=True)
label(ax, X32, 11.0, "+ ReLU + Dropout", fs=6.5, color="#aaaaaa", alpha=0.8)

# ── Connexions 91 → 64 ─────────────────────────────────────────────
for y0 in mlp_91_ys:
    for y1 in mlp_64_ys:
        arrow(ax, X91 + 0.21, y0, X64 - 0.21, y1, color=C_MLP, alpha=0.05)

# ── Connexions 64 → 32 ─────────────────────────────────────────────
for y0 in mlp_64_ys:
    for y1 in mlp_32_ys:
        arrow(ax, X64 + 0.21, y0, X32 - 0.21, y1, color=C_MLP, alpha=0.08)

# ── [1] neurone de sortie ──────────────────────────────────────────
draw_neurons(ax, X1, mlp_1_ys, C_OUT, r=0.32)
label(ax, X1, 11.3, "Linear(32→1)", fs=7.5, color=C_OUT, bold=True)
label(ax, X1, 11.0, "sigmoid(logit)", fs=6.5, color=C_OUT, alpha=0.85)

for y0 in mlp_32_ys:
    arrow(ax, X32 + 0.21, y0, X1 - 0.34, mlp_1_ys[0], color=C_OUT, alpha=0.2)

# Annotation résultat
label(ax, X1, 8.8,  "Score fraude", fs=8.5, color=C_OUT, bold=True)
label(ax, X1, 8.35, "∈ [0, 1]",     fs=7.5, color=C_OUT, alpha=0.85)
label(ax, X1, 7.0,  "> 0.5  →  ALERTE",  fs=7, color="#ff7043")
label(ax, X1, 6.55, "<= 0.5 →  Légitime", fs=7, color="#66bb6a")

# ════════════════════════════════════════════════════════════════════
# LÉGENDE EN BAS
# ════════════════════════════════════════════════════════════════════
legend_items = [
    (C_ACCT,  "Account (nœud)"),
    (C_TX,    "Transaction (nœud)"),
    (C_BANK,  "Bank (nœud)"),
    (C_PROJ,  "Espace commun [64 dim]"),
    (C_CONV,  "HeteroConv (SAGEConv × relation)"),
    (C_SKIP,  "Skip connection [91 dim]"),
    (C_MLP,   "MLP head"),
    (C_OUT,   "Score fraude"),
]
x0_leg, y0_leg = 1.0, 0.02
for i, (col, txt) in enumerate(legend_items):
    xi = x0_leg + i * 2.9
    c = plt.Circle((xi, y0_leg + 0.18), 0.14, facecolor=col, edgecolor=C_LINE,
                   linewidth=0.8, alpha=0.9, zorder=6, transform=ax.transData)
    ax.add_patch(c)
    label(ax, xi + 0.28, y0_leg + 0.18, txt, fs=6.8, color=C_LINE, ha="left")

# ════════════════════════════════════════════════════════════════════
# SÉPARATEURS VERTICAUX de couches
# ════════════════════════════════════════════════════════════════════
for xv, txt in [
    (3.4,  "Projection\nlinéaire"),
    (7.2,  "HeteroConv\nCouche 1"),
    (11.2, "HeteroConv\nCouche 2"),
    (15.3, "Skip\nConnection"),
    (18.7, "MLP\nHead"),
]:
    ax.plot([xv, xv], [0.5, 12.3], color=C_LINE, alpha=0.12, lw=0.8, ls="--", zorder=1)
    label(ax, xv, 12.5, txt, fs=7, color=C_LINE, alpha=0.55)

# ════════════════════════════════════════════════════════════════════
# NOTE INFÉRÉES
# ════════════════════════════════════════════════════════════════════
box(ax, 7.2, 0.4, 2.6, 0.55, "#ffe082", alpha=0.12, lw=1,
    label_txt="★ Relations inférées par OWL (transacted_with, same_bank_as)",
    label_color="#ffe082", label_fs=7)

# ════════════════════════════════════════════════════════════════════
# Sauvegarde
# ════════════════════════════════════════════════════════════════════
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"OK {OUTPUT_PATH}")


if __name__ == "__main__":
    pass
