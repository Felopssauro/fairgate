"""Agrega resultados/execucoes.csv em tabelas e figuras usadas no artigo."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).parent
OUT = RAIZ / "resultados"
df = pd.read_csv(OUT / "execucoes.csv")
CEN = ["base", "enviesado", "mitigado_dp", "mitigado_eo"]
ROT = {"base": "Base", "enviesado": "Enviesado", "mitigado_dp": "Mitigado-DP",
       "mitigado_eo": "Mitigado-EO"}
METR = ["diferenca_paridade_demografica", "razao_impacto_desigual", "diferenca_equalized_odds"]

# Tabela de metricas (uma linha por cenario; metricas nao dependem do perfil)
m = df[df.perfil == "completo"]
tab = []
for c in CEN:
    s = m[m.cenario == c]
    linha = {"cenario": ROT[c], "acuracia": f"{s.acuracia.mean():.3f} ± {s.acuracia.std():.3f}",
             "teste_acuracia_passou": f"{int(s.teste_acuracia_passou.sum())}/{len(s)}"}
    for attr in ["sex", "race_bin"]:
        for k in METR:
            col = s[f"{attr}_{k}"]
            linha[f"{attr}_{k}"] = f"{col.mean():.3f} ± {col.std():.3f}"
    tab.append(linha)
pd.DataFrame(tab).to_csv(OUT / "tabela_metricas.csv", index=False)

# Tabela de decisoes do portao: aprovacoes por cenario x perfil
dec = (df.groupby(["cenario", "perfil"]).fairgate_aprovou.agg(["sum", "count"])
       .reset_index())
dec.to_csv(OUT / "tabela_decisoes.csv", index=False)

# Tempos
tempos = {
    "fairgate_media_s": round(df.t_fairgate_s.mean(), 3),
    "fairgate_dp_s": round(df.t_fairgate_s.std(), 3),
    "teste_acuracia_media_s": round(df.drop_duplicates(["semente", "cenario"]).t_teste_acuracia_s.mean(), 3),
    "treino_media_s": df.drop_duplicates(["semente", "cenario"]).groupby("cenario").t_treino_s.mean().round(2).to_dict(),
}
(OUT / "tempos.json").write_text(json.dumps(tempos, indent=2))
print(pd.DataFrame(tab).to_string())
print(dec.to_string())
print(tempos)

# Figura 2: metricas por sexo, media e desvio-padrao, com limiares
plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
tons = ["0.75", "0.15", "0.45", "0.6"]
hatch = ["", "", "//", ".."]
info = [("diferenca_paridade_demografica", "Dif. paridade demográfica", 0.10, "máx."),
        ("razao_impacto_desigual", "Razão de impacto desigual", 0.80, "mín."),
        ("diferenca_equalized_odds", "Dif. equalized odds", 0.10, "máx.")]
for ax, (k, titulo, lim, tipo) in zip(axs, info):
    med = [m[m.cenario == c][f"sex_{k}"].mean() for c in CEN]
    dp = [m[m.cenario == c][f"sex_{k}"].std() for c in CEN]
    ax.bar(range(4), med, yerr=dp, color=tons, edgecolor="black", hatch=hatch, capsize=3, linewidth=0.6)
    ax.axhline(lim, ls="--", color="black", lw=0.9)
    ax.set_xticks(range(4), [ROT[c] for c in CEN], rotation=30, ha="right", fontsize=7.5)
    ax.set_title(titulo + "\n" + f"(limiar {tipo} " + f"{lim:.2f}".replace(".", ",") + ")", fontsize=8.5)
    ax.set_ylim(0, 1.05 if k == "razao_impacto_desigual" else max(0.5, max(med) * 1.25))
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig_metricas_sexo.png", dpi=300)

# Figura 1: arquitetura da esteira com o FairGate
fig, ax = plt.subplots(figsize=(7.2, 2.3))
ax.set_xlim(0, 100), ax.set_ylim(0, 30), ax.axis("off")
caixas = [(1, "Push / Pull\nrequest"), (19, "Treino do\nmodelo"), (37, "Teste de\nacurácia"),
          (55, "FairGate\n(perfil .yml)"), (79, "Deploy")]
for x, t in caixas:
    destaque = "FairGate" in t
    ax.add_patch(FancyBboxPatch((x, 14), 15 if x != 79 else 14, 10, boxstyle="round,pad=0.4",
                                fc="0.85" if destaque else "white", ec="black", lw=1.4 if destaque else 0.8))
    ax.text(x + (7.5 if x != 79 else 7), 19, t, ha="center", va="center", fontsize=8,
            weight="bold" if destaque else "normal")
for x0, x1 in [(16.4, 18.6), (34.4, 36.6), (52.4, 54.6)]:
    ax.annotate("", (x1, 19), (x0, 19), arrowprops=dict(arrowstyle="->", lw=0.8))
ax.annotate("", (78.6, 19), (70.4, 19), arrowprops=dict(arrowstyle="->", lw=0.8))
ax.text(74.5, 21.2, "exit 0", ha="center", fontsize=7)
ax.add_patch(FancyBboxPatch((55, 1), 15, 7, boxstyle="round,pad=0.4", fc="white", ec="black", lw=0.8, ls="--"))
ax.text(62.5, 4.5, "Build reprovado\n+ relatório JSON", ha="center", va="center", fontsize=7.5)
ax.annotate("", (62.5, 8.6), (62.5, 13.6), arrowprops=dict(arrowstyle="->", lw=0.8))
ax.text(63.5, 11, "exit 1", fontsize=7)
ax.text(44.5, 27.5, "GitHub Actions: job por cenário (matriz)", ha="center", fontsize=7.5, style="italic")
fig.tight_layout()
fig.savefig(OUT / "fig_arquitetura.png", dpi=300)
