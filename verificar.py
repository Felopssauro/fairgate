"""Compara os resultados recem-gerados em resultados/ com os publicados no
artigo (resultados_publicados/). Metricas e decisoes devem coincidir; tempos
dependem do hardware e so sao exibidos."""
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).parent
NOVO, REF = RAIZ / "resultados", RAIZ / "resultados_publicados"
ok = True

for nome in ["tabela_decisoes.csv", "tabela_metricas.csv"]:
    a, b = pd.read_csv(NOVO / nome), pd.read_csv(REF / nome)
    igual = a.equals(b)
    ok &= igual
    print(f"{nome}: {'IDENTICA' if igual else 'DIFERENTE'}")
    if not igual:
        print(a.compare(b).to_string())

cols = ["semente", "cenario", "perfil", "acuracia", "teste_acuracia_passou",
        "fairgate_aprovou", "n_violacoes"]
a = pd.read_csv(NOVO / "execucoes.csv")
b = pd.read_csv(REF / "execucoes.csv")
metr = [c for c in b.columns if c.startswith(("sex_", "race_bin_"))]
igual = a[cols + metr].equals(b[cols + metr])
ok &= igual
print(f"execucoes.csv (metricas e decisoes das 120 execucoes): {'IDENTICAS' if igual else 'DIFERENTES'}")

print("\nTempos (dependem do hardware, nao sao comparados):")
print(json.loads((NOVO / "tempos_sequenciais.json").read_text()))
sys.exit(0 if ok else 1)
