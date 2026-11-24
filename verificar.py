"""Compara os resultados recem-gerados em resultados/ com os publicados no
artigo (resultados_publicados/).

- Decisoes (teste de acuracia e FairGate) devem ser identicas: e o criterio de
  sucesso da reproducao.
- Metricas: identicas no mesmo hardware; em outro processador os modelos
  mitigados podem variar levemente, entao so a maior diferenca e informada.
- Tempos dependem do hardware e so sao exibidos.
"""
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).parent
NOVO, REF = RAIZ / "resultados", RAIZ / "resultados_publicados"
CHAVE = ["semente", "cenario", "perfil"]
DECISOES = ["teste_acuracia_passou", "fairgate_aprovou"]

a = pd.read_csv(NOVO / "execucoes.csv").sort_values(CHAVE).reset_index(drop=True)
b = pd.read_csv(REF / "execucoes.csv").sort_values(CHAVE).reset_index(drop=True)
assert len(a) == len(b) == 120, f"esperadas 120 execucoes, obtidas {len(a)}"
assert a[CHAVE].equals(b[CHAVE]), "conjunto de execucoes diferente"

dec_ok = a[DECISOES].equals(b[DECISOES])
print(f"Decisoes das 120 execucoes: {'IDENTICAS' if dec_ok else 'DIFERENTES'}")
if not dec_ok:
    m = (a[DECISOES] != b[DECISOES]).any(axis=1)
    print(pd.concat([a.loc[m, CHAVE + DECISOES], b.loc[m, DECISOES].add_suffix("_publicado")], axis=1).to_string())

metr = ["acuracia"] + [c for c in b.columns if c.startswith(("sex_", "race_bin_"))]
dif = (a[metr] - b[metr]).abs()
if dif.to_numpy().max() == 0:
    print("Metricas das 120 execucoes: IDENTICAS")
else:
    print(f"Metricas: maior diferenca absoluta = {dif.to_numpy().max():.4f}")
    print(dif.assign(cenario=a.cenario).groupby("cenario").max().round(4).to_string())

d1, d2 = pd.read_csv(NOVO / "tabela_decisoes.csv"), pd.read_csv(REF / "tabela_decisoes.csv")
print(f"tabela_decisoes.csv: {'IDENTICA' if d1.equals(d2) else 'DIFERENTE'}")
dec_ok &= d1.equals(d2)

print("\nTempos sequenciais (dependem do hardware, nao sao comparados):")
print(json.loads((NOVO / "tempos_sequenciais.json").read_text()))
sys.exit(0 if dec_ok else 1)
