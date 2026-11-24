"""Executa o experimento completo reproduzindo, localmente, os passos do workflow
.github/workflows/fairness.yml: treino -> teste de acuracia -> FairGate.

4 cenarios x 3 perfis x 10 sementes. Cada passo roda como subprocesso, tal como
no runner de CI, registrando codigo de saida e tempo de parede.
Saida: resultados/execucoes.csv
"""
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from treinar import CENARIOS

RAIZ = Path(__file__).parent
PERFIS = ["paridade", "equalized_odds", "completo"]
SEMENTES = range(10)
PY = sys.executable


def rodar(cmd):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=RAIZ, capture_output=True, text=True)
    return p.returncode, time.perf_counter() - t0


def uma_semente(s):
    linhas = []
    for c in CENARIOS:
        cod, t_treino = rodar([PY, "treinar.py", "--cenario", c, "--semente", str(s)])
        assert cod == 0, f"treino falhou: {c} s{s}"
        modelo, dados = f"modelos/{c}_s{s}.joblib", f"data/teste_s{s}.csv"
        cod_acc, t_acc = rodar([PY, "teste_acuracia.py", "--modelo", modelo, "--dados", dados])
        for perfil in PERFIS:
            rel = f"relatorios/{perfil}/{c}_s{s}.json"
            cod_fg, t_fg = rodar([PY, "fairgate.py", "--modelo", modelo, "--dados", dados,
                                  "--config", f"perfis/{perfil}.yml", "--relatorio", rel])
            r = json.loads((RAIZ / rel).read_text())
            linha = {"semente": s, "cenario": c, "perfil": perfil,
                     "acuracia": r["acuracia"], "teste_acuracia_passou": cod_acc == 0,
                     "fairgate_aprovou": cod_fg == 0, "n_violacoes": len(r["violacoes"]),
                     "t_treino_s": round(t_treino, 3), "t_teste_acuracia_s": round(t_acc, 3),
                     "t_fairgate_s": round(t_fg, 3)}
            for attr, m in r["atributos"].items():
                for k, v in m.items():
                    linha[f"{attr}_{k}"] = v
            linhas.append(linha)
    return linhas


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as ex:
        todas = [l for ls in ex.map(uma_semente, SEMENTES) for l in ls]
    (RAIZ / "resultados").mkdir(exist_ok=True)
    with open(RAIZ / "resultados" / "execucoes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(todas[0].keys()))
        w.writeheader()
        w.writerows(todas)
    print(f"{len(todas)} execucoes registradas")
