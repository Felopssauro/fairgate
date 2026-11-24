"""Mede a sobrecarga do FairGate em execucao SEQUENCIAL (sem paralelismo).

O experimento.py roda 4 sementes em paralelo, o que infla os tempos de parede
(ver resultados/tempos.json). Para a sobrecarga reportada no artigo, cada
modelo da semente 0 e auditado 5 vezes, uma de cada vez, pelo teste de
acuracia e pelo FairGate (perfil completo), como subprocessos, tal como no
runner de CI. Requer os modelos da semente 0 (gerados pelo experimento.py ou
por `python treinar.py --cenario <c> --semente 0`).

Saida: resultados/tempos_sequenciais.json no formato
  {cenario: {"fairgate": [media_s, dp_s], "acuracia": [media_s, dp_s]}}
"""
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from treinar import CENARIOS

RAIZ = Path(__file__).parent
REPETICOES = 5
PY = sys.executable


def cronometrar(cmd):
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=RAIZ, capture_output=True, text=True)
    return time.perf_counter() - t0


if __name__ == "__main__":
    saida = {}
    for c in CENARIOS:
        modelo, dados = f"modelos/{c}_s0.joblib", "data/teste_s0.csv"
        if not (RAIZ / modelo).exists():
            subprocess.run([PY, "treinar.py", "--cenario", c, "--semente", "0"], cwd=RAIZ, check=True)
        fg, acc = [], []
        for _ in range(REPETICOES):
            acc.append(cronometrar([PY, "teste_acuracia.py", "--modelo", modelo, "--dados", dados]))
            fg.append(cronometrar([PY, "fairgate.py", "--modelo", modelo, "--dados", dados,
                                   "--config", "perfis/completo.yml"]))
        saida[c] = {"fairgate": [round(statistics.mean(fg), 2), round(statistics.stdev(fg), 2)],
                    "acuracia": [round(statistics.mean(acc), 2), round(statistics.stdev(acc), 2)]}
        print(c, saida[c])
    (RAIZ / "resultados").mkdir(exist_ok=True)
    (RAIZ / "resultados" / "tempos_sequenciais.json").write_text(json.dumps(saida, indent=2))
