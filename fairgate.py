"""FairGate: portao de qualidade de fairness para esteiras de CI/CD.

Carrega um modelo treinado e um conjunto de teste rotulado, calcula metricas de
justica algoritmica por atributo sensivel e retorna codigo de saida 1 (build
reprovado) quando algum limiar configurado e violado.

Uso: python fairgate.py --modelo modelos/base_s0.joblib --dados data/teste_s0.csv \
                        --config perfis/completo.yml --relatorio relatorios/base_s0.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import yaml
from fairlearn.metrics import (demographic_parity_difference,
                               demographic_parity_ratio,
                               equalized_odds_difference)
from sklearn.metrics import accuracy_score

import wrappers  # noqa: F401  (necessario para desserializar modelos mitigados)


def auditar(modelo, dados, config):
    y = dados["y"].to_numpy()
    X = dados.drop(columns=["y", "race_bin"])
    pred = modelo.predict(X)
    lim = config["limiares"]
    resultado = {"acuracia": round(float(accuracy_score(y, pred)), 4),
                 "atributos": {}, "violacoes": []}
    for attr in config["atributos_sensiveis"]:
        s = dados[attr]
        m = {
            "diferenca_paridade_demografica": demographic_parity_difference(y, pred, sensitive_features=s),
            "razao_impacto_desigual": demographic_parity_ratio(y, pred, sensitive_features=s),
            "diferenca_equalized_odds": equalized_odds_difference(y, pred, sensitive_features=s),
        }
        m = {k: round(float(v), 4) for k, v in m.items()}
        resultado["atributos"][attr] = m
        regras = [("diferenca_paridade_demografica", ">", "paridade demografica"),
                  ("razao_impacto_desigual", "<", "impacto desigual"),
                  ("diferenca_equalized_odds", ">", "equalized odds")]
        for chave, op, nome in regras:
            if chave not in lim:
                continue  # metrica nao exigida neste perfil
            viola = m[chave] > lim[chave] if op == ">" else m[chave] < lim[chave]
            if viola:
                resultado["violacoes"].append(f"{attr}: {nome} {m[chave]} {op} {lim[chave]}")
    resultado["aprovado"] = not resultado["violacoes"]
    return resultado


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--dados", required=True)
    ap.add_argument("--config", default="perfis/completo.yml")
    ap.add_argument("--relatorio")
    a = ap.parse_args()
    t0 = time.perf_counter()
    config = yaml.safe_load(Path(a.config).read_text())
    r = auditar(joblib.load(a.modelo), pd.read_csv(a.dados), config)
    r["tempo_auditoria_s"] = round(time.perf_counter() - t0, 3)
    if a.relatorio:
        Path(a.relatorio).parent.mkdir(parents=True, exist_ok=True)
        Path(a.relatorio).write_text(json.dumps(r, indent=2, ensure_ascii=False))
    print(f"FairGate | acuracia={r['acuracia']}")
    for attr, m in r["atributos"].items():
        print(f"  {attr}: " + ", ".join(f"{k}={v}" for k, v in m.items()))
    if r["aprovado"]:
        print("RESULTADO: APROVADO - deploy liberado")
        sys.exit(0)
    print("RESULTADO: REPROVADO - deploy bloqueado")
    for v in r["violacoes"]:
        print(f"  ::error::{v}")
    sys.exit(1)


if __name__ == "__main__":
    main()
