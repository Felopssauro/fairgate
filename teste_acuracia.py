"""Teste de qualidade tradicional (linha de base): so verifica acuracia minima.
Representa a garantia de qualidade usual, que ignora fairness."""
import argparse
import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

import wrappers  # noqa: F401

ap = argparse.ArgumentParser()
ap.add_argument("--modelo", required=True)
ap.add_argument("--dados", required=True)
ap.add_argument("--minimo", type=float, default=0.80)
a = ap.parse_args()
d = pd.read_csv(a.dados)
acc = accuracy_score(d["y"], joblib.load(a.modelo).predict(d.drop(columns=["y", "race_bin"])))
print(f"Teste de acuracia: {acc:.4f} (minimo {a.minimo})")
sys.exit(0 if acc >= a.minimo else 1)
