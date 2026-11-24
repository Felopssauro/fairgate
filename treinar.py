"""Treina os modelos dos quatro cenarios do experimento FairGate.

Cenarios:
  base      - regressao logistica treinada nos dados originais.
  enviesado - mesmo modelo, mas com vies historico injetado nos rotulos de
              treino (50% dos rotulos positivos de mulheres trocados para negativos).
  mitigado_dp - regressao logistica com a reducao ExponentiatedGradient
              (Agarwal et al., 2018) sob restricao de paridade demografica nos
              grupos interseccionais sexo x raca.
  mitigado_eo - idem, sob restricao de equalized odds (Hardt et al., 2016).

Uso: python treinar.py --cenario base --semente 0
Saidas: modelos/<cenario>_s<semente>.joblib e data/teste_s<semente>.csv
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from wrappers import ModeloMitigado

RAIZ = Path(__file__).parent
COLUNAS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "sex",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income"]
CENARIOS = ["base", "enviesado", "mitigado_dp", "mitigado_eo"]
NUMERICAS = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss",
             "hours_per_week"]
CATEGORICAS = ["workclass", "education", "marital_status", "occupation",
               "relationship", "race", "sex", "native_country"]


def carregar_dados():
    df = pd.read_csv(RAIZ / "data" / "adult.csv", header=None, names=COLUNAS,
                     skipinitialspace=True, na_values="?")
    df = df.dropna().reset_index(drop=True)
    df["y"] = df["income"].str.startswith(">50K").astype(int)
    # raca binarizada para grupos estaveis: White x Non-white
    df["race_bin"] = np.where(df["race"] == "White", "White", "Non-white")
    return df.drop(columns=["income"])


def preprocessador():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
    ])


def injetar_vies(y, sexo, rng, taxa=0.5):
    """Simula decisoes historicas discriminatorias: troca positivos de mulheres."""
    y = y.copy()
    alvo = np.where((sexo == "Female") & (y == 1))[0]
    trocar = rng.choice(alvo, size=int(len(alvo) * taxa), replace=False)
    y[trocar] = 0
    return y


def treinar(cenario, semente):
    df = carregar_dados()
    treino, teste = train_test_split(df, test_size=0.3, random_state=semente,
                                     stratify=df["y"])
    X_tr = treino.drop(columns=["y", "race_bin"])
    y_tr = treino["y"].to_numpy()
    rng = np.random.default_rng(semente)

    if cenario == "enviesado":
        y_tr = injetar_vies(y_tr, treino["sex"].to_numpy(), rng)

    if cenario.startswith("mitigado"):
        prep = preprocessador().fit(X_tr)
        restricao = DemographicParity() if cenario == "mitigado_dp" else EqualizedOdds()
        mitigador = ExponentiatedGradient(
            LogisticRegression(max_iter=1000), constraints=restricao, eps=0.01)
        grupos = treino["sex"] + "|" + treino["race_bin"]
        mitigador.fit(prep.transform(X_tr), y_tr, sensitive_features=grupos)
        modelo = ModeloMitigado(prep, mitigador)
    else:
        modelo = Pipeline([("prep", preprocessador()),
                           ("clf", LogisticRegression(max_iter=1000))])
        modelo.fit(X_tr, y_tr)

    (RAIZ / "modelos").mkdir(exist_ok=True)
    joblib.dump(modelo, RAIZ / "modelos" / f"{cenario}_s{semente}.joblib")
    teste.to_csv(RAIZ / "data" / f"teste_s{semente}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cenario", choices=CENARIOS, required=True)
    ap.add_argument("--semente", type=int, default=0)
    a = ap.parse_args()
    treinar(a.cenario, a.semente)
