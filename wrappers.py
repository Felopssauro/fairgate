"""Classe auxiliar compartilhada entre o treino e a auditoria (necessaria para o joblib)."""


class ModeloMitigado:
    """Envolve o ExponentiatedGradient para expor predict() como um modelo comum."""

    def __init__(self, prep, mitigador):
        self.prep, self.mitigador = prep, mitigador

    def predict(self, X):
        return self.mitigador.predict(self.prep.transform(X), random_state=0)
