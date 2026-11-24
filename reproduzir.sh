#!/usr/bin/env bash
# Reproduz todos os resultados do artigo a partir do zero.
# Uso: bash reproduzir.sh            (experimento completo, ~25 min com 4 vCPUs)
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/4] 120 execucoes (4 cenarios x 3 perfis x 10 sementes)"
python experimento.py

echo "[2/4] Tabelas e figuras"
python analisar.py

echo "[3/4] Sobrecarga em execucao sequencial (5 repeticoes por modelo)"
python medir_tempos.py

echo "[4/4] Log de build reprovado (Enviesado, semente 0, perfil Paridade)"
python fairgate.py --modelo modelos/enviesado_s0.joblib --dados data/teste_s0.csv \
  --config perfis/paridade.yml > resultados/log_build_reprovado.txt || true
cat resultados/log_build_reprovado.txt

python verificar.py
