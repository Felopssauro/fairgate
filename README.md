# FairGate: artefato de pesquisa

Artefato do artigo **"FairGate: Auditoria Automatizada de Viés Algorítmico em Esteiras de CI/CD"**

O FairGate é um portão de qualidade (*quality gate*) para esteiras de CI/CD. Ele carrega um modelo de classificação treinado e um conjunto de teste rotulado, calcula métricas de justiça algorítmica por atributo sensível e encerra com código de saída 1 (build reprovado, deploy bloqueado) quando algum limiar configurado é violado. Este pacote contém o código, a base de dados, os perfis de limiares, o workflow do GitHub Actions e os registros brutos das 120 execuções relatadas no artigo, além de scripts para reproduzi-las e conferi-las.

**Selos pretendidos:** Disponível, Funcional, Reprodutível.

---

## 1. Estrutura

```
FairGate-artefato/
├── README.md                      este arquivo
├── LICENSE                        licença do código (MIT)
├── requirements.txt               dependências com versões fixas
├── reproduzir.sh                  reproduz tudo do zero e confere com o artigo
│
├── treinar.py                     treina um modelo (cenário × semente)
├── teste_acuracia.py              teste tradicional: acurácia mínima de 0,80 (linha de base)
├── fairgate.py                    o portão de fairness (contribuição do artigo)
├── wrappers.py                    classe auxiliar para serializar modelos mitigados
├── perfis/
│   ├── paridade.yml               Perfil A: paridade demográfica + regra dos 4/5
│   ├── equalized_odds.yml         Perfil B: equalized odds
│   └── completo.yml               Perfil C: os três limiares ao mesmo tempo
├── experimento.py                 roda as 120 execuções → resultados/execucoes.csv
├── analisar.py                    gera tabelas e figuras do artigo
├── medir_tempos.py                mede a sobrecarga em execução sequencial
├── verificar.py                   compara resultados novos com os publicados
├── .github/workflows/fairness.yml workflow do GitHub Actions
│
├── data/adult.csv                 base Adult Census Income (UCI), 48.842 registros
├── resultados_publicados/         resultados exatamente como usados no artigo
└── (modelos/, relatorios/, resultados/ são criados ao executar)
```

## 2. Requisitos

| Item | Usado no artigo | Mínimo recomendado |
|---|---|---|
| Sistema operacional | Linux (máquina virtual) | Linux ou macOS; Windows via WSL |
| CPU | 4 vCPUs | 2 núcleos |
| Memória | não registrada | 4 GB |
| Disco | ~650 MB após a execução completa (modelos e conjuntos de teste) | 1 GB livre |
| Python | 3.13 | 3.13 |
| Rede | só para `pip install` | idem |

As versões exatas das bibliotecas estão em `requirements.txt` (Fairlearn 0.15.0, scikit-learn 1.9.1, pandas 3.0.5, numpy 2.5.3, matplotlib 3.11.2, PyYAML 6.0.3, joblib 1.6.0). Não há dependência de GPU nem de serviços externos.

## 3. Instalação

```bash
unzip FairGate-artefato.zip && cd FairGate-artefato
python3.13 -m venv .venv
source .venv/bin/activate          # Windows/WSL: idem
pip install -r requirements.txt
```

## 4. Reprodução

### 4.1 Teste rápido (cerca de 5 minutos)

Reproduz um job do workflow para um cenário, exatamente com os comandos do `fairness.yml`:

```bash
python treinar.py --cenario enviesado --semente 0
python teste_acuracia.py --modelo modelos/enviesado_s0.joblib --dados data/teste_s0.csv
echo "código de saída do teste de acurácia: $?"      # 0 = passou
python fairgate.py --modelo modelos/enviesado_s0.joblib --dados data/teste_s0.csv \
       --config perfis/paridade.yml --relatorio relatorios/enviesado_s0.json
echo "código de saída do FairGate: $?"               # 1 = reprovou
```

Resultado esperado: o teste de acurácia passa (acurácia 0,8418), e o FairGate reprova com três violações, a mesma saída mostrada no artigo e em `resultados_publicados/log_build_reprovado.txt`. Esse é o resultado central do trabalho: o teste tradicional deixa passar um modelo enviesado que o portão bloqueia.

### 4.2 Reprodução completa (cerca de 25 minutos com 4 vCPUs)

```bash
bash reproduzir.sh
```

O script executa, em ordem:

1. `experimento.py`: 120 execuções (4 cenários × 3 perfis × 10 sementes) → `resultados/execucoes.csv`;
2. `analisar.py`: tabelas e figuras → `resultados/tabela_metricas.csv`, `tabela_decisoes.csv`, `tempos.json`, `fig_metricas_sexo.png`, `fig_arquitetura.png`;
3. `medir_tempos.py`: sobrecarga sequencial → `resultados/tempos_sequenciais.json`;
4. o log de build reprovado → `resultados/log_build_reprovado.txt`;
5. `verificar.py`: compara o que foi gerado com `resultados_publicados/`.

A maior parte do tempo vem do treino dos modelos mitigados (Exponentiated Gradient, 2 a 3 minutos cada sob execução paralela).

**Reprodução verificada (24/11/2026):** numa VM Linux nova (4 vCPUs Intel Xeon, Python 3.13.18, dependências de `requirements.txt`), `reproduzir.sh` levou cerca de 21 minutos. As 120 decisões, as 120 linhas de métricas, `tabela_metricas.csv`, `tabela_decisoes.csv` e `log_build_reprovado.txt` saíram idênticos aos publicados. Os tempos sequenciais do FairGate ficaram entre 1,79 e 1,99 s, contra 2,03 a 2,28 s no artigo. Todas as etapas usam sementes fixas. Na mesma configuração de hardware e software, métricas e decisões saem idênticas, e `verificar.py` informa `IDENTICAS`. Em outro processador, as métricas dos modelos mitigados podem variar um pouco por diferenças numéricas no treino (veja a seção 7). Por isso, `verificar.py` só considera falha quando alguma **decisão** muda (teste de acurácia ou FairGate), e para as métricas informa a maior diferença encontrada. Os tempos dependem do hardware e não são comparados.

## 5. Como os testes foram realizados

### 5.1 Ambiente

As 120 execuções foram feitas **localmente**, numa máquina virtual Linux com 4 vCPUs e Python 3.13. Para que a avaliação fosse fiel à esteira, `experimento.py` chama cada etapa do workflow como um **subprocesso independente**, com os mesmos comandos e argumentos do `fairness.yml`, e decide o resultado pelo **código de saída** do processo (0 = passou, 1 = reprovou), da mesma forma que o runner do GitHub Actions decide se um *step* falhou. Além disso, o workflow foi executado de verdade no GitHub Actions para a semente 0, nos três perfis (seção 7).

### 5.2 Os quatro cenários (`treinar.py`)

Todos os modelos são regressões logísticas com o mesmo pré-processamento (padronização dos atributos numéricos e one-hot dos categóricos). A base é dividida em 70% treino e 30% teste, estratificada pelo rótulo, com `random_state` igual à semente.

| Cenário | O que é | Papel no experimento |
|---|---|---|
| `base` | Treinado nos dados originais | Modelo "padrão" de uma equipe; revela o viés que já existe nos dados de 1994 |
| `enviesado` | 50% dos rótulos positivos de mulheres trocados para negativo antes do treino | Simula decisões históricas discriminatórias; é o modelo que o portão **deve** barrar |
| `mitigado_dp` | Exponentiated Gradient com restrição de paridade demográfica nos grupos sexo × raça | Deve passar no Perfil A |
| `mitigado_eo` | Exponentiated Gradient com restrição de equalized odds nos grupos sexo × raça | Deve passar no Perfil B |

Cada chamada salva `modelos/<cenario>_s<semente>.joblib` e o conjunto de teste `data/teste_s<semente>.csv`.

### 5.3 Teste tradicional de acurácia (`teste_acuracia.py`)

É a **linha de base**: representa a garantia de qualidade usual de uma esteira de ML, que só olha o desempenho global. Calcula a acurácia no conjunto de teste e sai com código 0 se ela for ≥ 0,80 (ajustável com `--minimo`), e 1 caso contrário. Não sabe nada sobre grupos.

### 5.4 O portão FairGate (`fairgate.py` + `perfis/*.yml`)

Para cada atributo sensível listado no perfil (`sex` e `race_bin`, raça agrupada em White × Non-white), calcula com o Fairlearn:

| Métrica | Significado | Ideal | Limiar |
|---|---|---|---|
| Diferença de paridade demográfica | Maior diferença entre grupos na taxa de previsões positivas | 0 | reprova se > 0,10 |
| Razão de impacto desigual | Menor taxa de positivos dividida pela maior (regra dos 4/5 da EEOC) | 1 | reprova se < 0,80 |
| Diferença de equalized odds | Maior diferença entre grupos na taxa de verdadeiros positivos ou de falsos positivos | 0 | reprova se > 0,10 |

O perfil escolhe quais limiares valem:

| Perfil | Arquivo | Limiares exigidos | Critério |
|---|---|---|---|
| A, Paridade | `paridade.yml` | paridade demográfica e razão de impacto | independência |
| B, Equalized odds | `equalized_odds.yml` | equalized odds | separação |
| C, Completo | `completo.yml` | os três | ambos |

O portão grava um relatório JSON com acurácia, métricas, lista de violações e tempo, imprime um resumo legível (as violações saem no formato `::error::`, que o GitHub Actions destaca na interface) e termina com código 0 (APROVADO) ou 1 (REPROVADO). No workflow, o perfil é escolhido pela variável `PERFIL_FAIRNESS`.

### 5.5 O experimento (`experimento.py`)

Para cada uma das 10 sementes (0 a 9) e cada um dos 4 cenários: treina o modelo, roda o teste de acurácia uma vez e roda o FairGate com cada um dos 3 perfis. Isso dá 4 × 3 × 10 = **120 execuções do portão**. As sementes rodam em paralelo (4 processos), e cada linha de `execucoes.csv` guarda decisões, métricas e tempos de parede.

### 5.6 Medição de sobrecarga (`medir_tempos.py`)

Como as sementes rodam em paralelo, os tempos de `execucoes.csv` e `tempos.json` ficam inflados pela disputa de CPU (cerca de 6 s por auditoria). Para medir o custo real do portão, os modelos da semente 0 são auditados **5 vezes em sequência**, um processo por vez, pelo teste de acurácia e pelo FairGate (perfil Completo). Esses são os números do artigo (cerca de 2 s por auditoria).

## 6. O que cada resultado significa

| Arquivo | Conteúdo | Onde aparece no artigo |
|---|---|---|
| `execucoes.csv` | Dados brutos: 120 linhas (semente, cenário, perfil, acurácia, decisão do teste de acurácia, decisão do FairGate, nº de violações, tempos e as 6 métricas) | Base de todas as tabelas |
| `tabela_metricas.csv` | Média ± desvio-padrão (10 sementes) da acurácia e das métricas por cenário; quantas vezes o teste de acurácia passou | Tabela de métricas e Figura 2 |
| `tabela_decisoes.csv` | Quantas das 10 execuções o FairGate aprovou, por cenário × perfil (`sum` = aprovações, `count` = execuções) | Tabela de decisões do portão |
| `tempos_sequenciais.json` | Média e desvio do tempo do FairGate e do teste de acurácia, em segundos, 5 repetições sequenciais | Parágrafo de sobrecarga |
| `tempos.json` | Tempos sob execução paralela e tempo médio de treino por cenário | Tempo de treino citado no artigo |
| `log_build_reprovado.txt` | Saída do FairGate para o Enviesado, semente 0, perfil Paridade | Listagem do log da esteira |
| `fig_metricas_sexo.png` | Métricas por sexo nos 4 cenários, com os limiares tracejados | Figura 2 |
| `fig_arquitetura.png` | Diagrama da esteira com o FairGate | Figura 1 |

**Leitura dos resultados principais:**

- **O teste de acurácia não detecta viés.** Todos os modelos passaram em 10/10 sementes (acurácia entre 0,828 e 0,848), inclusive o Enviesado.
- **O FairGate barrou o Enviesado em 30 de 30 execuções** (10 sementes × 3 perfis). A razão de impacto por sexo dele é 0,087: uma mulher recebe previsão de renda alta cerca de onze vezes menos que um homem.
- **O portão não reprova tudo.** O Mitigado-DP foi aprovado em 10/10 no Perfil Paridade, e o Mitigado-EO em 10/10 no Perfil Equalized odds, que são os perfis coerentes com cada mitigação.
- **Nenhum modelo passou no Perfil Completo** (0/40): cada mitigação corrige um critério e piora o outro, como prevê a teoria (independência e separação não valem juntas quando as taxas-base diferem).
- **O Base foi reprovado** nos perfis Paridade e Completo e aprovado em só 4/10 no Equalized odds, porque está perto do limiar (0,098 ± 0,018 contra 0,10). A decisão depende da divisão aleatória dos dados.
- **Custo:** cerca de 2 s por auditoria, contra cerca de 10 s de treino no Base e mais de 2 min nos mitigados.

## 7. Uso do GitHub Actions

O arquivo `.github/workflows/fairness.yml` define a esteira: um job `qualidade-modelo` em matriz (um por cenário) com os passos treino → teste de acurácia → FairGate → upload do relatório JSON, e um job `deploy` que só roda se todos os jobs da matriz passarem.

**Status:** as 120 execuções do artigo foram feitas localmente, como descrito na seção 5.1. Para validar a integração, o workflow também foi executado neste repositório, nos servidores do GitHub Actions, uma vez em cada perfil, com os modelos da semente 0 (12 jobs):

| Perfil | Base | Enviesado | Mitigado-DP | Mitigado-EO | Deploy | Execução |
|---|---|---|---|---|---|---|
| Paridade | reprovado | reprovado | **aprovado** | reprovado | não executado | [45400738158](https://github.com/Felopssauro/fairgate/actions/runs/45400738158) |
| Equalized odds | **aprovado** | reprovado | reprovado | **aprovado** | não executado | [45400823520](https://github.com/Felopssauro/fairgate/actions/runs/45400823520) |
| Completo | reprovado | reprovado | reprovado | reprovado | não executado | [45400828391](https://github.com/Felopssauro/fairgate/actions/runs/45400828391) |

As 12 decisões coincidem com as da semente 0 na execução local. O teste de acurácia passou nos 12 jobs. No runner, o passo do FairGate levou de 1 a 2 s, e o treino do Mitigado-EO cerca de 1 min. Na execução do perfil Completo, as métricas do Mitigado-EO diferiram das locais (por exemplo, razão de impacto por sexo de 0,561, contra 0,535), sem mudar a decisão. A causa provável são diferenças numéricas do hardware do runner no treino com Exponentiated Gradient; os outros três cenários deram valores idênticos em todas as execuções. A primeira execução ([45400462855](https://github.com/Felopssauro/fairgate/actions/runs/45400462855)) mostrou que, com o `fail-fast` padrão, a reprovação do cenário Base cancelava os demais jobs; por isso a matriz usa `fail-fast: false`.

Para executar num repositório seu:

1. Crie um repositório no GitHub e envie o conteúdo deste pacote para a branch `main` (a pasta `.github/` deve ficar na raiz).
2. O push dispara o workflow com o perfil padrão (`paridade`). Em **Actions**, os jobs `base`, `enviesado` e `mitigado_eo` aparecem reprovados no passo "Portao de fairness (FairGate)", com as violações destacadas; `mitigado_dp` é aprovado; e o job `deploy` não roda. A matriz usa `fail-fast: false`, para que a reprovação de um cenário não cancele os outros.
3. Para testar outro perfil, abra **Actions → FairGate - auditoria de fairness → Run workflow** e escolha `equalized_odds` ou `completo`. Outra opção é mudar o valor padrão de `PERFIL_FAIRNESS` no YAML e fazer um commit.
4. O relatório JSON de cada job fica disponível para download em **Artifacts**, na página da execução.

Cada job treina um único modelo (semente 0), e o mais lento é o `mitigado_eo`.

## 8. Limitações do artefato

- A base Adult é antiga (censo dos EUA de 1994) e o viés do cenário Enviesado foi injetado artificialmente.
- Apenas classificadores lineares e dois atributos sensíveis, com raça binarizada.
- Os tempos dependem do hardware; os valores do artigo foram medidos numa VM Linux com 4 vCPUs.
- O FairGate não calcula intervalos de confiança: decisões perto do limiar (caso do Base no Perfil B) variam com a semente.
