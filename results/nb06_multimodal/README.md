# NB06 — Multimodal Fusion: Resumo Executivo

**Data de conclusão**: 2026-04-25
**Framework**: sklearn MLPClassifier (consistente com NB04/NB05)
**Modelo oficial**: Model A — fusion of 3 probabilities

---

## 1. Objetivo

Combinar as predictions dos três ramos unimodais (tabular NB03, texto NB04, imagem NB05)
num sistema único de ranking para identificar animais com alta probabilidade de adoção lenta,
alimentando o NB07 (camada generativa LLM) e o dashboard Tableau.

## 2. Metodologia

- **Inputs de treino da fusão**: val set (2396 animais) — escolha metodológica para evitar
  data leakage. Predictions de treino dos ramos originais não estão disponíveis sem OOF.
- **Avaliação final**: test set (2997 animais), uma única vez.
- **Seleção de modelo**: 5-fold stratified CV no val set.
- **Quatro arquiteturas testadas**: Model A (fusão pura), Models B/B2/B3 (variantes enriquecidas).

## 3. Resultados

### Tabela comparativa final

| Baseline | Input dim | ROC-AUC val | ROC-AUC test | Observação |
|----------|-----------|-------------|--------------|-----------|
| NB03 Tabular (LightGBM) | 50 | 0.6758 | — | Baseline tabular forte |
| NB04 Texto (sklearn MLP) | 384 | 0.5499 | — | Teto de sinal ~0.55 |
| NB05 Imagem (sklearn MLP) | 1280 | 0.6249 | 0.6407 | Segundo melhor ramo |
| **NB06 Multimodal (Model A)** | **3** | **0.6823** | **0.6875** | **Oficial** |

### Métricas de ranking (Model A, test set)

- **Precision@50**: 0.8200 (lift 1.65x vs random)
- **Precision@100**: 0.8000 (lift 1.61x)
- **Precision@500**: 0.7620 (lift 1.53x)

### Tier performance (test set)

| Tier | n | % lentos | Lift vs base |
|------|---|----------|--------------|
| High | 625 | 74.4% | 1.50x |
| Medium | 751 | 56.3% | 1.13x |
| Low | 1621 | 37.2% | 0.75x |

Prevalência base (classe lenta): 49.7%.

## 4. Descobertas metodológicas

### 4.1 Fusão pura supera fusão enriquecida

Testámos três variantes de fusão enriquecida (ver célula 6). Todas sofreram overfit 
persistente (gap val-test > 0.08) mesmo com regularização 100× superior ao default.

**Interpretação:** o `proba_tab` produzido pelo LightGBM do NB03 é uma representação 
comprimida e robusta do sinal tabular. Re-expor as features cruas ao MLP de fusão 
introduz sensibilidade a pequenas diferenças distributivas entre val e test que o 
LightGBM absorvia implicitamente.

### 4.2 Ranking em vez de classificação binária

O Model A produz P(slow) numa gama estreita [0.43, 0.53]. A otimização de threshold binário
(F-beta β=2) colapsa em soluções degeneradas (marcar >90% dos animais). Adotámos ranking
com tiers alinhados com capacidade operacional realista (20% High, 30% Medium, 50% Low).

### 4.3 Paradoxo de Simpson em `Sterilized`

Análise qualitativa dos tiers revelou que animais esterilizados concentram-se no tier High
(50% vs 5% no Low), sugerindo que esterilização sinaliza adoção lenta. Contradiz o SHAP
do NB03, que mostrou efeito **acelerador** da esterilização. A diferença explica-se por
confounding com idade: esterilizados têm idade média 22.5 meses vs 6.0 meses dos não
esterilizados — a idade domina o sinal marginal.

## 5. Artefactos produzidos

- `model_A_fusion_probas.joblib` — **modelo oficial** (fusão pura)
- `model_B3_refuted.joblib`, `scaler_B3_refuted.joblib` — experimento refutado (preservado para auditoria)
- `val_predictions.parquet`, `test_predictions.parquet` — predictions do Model A
- `priority_queue.parquet`, `priority_queue.csv` — queue enriquecida para NB07 + Tableau
- `tier_config.json` — configuração de tiers
- `metrics.json` — métricas consolidadas
- `final_comparison.csv` — tabela comparativa dos 4 baselines

## 6. Próximos passos

- **NB07 — Camada generativa**: Ollama + Llama 3 reescreve descrições dos 625 animais do tier High.
  Dos quais 597 têm descrição razoável (≥5 palavras).
- **Tableau**: painéis Overview / Animal Explorer / Priority Queue alimentados por `priority_queue.csv`.
- **Trabalho futuro**: fusão intermédia com Keras Functional API + attention cross-modal
  (adiado pelo conflito TF/NumPy documentado no CLAUDE.md).
