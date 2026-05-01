# PetFinder Intelligence

> Sistema multimodal de previsão de adoção animal com camada generativa de reescrita de descrições

Projeto final de Pós-Graduação no ISAG. Combina dados tabulares, texto e imagem para prever que animais terão adoção lenta e reescreve as suas descrições com um modelo de linguagem local, gerando uma fila de prioridade rankeada para uso operacional do abrigo.

**Autor:** Rennan Damiani · **Ano:** 2026

---

## Resumo dos resultados

| Modelo | Modalidade | ROC-AUC |
|---|---|---|
| Baseline Tabular (LightGBM) | Estruturada | 0,676 (val) |
| Baseline Texto (MLP + MiniLM) | Descrições | 0,547 (val) |
| Baseline Imagem (MLP + MobileNetV2) | Fotografias | 0,625 (val) |
| **Modelo Multimodal (fusão)** | **Combinada** | **0,680 (test)** |

A fila de prioridade rankeada produzida pelo modelo multimodal apresenta **lift de 1,49×** sobre seleção aleatória no tier de alta prioridade (precision 73,9% contra prevalência base de 49,7%).

---

## Estrutura do pipeline

O projeto é organizado em 8 notebooks, executados sequencialmente. Cada notebook produz artefactos persistentes em disco para os notebooks seguintes consumirem.

| Notebook | Propósito |
|---|---|
| `01_EDA.ipynb` | Análise exploratória e decisões de pré-processamento |
| `02_preprocessing.ipynb` | Splits anti-leakage, scaling, embeddings de texto e imagem |
| `03_baseline_tabular.ipynb` | Modelo LightGBM sobre 50 features estruturadas |
| `04_baseline_texto.ipynb` | MLP sobre embeddings MiniLM |
| `05_baseline_imagem.ipynb` | MLP sobre embeddings MobileNetV2 |
| `06_multimodal.ipynb` | Fusão das 3 probabilidades + priority queue rankeada |
| `07_llm_rescue.ipynb` | Reescrita de descrições com Gemma 3 via Ollama |
| `08_tableau_prep.ipynb` | Preparação dos CSVs para visualização em Tableau |

---

## Reproduzir o pipeline

### Pré-requisitos

- Python 3.11
- Anaconda ou venv recomendado
- Para o NB07: [Ollama](https://ollama.com) instalado localmente com o modelo Gemma 3 (`ollama pull gemma3:4b`)

### Setup

1. **Clonar o repositório**
   ```bash
   git clone https://github.com/RennanRD/ProjetoFinal_PetFinder.git
   cd ProjetoFinal_PetFinder
   ```

2. **Descarregar o dataset original do Kaggle**

   O dataset não está incluído no repositório (licença Kaggle, ~1GB). Descarregar de:
   https://www.kaggle.com/c/petfinder-adoption-prediction/data

   Descompactar para a estrutura:
   ```
   data/train/train.csv
   data/train_images/
   ```

3. **Criar ambiente Python e instalar dependências**
   ```bash
   conda create -n petfinder python=3.11
   conda activate petfinder
   pip install -r requirements.txt
   ```

4. **Correr os notebooks pela ordem** (NB01 → NB08).

### Sobre a geração de embeddings de imagem (NB02)

Os embeddings de imagem foram gerados com TensorFlow 2.16 + MobileNetV2 num ambiente isolado, e estão pré-calculados em `data/cache/image_embeddings_mobilenetv2.npy` (73MB, incluído no repo).

O TensorFlow 2.16 tem incompatibilidade conhecida com NumPy 2.x. Para evitar conflitos com o resto do stack (que usa NumPy 2.x), o NB02 carrega o cache pré-calculado em vez de regenerar. Esta é uma decisão arquitetural deliberada — caching offline + consumo online.

Para regenerar do zero (raramente necessário), seguir as instruções no markdown da célula relevante do NB02.

---

## Visualização: Dashboard Kind Paws Shelter

A camada de visualização foi materializada num dashboard operacional em Tableau Desktop, dirigido a uma equipa fictícia de gestão de abrigo (Kind Paws Shelter). Tem duas vistas: **Overview** (leitura estratégica, indicadores populacionais) e **Priority Queue** (fila operacional rankeada).

Os dados consumidos pelo Tableau são gerados pelo NB08:
- `results/nb08_tableau/tableau_main.csv` — test set enriquecido com predições e reescritas
- `results/nb08_tableau/tableau_full.csv` — dataset completo para indicadores agregados

---

## Decisões metodológicas centrais

- **Anti-leakage por RescuerID:** divisão treino/validação/teste agrupada por voluntário (`StratifiedGroupKFold`), evitando que o modelo aprenda padrões de voluntário em vez de propriedades dos animais.
- **Caching offline de embeddings:** texto (MiniLM, 384 dim) e imagem (MobileNetV2, 1280 dim) calculados uma vez e persistidos em disco.
- **Fusão pura supera fusão enriquecida:** modelo final usa apenas as 3 probabilidades dos baselines, não as features tabulares brutas (que reintroduzem overfit mesmo com regularização agressiva).
- **Ranking sobre classificação binária:** o modelo produz scores numa gama estreita (0,43–0,53), inadequada para limiar binário — solução é ranking + tiers operacionais.
- **Iteração documentada do prompt LLM:** v1→v2 com regras anti-invenção e anti-suavização de factos clínicos.

Detalhes no relatório do projeto.

---

## Stack técnico

| Componente | Tecnologia |
|---|---|
| Tabular | LightGBM, scikit-learn |
| Texto | sentence-transformers (`all-MiniLM-L6-v2`) |
| Imagem | TensorFlow 2.16 + MobileNetV2 (offline, cached) |
| Fusão | scikit-learn MLPClassifier |
| LLM Rescue | Gemma 3 (4B) via Ollama |
| Visualização | Tableau Desktop |
| Análise | pandas, numpy, matplotlib, seaborn, SHAP |

---

## Estrutura de pastas

```
.
├── notebooks/                    # 8 notebooks Jupyter, executados sequencialmente
├── data/
│   ├── breed_labels.csv          # tabelas de lookup (incluídas)
│   ├── color_labels.csv
│   ├── state_labels.csv
│   ├── cache/                    # embeddings pré-calculados (incluídos)
│   ├── processed/                # outputs do NB02
│   └── train/                    # dataset original (descarregar do Kaggle)
├── results/                      # artefactos por notebook
│   ├── nb03_baseline_tabular/
│   ├── nb04_baseline_texto/
│   ├── nb05_baseline_imagem/
│   ├── nb06_multimodal/
│   ├── nb07_llm_rescue/
│   └── nb08_tableau/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Licença

Este projeto é académico e não tem fins comerciais. O dataset PetFinder.my é propriedade da Kaggle e dos seus titulares originais — consultar [termos de uso](https://www.kaggle.com/c/petfinder-adoption-prediction/rules) antes de redistribuir.
