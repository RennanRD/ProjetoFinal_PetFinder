# PetFinder Intelligence

## Sistema Multimodal de Previsão de Adoção e Otimização Assistida por IA Generativa

Projeto final da Pós-Graduação em Data Science & Business Intelligence — ISAG 2025/2026.

---

## Contexto

Este projeto dá continuidade a uma investigação anterior realizada na unidade curricular de Deep Learning, onde se demonstrou que a previsão de velocidade de adoção baseada exclusivamente em imagem estagna em ~63% de accuracy — um tecto informacional atribuído à ausência de contexto demográfico e textual.

O presente trabalho valida empiricamente essa hipótese através da construção de uma arquitetura multimodal (imagem + metadados tabulares + descrição textual), e estende o sistema com uma camada generativa local (LLM) para reescrita automática de descrições em casos de alto risco de adoção lenta.

## Pergunta de Negócio

> *"Conseguimos prever com precisão o tempo de permanência de um animal num abrigo, combinando a sua imagem, dados clínicos e descrição textual — e gerar automaticamente descrições mais apelativas para animais com elevado risco de adoção lenta, acelerando a sua saída do abrigo?"*

## Arquitetura

O sistema integra três modalidades de dados:

- **Ramo Visual:** imagens dos animais, processadas por MobileNetV2 (Transfer Learning)
- **Ramo Tabular:** metadados clínicos e demográficos (idade, raça, tamanho, vacinação, etc.)
- **Ramo Textual:** descrições em linguagem natural via sentence-transformers

Os três ramos são fundidos por concatenação de embeddings e seguidos de camadas densas de integração.

Adicionalmente, um LLM local (Llama 3 via Ollama) gera descrições otimizadas para animais classificados como "Adoção Lenta".

## Estrutura do Projeto
ProjetoFinal_PetFinder/
├── notebooks/              # Notebooks de análise e modelação
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_tabular.ipynb
│   ├── 04_baseline_texto.ipynb
│   ├── 05_baseline_imagem.ipynb
│   ├── 06_multimodal.ipynb
│   └── 07_llm_rescue.ipynb
├── src/                    # Código Python reutilizável
├── results/                # Modelos treinados e outputs
├── docs/                   # Documentação do projeto
└── data/                   # Dataset (não incluído no Git)
## Setup

### 1. Clonar o repositório

```bash
git clone https://github.com/<teu-username>/ProjetoFinal_PetFinder.git
cd ProjetoFinal_PetFinder
```

### 2. Criar ambiente conda

```bash
conda create -n petfinder python=3.11 -y
conda activate petfinder
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Obter os dados

Seguir as instruções em [`data/README.md`](data/README.md).

## Stack Técnico

- **Linguagem:** Python 3.11
- **Deep Learning:** TensorFlow 2.16 + Metal (aceleração Apple Silicon)
- **NLP:** sentence-transformers (embeddings), Ollama + Llama 3 (geração)
- **Visualização:** Matplotlib, Seaborn, Tableau
- **Desenvolvimento:** VS Code + Jupyter

## Autor

**Rennan Damiani**
Pós-Graduação em Data Science & Business Intelligence — ISAG 2025/2026

## Documentação Detalhada

Ver [`docs/PetFinder_Intelligence_Projeto.pdf`](docs/) para a especificação completa do projeto.
