# Amazônia Alerta

**Previsão de alta concentração mensal de focos de calor em municípios do bioma Amazônia utilizando Python, análise de dados e Machine Learning.**

## Sobre o projeto

O Amazônia Alerta é um projeto de Ciência de Dados que utiliza registros históricos de detecções de focos de calor disponibilizados pelo Instituto Nacional de Pesquisas Espaciais (INPE) para investigar padrões temporais e construir um modelo preditivo.

O objetivo é estimar se determinado município do bioma Amazônia apresentará **30 ou mais detecções de focos de calor no mês seguinte**, com base no comportamento observado nos meses anteriores.

O projeto foi desenvolvido como estudo prático de análise de dados, engenharia de atributos, classificação supervisionada, validação temporal e boas práticas de desenvolvimento em Python.

## Fonte dos dados

Os dados foram obtidos do Programa Queimadas do INPE, utilizando os arquivos anuais de focos de calor do satélite de referência, referentes a 2022, 2023, 2024 e 2025.

Fonte: [Programa Queimadas — INPE](https://terrabrasilis.dpi.inpe.br/queimadas/portal/)

Foram selecionados registros do Brasil classificados no bioma Amazônia, posteriormente agregados por município e mês.

**Importante:** uma detecção de foco de calor não corresponde necessariamente a um incêndio individual ou à área efetivamente queimada.

## Tecnologias utilizadas

- **Python:** processamento e organização do projeto.
- **Pandas e NumPy:** manipulação e análise de dados.
- **Matplotlib:** visualização exploratória.
- **Scikit-learn:** pré-processamento, treinamento e avaliação do modelo.
- **PostgreSQL e SQL:** armazenamento e consultas analíticas.
- **Joblib:** persistência do modelo treinado.
- **Pytest:** testes automatizados.
- **Jupyter Notebook:** análise exploratória.

## Metodologia

### 1. Preparação dos dados

Os registros anuais foram consolidados, filtrados para o bioma Amazônia e agregados por município e mês.

Foi construída uma base mensal com 542 municípios, incluindo meses sem detecções registradas.

### 2. Engenharia de atributos

Para cada município e mês, foram utilizadas as seguintes variáveis preditoras:

- Quantidade de detecções no mês atual.
- Quantidade de detecções no mês anterior.
- Média de detecções dos últimos três meses, incluindo o atual.
- Mês do calendário correspondente à previsão.
- Estado do município.

A variável-alvo indica se a contagem de detecções no mês seguinte será igual ou superior a 30.

Esse limite foi definido como critério operacional de classificação a partir da distribuição histórica de 2022–2024. Não representa um limiar oficial de risco estabelecido pelo INPE.

### 3. Treinamento e validação

Foi utilizada uma regressão logística com pré-processamento das variáveis numéricas e categóricas.

A divisão temporal foi realizada da seguinte maneira:

| Etapa | Período |
|---|---|
| Treinamento inicial | 2022–2023 |
| Validação e escolha do limiar | 2024 |
| Treinamento final | 2022–2024 |
| Teste final | 2025 |

O limiar de decisão de 0,60 foi selecionado com base no F1-score entre os valores avaliados em 2024.

### 4. Comparação com baseline

O modelo foi comparado a uma regra simples: prever alta concentração no mês seguinte quando o município já apresentou pelo menos 30 detecções no mês atual.

Essa comparação permite avaliar o benefício do modelo em relação a uma abordagem sem Machine Learning.

## Resultados

**Avaliação temporal final — 2025**

| Métrica | Baseline | Regressão logística |
|---|---:|---:|
| Precision | 49,15% | 40,80% |
| Recall | 49,01% | **66,20%** |
| F1-score | 0,4908 | **0,5048** |

A regressão logística apresentou maior capacidade de identificar períodos de alta concentração, mas também produziu mais falsos positivos.

O modelo alcançou Average Precision de **0,5052** no conjunto de teste de 2025.

Esses resultados demonstram o potencial da abordagem, mas também evidenciam limitações relevantes para sua aplicação prática.

## Estrutura do projeto

```text
amazonia-alerta/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── modelo_amazonia_alerta.joblib
├── notebooks/
│   └── 01_analise_exploratoria.ipynb
├── scripts/
│   ├── consolidar_dados.py
│   ├── construir_base_mensal.py
│   ├── inspecionar_dados.py
│   ├── preparar_dados_ml.py
│   ├── prever.py
│   └── treinar_modelo.py
├── sql/
│   ├── schema.sql
│   └── analise_exploratoria.sql
├── tests/
├── .gitignore
└── requirements.txt
```

## Como executar

Recomenda-se Python 3.12.

Clone o repositório e crie um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Baixe os arquivos anuais de focos de calor do satélite de referência do INPE, referentes a 2022–2025, e coloque os CSVs em `data/raw/`, seguindo os nomes esperados pelo script de processamento.

Execute as etapas:

```bash
python scripts/construir_base_mensal.py
python scripts/preparar_dados_ml.py
python scripts/treinar_modelo.py
python scripts/prever.py
```

Para executar os testes:

```bash
python -m pytest tests/ -q
```

O script de previsão demonstra uma inferência retrospectiva para dezembro de 2025, utilizando os dados disponíveis até novembro daquele ano.

## Limitações

- O modelo considera apenas municípios identificados no universo histórico utilizado para construir a base.
- A definição de alta concentração é estatística e não corresponde a uma classificação oficial de risco ambiental.
- A validação de 2024 não é totalmente independente da definição do limiar de 30 detecções, estabelecido com dados de 2022–2024.
- A frequência da classe positiva e o desempenho do modelo variam entre anos.
- As pontuações produzidas pela regressão logística não passaram por calibração probabilística.
- A previsão depende da disponibilidade de dados mensais completos e atualizados.
- O projeto não considera, nesta versão, variáveis meteorológicas, características da vegetação ou informações de uso do solo.
- O sistema é um protótipo educacional e não deve ser utilizado como ferramenta oficial de prevenção ou resposta a incêndios.

## Possíveis evoluções

- Incorporar variáveis meteorológicas e ambientais.
- Avaliar modelos adicionais e estratégias de calibração.
- Implementar validação temporal com múltiplas janelas.
- Criar uma API para disponibilizar previsões.
- Desenvolver visualizações geográficas dos resultados.
- Automatizar a atualização dos dados.

## Qualidade de código

O projeto inclui **17 testes automatizados**, abrangendo integridade das bases, consistência de atributos históricos e comportamento da função de inferência.

---

*Projeto de portfólio desenvolvido para aplicar conhecimentos de Python, Ciência de Dados, Machine Learning e Engenharia de Software a um problema de interesse ambiental.*
