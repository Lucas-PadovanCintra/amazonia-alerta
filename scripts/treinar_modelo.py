
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone
import joblib

CAMINHO_DADOS = Path("data/processed/base_modelagem.csv")

df = pd.read_csv(CAMINHO_DADOS)

# Variáveis que estarão disponíveis no momento da previsão
variaveis_numericas = [
    "focos_mes_atual",
    "focos_mes_anterior",
    "media_3_meses",
]

variaveis_categoricas = [
    "mes_previsao",
    "estado",
]

variaveis = variaveis_numericas + variaveis_categoricas
alvo = "alta_concentracao_proximo_mes"

# Separação temporal
treino = df[df["ano_previsao"] <= 2023]
validacao = df[df["ano_previsao"] == 2024]
teste = df[df["ano_previsao"] == 2025]

print("Treinamento:", len(treino))
print("Validação:", len(validacao))
print("Teste reservado:", len(teste))

X_treino = treino[variaveis]
y_treino = treino[alvo]

X_validacao = validacao[variaveis]
y_validacao = validacao[alvo]

# Preparação das variáveis
preprocessamento = ColumnTransformer(
    transformers=[
        ("numericas", StandardScaler(), variaveis_numericas),
        (
            "categoricas",
            OneHotEncoder(handle_unknown="ignore"),
            variaveis_categoricas,
        ),
    ]
)

# Pipeline: preparação + modelo
modelo = Pipeline(
    steps=[
        ("preprocessamento", preprocessamento),
        (
            "classificador",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)

# Treinamento
modelo.fit(X_treino, y_treino)

# Previsões para validação (2024)
probabilidades = modelo.predict_proba(X_validacao)[:, 1]

# Limiar inicial de 50%
previsoes = (probabilidades >= 0.5).astype(int)

print("\nResultados da validação — 2024")
print(
    classification_report(
        y_validacao,
        previsoes,
        zero_division=0,
    )
)

print("Matriz de confusão:")
print(confusion_matrix(y_validacao, previsoes))

print(
    "\nAverage Precision (AP):",
    round(average_precision_score(y_validacao, probabilidades), 4),
)

print(
    "Proporção de positivos na validação:",
    round(y_validacao.mean(), 4),
)

# Baseline: repetir a classificação do mês atual
previsoes_baseline = (
    validacao["focos_mes_atual"] >= 30
).astype(int)

print("\n=== COMPARAÇÃO COM BASELINE ===")

resultados = pd.DataFrame(
    {
        "Métrica": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
        ],
        "Baseline": [
            accuracy_score(y_validacao, previsoes_baseline),
            precision_score(
                y_validacao,
                previsoes_baseline,
                zero_division=0,
            ),
            recall_score(
                y_validacao,
                previsoes_baseline,
                zero_division=0,
            ),
            f1_score(
                y_validacao,
                previsoes_baseline,
                zero_division=0,
            ),
        ],
        "Regressão logística": [
            accuracy_score(y_validacao, previsoes),
            precision_score(
                y_validacao,
                previsoes,
                zero_division=0,
            ),
            recall_score(
                y_validacao,
                previsoes,
                zero_division=0,
            ),
            f1_score(
                y_validacao,
                previsoes,
                zero_division=0,
            ),
        ],
    }
)

print(resultados.round(4).to_string(index=False))

print("\nMatriz de confusão do baseline:")
print(confusion_matrix(y_validacao, previsoes_baseline))


# Avaliar diferentes limiares de decisão
limiares = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

comparacao_limiares = []

for limiar in limiares:
    previsoes_limiar = (
        probabilidades >= limiar
    ).astype(int)

    comparacao_limiares.append(
        {
            "Limiar": limiar,
            "Precision": precision_score(
                y_validacao,
                previsoes_limiar,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_validacao,
                previsoes_limiar,
                zero_division=0,
            ),
            "F1-score": f1_score(
                y_validacao,
                previsoes_limiar,
                zero_division=0,
            ),
            "Alertas emitidos": int(previsoes_limiar.sum()),
        }
    )

tabela_limiares = pd.DataFrame(comparacao_limiares)

print("\n=== COMPARAÇÃO DE LIMIARES ===")
print(tabela_limiares.round(4).to_string(index=False))

# Configuração escolhida na validação de 2024
LIMIAR_FINAL = 0.60

# Reunir todos os dados anteriores a 2025
treino_final = df[df["ano_previsao"] <= 2024]

X_treino_final = treino_final[variaveis]
y_treino_final = treino_final[alvo]

X_teste = teste[variaveis]
y_teste = teste[alvo]

# Criar uma cópia independente do pipeline
modelo_final = clone(modelo)

# Treinar com o histórico ampliado
modelo_final.fit(X_treino_final, y_treino_final)

# Prever probabilidades para 2025
probabilidades_teste = modelo_final.predict_proba(X_teste)[:, 1]

# Aplicar o limiar definido anteriormente
previsoes_teste = (
    probabilidades_teste >= LIMIAR_FINAL
).astype(int)

print("\n=== AVALIAÇÃO FINAL — 2025 ===")

print("Observações de treinamento:", len(treino_final))
print("Observações de teste:", len(teste))

print(
    classification_report(
        y_teste,
        previsoes_teste,
        zero_division=0,
    )
)

print("Matriz de confusão:")
print(confusion_matrix(y_teste, previsoes_teste))

print(
    "Average Precision:",
    round(
        average_precision_score(
            y_teste,
            probabilidades_teste,
        ),
        4,
    ),
)

# Baseline no mesmo conjunto de teste
previsoes_baseline_teste = (
    teste["focos_mes_atual"] >= 30
).astype(int)

print("\n=== COMPARAÇÃO FINAL COM BASELINE ===")

comparacao_final = pd.DataFrame(
    {
        "Métrica": ["Precision", "Recall", "F1-score"],
        "Baseline": [
            precision_score(y_teste, previsoes_baseline_teste, zero_division=0),
            recall_score(y_teste, previsoes_baseline_teste, zero_division=0),
            f1_score(y_teste, previsoes_baseline_teste, zero_division=0),
        ],
        "Regressão logística": [
            precision_score(y_teste, previsoes_teste, zero_division=0),
            recall_score(y_teste, previsoes_teste, zero_division=0),
            f1_score(y_teste, previsoes_teste, zero_division=0),
        ],
    }
)

print(comparacao_final.round(4).to_string(index=False))


# Salvar o pipeline final treinado
CAMINHO_MODELO = Path("models/modelo_amazonia_alerta.joblib")

CAMINHO_MODELO.parent.mkdir(
    parents=True,
    exist_ok=True,
)

joblib.dump(modelo_final, CAMINHO_MODELO)

print(f"\nModelo salvo em: {CAMINHO_MODELO}")
