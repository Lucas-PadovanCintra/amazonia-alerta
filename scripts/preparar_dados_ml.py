
from pathlib import Path

import pandas as pd

CAMINHO_ENTRADA = Path("data/processed/focos_mensais.csv")
CAMINHO_SAIDA = Path("data/processed/base_modelagem.csv")

LIMITE_ALTA_CONCENTRACAO = 30

# 1. Carregar e ordenar os dados
df = pd.read_csv(
    CAMINHO_ENTRADA,
    parse_dates=["mes"]
)

df = df.sort_values(
    ["estado", "municipio", "mes"]
).reset_index(drop=True)

# 2. Agrupar o histórico de cada município
grupo = df.groupby(["estado", "municipio"])

# 3. Criar variáveis históricas
df["focos_mes_atual"] = df["qtd_focos"]

df["focos_mes_anterior"] = grupo["qtd_focos"].shift(1)

df["media_3_meses"] = (
    grupo["qtd_focos"]
    .transform(lambda serie: serie.rolling(3, min_periods=3).mean())
)

# 4. Criar variáveis de calendário
df["mes_previsao"] = (
    df["mes"] + pd.DateOffset(months=1)
).dt.month

# 5. Criar o alvo: contagem no mês seguinte
df["focos_proximo_mes"] = grupo["qtd_focos"].shift(-1)

# 6. Manter apenas linhas com histórico e alvo disponíveis
df = df.dropna(
    subset=[
        "focos_mes_anterior",
        "media_3_meses",
        "focos_proximo_mes"
    ]
).copy()

# 7. Transformar o alvo em classificação binária
df["alta_concentracao_proximo_mes"] = (
    df["focos_proximo_mes"] >= LIMITE_ALTA_CONCENTRACAO
).astype(int)

# 8. Separar o período da previsão
df["ano_previsao"] = (
    df["mes"] + pd.DateOffset(months=1)
).dt.year

# 9. Salvar a base
CAMINHO_SAIDA.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(CAMINHO_SAIDA, index=False)

print("Dimensões da base de modelagem:", df.shape)

print("\nDistribuição dos anos previstos:")
print(df["ano_previsao"].value_counts().sort_index())

print("\nDistribuição da variável-alvo:")
print(df["alta_concentracao_proximo_mes"].value_counts())

print("\nPrimeiras linhas:")
print(
    df[
        [
            "estado",
            "municipio",
            "mes",
            "focos_mes_atual",
            "focos_mes_anterior",
            "media_3_meses",
            "mes_previsao",
            "alta_concentracao_proximo_mes"
        ]
    ].head()
)

print(f"\nArquivo salvo em: {CAMINHO_SAIDA}")
