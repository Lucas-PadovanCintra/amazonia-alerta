
from pathlib import Path

import joblib
import pandas as pd


CAMINHO_MODELO = Path("models/modelo_amazonia_alerta.joblib")
CAMINHO_DADOS = Path("data/processed/base_modelagem.csv")

LIMIAR_ALERTA = 0.60

VARIAVEIS = [
    "focos_mes_atual",
    "focos_mes_anterior",
    "media_3_meses",
    "mes_previsao",
    "estado",
]


def gerar_alertas(modelo, dados, variaveis, limiar=0.60):
    resultado = dados.copy()

    resultado["probabilidade"] = modelo.predict_proba(
        resultado[variaveis]
    )[:, 1]

    resultado["alerta"] = (
        resultado["probabilidade"] >= limiar
    ).astype(int)

    return resultado


def main():
    # Carregar o pipeline treinado
    modelo = joblib.load(CAMINHO_MODELO)

    # Carregar os dados de modelagem
    df = pd.read_csv(CAMINHO_DADOS, parse_dates=["mes"])

    # Selecionar novembro de 2025 para prever dezembro
    dados_previsao = df[
        df["mes"] == "2025-11-01"
    ].copy()

    # Gerar previsões
    dados_previsao = gerar_alertas(
        modelo,
        dados_previsao,
        VARIAVEIS,
        LIMIAR_ALERTA,
    )

    # Ordenar municípios por probabilidade
    resultado = dados_previsao[
        [
            "estado",
            "municipio",
            "mes_previsao",
            "probabilidade",
            "alerta",
        ]
    ].sort_values(
        "probabilidade",
        ascending=False,
    )

    print("\n=== AMAZÔNIA ALERTA ===")
    print("Previsão retrospectiva para dezembro de 2025")
    print(f"Municípios analisados: {len(resultado)}")
    print(f"Alertas emitidos: {resultado['alerta'].sum()}")

    print("\n10 municípios com maior probabilidade:")
    print(resultado.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
