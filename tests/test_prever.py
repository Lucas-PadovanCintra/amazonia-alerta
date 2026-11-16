
import numpy as np
import pandas as pd

from scripts.prever import gerar_alertas


class ModeloSimulado:
    def predict_proba(self, dados):
        assert list(dados.columns) == ["focos_mes_atual"]

        return np.array([
            [0.80, 0.20],
            [0.40, 0.60],
            [0.10, 0.90],
        ])


def criar_dados_teste():
    return pd.DataFrame({
        "focos_mes_atual": [5, 20, 50],
        "focos_proximo_mes": [10, 40, 60],
    })


def test_geracao_de_alertas():
    modelo = ModeloSimulado()
    dados = criar_dados_teste()

    resultado = gerar_alertas(
        modelo,
        dados,
        ["focos_mes_atual"],
        limiar=0.60,
    )

    assert resultado["alerta"].tolist() == [0, 1, 1]


def test_probabilidades():
    modelo = ModeloSimulado()
    dados = criar_dados_teste()

    resultado = gerar_alertas(
        modelo,
        dados,
        ["focos_mes_atual"],
        limiar=0.60,
    )

    np.testing.assert_allclose(
        resultado["probabilidade"].to_numpy(),
        [0.20, 0.60, 0.90],
    )


def test_dados_originais_preservados():
    modelo = ModeloSimulado()
    dados = criar_dados_teste()
    copia_original = dados.copy(deep=True)

    gerar_alertas(
        modelo,
        dados,
        ["focos_mes_atual"],
        limiar=0.60,
    )

    pd.testing.assert_frame_equal(dados, copia_original)


def test_sem_vazamento_da_variavel_futura():
    modelo = ModeloSimulado()
    dados = criar_dados_teste()

    resultado = gerar_alertas(
        modelo,
        dados,
        ["focos_mes_atual"],
        limiar=0.60,
    )

    assert "focos_proximo_mes" in resultado.columns
    assert "alerta" in resultado.columns
