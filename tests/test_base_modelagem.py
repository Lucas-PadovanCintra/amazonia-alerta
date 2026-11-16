
from pathlib import Path

import pandas as pd

CAMINHO = Path("data/processed/base_modelagem.csv")


def carregar_base():
    return pd.read_csv(CAMINHO, parse_dates=["mes"])


def test_sem_valores_nulos():
    df = carregar_base()
    assert df.isna().sum().sum() == 0


def test_chave_unica():
    df = carregar_base()
    assert not df.duplicated(
        subset=["estado", "municipio", "mes"]
    ).any()


def test_classes_validas():
    df = carregar_base()
    assert set(df["alta_concentracao_proximo_mes"].unique()) <= {0, 1}


def test_alvo_corresponde_contagem():
    df = carregar_base()

    esperado = (
        df["focos_proximo_mes"] >= 30
    ).astype(int)

    assert (
        df["alta_concentracao_proximo_mes"] == esperado
    ).all()


def test_ano_previsao():
    df = carregar_base()

    ano_esperado = (
        df["mes"] + pd.DateOffset(months=1)
    ).dt.year

    assert (df["ano_previsao"] == ano_esperado).all()


def test_ultimo_mes_previsto():
    df = carregar_base()

    mes_previsao = (
        df["mes"] + pd.DateOffset(months=1)
    )

    assert mes_previsao.max() == pd.Timestamp("2025-12-01")


def test_continuidade_mensal():
    df = carregar_base()

    df = df.sort_values(
        ["estado", "municipio", "mes"]
    )

    for _, grupo in df.groupby(["estado", "municipio"]):
        meses = grupo["mes"].dt.to_period("M")
        diferencas = meses.astype(int).diff().dropna()

        assert (diferencas == 1).all()


def test_variaveis_historicas():
    base = pd.read_csv(
        "data/processed/focos_mensais.csv",
        parse_dates=["mes"],
    )

    base = base.sort_values(
        ["estado", "municipio", "mes"]
    )

    grupo = base.groupby(["estado", "municipio"])

    base["anterior_esperado"] = (
        grupo["qtd_focos"].shift(1)
    )

    base["media_esperada"] = (
        grupo["qtd_focos"]
        .transform(
            lambda serie: serie.rolling(
                3,
                min_periods=3,
            ).mean()
        )
    )

    modelagem = carregar_base()

    comparacao = modelagem.merge(
        base[
            [
                "estado",
                "municipio",
                "mes",
                "anterior_esperado",
                "media_esperada",
            ]
        ],
        on=["estado", "municipio", "mes"],
        validate="one_to_one",
    )

    assert len(comparacao) == len(modelagem)

    pd.testing.assert_series_equal(
        comparacao["focos_mes_anterior"],
        comparacao["anterior_esperado"],
        check_names=False,
    )

    pd.testing.assert_series_equal(
        comparacao["media_3_meses"],
        comparacao["media_esperada"],
        check_names=False,
    )
