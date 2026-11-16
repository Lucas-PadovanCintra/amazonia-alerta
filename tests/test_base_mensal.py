from pathlib import Path

import pandas as pd
import pytest


CAMINHO_BASE = Path("data/processed/focos_mensais.csv")


@pytest.fixture
def base_mensal():
    return pd.read_csv(CAMINHO_BASE, parse_dates=["mes"])


def test_quantidade_de_linhas(base_mensal):
    assert len(base_mensal) == 542 * 48


def test_chave_unica(base_mensal):
    duplicados = base_mensal.duplicated(
        subset=["estado", "municipio", "mes"]
    )

    assert not duplicados.any()


def test_contagens_nao_negativas(base_mensal):
    assert (base_mensal["qtd_focos"] >= 0).all()


def test_ausencia_de_valores_nulos(base_mensal):
    assert not base_mensal.isna().any().any()


def test_periodo_temporal(base_mensal):
    assert base_mensal["mes"].min() == pd.Timestamp("2022-01-01")
    assert base_mensal["mes"].max() == pd.Timestamp("2025-12-01")
