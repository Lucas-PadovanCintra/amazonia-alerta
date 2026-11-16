from pathlib import Path
import pandas as pd

# Pasta onde estão os arquivos originais
pasta_dados = Path("data/raw")

# Localizar os arquivos CSV de focos de calor
arquivos = sorted(pasta_dados.glob("focos_ams_ref_*.csv"))

tabelas = []

for arquivo in arquivos:
    print(f"Lendo: {arquivo.name}")

    tabela = pd.read_csv(arquivo)

    # Identificar o ano de origem do arquivo
    ano = int(arquivo.stem.split("_")[-1])
    tabela["ano_arquivo"] = ano

    tabelas.append(tabela)

# Juntar as tabelas verticalmente
df = pd.concat(tabelas, ignore_index=True)

print("\nQuantidade de arquivos:", len(arquivos))
print("Dimensões da base consolidada:", df.shape)

print("\nRegistros por ano de origem:")
print(df["ano_arquivo"].value_counts().sort_index())

print("\nColunas da base consolidada:")
print(df.columns.tolist())

# Selecionar registros brasileiros do bioma Amazônia
amazonia = df.loc[
    (df["pais"] == "Brasil") &
    (df["bioma"] == "Amazônia")
].copy()

# Converter a coluna de datas
amazonia["data_pas"] = pd.to_datetime(
    amazonia["data_pas"],
    errors="coerce"
)

# Extrair o ano real da detecção
amazonia["ano"] = amazonia["data_pas"].dt.year

print("\nDimensões do recorte amazônico:")
print(amazonia.shape)

print("\nRegistros do bioma Amazônia por ano:")
print(amazonia["ano_arquivo"].value_counts().sort_index())

print("\nDatas inválidas:")
print(amazonia["data_pas"].isna().sum())

print("\nRegistros cujo ano da detecção difere do arquivo:")
print(
    (
        amazonia["ano"] != amazonia["ano_arquivo"]
    ).sum()
)

print("\nValores ausentes no recorte amazônico:")
print(amazonia.isna().sum())

print("\nIdentificadores de focos repetidos:")
print(amazonia["foco_id"].duplicated().sum())

print("\nMunicípios distintos no histórico completo:")

municipios = (
    amazonia[["estado", "municipio"]]
    .drop_duplicates()
)

print(len(municipios))

print("\nMunicípios distintos por estado:")

print(
    municipios["estado"]
    .value_counts()
    .sort_index()
)

print("\nMunicípios distintos por ano:")

for ano in sorted(amazonia["ano"].unique()):
    dados_ano = amazonia[amazonia["ano"] == ano]

    quantidade = (
        dados_ano[["estado", "municipio"]]
        .drop_duplicates()
        .shape[0]
    )

    print(f"{ano}: {quantidade} municípios")

    # Definir o universo de municípios usando apenas 2022–2024
dados_treinamento = amazonia[
    amazonia["ano"] <= 2024
].copy()

municipios_treinamento = (
    dados_treinamento[["estado", "municipio"]]
    .drop_duplicates()
)

print("\nMunicípios no período de treinamento:")
print(len(municipios_treinamento))

# Identificar municípios presentes em 2025,
# mas ausentes no histórico de treinamento
municipios_2025 = (
    amazonia[amazonia["ano"] == 2025][["estado", "municipio"]]
    .drop_duplicates()
)

municipios_novos = municipios_2025.merge(
    municipios_treinamento,
    on=["estado", "municipio"],
    how="left",
    indicator=True
)

municipios_novos = municipios_novos[
    municipios_novos["_merge"] == "left_only"
]

print("\nMunicípios de 2025 ausentes no treinamento:")
print(len(municipios_novos))

print("\nQuais são esses municípios?")
print(municipios_novos[["estado", "municipio"]])
