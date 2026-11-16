import pandas as pd

caminho = "data/raw/focos_ams_ref_2025.csv"

df = pd.read_csv(caminho)

print("Dimensões do dataset:")
print(df.shape)

print("\nTipos das colunas:")
print(df.dtypes)

print("\nPrimeiras cinco linhas:")
print(df.head())

print("\nQuantidade de registros por estado:")
print(df["estado"].value_counts())

print("\nQuantidade de registros por bioma:")
print(df["bioma"].value_counts())

print("\nQuantidade de países:")
print(df["pais"].value_counts())

print("\nValores ausentes por coluna:")
print(df.isna().sum())

print("\nIntervalo das datas:")
print(df["data_pas"].min())
print(df["data_pas"].max())

print("\nRegistros do bioma Amazônia no Brasil:")
amazonia = df[
    (df["pais"] == "Brasil") &
    (df["bioma"] == "Amazônia")
]

print(amazonia.shape)

print("\nEstados com registros no bioma Amazônia:")
print(amazonia["estado"].value_counts())

print("\nQuantidade de municípios distintos:")
print(amazonia[["estado", "municipio"]].drop_duplicates().shape[0])

# Converter a coluna de texto para datetime
amazonia = amazonia.copy()

amazonia["data_pas"] = pd.to_datetime(
    amazonia["data_pas"],
    errors="coerce"
)

# Criar uma coluna com o mês da detecção
amazonia["mes"] = amazonia["data_pas"].dt.month

# Contar as detecções por mês
focos_por_mes = (
    amazonia.groupby("mes")
    .size()
    .sort_index()
)

print("\nFocos de calor por mês em 2025:")
print(focos_por_mes)

print("\nDatas inválidas após conversão:")
print(amazonia["data_pas"].isna().sum())

print("\nMunicípios com registros por mês:")
print(
    amazonia.groupby("mes")[["estado", "municipio"]]
    .apply(lambda grupo: grupo.drop_duplicates().shape[0])
)
