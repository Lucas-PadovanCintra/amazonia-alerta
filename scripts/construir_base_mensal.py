from pathlib import Path

import pandas as pd


PASTA_RAW = Path("data/raw")
PASTA_PROCESSED = Path("data/processed")

# 1. Ler e consolidar os arquivos históricos
arquivos = sorted(PASTA_RAW.glob("focos_ams_ref_*.csv"))


# Verificar se os quatro arquivos esperados estão disponíveis
nomes_esperados = {
    f"focos_ams_ref_{ano}.csv"
    for ano in range(2022, 2026)
}

nomes_encontrados = {arquivo.name for arquivo in arquivos}

if nomes_encontrados != nomes_esperados:
    faltantes = nomes_esperados - nomes_encontrados
    inesperados = nomes_encontrados - nomes_esperados

    raise ValueError(
        "Arquivos de entrada incorretos. "
        f"Faltantes: {sorted(faltantes)}. "
        f"Inesperados: {sorted(inesperados)}."
    )


tabelas = []

for arquivo in arquivos:
    tabela = pd.read_csv(arquivo)
    tabelas.append(tabela)

df = pd.concat(tabelas, ignore_index=True)

# 2. Filtrar o bioma Amazônia brasileiro
amazonia = df.loc[
    (df["pais"] == "Brasil")
    & (df["bioma"] == "Amazônia")
].copy()

# 3. Preparar a coluna temporal
amazonia["data_pas"] = pd.to_datetime(amazonia["data_pas"])

amazonia["mes"] = (
    amazonia["data_pas"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

# 4. Definir os municípios com base em 2022–2024
municipios = (
    amazonia.loc[
        amazonia["mes"] < "2025-01-01",
        ["estado", "municipio"]
    ]
    .drop_duplicates()
)

# 5. Construir uma grade de municípios e meses
meses = pd.DataFrame({
    "mes": pd.date_range(
        start="2022-01-01",
        end="2025-12-01",
        freq="MS"
    )
})

grade = municipios.merge(meses, how="cross")

# 6. Contar as detecções por município e mês
contagens = (
    amazonia.groupby(
        ["estado", "municipio", "mes"]
    )
    .size()
    .reset_index(name="qtd_focos")
)

# 7. Juntar as contagens à grade completa
base_mensal = grade.merge(
    contagens,
    on=["estado", "municipio", "mes"],
    how="left",
    validate="one_to_one"
)

# Meses sem detecções registradas recebem zero
base_mensal["qtd_focos"] = (
    base_mensal["qtd_focos"]
    .fillna(0)
    .astype(int)
)

# 8. Organizar e salvar
base_mensal = base_mensal.sort_values(
    ["estado", "municipio", "mes"]
).reset_index(drop=True)

PASTA_PROCESSED.mkdir(parents=True, exist_ok=True)

caminho_saida = PASTA_PROCESSED / "focos_mensais.csv"

base_mensal.to_csv(caminho_saida, index=False)

# 9. Verificações
print("Dimensões:", base_mensal.shape)

print("Municípios:", len(municipios))

print("Meses:", len(meses))

print("Meses com zero detecções:",
      (base_mensal["qtd_focos"] == 0).sum())

print("Total de detecções na base mensal:",
      base_mensal["qtd_focos"].sum())

print("\nPrimeiras linhas:")
print(base_mensal.head())

print("\nArquivo salvo em:", caminho_saida)
