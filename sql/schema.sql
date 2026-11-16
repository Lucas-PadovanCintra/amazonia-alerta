CREATE TABLE IF NOT EXISTS focos_mensais (
    estado VARCHAR(50) NOT NULL,
    municipio VARCHAR(150) NOT NULL,
    mes DATE NOT NULL,
    qtd_focos INTEGER NOT NULL,

    CONSTRAINT pk_focos_mensais
        PRIMARY KEY (estado, municipio, mes),

    CONSTRAINT chk_qtd_focos_nao_negativa
        CHECK (qtd_focos >= 0)
);
