-- 1. Total de detecções por ano
SELECT
    EXTRACT(YEAR FROM mes)::INTEGER AS ano,
    SUM(qtd_focos) AS total_focos
FROM focos_mensais
GROUP BY ano
ORDER BY ano;


-- 2. Total de detecções por estado
SELECT
    estado,
    SUM(qtd_focos) AS total_focos
FROM focos_mensais
GROUP BY estado
ORDER BY total_focos DESC;


-- 3. Distribuição das detecções por mês do ano
SELECT
    EXTRACT(MONTH FROM mes)::INTEGER AS numero_mes,
    SUM(qtd_focos) AS total_focos
FROM focos_mensais
GROUP BY numero_mes
ORDER BY numero_mes;
