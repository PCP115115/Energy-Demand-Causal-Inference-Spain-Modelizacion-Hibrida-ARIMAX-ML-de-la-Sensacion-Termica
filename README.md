# Energy-Demand-Causal-Inference-Spain-Modelizacion-Hibrida-ARIMAX-ML-de-la-Sensacion-Termica
Este proyecto de investigación analiza la relación causal entre la sensación térmica y la demanda energética diaria en España (2021-2025). El estudio desafía el uso tradicional de la temperatura simple, aplicando técnicas de ML y DL junto a métodos de econometría clásica, proponiendo una nueva variable validada empíricamente midiendo estrés térmico

Metodología Híbrida:El análisis triangula resultados mediante dos enfoques metodológicos:
  Econometría Clásica: Modelos ARIMAX con componentes estacionales y dummies de calendario para capturar la inercia estructural de la demanda.
  Inferencia Causal (Machine Learning): Implementación de Double Machine Learning (DML) con Redes Temporales Convolucionales (TCNs) y Causal Forests para aislar el efecto puro del clima y capturar no-linealidades complejas.

Principales Hallazgos:
  Relación No Lineal en "U": Se identifica empíricamente una "Zona de Confort" entre 35 y 45 kJ/kg. Fuera de este rango, la demanda se dispara.
  Cuantificación del Impacto: El modelo DML estima un efecto causal de 134.10 MW adicionales de demanda por cada unidad (kJ/kg) de desviación del confort, refinando la estimación lineal de 112.2 MW.
  Asimetría Estacional: El sistema es más sensible al calor. El impacto marginal en verano (274.67 MW) supera ampliamente al de invierno (184.24 MW), sugiriendo una electrificación más intensa en refrigeración que en calefacción.
  Elasticidad Calendario-Clima: Aunque el consumo base varía drásticamente entre días laborables y fines de semana, la reacción ante una ola de calor es inelástica al calendario.

Este repositorio contiene el código en R/Python, los datasets procesados y el paper completo con el diagnóstico de series temporales y validación de modelos.
