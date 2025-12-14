Modelización de la Demanda Energética en España: Análisis de la Sensación Térmica e Impacto Climático  
**Descripción:**  
Este repositorio contiene el estudio completo y el código para modelar la demanda energética diaria en España utilizando variables climáticas y de calendario. La investigación integra econometría clásica (ARIMAX) con técnicas avanzadas de inferencia causal (Double Machine Learning y Causal Forests) para analizar cómo el confort térmico —medido mediante la entalpía— afecta al consumo eléctrico.

**Hallazgos Principales:**
- Existe una relación no lineal en forma de U entre la entalpía y la demanda energética, con una zona de confort óptima entre **35–45 kJ/kg**.
- Cada unidad de desviación de esta zona incrementa la demanda entre **112–134 MW**, siendo el impacto del calor en verano mayor que el frío en invierno.
- Los patrones semanales (mayor demanda los miércoles, menor los domingos) definen estructuralmente la demanda, pero la sensibilidad climática se mantiene constante entre días laborables y fines de semana.
- Aunque el clima es un factor significativo, los factores socioeconómicos e inerciales explican la mayor parte de la variabilidad de la demanda.

**Contenido:**
- Documento completo de investigación (PDF) con metodología, diagnóstico, desarrollo de modelos y conclusiones.
- Scripts en Python y R para preprocesamiento de datos, estimación de modelos (ARIMAX, DML, Causal Forests) y visualización.
- Datasets y código para reproducir el análisis y los resultados.

Este proyecto ofrece un enfoque robusto y multimetódico para la modelización de la demanda energética, partiendo de un exhaustivo y riguroso análisis de los datos estudiados, prosiguiendo por una modelización clásica (ARIMAX y derivados) y validando y aportando robustez al estudio con técnicas de ML y DL de vanguardia.

