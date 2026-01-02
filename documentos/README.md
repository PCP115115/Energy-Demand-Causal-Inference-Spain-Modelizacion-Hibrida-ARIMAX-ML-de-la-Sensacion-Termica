# ⚡ Modelización Híbrida de la Demanda Energética en España: Inferencia Causal y Sensación Térmica

![Status](https://img.shields.io/badge/Status-En_Desarrollo-orange)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![R](https://img.shields.io/badge/R-Reciente-blue?logo=r&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> **Investigación 2021-2025**: Un enfoque multidisciplinar que integra **Econometría Clásica (ARIMAX)** con **Double Machine Learning (DML)** y **Redes Neuronales (TCN)** para aislar el impacto causal del confort térmico en la red eléctrica española.

---

## 📖 Descripción del Proyecto

Este repositorio contiene el código fuente, los datasets y la documentación de una investigación exhaustiva sobre la elasticidad de la demanda eléctrica en España frente a variables climáticas complejas.

A diferencia de los modelos tradicionales que usan la temperatura simple, este estudio introduce la **Entalpía** como métrica de sensación térmica y aplica técnicas de **Inferencia Causal** para cuantificar efectos no lineales, eliminando el sesgo introducido por variables de calendario e inercia socioeconómica.

### 🎯 Objetivos
1. **Desafiar** el uso de la temperatura seca como único predictor climático.
2. **Cuantificar** el impacto energético (en MW) de desviarse de la "Zona de Confort".
3. **Comparar** la robustez de modelos clásicos (ARIMA/Regresión) vs. modelos de vanguardia (Causal Forests/DML).

---

## 📊 Hallazgos Principales (Key Findings)

El análisis de datos y la modelización han revelado patrones críticos para la planificación energética:

* **🌡️ La "U" de la Demanda:** Existe una relación no lineal robusta. La demanda se minimiza en una **Zona de Confort de 35–45 kJ/kg** (entalpía).
* **📈 Impacto Causal:** Mediante Double Machine Learning, se estima que cada unidad (kJ/kg) de desviación fuera de la zona de confort incrementa la demanda en **~134.10 MW**.
* **⚖️ Asimetría Estacional:** El sistema es mucho más sensible al calor (**274 MW** marginales en verano) que al frío (**184 MW** en invierno).
* **🗓️ Elasticidad:** Aunque el consumo base varía drásticamente entre laborables y festivos, la sensibilidad al clima es inelástica al calendario.

---

## 🛠️ Metodología Híbrida

El proyecto triangula resultados utilizando dos enfoques metodológicos complementarios alojados en la carpeta `/src`:

### 1. Econometría Clásica (R)
* **Modelos ARIMAX:** Ajuste de series temporales con regresores exógenos.
* **Diagnóstico:** Tests de raíces unitarias, estacionalidad, y heterocedasticidad (Newey-West HAC).
* **Validación de No-Linealidad:** Tests de Ramsey RESET y curvas Loess.

### 2. Machine Learning & Inferencia Causal (Python)
* **Double Machine Learning (DML):** Uso del teorema FWL (Frisch-Waugh-Lovell) para ortogonalizar regresores.
* **TCN (Temporal Convolutional Networks):** Redes profundas con *dilated convolutions* para capturar dependencias a largo plazo (memoria de 30 días).
* **MC Dropout:** Implementación Bayesiana para estimar incertidumbre en las predicciones.

---

## 📂 Estructura del Repositorio

```text
├── 📁 documentos/         # Paper de investigación, licencias y documentación teórica
├── 📁 figuras/            # Gráficos generados (correlaciones, residuos, predicciones)
├── 📁 src/                # Código fuente del proyecto
│   ├── 🐍 Causal_forest_...py         # Script de Causal Forests
│   ├── 🐍 Doble_machine_learning...py # Implementación DML con TCN y TensorFlow
│   ├── 📊 Desarrollo de modelos...R   # Scripts de R para ARIMAX y tests estadísticos
│   ├── 📉 Graficación...R             # Scripts para generar visualizaciones (Ggplot2)
│   └── ...
├── 📁 otros/              # Datos brutos y archivos auxiliares
└── README.md              # Este archivo
