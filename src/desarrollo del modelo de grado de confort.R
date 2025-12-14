library(tidyverse)
library(lubridate)
library(forecast)
library(tseries)

# --- 1. PREPARACIÓN DE DATOS ---

# Ordenar por fecha y formatear
dt <- dt %>%
  mutate(fecha = as.Date(fecha)) %>%
  arrange(fecha)

dt_model <- dt %>%
  mutate(
    # Variable 'distancia_confort' (Función a trozos)
    distancia_confort = case_when(
      Entalpia < 35 ~ abs(Entalpia - 35), # Zona Frío
      Entalpia > 45 ~ abs(Entalpia - 45), # Zona Calor
      TRUE ~ 0                            # Zona Confort
    ),
    # Dummies: Forzamos 'ordered = FALSE' para obtener coeficientes clásicos
    dia_semana = factor(wday(fecha, label = TRUE, abbr = FALSE, week_start = 1), ordered = FALSE)
  )

# --- 2. MATRIZ DE REGRESORES (ARIMAX) ---

# Creamos la matriz de exógenas.
# model.matrix genera dummies (Lunes queda como base). Eliminamos la columna de intercepto [, -1]
xreg_matrix <- model.matrix(~ distancia_confort + dia_semana, data = dt_model)[, -1]

# Serie temporal (Frecuencia 7 = Semanal)
y_ts <- ts(dt_model$demanda_MW, frequency = 7)

# --- 3. ESTIMACIÓN DEL MODELO ---

# Ajuste riguroso (stepwise = FALSE)
modelo_arima <- auto.arima(
  y_ts,
  xreg = xreg_matrix,
  seasonal = TRUE,
  stepwise = FALSE,
  approximation = FALSE
)

# --- 4. RESULTADOS Y DIAGNÓSTICO ---

cat("=== RESUMEN DEL MODELO (Coeficientes Clásicos) ===\n")
# Ahora verás 'dia_semanaMartes', 'dia_semanaMiercoles', etc. (Vs Lunes)
summary(modelo_arima)

cat("\n=== TEST DE RESIDUOS (Ljung-Box) ===\n")
checkresiduals(modelo_arima)

cat("\n=== PRECISIÓN ===\n")
print(accuracy(modelo_arima))

# --- 5. GRÁFICA LOESS (SUAVIZADA) ---

dt_plot <- dt_model %>%
  mutate(fitted_values = fitted(modelo_arima))

ggplot(dt_plot, aes(x = fecha)) +
  # Curva Demanda Real
  geom_smooth(aes(y = demanda_MW, color = "Demanda Real"), 
              method = "loess", span = 0.1, se = FALSE, linewidth = 1) +
  # Curva Predicción
  geom_smooth(aes(y = fitted_values, color = "Predicción Modelo"), 
              method = "loess", span = 0.1, se = FALSE, linetype = "dashed", linewidth = 1) +
  scale_color_manual(values = c("Demanda Real" = "#2c3e50", "Predicción Modelo" = "#e74c3c")) +
  labs(
    title = "Ajuste del Modelo ARIMAX: Demanda Eléctrica (Suavizado)",
    subtitle = "Comparativa de tendencias mediante LOESS (span=0.1)",
    y = "Demanda (MW)", x = "Fecha", color = ""
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")
