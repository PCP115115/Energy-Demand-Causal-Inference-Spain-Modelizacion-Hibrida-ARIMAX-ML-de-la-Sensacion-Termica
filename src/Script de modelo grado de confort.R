# --- 1. LIBRERÍAS NECESARIAS ---
library(ggplot2)
library(dplyr)
library(tidyr)

# --- 2. PREPARACIÓN DE FUNCIONES ---

# Función para realizar Validación Cruzada (K-Fold CV)
calcular_cv_error <- function(data, formula, k = 5) {
  set.seed(123) 
  
  # Asignamos aleatoriamente cada fila a un grupo
  data$fold <- sample(1:k, nrow(data), replace = TRUE)
  errores_cuadraticos <- c()
  
  for(i in 1:k) {
    train <- data[data$fold != i, ]
    test <- data[data$fold == i, ]
    
    modelo <- lm(formula, data = train)
    predicciones <- predict(modelo, test)
    
    errores_cuadraticos <- c(errores_cuadraticos, mean((test$demanda_MW - predicciones)^2))
  }
  return(sqrt(mean(errores_cuadraticos)))
}

# --- 3. BUCLE DE OPTIMIZACIÓN DEL UMBRAL ---

umbrales <- 36:60

resultados <- data.frame(
  Umbral = umbrales,
  R2 = NA,
  AIC = NA,
  RMSE_CV = NA
)

print("Calculando métricas... Si vuelve a fallar, revisa nombres con 'names(dt)'")

for (i in 1:length(umbrales)) {
  ub <- umbrales[i]
  
  dt_temp <- dt
  
  # --- CORRECCIÓN AQUÍ: Usamos 'Entalpia' con E mayúscula ---
  
  # GC_frio: Valor absoluto de (35 - Entalpia) si Entalpia < 35, sino 0
  dt_temp$GC_frio <- pmax(35 - dt_temp$Entalpia, 0)
  
  # GC_calor: Valor absoluto de (Entalpia - Umbral) si Entalpia > Umbral, sino 0
  dt_temp$GC_calor <- pmax(dt_temp$Entalpia - ub, 0)
  
  # --- AJUSTE DEL MODELO ---
  modelo <- lm(demanda_MW ~ GC_frio + GC_calor, data = dt_temp)
  
  # --- GUARDADO DE MÉTRICAS ---
  resultados$R2[i] <- summary(modelo)$r.squared
  resultados$AIC[i] <- AIC(modelo)
  
  # Validación cruzada
  resultados$RMSE_CV[i] <- calcular_cv_error(dt_temp, demanda_MW ~ GC_frio + GC_calor)
}

# --- 4. RESULTADOS Y GRÁFICO ---

mejor_aic <- resultados[which.min(resultados$AIC), ]
mejor_cv <- resultados[which.min(resultados$RMSE_CV), ]
mejor_r2 <- resultados[which.max(resultados$R2), ]

cat("\n--- RESULTADOS DEL ANÁLISIS ---\n")
cat("Mejor R2 (Ajuste):", mejor_r2$Umbral, "\n")
cat("Mejor AIC (Modelo):", mejor_aic$Umbral, "\n")
cat("Mejor CV (Predicción):", mejor_cv$Umbral, "\n")

# Gráfico
resultados_norm <- resultados %>%
  mutate(
    R2_Scaled = (R2 - min(R2)) / (max(R2) - min(R2)),
    AIC_Inverted = 1 - (AIC - min(AIC)) / (max(AIC) - min(AIC)),
    CV_Inverted = 1 - (RMSE_CV - min(RMSE_CV)) / (max(RMSE_CV) - min(RMSE_CV))
  ) %>%
  pivot_longer(
    cols = c(R2_Scaled, AIC_Inverted, CV_Inverted), 
    names_to = "Metrica", 
    values_to = "Valor"
  )

etiquetas_metrica <- c(
  "AIC_Inverted" = "AIC (Invertido)", 
  "CV_Inverted" = "Validación Cruzada (Invertido)", 
  "R2_Scaled" = "R-Cuadrado"
)

ggplot(resultados_norm, aes(x = Umbral, y = Valor, color = Metrica)) +
  geom_line(size = 1.2) +
  geom_vline(xintercept = 45, linetype = "dashed", color = "black") + 
  annotate("text", x = 46, y = 0.5, label = "Propuesta (45)", hjust = 0) +
  scale_color_manual(values = c("red", "blue", "darkgreen"), labels = etiquetas_metrica) +
  labs(
    title = "Optimización del Umbral Superior de Confort",
    x = "Límite Superior de Entalpía",
    y = "Desempeño Normalizado"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")
