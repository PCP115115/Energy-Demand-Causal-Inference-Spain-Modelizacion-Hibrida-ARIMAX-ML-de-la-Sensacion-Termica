library(ggplot2)
library(dplyr)

# 1. Extraemos los residuos y los datos usados
# Aseguramos que las longitudes coincidan usando los mismos datos filtrados
datos_grafico <- dt %>%
  select(dif7_demanda, dif7_entalpia) %>%
  na.omit()

# Añadimos los residuos del modelo 2 a este dataframe
# OJO: Asegúrate de que 'modelo_armax_2' es el modelo de entalpía
datos_grafico$Residuos <- as.numeric(residuals(modelo_armax_2))

# 2. Generamos el Gráfico
grafico_diagnostico <- ggplot(datos_grafico, aes(x = dif7_entalpia, y = Residuos)) +
  geom_point(alpha = 0.5, color = "grey30") + # Puntos dispersos
  
  # Línea roja de referencia (el cero perfecto)
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", size = 1) +
  
  # Línea azul suavizada (LOESS) que muestra la TENDENCIA del error
  geom_smooth(method = "loess", color = "blue", se = TRUE, size = 1.2) +
  
  labs(title = "Evidencia de Mala Especificación Funcional",
       subtitle = "Patrón sistemático en los residuos vs. Entalpía",
       x = "Diferencial de Entalpía (Variables Explicativa)",
       y = "Residuos del Modelo ARMAX") +
  theme_minimal()

print(grafico_diagnostico)

#prueba VIF para las variables semiexógenas del modelo no loineal rival:
modelo_para_probar_vif_semiexogeno <- lm(dt$dif7_demanda~dt$dif7_entalpia_sq+dt$dif7_entalpia_cu, data=dt)
vif_semiexogeno <- vif(modelo_para_probar_vif_semiexogeno)
print(vif_semiexogeno)













grafico_realidad <- ggplot(datos_grafico, aes(x = dif7_entalpia, y = dif7_demanda)) +
  geom_point(alpha = 0.4) +
  
  # Intentamos ajustar una RECTA (lo que hace tu modelo actual) en ROJO
  geom_smooth(method = "lm", color = "red", se = FALSE, linetype = "dashed") +
  
  # Ajustamos una CURVA flexible (lo que haría un GAM) en AZUL
  geom_smooth(method = "gam", color = "blue", size = 1.2) +
  
  labs(title = "Relación No Lineal: Demanda vs Entalpía",
       subtitle = "Comparación: Ajuste Lineal (Rojo) vs. Flexible (Azul)",
       x = "Diferencial de Entalpía",
       y = "Diferencial de Demanda") +
  theme_minimal()

print(grafico_realidad)

