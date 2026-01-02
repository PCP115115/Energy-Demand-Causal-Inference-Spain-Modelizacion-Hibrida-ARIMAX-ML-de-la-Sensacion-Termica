library(dplyr)
library(dplyr)
library(forecast)
library(lmtest)
library(ggplot2)
library(ggfortify)

#Diferenciación semanal de las variables x e y:

dt <- dt %>%
  arrange(fecha) %>% 
  mutate(
    dif7_demanda  = demanda_MW - lag(demanda_MW, n = 7),
    dif7_gsp      = GSP - lag(GSP, n = 7),
    dif7_inercia  = Inercia_Termica - lag(Inercia_Termica, n = 7),
    dif7_entalpia = Entalpia - lag(Entalpia, n = 7)
  ) %>%
  na.omit() 


#Modelo de regresion multiple entero para VIF:
library(car)
library(lmtest)
library(sandwich)
regresion_multiple_diff7_entero <- lm(dt$dif7_demanda~dif7_gsp+dif7_inercia+dif7_entalpia, data=dt)
print(summary(regresion_multiple_diff7_entero))
print(vif(regresion_multiple_diff7_entero))
print(dwtest(regresion_multiple_diff7_entero))
print(bgtest(regresion_multiple_diff7_entero, order = 7))
print(bptest(regresion_multiple_diff7_entero))
errores_hac <- NeweyWest(regresion_multiple_diff7_entero, lag = 7) 

coeftest(regresion_multiple_diff7_entero, vcov = errores_hac)

#Corremos dos regresiones múltiples
#1. Con IT
regresion_multiple_diff7_con_it <- lm(dt$dif7_demanda~dif7_gsp+dif7_inercia, data=dt)
print(summary(regresion_multiple_diff7_con_it))
library(lmtest)
print("Test DW (autocorrelación diaria)")
print(dwtest(regresion_multiple_diff7_con_it))
print("Test BG (autocorrelación SEMANAL")
print(bgtest(regresion_multiple_diff7_con_it, order=7))

#2.con h
regresion_multiple_diff7_con_h <- lm(dt$dif7_demanda~dif7_gsp+dif7_entalpia, data=dt)
print(summary(regresion_multiple_diff7_con_h))
print(dwtest(regresion_multiple_diff7_con_h))
print(bgtest(regresion_multiple_diff7_con_h))

#Modelo ARMAX con las diferenciacios semanales en su totalidad:

mis_regresores <- cbind(dt$dif7_gsp, dt$dif7_inercia, dt$dif7_entalpia)


modelo_armax_entero <- auto.arima(
  dt$dif7_demanda,
  xreg = mis_regresores,  # <--- AQUÍ está la clave
  d = 0,
  seasonal = TRUE,
  stepwise = FALSE,
  approximation = FALSE,
  lambda = TRUE # Ojo: Si te da error, cambia TRUE por "auto"
)
print(summary(modelo_armax_entero))
print(coeftest(modelo_armax_entero))
print(checkresiduals(modelo_armax_entero))





p1 <- autoplot(residuals(modelo_armax_entero), ts.colour = 'black', ts.linetype = 'solid') +
  geom_hline(yintercept = 0, color = "blue", linetype = "dashed") +
  labs(title = "Residuos ARMAX Modelo entero",
       subtitle = "Evolución temporal para detectar heterocedasticidad",
       y = "Residuals", x = "Tiempo") +
  theme_minimal()
print(p1)

# 6.2 QQ-Plot
dev.new() # Abre una nueva ventana gráfica si es necesario
qqnorm(residuals(modelo_armax_entero), main = "QQ-Plot Modelo armax entero")



#MODELO ARIMAX PERO SOLO CON LA ENTALPIA:
mis_regresores_segundo <- cbind(Entalpia = dt$dif7_entalpia) 

modelo_armax_segundo <- auto.arima(
  dt$dif7_demanda,
  xreg = mis_regresores_segundo,  
  d = 0,
  seasonal = TRUE,
  stepwise = FALSE,
  approximation = FALSE,
  lambda = "auto"   # <--- AQUÍ: Con comillas
)

print(summary(modelo_armax_segundo))
print(coeftest(modelo_armax_segundo))
print(checkresiduals(modelo_armax_segundo))











































































# Gráfico de dispersión: Diferencia Demanda vs Diferencia Entalpía
ggplot(dt, aes(x = dif7_entalpia, y = dif7_demanda)) +
  geom_point(alpha = 0.3, color = "darkblue") + # Puntos semitransparentes
  geom_smooth(method = "lm", color = "red", se = FALSE) + # Línea recta (Tu modelo actual)
  geom_smooth(method = "loess", color = "green", se = FALSE) + # Línea flexible (Realidad)
  labs(title = "Análisis de No Linealidad",
       subtitle = "Rojo: Asunción del modelo lineal | Verde: Tendencia real de los datos",
       x = "Variación de Entalpía (dif7)",
       y = "Variación de Demanda (dif7)") +
  theme_minimal()

library(lmtest)
library(ggplot2)

# Crear el gráfico
p_nolineal <- ggplot(dt, aes(x = dif7_entalpia, y = dif7_demanda)) +
  # 1. Los datos reales (puntos)
  geom_point(alpha = 0.3, color = "gray50") + 
  
  # 2. Tu modelo actual (Línea Roja - Recta)
  geom_smooth(method = "lm", aes(color = "Modelo Lineal (Actual)"), se = FALSE, size = 1) + 
  
  # 3. La realidad (Línea Azul - Curva)
  geom_smooth(method = "loess", aes(color = "Tendencia Real (Loess)"), se = FALSE, size = 1) +
  
  # Líneas de referencia
  geom_vline(xintercept = 0, linetype = "dashed", alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
  
  # Colores y títulos
  scale_color_manual(name = "Comparación", values = c("Modelo Lineal (Actual)" = "red", "Tendencia Real (Loess)" = "blue")) +
  labs(
    title = "Evidencia de No Linealidad",
    subtitle = "Divergencia entre el ajuste lineal y el comportamiento real de la demanda",
    x = "Variación de Entalpía (Semanal)",
    y = "Variación de Demanda (Semanal)",
    caption = "Nota: La curva azul indica que la respuesta de la demanda cambia según la intensidad del clima."
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

# Mostrar gráfico
print(p_nolineal)

modelo_lineal_auxiliar <- lm(dif7_demanda ~ dif7_entalpia, data = dt)

# 2. Ejecutamos el Ramsey RESET Test
#    power = 2:3 -> Probamos si faltan términos al cuadrado (x^2) o al cubo (x^3)
#    type = "fitted" -> Es la forma estándar del test
resultado_reset <- resettest(modelo_lineal_auxiliar, power = 2:3, type = "fitted")

print(resultado_reset)




# 1. Crear las variables de potencia (La "ingeniería de características")
#    Elevamos al cuadrado y al cubo la variable diferenciada
dt$dif7_entalpia_sq <- dt$dif7_entalpia^2
dt$dif7_entalpia_cu <- dt$dif7_entalpia^3

# 2. Agruparlas en la matriz de regresores
#    Esto le dice al modelo: "Usa la recta, la curva U y la curva S a la vez"
mis_regresores_cubico <- cbind(
  Lineal   = dt$dif7_entalpia, 
  Cuadrado = dt$dif7_entalpia_sq,
  Cubico   = dt$dif7_entalpia_cu
)

# 3. Entrenar el modelo ARMAX
modelo_armax_cubico <- auto.arima(
  dt$dif7_demanda,
  xreg = mis_regresores_cubico,
  d = 0,             # Ya está diferenciada
  seasonal = TRUE,   # Busca patrones estacionales en los residuos
  stepwise = FALSE,  # Búsqueda exhaustiva (más lento pero mejor)
  approximation = FALSE,
  lambda = NULL      # IMPORTANTE: Quitamos Box-Cox para que no interfiera con el polinomio
)

# 4. Ver resultados
print(summary(modelo_armax_cubico))
print(coeftest(modelo_armax_cubico))

# 5. Comprobar si los residuos mejoraron (esperamos que el p-valor de Ljung-Box suba)
print(checkresiduals(modelo_armax_cubico))


