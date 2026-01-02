library(forecast)
library(lmtest)
library(dplyr)

# Preparamos los datos
dt2 <- dt %>% 
  select(dif7_demanda, dif7_entalpia, dif7_gsp) %>%
  na.omit()

regresores_cambios <- cbind(
  entalpia = dt2$dif7_entalpia,
  gsp      = dt2$dif7_gsp
)

# Calculamos lambda
lambda_opt <- BoxCox.lambda(dt2$dif7_demanda)

# Modelo correctamente especificado
modelo_armax_cambiado <- auto.arima(
  dt2$dif7_demanda,
  xreg = regresores_cambios,
  d = 0,
  seasonal = TRUE,
  stepwise = FALSE,
  approximation = FALSE,
  lambda = lambda_opt
)

# Residuos y test de Ljung-Box
residuos <- residuals(modelo_armax_cambiado)

Box.test(residuos, type = "Ljung-Box", lag = 7)
modelo_armax_cambiado$lambda



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

