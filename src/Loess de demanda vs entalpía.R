library(ggplot2)
ggplot(data=dt, aes(x = dt$Entalpia, y = dt$demanda_MW)) + geom_smooth(method = "loess", span = 0.3, se = FALSE, color = "blue", linewidth = 1.2)
