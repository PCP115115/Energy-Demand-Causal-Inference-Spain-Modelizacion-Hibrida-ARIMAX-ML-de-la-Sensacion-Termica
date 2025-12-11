import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from econml.dml import CausalForestDML

# Configuración visual
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ---------------------------------------------------------
# 1. CARGA DE DATOS
# ---------------------------------------------------------
file_path = "C:\\Users\\pedro\\Desktop\\D.Elec-Tiempo\\Importantes\\dataset_bruto_conjunto_final.csv"
df = pd.read_csv(file_path)
df['fecha'] = pd.to_datetime(df['fecha'])

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------
# T: Distancia a zona de confort
def calcular_distancia(entalpia):
    if entalpia < 35:
        return 35 - entalpia
    elif 35 <= entalpia <= 45:
        return 0
    else:
        return entalpia - 45

df['distancia_confort'] = df['Entalpia'].apply(calcular_distancia)

# X: Heterogeneidad
df['day_of_week'] = df['fecha'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['month'] = df['fecha'].dt.month
def get_season(month):
    if month in [12, 1, 2]: return 'Invierno'
    elif month in [3, 4, 5]: return 'Primavera'
    elif month in [6, 7, 8]: return 'Verano'
    else: return 'Otoño'
df['season'] = df['month'].apply(get_season)

# Dummies para el modelo
season_dummies = pd.get_dummies(df['season'], prefix='season', drop_first=False)
df = pd.concat([df, season_dummies], axis=1)

# ---------------------------------------------------------
# 3. ENTRENAMIENTO ROBUSTO (FULL DATASET)
# ---------------------------------------------------------
# Definimos variables
feature_cols = ['is_weekend', 'month'] + [col for col in df.columns if 'season_' in col]
X = df[feature_cols]
Y = df['demanda_MW']
T = df['distancia_confort']

print(f"Entrenando modelo Causal Forest con el dataset completo ({len(df)} registros)...")
print("Nota: EconML usa 'Cross-Fitting' y 'Honest Splitting' internamente para evitar overfitting.")

# Modelos Base (Nuisance)
# Usamos XGBoost. EconML hará cross-validation interna con ellos.
model_y = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model_t = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# Causal Forest DML
est = CausalForestDML(
    model_y=model_y,
    model_t=model_t,
    discrete_treatment=False,
    n_estimators=300,       # Aumentamos un poco para más robustez
    min_samples_leaf=10,
    cv=3,                   # 3-Fold Cross-Fitting interno (PROTECCIÓN CONTRA OVERFITTING)
    random_state=42
)

# Ajustamos al dataset completo
est.fit(Y, T, X=X, W=None)

# ---------------------------------------------------------
# 4. EXTRACCIÓN DE RESULTADOS (CATE)
# ---------------------------------------------------------
# Calculamos el efecto para cada día
cate_preds = est.effect(X)
df['Efecto_CATE'] = cate_preds

# ---------------------------------------------------------
# 5. TABLAS DE RESULTADOS EXACTOS
# ---------------------------------------------------------
print("\n" + "="*70)
print(" RESULTADOS HISTÓRICOS COMPLETOS (2021-2025)")
print(" Interpretación: Aumento de Demanda (MW) por cada 1 unidad de distancia al confort")
print("="*70)

# --- A. Tabla por Estación ---
tabla_estacion = df.groupby('season')['Efecto_CATE'].agg(
    Impacto_Promedio_MW=('mean'),
    Desviacion=('std'),
    Dias_Totales=('count')
).sort_values('Impacto_Promedio_MW', ascending=False)

print("\n>>> 1. IMPACTO POR ESTACIÓN:")
print(tabla_estacion.round(2))

# --- B. Tabla por Mes ---
tabla_mes = df.groupby('month')['Efecto_CATE'].agg(
    Impacto_Promedio_MW=('mean')
).sort_index()
nombres_meses = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio', 
                 7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
tabla_mes.index = tabla_mes.index.map(nombres_meses)

print("\n>>> 2. IMPACTO POR MES:")
print(tabla_mes.round(2))

# --- C. Tabla Laborable vs Finde ---
df['Tipo_Dia'] = df['is_weekend'].map({0: 'Entre Semana', 1: 'Fin de Semana'})
tabla_finde = df.groupby('Tipo_Dia')['Efecto_CATE'].agg(
    Impacto_Promedio_MW=('mean'),
    Desviacion=('std')
)
print("\n>>> 3. IMPACTO TIPO DE DÍA:")
print(tabla_finde.round(2))

# ---------------------------------------------------------
# 6. GRÁFICOS PARA PAPER (SET COMPLETO DE 3 FIGURAS)
# ---------------------------------------------------------

# Configuración global estética
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# --- PREPARACIÓN DE DATOS (Común para todos los gráficos) ---
df_plot = df.copy()

# Mapeos bonitos
mapa_finde = {0: "Día Laborable", 1: "Fin de Semana"}
df_plot['Tipo de Día'] = df_plot['is_weekend'].map(mapa_finde)
df_plot['Estación'] = df_plot['season'] # Ya está en español
orden_estaciones = ['Invierno', 'Primavera', 'Verano', 'Otoño']

# =========================================================
# FIGURA 1: DISTRIBUCIÓN DEL EFECTO (HISTOGRAMA)
# =========================================================
plt.figure(figsize=(10, 6))
sns.histplot(data=df_plot, x="Efecto_CATE", kde=True, color="#444444", element="step")
plt.axvline(df_plot['Efecto_CATE'].mean(), color='red', linestyle='--', label='Media Global')
plt.axvline(0, color='black', linewidth=1)

plt.title("Distribución del Efecto Causal Estimado (CATE)", weight='bold', pad=15)
plt.xlabel("Aumento de Demanda (MW)")
plt.ylabel("Frecuencia (Días)")
sns.despine()
plt.legend()
plt.tight_layout()
plt.show()

# =========================================================
# FIGURA 2: IMPACTO SOLO POR ESTACIÓN (LIMPIO)
# =========================================================
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_plot, 
    x="Estación", 
    y="Efecto_CATE", 
    order=orden_estaciones,
    palette="Greys", # Escala de grises elegante
    capsize=0.1,
    errorbar=('ci', 95)
)
plt.axhline(0, color='black', linewidth=0.8)
plt.title("Impacto Promedio por Estación (Sin desglosar días)", weight='bold', pad=15)
plt.ylabel("Aumento de Demanda (MW)")
plt.xlabel("")
sns.despine(left=True)
plt.tight_layout()
plt.show()

# =========================================================
# FIGURA 3: LA GRÁFICA DE INTERACCIÓN (LA IMPORTANTE)
# =========================================================
plt.figure(figsize=(10, 6))

ax = sns.pointplot(
    data=df_plot, 
    x="Estación", 
    y="Efecto_CATE", 
    hue="Tipo de Día", 
    order=orden_estaciones,
    markers=["o", "s"],
    linestyles=["-", "--"], 
    capsize=0.1, 
    dodge=0.2, 
    palette=["#333333", "#888888"], 
    errorbar=('ci', 95)
)

plt.axhline(0, color='black', linewidth=0.8, linestyle=':', alpha=0.5)
plt.title("Heterogeneidad del Efecto: Estación vs. Tipo de Día", fontsize=14, pad=20, weight='bold')
plt.ylabel("Aumento de Demanda (MW)")
plt.xlabel("")
sns.despine(left=True, bottom=True)
plt.legend(title=None, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.tight_layout()
plt.show()

# =========================================================
# FIGURA 4: DETALLE MENSUAL DEL IMPACTO (NUEVA)
# =========================================================
plt.figure(figsize=(12, 6))

# 1. Aseguramos que tenemos la columna de mes y creamos etiquetas legibles
if 'month' not in df_plot.columns:
    df_plot['month'] = df['month'] # Recuperamos la columna original si falta

mapa_meses = {
    1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
}
df_plot['Nombre_Mes'] = df_plot['month'].map(mapa_meses)

# 2. Creamos el gráfico de barras mensual
sns.barplot(
    data=df_plot,
    x="Nombre_Mes",
    y="Efecto_CATE",
    # Ordenamos explícitamente para que no salga alfabético (Abril primero)
    order=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
           'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'],
    palette="Greys",      # Misma paleta profesional que las anteriores
    capsize=0.1,          # Remates en las barras de error
    errorbar=('ci', 95)   # Intervalo de confianza del 95%
)

# 3. Detalles finales
plt.axhline(0, color='black', linewidth=0.8)
plt.title("Evolución Mensual del Impacto Causal en la Demanda", weight='bold', pad=15)
plt.ylabel("Aumento de Demanda (MW)")
plt.xlabel("") # Se entiende por las etiquetas
sns.despine(left=True)

plt.tight_layout()
plt.show()