#En este modelo utilizamos la entalpia como distancia a zona de confort como en el paper.
# ==========================================
# BLOQUE 1. CARGA DE LIBRERIAS Y CONFIGURACIÓN
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import statsmodels.api as sm

# Estilo y Semillas para reproducibilidad
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper")
np.random.seed(42)
tf.random.set_seed(42)

print("Bloque 1: Librerías cargadas. Configuración lista.")

# ==========================================
# BLOQUE 2. CARGAR DATASET
# ==========================================
# Ajusta la ruta si es necesario
file_path = r"C:\Users\pedro\Desktop\D.Elec-Tiempo\Importantes\dataset_bruto_conjunto_final.csv"

try:
    df = pd.read_csv(file_path)
    print(f"Dataset cargado desde ruta local: {df.shape}")
except FileNotFoundError:
    print("Ruta específica no encontrada, buscando en directorio actual...")
    df = pd.read_csv("dataset_bruto_conjunto_final.csv")

# IMPORTANTE: Ordenar cronológicamente
df['fecha'] = pd.to_datetime(df['fecha'])
df = df.sort_values('fecha').reset_index(drop=True)

# ==========================================
# BLOQUE 3, 4 y 5 (CORREGIDO Y REORDENADO): FEATURE ENGINEERING, ESCALADO Y SECUENCIAS
# ==========================================

# --- 1. CREACIÓN DE VARIABLES (Feature Engineering) ---

# A) Transformación "Distancia al Confort"
def calcular_distancia_confort(h):
    if h < 35:
        return abs(35 - h) # Estrés por frío
    elif 35 <= h <= 45:
        return 0.0         # Zona de confort
    else:
        return h - 45      # Estrés por calor

df['Distancia_Confort'] = df['Entalpia'].apply(calcular_distancia_confort)

# B) Variables Dummy (Día de la semana) - ESTO DEBE IR ANTES DE DEFINIR control_cols
df['dia_semana'] = df['fecha'].dt.dayofweek
dummies_dia = pd.get_dummies(df['dia_semana'], prefix='dia', drop_first=False)
# Unimos las dummies al dataframe principal si no están ya
df = pd.concat([df, dummies_dia], axis=1)

# --- 2. DEFINICIÓN DE ROLES ---

target_col = 'demanda_MW'
treatment_col = 'Distancia_Confort'
# AHORA SÍ podemos usar dummies_dia.columns porque ya existe
control_cols = ['Inercia_Termica', 'GSP'] + dummies_dia.columns.tolist()

print(f"Bloque 3: Variables listas. Controles seleccionados: {len(control_cols)}")

# --- 3. DIVISIÓN TEMPORAL (Train/Test) ---

# Definimos el punto de corte (80% train, 20% test)
train_size = int(len(df) * 0.8)
WINDOW_SIZE = 30 # Ventana de memoria de la TCN

# Dataset de Entrenamiento (Estricto)
df_train = df.iloc[:train_size].copy()

# Dataset de Test con "Buffer" (Memoria)
# Cogemos los datos de test, pero le pegamos los 30 días anteriores del train
# para evitar perder datos al inicio del test (Data Leakage solucionado correctamente)
df_test_buffered = df.iloc[train_size - WINDOW_SIZE:].copy()

print(f"Corte temporal realizado. Train hasta: {df_train['fecha'].max()}")
print(f"Test comienza a predecir desde: {df['fecha'].iloc[train_size]}")

# --- 4. ESCALADO (Fit solo en Train) ---

scaler_x = StandardScaler()
scaler_y = StandardScaler()
scaler_t = StandardScaler()

# FIT: Aprendemos la media y desviación SOLO de los datos de entrenamiento
scaler_x.fit(df_train[control_cols])
scaler_y.fit(df_train[[target_col]])
scaler_t.fit(df_train[[treatment_col]])

# TRANSFORM: Aplicamos esa media a los datos (Train y Test)
X_train_arr = scaler_x.transform(df_train[control_cols])
Y_train_arr = scaler_y.transform(df_train[[target_col]])
T_train_arr = scaler_t.transform(df_train[[treatment_col]])

X_test_arr = scaler_x.transform(df_test_buffered[control_cols])
Y_test_arr = scaler_y.transform(df_test_buffered[[target_col]])
T_test_arr = scaler_t.transform(df_test_buffered[[treatment_col]])

# --- 5. GENERACIÓN DE SECUENCIAS (Tensores 3D) ---

def create_sequences(X, Y, T, window_size):
    Xs, Ys, Ts = [], [], []
    # Recorremos desde el inicio hasta len - window
    for i in range(len(X) - window_size):
        Xs.append(X[i:(i + window_size)])
        Ys.append(Y[i + window_size])
        Ts.append(T[i + window_size])
    return np.array(Xs), np.array(Ys), np.array(Ts)

# Generamos secuencias por separado
X_train, Y_train, T_train = create_sequences(X_train_arr, Y_train_arr, T_train_arr, WINDOW_SIZE)
X_test, Y_test, T_test = create_sequences(X_test_arr, Y_test_arr, T_test_arr, WINDOW_SIZE)

print("\n--- RESUMEN DE DIMENSIONES (CORREGIDO) ---")
print(f"X_train shape: {X_train.shape} (Muestras, Ventana, Variables)")
print(f"X_test shape:  {X_test.shape}")
print("Nota: El Data Leakage ha sido eliminado. El orden de ejecución es correcto.")

# ==========================================
# BLOQUE 6. ARQUITECTURA TCN CON MC DROPOUT
# ==========================================
def build_tcn_mc_dropout(input_shape, dropout_rate=0.2):
    inputs = layers.Input(shape=input_shape)
    
    # Capa 1: Dilatación 1
    x = layers.Conv1D(64, 3, dilation_rate=1, padding='causal', activation='relu')(inputs)
    x = layers.Dropout(dropout_rate)(x, training=True) # training=True activa MC Dropout
    
    # Capa 2: Dilatación 2
    x = layers.Conv1D(64, 3, dilation_rate=2, padding='causal', activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x, training=True)
    
    # Capa 3: Dilatación 4
    x = layers.Conv1D(64, 3, dilation_rate=4, padding='causal', activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x, training=True)
    
    # Capa 4: Dilatación 8 (Visión de ~25-30 días atrás)
    x = layers.Conv1D(64, 3, dilation_rate=8, padding='causal', activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x, training=True)
    
    # Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x, training=True)
    
    outputs = layers.Dense(1)(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    # Learning rate bajo para estabilidad
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss='mse')
    return model

# Instanciar modelos
model_y = build_tcn_mc_dropout((WINDOW_SIZE, X_train.shape[2]))
model_t = build_tcn_mc_dropout((WINDOW_SIZE, X_train.shape[2]))

print("Bloque 6: Arquitecturas compiladas correctamente.")

# ==========================================
# BLOQUE 7. ENTRENAMIENTO
# ==========================================
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
EPOCHS = 45
BATCH_SIZE = 32

print("\n--- Entrenando Modelo Nuisance Y (Demanda) ---")
hist_y = model_y.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                     validation_split=0.1, callbacks=[early_stop], verbose=0)
print(f"Modelo Y completado. Loss final: {hist_y.history['loss'][-1]:.4f}")

print("\n--- Entrenando Modelo Nuisance T (Confort) ---")
hist_t = model_t.fit(X_train, T_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                     validation_split=0.1, callbacks=[early_stop], verbose=0)
print(f"Modelo T completado. Loss final: {hist_t.history['loss'][-1]:.4f}")

# ==========================================
# BLOQUE 8. INFERENCIA MONTE CARLO (AVERAGING)
# ==========================================
#IMPORTANTE: Para cuando queramos sacar el resultado final de la estimación, para que no varie mucho, hacemos una iteracion en las simulacioens
# de monte-carlo de 500 o incluso de 1000 para que asi el resultado sea mas estable.
# Hay que tener en cuenta que tardará un ratillo en cargar asi que, hazlo cuando ya sea para pasarlo al paper.
# Esta función es la clave para que el MC Dropout funcione como técnica bayesiana
def predict_mc_dropout(model, X, n_iter=50):
    print(f"   -> Ejecutando {n_iter} pases estocásticos...")
    preds = []
    # Realizamos n_iter predicciones con distintas "máscaras" de dropout
    for _ in range(n_iter):
        preds.append(model.predict(X, verbose=0))
    
    # Promediamos los resultados (E[Y|X])
    preds = np.array(preds)
    mean_preds = preds.mean(axis=0)
    return mean_preds

print(f"\nGenerando predicciones robustas (MC Dropout)...")
y_pred_robust = predict_mc_dropout(model_y, X_test, n_iter=500)
t_pred_robust = predict_mc_dropout(model_t, X_test, n_iter=500)

# Diagnóstico de R2
r2_y = r2_score(Y_test, y_pred_robust)
r2_t = r2_score(T_test, t_pred_robust)
print(f"\nDIAGNÓSTICO FINAL:")
print(f"R^2 Demanda (MC Robust): {r2_y:.4f}")
print(f"R^2 Distancia Confort (MC Robust): {r2_t:.4f}")

# ==========================================
# BLOQUE 9. CÁLCULO DE EFECTO CAUSAL
# ==========================================
# Cálculo de Residuos (Variación no explicada por controles)
res_y = Y_test - y_pred_robust
res_t = T_test - t_pred_robust

# Regresión FWL con errores HAC (Newey-West)
# maxlags=30 ajustado a la ventana temporal
ols_model = sm.OLS(res_y, res_t).fit(cov_type='HAC', cov_kwds={'maxlags': WINDOW_SIZE})

print("\n" + "="*50)
print("RESULTADOS FINAL DOUBLE MACHINE LEARNING")
print("="*50)
print(ols_model.summary())

# Interpretación
coef_causal = ols_model.params[0]
pval = ols_model.pvalues[0]
std_y = scaler_y.scale_[0]
std_t = scaler_t.scale_[0]
efecto_mw = coef_causal * (std_y / std_t)

print(f"\n--- INTERPRETACIÓN DE NEGOCIO ---")
print(f"Por cada 1 kJ/kg que nos alejamos de la zona de confort (35-45),")
print(f"la demanda aumenta: {efecto_mw:.2f} MW")
print(f"P-Valor (Robust): {pval:.6f}")

if pval < 0.05:
    print(">> CONCLUSIÓN: Relación ESTADÍSTICAMENTE SIGNIFICATIVA.")
else:
    print(">> CONCLUSIÓN: No significativa (revisar datos/modelo).")

# ==========================================
# BLOQUE 10. VISUALIZACIÓN FINAL
# ==========================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Validación: Nueva Variable vs Demanda Real")
plt.scatter(df['Distancia_Confort'], df['demanda_MW'], alpha=0.1, c='green')
plt.xlabel("Distancia a Confort (kJ/kg)")
plt.ylabel("Demanda (MW)")

plt.subplot(1, 2, 2)
plt.title(f"Efecto Causal Aislado (Pendiente={efecto_mw:.1f})")
plt.scatter(res_t, res_y, alpha=0.5, c='teal', s=15, label='Residuos')
# Línea de tendencia
x_vals = np.linspace(res_t.min(), res_t.max(), 100)
plt.plot(x_vals, x_vals * coef_causal, 'r--', lw=2, label='Efecto Causal')
plt.xlabel("Residuos Distancia Confort")
plt.ylabel("Residuos Demanda")
plt.legend()

plt.tight_layout()
plt.show()