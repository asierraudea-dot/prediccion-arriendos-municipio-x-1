import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import joblib
import os
from datetime import datetime

# ==================== CONFIG ====================
st.set_page_config(
    page_title="SENA Predict",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 SENA Predict")
st.markdown("**Predicción de demanda de aprendices en programas de formación**")

# ==================== CARGA DE DATOS ====================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data_historico.csv')
    except:
        st.error("No se encontró data_historico.csv")
        st.stop()
    
    # Limpieza
    df['FECHA_INICIO_FICHA'] = pd.to_datetime(df['FECHA_INICIO_FICHA'], errors='coerce')
    df['AÑO'] = df['FECHA_INICIO_FICHA'].dt.year.fillna(df.get('AÑO', 2024))
    df['MES'] = df['FECHA_INICIO_FICHA'].dt.month
    return df

df = load_data()

# ==================== PREPROCESAMIENTO ====================
def prepare_features(df):
    df = df.copy()
    df['MUNICIPIO'] = df['NOMBRE_MUNICIPIO_CURSO']
    df['PROGRAMA'] = df['NOMBRE_PROGRAMA_FORMACION'].fillna('').str.lower()
    df['NIVEL'] = df['NIVEL_FORMACION']
    df['SECTOR'] = df['NOMBRE_NUEVO_SECTOR'].fillna('OTRO')
    df['MES_INICIO'] = df['MES'].fillna(1)
    return df

df_prep = prepare_features(df)

# ==================== ENTRENAMIENTO DEL MODELO ====================
@st.cache_resource
def train_model(df_prep):
    features = ['MUNICIPIO', 'PROGRAMA', 'NIVEL', 'SECTOR', 'MES_INICIO', 'DURACION_PROGRAMA']
    X = pd.get_dummies(df_prep[features], drop_first=True)
    y = df_prep['TOTAL_APRENDICES']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    return model, X.columns.tolist(), mae, r2

model, feature_names, mae, r2 = train_model(df_prep)

# Guardar modelo
if not os.path.exists('model.pkl'):
    joblib.dump((model, feature_names), 'model.pkl')

# ==================== SIDEBAR ====================
st.sidebar.header("📊 Métricas del Modelo")
st.sidebar.metric("MAE", f"{mae:.1f} aprendices")
st.sidebar.metric("R²", f"{r2:.3f}")

municipio_filtro = st.sidebar.selectbox("Filtrar por Municipio", ['Todos'] + sorted(df['NOMBRE_MUNICIPIO_CURSO'].unique()))

# ==================== TABS ====================
tab1, tab2, tab3 = st.tabs(["🔮 Predictor", "📈 Análisis Histórico", "🌍 Dashboard"])

with tab1:
    st.header("Predicción de Inscripciones")
    col1, col2 = st.columns(2)
    
    with col1:
        mun_sel = st.selectbox("Municipio", sorted(df['NOMBRE_MUNICIPIO_CURSO'].unique()), key="mun")
        prog_input = st.text_input("Nombre del Programa", "MANIPULACION HIGIENICA DE ALIMENTOS")
        nivel_sel = st.selectbox("Nivel de Formación", df['NIVEL_FORMACION'].unique())
    
    with col2:
        sector_sel = st.selectbox("Sector", sorted(df['NOMBRE_NUEVO_SECTOR'].dropna().unique()))
        duracion_sel = st.number_input("Duración del Programa (horas)", min_value=10, value=48, step=1)
        mes_sel = st.slider("Mes de Inicio", 1, 12, value=3)
    
    if st.button("🚀 Predecir Número de Aprendices", type="primary"):
        input_df = pd.DataFrame([{
            'MUNICIPIO': mun_sel,
            'PROGRAMA': prog_input.lower(),
            'NIVEL': nivel_sel,
            'SECTOR': sector_sel,
            'MES_INICIO': mes_sel,
            'DURACION_PROGRAMA': duracion_sel
        }])
        
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)
        
        prediccion = model.predict(input_encoded)[0]
        st.success(f"**{int(round(prediccion))} aprendices esperados**")
        st.balloons()

with tab2:
    st.header("Datos Históricos")
    if municipio_filtro != 'Todos':
        df_view = df[df['NOMBRE_MUNICIPIO_CURSO'] == municipio_filtro]
    else:
        df_view = df
    st.dataframe(df_view, use_container_width=True)

with tab3:
    st.header("Tendencias Regionales")
    col1, col2 = st.columns(2)
    
    with col1:
        yearly = df.groupby('AÑO')['TOTAL_APRENDICES'].sum().reset_index()
        fig1 = px.line(yearly, x='AÑO', y='TOTAL_APRENDICES', title="Evolución Total de Aprendices")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        sector_trend = df.groupby(['AÑO', 'NOMBRE_NUEVO_SECTOR'])['TOTAL_APRENDICES'].sum().reset_index()
        fig2 = px.bar(sector_trend, x='AÑO', y='TOTAL_APRENDICES', color='NOMBRE_NUEVO_SECTOR', title="Por Sector")
        st.plotly_chart(fig2, use_container_width=True)

st.caption("App desarrollada a partir del histórico PE04_HISTÓRICO_PREVIOS.xlsx")
