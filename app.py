import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# CLASE DE ANÁLISIS (POO)
# ==============================

class DataAnalyzer:

    def __init__(self, df):
        self.df = df

    def valores_nulos(self):
        return self.df.isnull().sum()

    def estadisticas(self):
        return self.df.describe()

    def variables_numericas(self):
        return self.df.select_dtypes(include=np.number).columns

    def variables_categoricas(self):
        return self.df.select_dtypes(include="object").columns


# ==============================
# CONFIG STREAMLIT
# ==============================

st.set_page_config(page_title="Bank Marketing EDA", layout="wide")

st.title("📊 Bank Marketing - Análisis Exploratorio de Datos")

menu = st.sidebar.selectbox(
    "Navegación",
    ["Home", "Carga Dataset", "EDA", "Análisis Interactivo", "Conclusiones"]
)

# ==============================
# HOME
# ==============================

if menu == "Home":

    st.header("Presentación del Proyecto")

    st.write("""
    Este proyecto desarrolla un análisis exploratorio del dataset
    Bank Marketing para identificar patrones que influyen en la
    aceptación de campañas de marketing.
    """)

    st.subheader("Autor")
    st.write("Ruth Llalla")

    st.subheader("Tecnologías")

    st.write("""
    - Python
    - Pandas
    - NumPy
    - Streamlit
    - Matplotlib
    - Seaborn
    """)

# ==============================
# CARGA DATASET
# ==============================

if menu == "Carga Dataset":

    st.header("Carga del Dataset")

    archivo = st.file_uploader("Sube el archivo BankMarketing.csv", type=["csv"])

    if archivo:

        df = pd.read_csv(archivo, sep=";")

        st.success("Dataset cargado correctamente")

        st.dataframe(df.head())

        st.write("Dimensiones del dataset")

        st.write(df.shape)

# ==============================
# EDA PRINCIPAL
# ==============================

if menu == "EDA":

    st.header("Análisis Exploratorio")

    archivo = st.file_uploader("Sube dataset", type=["csv"])

    if archivo:

        df = pd.read_csv(archivo, sep=";")

        analyzer = DataAnalyzer(df)

        tab1, tab2, tab3, tab4 = st.tabs([
            "Información",
            "Estadísticas",
            "Distribuciones",
            "Variables Categóricas"
        ])

        # ---------------------
        # TAB 1
        # ---------------------

        with tab1:

            st.subheader("Información general")

            st.write("Dimensiones")

            st.write(df.shape)

            st.write("Tipos de datos")

            st.dataframe(df.dtypes)

            st.write("Valores faltantes")

            st.dataframe(analyzer.valores_nulos())

        # ---------------------
        # TAB 2
        # ---------------------

        with tab2:

            st.subheader("Estadísticas descriptivas")

            st.dataframe(analyzer.estadisticas())

        # ---------------------
        # TAB 3
        # ---------------------

        with tab3:

            st.subheader("Distribución de variables numéricas")

            numericas = analyzer.variables_numericas()

            variable = st.selectbox("Selecciona variable", numericas)

            fig, ax = plt.subplots()

            sns.histplot(df[variable], kde=True, ax=ax)

            st.pyplot(fig)

        # ---------------------
        # TAB 4
        # ---------------------

        with tab4:

            st.subheader("Variables categóricas")

            categoricas = analyzer.variables_categoricas()

            variable = st.selectbox("Selecciona variable categórica", categoricas)

            fig, ax = plt.subplots()

            sns.countplot(data=df, x=variable, ax=ax)

            plt.xticks(rotation=45)

            st.pyplot(fig)

# ==============================
# ANÁLISIS INTERACTIVO
# ==============================

if menu == "Análisis Interactivo":

    st.header("Análisis dinámico")

    archivo = st.file_uploader("Sube dataset", type=["csv"])

    if archivo:

        df = pd.read_csv(archivo, sep=";")

        numericas = df.select_dtypes(include=np.number).columns

        variable = st.selectbox("Variable numérica", numericas)

        rango = st.slider(
            "Selecciona rango",
            float(df[variable].min()),
            float(df[variable].max()),
            (float(df[variable].min()), float(df[variable].max()))
        )

        df_filtrado = df[(df[variable] >= rango[0]) & (df[variable] <= rango[1])]

        st.write("Datos filtrados")

        st.dataframe(df_filtrado)

        fig, ax = plt.subplots()

        sns.histplot(df_filtrado[variable], kde=True, ax=ax)

        st.pyplot(fig)

        if st.checkbox("Mostrar estadísticas"):

            st.dataframe(df_filtrado.describe())

# ==============================
# CONCLUSIONES
# ==============================

if menu == "Conclusiones":

    st.header("Hallazgos principales")

    st.write("""
    - La duración de las llamadas parece tener relación con la aceptación de la campaña.

    - Algunos perfiles demográficos presentan mayor probabilidad de aceptar la oferta.

    - La repetición excesiva de contactos no necesariamente mejora los resultados.

    - Las condiciones económicas pueden influir en el comportamiento del cliente.

    - El análisis exploratorio permite detectar patrones útiles antes de aplicar modelos predictivos.
    """)