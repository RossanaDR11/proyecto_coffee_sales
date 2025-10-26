import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuración de la página y carga de datos ---
st.set_page_config(layout="wide") # Usar el ancho completo de la página
st.title("☕ Análisis de Ventas de Café y Tendencia Anual")
st.markdown("Exploración detallada de ingresos diarios, estacionalidad y el rendimiento del producto estrella.")

# Nombre del archivo de datos
DATA_FILE = 'Coffe_sales_clean.csv'

# Cargar los datos (con manejo de errores y caché)
@st.cache_data
def load_data(filepath):
    """Carga los datos y realiza la limpieza inicial."""
    try:
        df = pd.read_csv(filepath)
        # Convertir 'Date' a datetime si existe
        if 'Date' in df.columns:
             df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo '{filepath}'. Asegúrate de que está en la misma carpeta que la aplicación.")
        return None
    except Exception as e:
        st.error(f"Error al cargar o procesar el archivo: {e}")
        return None

# Cargar los datos
df_raw = load_data(DATA_FILE)

if df_raw is None: 
    st.stop()

# --- Preparación de datos (DataFrame Principal) ---
df = df_raw.copy()

# Definir el orden correcto de los meses para el análisis de estacionalidad
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']

try:
    # Asegurarse de que 'Month_name' es categórica y está ordenada
    df['Month_name'] = pd.Categorical(df['Month_name'], categories=month_order, ordered=True)
except ValueError:
    st.warning("Advertencia: No todos los meses esperados están en los datos.")

# ----------------------------------------------------
# --- 1. PREPARACIÓN: INGRESO POR DÍA (Gráfico de Pastel) ---
# ----------------------------------------------------
sales_by_day_money = None
dia_pico_ingresos = "N/A"

if 'Weekday' in df.columns and 'Money' in df.columns and 'Weekdaysort' in df.columns:
    # Agrupamos por día y por el clasificador para mantener el orden
    sales_by_day_money = df.groupby(['Weekday', 'Weekdaysort'])['Money'].sum().reset_index()
    # Ordenamos por la columna 'Weekdaysort' (asumiendo 1=Lunes, 7=Domingo)
    sales_by_day_money = sales_by_day_money.sort_values('Weekdaysort')
    # Cálculo para la métrica
    dia_pico_ingresos = sales_by_day_money.loc[sales_by_day_money['Money'].idxmax(), 'Weekday']

# ----------------------------------------------------
# --- 2. PREPARACIÓN: CAFÉ MÁS VENDIDO Y TENDENCIA MENSUAL (Gráfico de Barras) ---
# ----------------------------------------------------
cafe_mas_vendido_ano = "N/A"
ventas_cafe_estrella_mes = None
mes_pico_estrella = "N/A"
max_ventas_estrella = 0

if 'Coffee_Name' in df.columns:
    # 1. Identificar el café más vendido en todo el conjunto de datos
    cafe_mas_vendido_ano = df['Coffee_Name'].value_counts().idxmax()
    
    # 2. Filtrar solo las ventas de ese café
    df_cafe_mas_vendido = df[df['Coffee_Name'] == cafe_mas_vendido_ano]
    
    # 3. Agrupar por mes y contar ventas, reindexando para asegurar el orden
    ventas_cafe_estrella_mes = df_cafe_mas_vendido.groupby('Month_name').size().reset_index(name='Ventas')
    ventas_cafe_estrella_mes = ventas_cafe_estrella_mes.set_index('Month_name').reindex(month_order).fillna(0).reset_index()
    ventas_cafe_estrella_mes['Ventas'] = ventas_cafe_estrella_mes['Ventas'].astype(int)
    
    # Cálculo para las métricas
    max_ventas_estrella = ventas_cafe_estrella_mes['Ventas'].max()
    mes_pico_estrella = ventas_cafe_estrella_mes.loc[ventas_cafe_estrella_mes['Ventas'].idxmax(), 'Month_name']

# ----------------------------------------------------
# --- 3. PREPARACIÓN: VENTAS POR HORA DEL DÍA (Gráfico de Conteo) ---
# ----------------------------------------------------
sales_by_hour = None
peak_hour = "N/A"
sales_at_peak = 0

if 'Hour_of_Day' in df.columns:
    # Contar las ventas por hora
    sales_by_hour = df['Hour_of_Day'].value_counts().sort_index()
    
    # Calcular la hora pico
    if not sales_by_hour.empty:
        peak_hour = sales_by_hour.idxmax()
        sales_at_peak = sales_by_hour.max()

# ----------------------------------------------------
# --- 4. PREPARACIÓN: PORCENTAJE DE VENTAS POR TIPO DE CAFÉ ---
# ----------------------------------------------------
porcentajes_cafe = None
top_coffee_name = "N/A"
top_coffee_percent = 0.0

if 'Coffee_Name' in df.columns:
    # Calcular el porcentaje de cada café
    porcentajes_cafe = df['Coffee_Name'].value_counts(normalize=True) * 100
    
    # Calcular la métrica del más vendido
    if not porcentajes_cafe.empty:
        top_coffee_name = porcentajes_cafe.idxmax()
        top_coffee_percent = porcentajes_cafe.max()

# ----------------------------------------------------
# --- 5. PREPARACIÓN: VOLUMEN DE VENTAS POR DÍA DE LA SEMANA (NUEVA) ---
# ----------------------------------------------------
sales_volume_by_weekday = None
peak_sales_day = "N/A"

if 'Weekday' in df.columns and 'Weekdaysort' in df.columns:
    # Agrupar por día de la semana y contar el número de transacciones (volumen)
    sales_volume_by_weekday = df.groupby(['Weekday', 'Weekdaysort']).size().reset_index(name='Ventas')
    sales_volume_by_weekday = sales_volume_by_weekday.sort_values('Weekdaysort')
    
    # Cálculo para la métrica
    if not sales_volume_by_weekday.empty:
        peak_sales_day = sales_volume_by_weekday.loc[sales_volume_by_weekday['Ventas'].idxmax(), 'Weekday']

# ----------------------------------------------------
# --- 6. PREPARACIÓN: VOLUMEN DE VENTAS POR MES (NUEVA) ---
# ----------------------------------------------------
sales_volume_by_month = None
peak_sales_month = "N/A"
peak_sales_month_volume = 0

if 'Month_name' in df.columns:
    # Agrupar por mes y contar el número de transacciones (volumen)
    sales_volume_by_month = df.groupby('Month_name').size().reset_index(name='Ventas')
    sales_volume_by_month = sales_volume_by_month.set_index('Month_name').reindex(month_order).fillna(0).reset_index()
    sales_volume_by_month['Ventas'] = sales_volume_by_month['Ventas'].astype(int)
    
    # Cálculo para la métrica
    if not sales_volume_by_month.empty:
        peak_sales_month_volume = sales_volume_by_month['Ventas'].max()
        peak_sales_month = sales_volume_by_month.loc[sales_volume_by_month['Ventas'].idxmax(), 'Month_name']


# ==================================================================
# --- SECCIÓN A: DISTRIBUCIÓN DE INGRESOS (Pastel) ---
# ==================================================================
st.header("💰 Distribución Porcentual de Ingresos por Día")
st.markdown("Esta vista revela la contribución financiera de cada día de la semana, esencial para la rentabilidad.")

if sales_by_day_money is not None:
    col_pie_chart, col_pie_info = st.columns([1.5, 1])

    with col_pie_chart:
        st.subheader("Ingresos Totales (€) por Día de la Semana")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = sns.color_palette('Set2', len(sales_by_day_money))
        
        ax.pie(sales_by_day_money['Money'], 
               labels=sales_by_day_money['Weekday'], 
               autopct='**%1.1f%%**',
               startangle=90, 
               colors=colors,
               wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
               textprops={'fontsize': 10})
        
        ax.set_title('Distribución Porcentual de Ingresos', fontsize=16, pad=20)
        ax.axis('equal') 
        
        st.pyplot(fig)

    with col_pie_info:
        st.subheader("Análisis de Flujo de Efectivo Diario")
        st.metric(label="Día con Mayor Ingreso", value=dia_pico_ingresos, delta=f"Total: {sales_by_day_money['Money'].max():,.2f} €")
        
        st.write("""
        **Interpretación Clave:**

        * **Prioridad Operativa:** El día con mayor ingreso es crucial, ya que impulsa la mayor parte de la facturación.
        * **Rentabilidad:** Un alto porcentaje aquí puede indicar dónde se venden productos de mayor valor o margen.
        """)
        st.dataframe(sales_by_day_money[['Weekday', 'Money']].rename(columns={'Money': 'Ingresos (€)'}), use_container_width=True)

else:
    st.info("Faltan datos de Ingresos ('Money') para el análisis por día.")

# ==================================================================
# --- SECCIÓN E (NUEVA): VOLUMEN DE VENTAS POR DÍA ---
# ==================================================================
st.divider()
st.header("🗓️ Volumen de Ventas (Transacciones) por Día de la Semana")
st.markdown("Analiza la cantidad de pedidos o transacciones, lo cual es vital para la gestión de colas y personal.")

if sales_volume_by_weekday is not None:
    col_volume_day_chart, col_volume_day_info = st.columns([1.5, 1])

    with col_volume_day_chart:
        st.subheader("Cantidad de Ventas por Día")
        
        fig_volume_day, ax_volume_day = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Weekday', y='Ventas', data=sales_volume_by_weekday, ax=ax_volume_day, palette='viridis')
        
        ax_volume_day.set_title('Volumen de Ventas por Día de la Semana')
        ax_volume_day.set_xlabel('Día de la Semana')
        ax_volume_day.set_ylabel('Número de Ventas')
        plt.tight_layout()
        
        st.pyplot(fig_volume_day)

    with col_volume_day_info:
        st.subheader("Gestión Operativa Diaria")
        st.metric(label="Día con Mayor Volumen de Pedidos", 
                  value=peak_sales_day, 
                  delta=f"Total: {sales_volume_by_weekday['Ventas'].max():,}")
                  
        st.write("""
        **Comentarios de Operación:**
        
        * **Contraste con Ingresos:** Es importante comparar este volumen con la Sección A (Ingresos). Si el día de mayor volumen no es el de mayor ingreso, sugiere que ese día se venden productos de menor coste.
        * **Personal y Rapidez:** El **{peak_sales_day}** exige máxima eficiencia y más personal para manejar el alto flujo de clientes.
        """)
        st.dataframe(sales_volume_by_weekday[['Weekday', 'Ventas']].set_index('Weekday'), use_container_width=True)

else:
    st.info("Faltan datos de 'Weekday' para el análisis de volumen de ventas.")

# ==================================================================
# --- SECCIÓN F (NUEVA): VOLUMEN DE VENTAS POR MES ---
# ==================================================================
st.divider()
st.header("🗓️ Tendencia de Volumen de Ventas (Transacciones) por Mes")
st.markdown("Identifica la estacionalidad general de la demanda de café a lo largo del año.")

if sales_volume_by_month is not None:
    col_volume_month_chart, col_volume_month_info = st.columns([2, 1])

    with col_volume_month_chart:
        st.subheader("Cantidad Total de Ventas por Mes")
        
        fig_volume_month, ax_volume_month = plt.subplots(figsize=(10, 5))
        sns.lineplot(x='Month_name', y='Ventas', data=sales_volume_by_month, marker='o', ax=ax_volume_month, color='darkorange')
        
        ax_volume_month.set_title('Tendencia de Volumen de Ventas Mensuales')
        ax_volume_month.set_xlabel('Mes')
        ax_volume_month.set_ylabel('Número de Ventas')
        plt.setp(ax_volume_month.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.tight_layout()
        
        st.pyplot(fig_volume_month)

    with col_volume_month_info:
        st.subheader("Estrategias Estacionales")
        st.metric(label="Mes de Mayor Volumen Total", 
                  value=peak_sales_month, 
                  delta=f"Ventas: {peak_sales_month_volume:,}")
                  
        st.write("""
        **Observaciones Clave:**
        
        * **Estacionalidad Global:** Este gráfico muestra los picos de demanda a lo largo del año. El **{peak_sales_month}** es el mes crucial para maximizar la capacidad de producción.
        * **Planificación de Inventario:** Asegúrate de que el inventario general y la materia prima estén asegurados antes del inicio del mes pico.
        * **Marketing:** Las campañas de marketing deben intensificarse en los meses previos al pico, o enfocarse en los meses bajos para estabilizar las ventas.
        """)  
        st.dataframe(sales_volume_by_month.set_index('Month_name'), use_container_width=True)

else:
    st.info("Faltan datos de 'Month_name' para el análisis de volumen de ventas mensuales.")


# ==================================================================
# --- SECCIÓN C: TENDENCIA DIARIA POR HORA ---
# ==================================================================
st.divider()
st.header("🕒 Distribución de Ventas por Hora del Día")
st.markdown("Identifica los períodos pico para la optimización del servicio y la eficiencia operativa.")

if 'Hour_of_Day' in df.columns:
    
    col_graph_hour, col_info_hour = st.columns([2, 1])

    with col_graph_hour:
        st.subheader("Ventas de Café por Hora del Día")
        
        fig_hour, ax_hour = plt.subplots(figsize=(10, 5))
        
        # Generar el gráfico de conteo
        sns.countplot(data=df, x='Hour_of_Day', order=sorted(df['Hour_of_Day'].unique()), ax=ax_hour, palette='flare')
        
        ax_hour.set_title('Ventas de Café por Hora del Día')
        ax_hour.set_xlabel('Hora del Día')
        ax_hour.set_ylabel('Cantidad de Ventas')
        plt.tight_layout()
        
        st.pyplot(fig_hour)

    with col_info_hour:
        st.subheader("Métricas Operativas")
        
        # Métrica clave
        st.metric(label="Hora Pico de Ventas", 
                  value=f"{peak_hour}:00 hrs", 
                  delta=f"Ventas: {sales_at_peak:,}")
                  
        st.write("""
        **Optimización de la Operación:**
        
        Este gráfico es fundamental para la **planificación operativa**.
        
        * **Personal:** La **hora pico ({peak_hour}:00 hrs)** requiere la mayor dotación de personal para manejar el volumen de clientes de manera eficiente.
        * **Inventario:** Asegúrate de que los productos estén completamente preparados antes de la subida de la mañana/tarde.
        """)
        
        st.dataframe(sales_by_hour.to_frame(name='Ventas').rename_axis('Hora del Día'), use_container_width=True)

else:
    st.info("No se puede mostrar el análisis por hora. Revise la columna 'Hour_of_Day'.")


# ==================================================================
# --- SECCIÓN B: TENDENCIA DEL CAFÉ ESTRELLA (Barra con Layout 2:1) ---
# ==================================================================
st.divider()
st.header("✨ Tendencia Anual del Producto Estrella")
st.markdown(f"Análisis mensual de ventas para el café más popular del año: **{cafe_mas_vendido_ano}**")

if ventas_cafe_estrella_mes is not None and cafe_mas_vendido_ano != "N/A":
    
    col_grafico_estrella, col_info_estrella = st.columns([2, 1])

    with col_grafico_estrella:
        st.subheader(f"Ventas Mensuales de {cafe_mas_vendido_ano}")
        
        fig_estrella, ax_estrella = plt.subplots(figsize=(10, 5))
        
        sns.barplot(x='Month_name', y='Ventas', data=ventas_cafe_estrella_mes, ax=ax_estrella, palette='mako')
        
        ax_estrella.set_title(f'Ventas mensuales de {cafe_mas_vendido_ano}')
        ax_estrella.set_xlabel('Mes')
        ax_estrella.set_ylabel('Cantidad de Ventas')
        plt.setp(ax_estrella.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.tight_layout()
        
        st.pyplot(fig_estrella)

    with col_info_estrella:
        st.subheader("Métricas de Estacionalidad del Producto")
        
        st.metric(label="Mes de Mayor Demanda del Producto", 
                  value=mes_pico_estrella, 
                  delta=f"Ventas: {max_ventas_estrella:,}")
                  
        min_ventas_estrella = ventas_cafe_estrella_mes['Ventas'].min()
        st.metric(label="Ventas Mínimas",
                  value=f"{min_ventas_estrella:,}")
        
        st.write("""
        **Estrategia Basada en Tendencias:**
        
        Este análisis detalla el ciclo de vida estacional de su producto más vendido. 
        
        * **Anticipación:** Las compras de ingredientes específicos deben ser máximas antes del mes pico.
        * **Promociones:** Use este gráfico para decidir si promover este café en meses bajos o centrarse en alternativas.
        """)
        st.dataframe(ventas_cafe_estrella_mes.set_index('Month_name'), use_container_width=True)

else:
    st.info("No se puede determinar la tendencia mensual del café más vendido. Revise la columna 'Coffee_Name'.")


# ==================================================================
# --- SECCIÓN D: POPULARIDAD DE PRODUCTOS (Barra de Porcentajes) ---
# ==================================================================
st.divider()
st.header("📈 Popularidad Relativa: Porcentaje de Ventas por Tipo de Café")
st.markdown("Compara visualmente la cuota de mercado de cada producto dentro del total de ventas de café.")

if porcentajes_cafe is not None and not porcentajes_cafe.empty:
    
    col_graph_percent, col_info_percent = st.columns([2, 1])

    with col_graph_percent:
        st.subheader("Distribución Porcentual de Ventas")
        
        fig_percent, ax_percent = plt.subplots(figsize=(10, 6))
        
        sns.barplot(x=porcentajes_cafe.index, y=porcentajes_cafe.values, ax=ax_percent, palette='coolwarm')
        
        ax_percent.set_title('Porcentaje de Ventas por Tipo de Café')
        ax_percent.set_xlabel('Tipo de Café')
        ax_percent.set_ylabel('Porcentaje (%)')
        plt.setp(ax_percent.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.tight_layout()
        
        st.pyplot(fig_percent)

    with col_info_percent:
        st.subheader("Conclusiones de Mercado")
        
        st.metric(label="Líder del Mercado", 
                  value=top_coffee_name, 
                  delta=f"Cuota: {top_coffee_percent:.1f}%")
                  
        st.write(f"""
        **Análisis de Concentración de Demanda:**
        
        * **Dependencia:** La alta concentración en **{top_coffee_name}** sugiere una dependencia del producto estrella.
        * **Diversificación:** Se recomienda analizar los productos con baja cuota para encontrar oportunidades de diversificación o descontinuación.
        """)
        
        st.dataframe(porcentajes_cafe.to_frame(name='Porcentaje (%)').rename_axis('Tipo de Café').sort_values(by='Porcentaje (%)', ascending=False), use_container_width=True)

else:
    st.info("No se puede calcular el porcentaje de ventas. Revise la columna 'Coffee_Name'.")


# ==================================================================
# Sección final: conclusión y recomendaciones
st.divider()
st.header("📌 Conclusión y Recomendaciones Finales")
st.markdown("Resumen ejecutivo y acciones sugeridas basadas en el análisis de ventas.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Métricas clave")
    st.metric("Producto estrella (año)", cafe_mas_vendido_ano)
    st.metric("Líder de cuota", f"{top_coffee_percent:.1f}%", delta=top_coffee_name)
    st.metric("Mes pico (producto estrella)", mes_pico_estrella, delta=f"{max_ventas_estrella:,} ventas")

with col2:
    st.subheader("Operación y demanda")
    st.metric("Día con mayor ingreso (€)", dia_pico_ingresos, delta=f"{sales_by_day_money['Money'].max():,.2f} €" if sales_by_day_money is not None else "N/A")
    st.metric("Día con mayor volumen (transacciones)", peak_sales_day, delta=f"{sales_volume_by_weekday['Ventas'].max():,}" if sales_volume_by_weekday is not None else "N/A")
    st.metric("Hora pico de ventas", f"{peak_hour}:00", delta=f"{sales_at_peak:,}" if sales_by_hour is not None else "N/A")

st.markdown("### Estrategia de Marketing y Operaciones")
st.markdown("- Asegurar inventario y materia prima antes del mes pico del producto estrella.")
st.markdown("- Diseñar promociones específicas para meses de baja demanda y probar impulsar productos con baja cuota para diversificar ingresos.")
st.markdown("- Aumentar stock y capacidad operativa.\n   - Lanzar teasers en redes sociales y email para generar expectativa.\n    - Activación (mes pico y horas pico)\n   - Promoción 'Producto del Mes' con oferta limitada; usar horarios específicos para promociones flash.\n   - Ofrecer combos (café + snack) y upsells para subir el ticket medio.\n   - Publicidad segmentada por hora/día (geolocalización y audiencias similares).\n  - Retención y optimización (post-pico)\n   - Implementar programa de lealtad y cupones para compras futuras.\n   - Recopilar feedback para mejorar la experiencia del cliente.")



