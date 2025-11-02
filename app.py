import streamlit as st
import pandas as pd
import os
import json
import plotly.graph_objects as go
from io import BytesIO
import numpy as np # Importado para cálculos matemáticos
import bcrypt # Importado para hashear contraseñas
import sys # Importado para compatibilidad de rutas de archivos

# --- Configuración de la Página ---
st.set_page_config(
    page_title="ComVida",
    page_icon="🍋‍🟩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constantes y Variables Globales ---

# Determinar la ruta base para los archivos (funciona en Streamlit Cloud y local)
BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0] if hasattr(sys, 'argv') else __file__))

# --- MODIFICADO: Esta es ahora la carpeta RAÍZ de todos los pacientes ---
BASE_DIRECTORIO_PACIENTES = os.path.join(BASE_DIR, "pacientes") 
DB_ALIMENTOS_PATH = os.path.join(BASE_DIR, "alimentos.csv")
DB_USUARIOS_PATH = os.path.join(BASE_DIR, "usuarios.json")
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")


# --- Funciones de Autenticación y Gestión de Usuarios ---

def hash_password(password):
    """Genera un hash bcrypt para una contraseña."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed_password):
    """Verifica si una contraseña coincide con su hash."""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    except ValueError:
        return False

def cargar_usuarios():
    """Carga los usuarios desde usuarios.json. Crea uno por defecto si no existe."""
    if not os.path.exists(DB_USUARIOS_PATH):
        # Crear usuario admin por defecto si el archivo no existe
        default_pass = hash_password("admin123")
        default_users = {
            "admin": {
                "password": default_pass,
                "rol": "admin"
            }
        }
        guardar_usuarios(default_users)
        return default_users
    
    try:
        with open(DB_USUARIOS_PATH, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback por si el archivo está corrupto o no se encuentra
        return {}

def guardar_usuarios(usuarios_data):
    """Guarda el diccionario de usuarios en usuarios.json."""
    try:
        with open(DB_USUARIOS_PATH, 'w') as f:
            json.dump(usuarios_data, f, indent=4)
    except IOError as e:
        st.error(f"Error crítico al guardar datos de usuario: {e}")

# --- Funciones de Carga de Datos ---

@st.cache_data
def cargar_base_de_datos_alimentos(filepath=DB_ALIMENTOS_PATH):
    """
    Carga y limpia la base de datos de alimentos desde el CSV.
    Maneja el encabezado de múltiples líneas y la limpieza de datos.
    """
    try:
        # Usamos header=0 para leer la primera línea como encabezado
        # Usamos skiprows=[1] para saltar la línea de unidades (kcal, g, etc.)
        df = pd.read_csv(filepath, delimiter=';', header=0, skiprows=[1])
        
        # Limpieza profunda de nombres de columnas
        columnas_limpias = []
        for col in df.columns:
            col_limpia = str(col).replace('\r\n', ' ').replace('\n', ' ').replace('  ', ' ').strip()
            columnas_limpias.append(col_limpia)
        df.columns = columnas_limpias

        # Renombrar columnas clave para acceso fácil y estandarizado
        columnas_renombrar = {
            'Energía <ENERC>': 'Kcal',
            'Energía <ENERC>.1': 'Kj',
            'Proteínas <PROCNT>': 'Proteínas',
            'Grasa total <FAT>': 'Grasas',
            'Carbohidratos totales <CHOCDF>': 'Carbohidratos',
            'Fibra dietaria <FIBTG>': 'Fibra',
            'Agua <WATER>': 'Agua',
            'Calcio <CA>': 'Calcio',
            'Fósforo <P>': 'Fósforo',
            'Zinc <ZN>': 'Zinc',
            'Hierro <FE>': 'Hierro',
            'Vitamina C <VITC>': 'Vitamina C',
            'Sodio <NA>': 'Sodio',
            'Potasio <K>': 'Potasio',
            # --- Columnas de micronutrientes agregadas ---
            'β caroteno equivalentes totales <CARTBQ>': 'Beta-Caroteno',
            'Vitamina A equivalentes totales <VITA>': 'Vitamina A',
            'Tiamina <THIA>': 'Tiamina',
            'Riboflavina <RIBF>': 'Riboflavina',
            'Niacina <NIA>': 'Niacina',
            'Ácido fólico': 'Acido Folico' # Evitar tildes en nombres de columnas
        }
        df = df.rename(columns=columnas_renombrar)

        # Convertir todas las columnas de nutrientes a numérico
        cols_numericas = [
            'Kcal', 'Proteínas', 'Grasas', 'Carbohidratos', 'Fibra', 'Agua',
            'Calcio', 'Fósforo', 'Zinc', 'Hierro', 'Vitamina C', 'Sodio', 'Potasio',
            # --- Columnas de micronutrientes agregadas ---
            'Beta-Caroteno', 'Vitamina A', 'Tiamina', 'Riboflavina', 'Niacina', 'Acido Folico'
        ]
        
        for col in cols_numericas:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', '.'), 
                    errors='coerce'
                ).fillna(0)
            else:
                # Si una columna renombrada no existía, la crea con 0s
                # st.warning(f"Columna esperada '{col}' no encontrada en el CSV. Se creará como columna vacía.")
                df[col] = 0
                
        return df

    except FileNotFoundError:
        st.error(f"Error Crítico: No se encontró el archivo '{filepath}'. Asegúrese de que 'alimentos.csv' esté en el mismo directorio.")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar la base de datos de alimentos: {e}")
        st.stop()

# --- Funciones de Cálculo Nutricional ---

def calcular_imc(peso, talla_cm):
    """Calcula el IMC y retorna el valor y un diagnóstico simple."""
    if talla_cm == 0 or peso == 0:
        return 0, "Sin datos"
        
    talla_m = talla_cm / 100
    imc = peso / (talla_m ** 2)
    
    if imc < 18.5:
        diagnostico = "Bajo Peso"
    elif 18.5 <= imc < 25:
        diagnostico = "Peso Normal"
    elif 25 <= imc < 30:
        diagnostico = "Sobrepeso"
    else:
        diagnostico = "Obesidad"
        
    return imc, diagnostico

def calcular_get(sexo, peso, talla_cm, edad, actividad, formula, masa_magra=0):
    """
    Calcula el Gasto Energético Total (GET) usando una de varias fórmulas.
    Requiere masa_magra para la fórmula de Cunningham.
    """
    if peso == 0 or talla_cm == 0 or edad == 0:
        return 0
        
    geb = 0
    
    # 1. Calcular GEB (Gasto Energético Basal)
    if formula == "Mifflin-St Jeor":
        if sexo == 'Masculino':
            geb = (10 * peso) + (6.25 * talla_cm) - (5 * edad) + 5
        else: # Femenino
            geb = (10 * peso) + (6.25 * talla_cm) - (5 * edad) - 161
    
    elif formula == "Harris-Benedict":
        if sexo == 'Masculino':
            geb = 88.362 + (13.397 * peso) + (4.799 * talla_cm) - (5.677 * edad)
        else: # Femenino
            geb = 447.593 + (9.247 * peso) + (3.098 * talla_cm) - (4.330 * edad)
    
    elif formula == "Cunningham":
        if masa_magra > 0:
            geb = 500 + (22 * masa_magra)
        else:
            geb = 0 # No se puede calcular sin masa magra
    
    else: # Fallback a Mifflin por si acaso
        if sexo == 'Masculino':
            geb = (10 * peso) + (6.25 * talla_cm) - (5 * edad) + 5
        else: # Femenino
            geb = (10 * peso) + (6.25 * talla_cm) - (5 * edad) - 161
            
    # 2. Factor de Actividad
    if actividad == 'Ligera':
        factor = 1.375
    elif actividad == 'Moderada':
        factor = 1.55
    else: # Intensa
        factor = 1.725
        
    get = geb * factor
    return get

# --- Nuevas Funciones de Composición Corporal y Somatotipo (ISAK) ---

def get_densidad_durnin(sexo, edad, L):
    """Calcula la Densidad Corporal (D) según Durnin & Womersley (1974)."""
    # L = log10(Suma de 4 pliegues)
    
    if sexo == 'Masculino':
        if edad < 17:
            D = 1.1533 - (0.0643 * L)
        elif edad <= 19:
            D = 1.1620 - (0.0630 * L)
        elif edad <= 29:
            D = 1.1631 - (0.0632 * L)
        elif edad <= 39:
            D = 1.1422 - (0.0544 * L)
        elif edad <= 49:
            D = 1.1620 - (0.0700 * L)
        else: # 50+
            D = 1.1715 - (0.0779 * L)
    else: # Femenino
        if edad < 17:
            D = 1.1369 - (0.0598 * L)
        elif edad <= 19:
            D = 1.1549 - (0.0678 * L)
        elif edad <= 29:
            D = 1.1599 - (0.0717 * L)
        elif edad <= 39:
            D = 1.1423 - (0.0632 * L)
        elif edad <= 49:
            D = 1.1333 - (0.0612 * L)
        else: # 50+
            D = 1.1339 - (0.0645 * L)
    return D

# --- FUNCIÓN ORIGINAL MODIFICADA (2C) ---
def calcular_composicion_2c_durnin_siri(peso, sexo, edad, pliegues):
    """
    Calcula la composición corporal (2 Componentes: Grasa, Magra) 
    usando Durnin & Womersley (con corrección de edad) y Siri.
    """
    pliegue_biceps = pliegues.get('Bicipital', 0.0)
    pliegue_triceps = pliegues.get('Tricipital', 0.0)
    pliegue_subescapular = pliegues.get('Subescapular', 0.0)
    pliegue_suprailiaco = pliegues.get('Suprailíaco', 0.0)
    
    suma_4_pliegues = pliegue_biceps + pliegue_triceps + pliegue_subescapular + pliegue_suprailiaco
    
    if suma_4_pliegues <= 0:
        return {'masa_grasa': 0, 'masa_magra': 0, 'porc_grasa': 0, 'diag_grasa': "Sin datos de pliegues"}
        
    L = np.log10(suma_4_pliegues)
    
    # Esta es la función get_densidad_durnin que ya tenías
    densidad = get_densidad_durnin(sexo, edad, L) 
    
    # Fórmula de Siri (1961) para % Grasa
    if densidad <= 0:
        return {'masa_grasa': 0, 'masa_magra': 0, 'porc_grasa': 0, 'diag_grasa': "Error en cálculo de densidad"}

    porc_grasa = ((4.95 / densidad) - 4.5) * 100
    
    if porc_grasa < 0: porc_grasa = 0
    if porc_grasa > 60: porc_grasa = 60 # Límite
    
    masa_grasa = peso * (porc_grasa / 100)
    masa_magra = peso - masa_grasa
    
    # Diagnóstico de grasa
    if porc_grasa < 10:
        diagnostico = "Nivel de grasa muy bajo"
    elif 10 <= porc_grasa < 20:
        diagnostico = "Nivel de grasa saludable (Atleta)"
    elif 20 <= porc_grasa < 30:
        diagnostico = "Nivel de grasa saludable (Promedio)"
    elif 30 <= porc_grasa < 40:
        diagnostico = "Nivel de grasa elevado (Sobrepeso)"
    else:
        diagnostico = "Nivel de grasa muy elevado (Obesidad)"
        
    return {
        'masa_grasa': masa_grasa, 
        'masa_magra': masa_magra, 
        'porc_grasa': porc_grasa, 
        'diag_grasa': diagnostico
    }

# --- NUEVA FUNCIÓN (5C) ---
def calcular_composicion_5c_kerr(peso, talla_cm, sexo, edad, raza, pliegues, circs, diams):
    """
    Calcula el modelo de 5 componentes (ISAK - Kerr, 1988).
    MG (Durnin/Siri), MM (Lee), MO (Rocha), MR (fijo), MP (diferencia).
    """
    
    # --- 0. Valores iniciales y conversiones ---
    H = talla_cm / 100.0
    sexo_num = 1 if sexo == 'Masculino' else 0
    
    if raza == 'Asiático':
        raza_val = 1.2
    elif raza == 'Africano':
        raza_val = 1.4
    else: # Caucásico
        raza_val = 0.0

    # Default dict para retornar en caso de error
    default_return = {
        'mg': 0, 'mm': 0, 'mo': 0, 'mr': 0, 'mp': 0,
        'porc_grasa_siri': 0, 'porc_grasa_brozek': 0, 'dc': 0, 'error': "Datos insuficientes"
    }

    if H <= 0 or peso <= 0:
        return default_return

    # --- 1. Masa Grasa (MG) ---
    p_tri = pliegues.get('Tricipital', 0.0)
    p_sub = pliegues.get('Subescapular', 0.0)
    p_bic = pliegues.get('Bicipital', 0.0)
    p_sup = pliegues.get('Suprailíaco', 0.0)
    
    suma_4_pliegues = p_tri + p_sub + p_bic + p_sup
    
    if suma_4_pliegues <= 0:
        return {**default_return, 'error': "Pliegues para DC (Durnin) no ingresados."}

    log_suma = np.log10(suma_4_pliegues)
    
    # Fórmulas de Densidad de Durnin (1974) (sin edad, como pide ISAK)
    if sexo == 'Masculino':
        DC = 1.1765 - (0.0744 * log_suma)
    else: # Femenino
        DC = 1.1567 - (0.0717 * log_suma)

    if DC <= 0:
        return {**default_return, 'error': "Error en cálculo de Densidad Corporal."}

    # %GC (Siri y Brozek)
    porc_grasa_siri = (495.0 / DC) - 450.0
    porc_grasa_brozek = (457.0 / DC) - 414.2
    
    # Usamos Siri para la Masa Grasa
    masa_grasa = (porc_grasa_siri / 100.0) * peso

    # --- 2. Masa Muscular (MM) - Lee et al. (2000) ---
    p_muslo_f = pliegues.get('Muslo (frontal)', 0.0)
    p_pant = pliegues.get('Pantorrilla Medial', 0.0)
    
    c_brazo = circs.get('Brazo (relajado)', 0.0)
    c_muslo = circs.get('Muslo (medial)', 0.0)
    c_pant = circs.get('Pantorrilla (máxima)', 0.0)

    # Circunferencias corregidas (cm)
    c_brazo_corr = c_brazo - (np.pi * (p_tri / 10.0))
    c_muslo_corr = c_muslo - (np.pi * (p_muslo_f / 10.0))
    c_pant_corr = c_pant - (np.pi * (p_pant / 10.0))

    masa_muscular = 0
    if c_brazo_corr > 0 and c_muslo_corr > 0 and c_pant_corr > 0:
        term1 = 0.00744 * (c_brazo_corr**2 / H)
        term2 = 0.00088 * (c_muslo_corr**2 / H)
        term3 = 0.00441 * (c_pant_corr**2 / H)
        
        masa_muscular = (H**2 * (term1 + term2 + term3)) + (2.4 * sexo_num) - (0.048 * edad) + raza_val + 7.8
    else:
        return {**default_return, 'error': "Circunferencias o pliegues para Masa Muscular (Lee) no ingresados."}

    # --- 3. Masa Ósea (MO) - Rocha (1975) ---
    d_muñeca = diams.get('Muñeca (bi-estiloideo)', 0.0)
    d_femur = diams.get('Fémur (bi-condilar)', 0.0)
    
    masa_osea = 0
    if d_muñeca > 0 and d_femur > 0:
        # Implementando la fórmula exactamente como se proporcionó
        masa_osea = 3.02 * ( (H**2) * (d_muñeca + d_femur) * 400.0 / 1000.0 )
    else:
        return {**default_return, 'error': "Diámetros para Masa Ósea (Rocha) no ingresados."}


    # --- 4. Masa Residual (MR) ---
    if sexo == 'Masculino':
        masa_residual = 0.241 * peso
    else: # Femenino
        masa_residual = 0.209 * peso

    # --- 5. Masa de Piel (MP) ---
    masa_piel = peso - (masa_grasa + masa_muscular + masa_osea + masa_residual)
    # La piel no puede ser negativa, si lo es, es un error de medición
    if masa_piel < 0: masa_piel = 0 

    return {
        'mg': masa_grasa,
        'mm': masa_muscular,
        'mo': masa_osea,
        'mr': masa_residual,
        'mp': masa_piel,
        'porc_grasa_siri': porc_grasa_siri,
        'porc_grasa_brozek': porc_grasa_brozek,
        'dc': DC,
        'error': None
    }
# --- FIN FUNCIONES 5C ---


def calcular_somatotipo(peso, talla_cm, pliegues, circs, diams):
    """
    Calcula el Somatotipo de Heath-Carter.
    """
    talla_m = talla_cm / 100
    if talla_m == 0 or peso == 0:
        return 0, 0, 0
        
    # Pliegues (mm)
    triceps = pliegues.get('Tricipital', 0.0)
    subescapular = pliegues.get('Subescapular', 0.0)
    suprailiaco = pliegues.get('Suprailíaco', 0.0)
    pantorrilla = pliegues.get('Pantorrilla Medial', 0.0)
    
    # Diámetros (cm) - deben convertirse a mm para algunas fórmulas, pero aquí se usan en cm
    humero_diam = diams.get('Húmero (bi-epicondilar)', 0.0)
    femur_diam = diams.get('Fémur (bi-condilar)', 0.0)
    
    # Circunferencias (cm)
    brazo_circ = circs.get('Brazo (relajado)', 0.0)
    pantorrilla_circ = circs.get('Pantorrilla (máxima)', 0.0)

    # 1. ENDOMORFIA
    X = (triceps + subescapular + suprailiaco) * (170.18 / talla_cm)
    if X <= 0:
        endo = 0
    else:
        endo = -0.7182 + (0.1451 * X) - (0.00068 * (X**2)) + (0.0000014 * (X**3))
        if endo < 0.1: endo = 0.1
        
    # 2. MESOMORFIA
    brazo_circ_corr = brazo_circ - (triceps / 10)
    pantorrilla_circ_corr = pantorrilla_circ - (pantorrilla / 10)
    
    if humero_diam <= 0 or femur_diam <= 0 or brazo_circ_corr <= 0 or pantorrilla_circ_corr <= 0:
        meso = 0
    else:
        meso = (0.858 * humero_diam) + \
               (0.601 * femur_diam) + \
               (0.188 * brazo_circ_corr) + \
               (0.161 * pantorrilla_circ_corr) - \
               (0.131 * talla_cm) + 4.5
        if meso < 0.1: meso = 0.1

    # 3. ECTOMORFIA
    HWR = talla_cm / (peso ** (1/3))
    
    if HWR > 40.75:
        ecto = (0.732 * HWR) - 28.58
    elif HWR > 38.25:
        ecto = (0.463 * HWR) - 17.63
    else:
        ecto = 0.1
        
    # Redondear a 1 decimal
    return round(endo, 1), round(meso, 1), round(ecto, 1)

def clasificar_somatotipo(endo, meso, ecto):
    """
    Clasifica el somatotipo en una categoría diagnóstica simple 
    basada en el componente dominante.
    """
    componentes = {'Endomorfo': endo, 'Mesomorfo': meso, 'Ectomorfo': ecto}
    
    # Ordenar componentes por valor
    ordenados = sorted(componentes.items(), key=lambda item: item[1], reverse=True)
    
    d1_nombre, d1_val = ordenados[0]
    d2_nombre, d2_val = ordenados[1]
    d3_nombre, d3_val = ordenados[2]

    # Clasificación basada en diferencias
    if (d1_val - d2_val) < 1.0:
        if (d2_val - d3_val) < 1.0:
            return "Central" # Ejs: 4-4-3, 3-3-3
        else:
            # Dos componentes dominantes
            return f"{d1_nombre}-{d2_nombre}" # Ejs: 5-5-2, 4-4-1
    else:
        if (d2_val - d3_val) < 1.0:
            return f"{d1_nombre} balanceado" # Ejs: 5-3-3, 6-4-4
        else:
            return f"{d1_nombre} (dominante)" # Ejs: 6-3-1, 5-3-1

# --- NUEVA FUNCIÓN DE EXPLICACIÓN ---
def obtener_explicacion_somatotipo(clasificacion):
    """
    Retorna una breve explicación para una clasificación de somatotipo dada.
    """
    explicaciones = {
        "Central": "Indica un desarrollo equilibrado entre los tres componentes (grasa, músculo y linealidad). Ningún componente domina claramente sobre los otros.",
        "Endomorfo (dominante)": "Predomina el componente endomorfo. Indica una alta adiposidad relativa, con tendencia a acumular grasa corporal.",
        "Mesomorfo (dominante)": "Predomina el componente mesomorfo. Indica un alto desarrollo músculo-esquelético relativo, con una complexión robusta y atlética.",
        "Ectomorfo (dominante)": "Predomina el componente ectomorfo. Indica una baja adiposidad y poco músculo, con una complexión delgada y lineal.",
        "Endomorfo balanceado": "Indica un predominio del componente endomorfo (grasa), pero con un desarrollo muscular y lineal similar entre sí.",
        "Mesomorfo balanceado": "Indica un predominio del componente mesomorfo (músculo), pero con un desarrollo de grasa y linealidad similar entre sí.",
        "Ectomorfo balanceado": "Indica un predominio del componente ectomorfo (linealidad), pero con un desarrollo de grasa y músculo similar entre sí.",
        "Endomorfo-Mesomorfo": "Indica un desarrollo alto y equilibrado en grasa y músculo, con menor linealidad. Es una complexión robusta y con grasa.",
        "Endomorfo-Ectomorfo": "Una clasificación muy poco común. Indica un equilibrio entre grasa y linealidad, con bajo desarrollo muscular.",
        "Mesomorfo-Ectomorfo": "Indica un desarrollo alto y equilibrado en músculo y linealidad, con baja adiposidad. Es la complexión 'atlética-delgada' (ej. saltadores).",
        "N/A": "No hay datos suficientes para clasificar el somatotipo."
    }
    # Retorna la explicación o un texto por defecto si no se encuentra
    return explicaciones.get(clasificacion, "No se pudo generar una explicación para esta clasificación.")
# --- FIN NUEVA FUNCIÓN ---

def crear_grafico_somatotipo(endo, meso, ecto):
    """
    Genera un gráfico de Plotly (Somatocarta) para el somatotipo dado.
    """
    
    # Calcular coordenadas X, Y de Heath-Carter
    x = ecto - endo
    y = (2 * meso) - (endo + ecto)

    # Puntos del polígono de la somatocarta (19 puntos estándar)
    boundary_x = [
        0.0, -1.0, -2.0, -3.0, -4.0, -4.0, -3.0, -2.0, -1.0, 
        0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 0.0
    ]
    boundary_y = [
        1.0, 2.0, 4.0, 7.0, 11.0, 10.0, 9.0, 8.0, 7.0, 
        6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 7.0, 4.0, 2.0, 1.0
    ]

    fig = go.Figure()

    # 1. Dibujar el polígono de la Somatocarta (con relleno)
    fig.add_trace(go.Scatter(
        x=boundary_x,
        y=boundary_y,
        fill="toself",
        fillcolor='rgba(230, 230, 230, 0.5)',
        line=dict(color='black', width=1),
        name='Somatocarta'
    ))

    # 2. Dibujar ejes X e Y (para referencia)
    fig.add_trace(go.Scatter(
        x=[-8, 8], y=[0, 0], 
        mode='lines', line=dict(color='grey', width=1, dash='dot'), 
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[-10, 18], 
        mode='lines', line=dict(color='grey', width=1, dash='dot'), 
        showlegend=False
    ))

    # 3. Dibujar el punto del paciente
    fig.add_trace(go.Scatter(
        x=[x],
        y=[y],
        mode='markers',
        marker=dict(color='red', size=12, line=dict(color='black', width=1)),
        name='Paciente'
    ))

    # 4. Configurar layout
    fig.update_layout(
        title='Somatocarta',
        xaxis=dict(range=[-8, 8], title='X (Ecto - Endo)'),
        yaxis=dict(range=[-10, 18], title='Y (2*Meso - (Endo+Ecto))'),
        width=500,
        height=500,
        showlegend=False,
        # Forzar aspecto 1:1
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1
    )

    return fig


# --- Funciones de Manejo de Pacientes (MODIFICADAS) ---

# --- NUEVO: Helper para obtener el directorio del usuario ---
def get_directorio_pacientes_usuario(username):
    """Retorna la ruta al directorio de pacientes para un usuario específico."""
    if not username:
        return None
    # Sanitizar el username por si acaso (aunque el login ya lo valida)
    safe_username = "".join(c for c in username if c.isalnum() or c in ('_', '-')).rstrip()
    user_dir = os.path.join(BASE_DIRECTORIO_PACIENTES, safe_username)
    return user_dir

# --- MODIFICADO: Ahora solo crea el directorio BASE ---
def inicializar_pacientes():
    """Crea el directorio BASE de pacientes si no existe."""
    if not os.path.exists(BASE_DIRECTORIO_PACIENTES):
        os.makedirs(BASE_DIRECTORIO_PACIENTES)

# --- MODIFICADO: Acepta 'username' ---
def listar_pacientes(username):
    """Retorna una lista de pacientes (sin .json) para el usuario especificado."""
    user_dir = get_directorio_pacientes_usuario(username)
    if not user_dir or not os.path.exists(user_dir):
        return [] # No directory for this user yet

    pacientes = []
    if os.path.exists(user_dir):
        for f in os.listdir(user_dir):
            if f.endswith('.json'):
                pacientes.append(f.replace('.json', ''))
    return pacientes

# --- MODIFICADO: Acepta 'username' ---
def cargar_paciente(username, nombre_archivo):
    """Carga los datos de un paciente desde el directorio del usuario."""
    user_dir = get_directorio_pacientes_usuario(username)
    if not user_dir:
        st.error("Error: No se pudo identificar el directorio del usuario.")
        return None
        
    filepath = os.path.join(user_dir, f"{nombre_archivo}.json")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            datos_paciente = json.load(f)
            # Asegurarse de que los campos clave existen
            datos_paciente.setdefault('pliegues', {})
            datos_paciente.setdefault('circunferencias', {}) # Agregado
            datos_paciente.setdefault('diametros', {}) # Agregado
            datos_paciente.setdefault('composicion', {})
            datos_paciente.setdefault('dieta_actual', [])
            return datos_paciente
    except Exception as e:
        st.error(f"Error al cargar el paciente {nombre_archivo}: {e}")
        return None

# --- MODIFICADO: Acepta 'username' y crea el dir del usuario ---
def guardar_paciente(username, datos_paciente):
    """Guarda o actualiza los datos de un paciente en el directorio del usuario."""
    user_dir = get_directorio_pacientes_usuario(username)
    if not user_dir:
        st.error("Error: No se pudo identificar el directorio del usuario para guardar.")
        return None

    nombre = datos_paciente.get('nombre')
    if not nombre:
        st.error("Error: El paciente no tiene nombre. No se puede guardar.")
        return None
        
    # --- NUEVO: Crear el directorio del usuario si no existe ---
    if not os.path.exists(user_dir):
        try:
            os.makedirs(user_dir)
        except OSError as e:
            st.error(f"Error crítico al crear directorio de paciente: {e}")
            return None
    # --- FIN NUEVO ---
        
    # Limpiar nombre para crear archivo
    nombre_archivo = nombre.replace(' ', '_').replace('.', '').lower()
    filepath = os.path.join(user_dir, f"{nombre_archivo}.json")
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(datos_paciente, f, indent=4, ensure_ascii=False)
        return nombre_archivo
    except Exception as e:
        st.error(f"Error al guardar el paciente: {e}")
        return None

# --- MODIFICADO: Acepta 'username' ---
def eliminar_paciente(username, nombre_archivo):
    """Elimina un archivo .json de paciente del directorio del usuario."""
    user_dir = get_directorio_pacientes_usuario(username)
    if not user_dir:
        st.error("Error: No se pudo identificar el directorio del usuario.")
        return False
        
    try:
        filepath = os.path.join(user_dir, f"{nombre_archivo}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        else:
            st.warning(f"No se encontró el archivo {filepath} para eliminar.")
            return False
    except Exception as e:
        st.error(f"Error al eliminar el paciente: {e}")
        return False
# --- FIN DE FUNCIONES MODIFICADAS ---

# --- Funciones de Utilidad (Exportación) ---

@st.cache_data
def generar_excel_dieta(df_dieta, df_resumen_comidas, df_macros):
    """
    Genera un archivo Excel en memoria con varias pestañas:
    1. Dieta Detallada (completa)
    2. Resumen por Comidas (pivote)
    3. Adecuación de Macros
    4. Pestañas individuales por cada Tiempo de Comida
    """
    output = BytesIO()
    
    # Asegurarse de que openpyxl está disponible
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            
            # --- Hoja 1: Dieta Detallada (Completa) - Se mantiene ---
            df_dieta_export = df_dieta.copy()
            for col in df_dieta_export.columns:
                if df_dieta_export[col].dtype == 'object':
                    df_dieta_export[col] = pd.to_numeric(df_dieta_export[col], errors='ignore')
            df_dieta_export.to_excel(writer, sheet_name='Dieta_Detallada', index=False)
            
            # --- Hoja 2: Resumen por Comidas (Se mantiene) ---
            df_resumen_comidas.to_excel(writer, sheet_name='Resumen_Comidas')
            
            # --- Hoja 3: Adecuación de Macros (Se mantiene) ---
            df_macros.to_excel(writer, sheet_name='Adecuacion_Macros')

            # --- NUEVO: Pestañas por Tiempo de Comida ---
            tiempos_de_comida_orden = [
                "Desayuno", "Colación Mañana", "Almuerzo", 
                "Colación Tarde", "Cena", "Colación Noche"
            ]
            
            # Columnas relevantes para estas pestañas
            columnas_exportar = [
                'Código', 'Alimento', 'Gramos', 'Kcal', 'Proteínas', 'Grasas', 'Carbohidratos', 'Fibra',
                'Agua', 'Calcio', 'Fósforo', 'Zinc', 'Hierro', 'Vitamina C', 'Sodio', 'Potasio',
                'Beta-Caroteno', 'Vitamina A', 'Tiamina', 'Riboflavina', 'Niacina', 'Acido Folico'
            ]
            
            columnas_presentes = [col for col in columnas_exportar if col in df_dieta.columns]
            
            # Usar el df_dieta_export ya limpio y numérico
            grupos = df_dieta_export.groupby('Tiempo Comida')
            
            for tiempo in tiempos_de_comida_orden:
                if tiempo in grupos.groups:
                    df_tiempo = grupos.get_group(tiempo)
                    df_tiempo_export = df_tiempo[columnas_presentes]
                    
                    # Sanear nombre de la pestaña (ej. "Colación Mañana" -> "Colacion_Manana")
                    nombre_pestaña = tiempo.replace(' ', '_').replace('ñ', 'n')
                    df_tiempo_export.to_excel(writer, sheet_name=nombre_pestaña, index=False)
            # --- FIN MODIFICACIÓN ---
            
    except ImportError:
        st.error("Se necesita la librería 'openpyxl'. Por favor, instálala con: pip install openpyxl")
        return None
    except Exception as e:
        st.error(f"Error al generar el Excel: {e}")
        return None
        
    return output.getvalue()

# --- FUNCIÓN DE EXPORTACIÓN (ACTUALIZADA) ---
@st.cache_data
def generar_excel_composicion(paciente_data):
    """
    Genera un archivo Excel en memoria con el resumen de la 
    evaluación corporal del paciente.
    """
    output = BytesIO()
    pa = paciente_data
    
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Hoja 1: Resumen Paciente
            resumen_info = {
                'Dato': ['Nombre', 'Edad', 'Sexo', 'Peso (kg)', 'Talla (cm)', 'Raza'],
                'Valor': [
                    pa.get('nombre', 'N/A'), pa.get('edad', 0), pa.get('sexo', 'N/A'), 
                    pa.get('peso', 0), pa.get('talla_cm', 0), pa.get('raza', 'N/A')
                ]
            }
            df_resumen = pd.DataFrame(resumen_info)
            df_resumen.to_excel(writer, sheet_name='Resumen_Paciente', index=False)
            
            # Hoja 2: Evaluación General
            eval_info = {
                'Métrica': ['IMC', 'Diagnóstico IMC', 'GET (kcal)', 'Fórmula GET'],
                'Valor': [f"{pa.get('imc', 0):.2f}", pa.get('diagnostico_imc', 'N/A'), f"{pa.get('get', 0):.0f}", pa.get('formula_get', 'N/A')]
            }
            df_eval = pd.DataFrame(eval_info)
            df_eval.to_excel(writer, sheet_name='Evaluacion_General', index=False)
            
            # Hoja 3: Composición Corporal (2C)
            comp = pa.get('composicion', {})
            comp_2c = comp.get('modelo_2c', {}) # LEER DESDE NUEVA ESTRUCTURA
            comp_info = {
                'Métrica': ['% Grasa Corporal (2C)', 'Masa Grasa (2C) (kg)', 'Masa Magra (2C) (kg)', 'Diagnóstico Grasa (2C)'],
                'Valor': [
                    f"{comp_2c.get('porc_grasa', 0):.1f}%", 
                    f"{comp_2c.get('masa_grasa', 0):.1f}", 
                    f"{comp_2c.get('masa_magra', 0):.1f}", 
                    comp_2c.get('diag_grasa', 'N/A')
                ]
            }
            df_comp = pd.DataFrame(comp_info)
            df_comp.to_excel(writer, sheet_name='Composicion_Corporal_2C', index=False)
            
            # Hoja 4: Somatotipo
            som = comp.get('somatotipo', {})
            som_info = {
                'Componente': ['Endomorfia', 'Mesomorfia', 'Ectomorfia', 'Clasificación'],
                'Valor': [som.get('endo', 0), som.get('meso', 0), som.get('ecto', 0), som.get('clasificacion', 'N/A')]
            }
            df_som = pd.DataFrame(som_info)
            df_som.to_excel(writer, sheet_name='Somatotipo', index=False)
            
            # Hoja 5: Medidas (Pliegues, Circs, Diams)
            df_pliegues = pd.Series(pa.get('pliegues', {})).reset_index()
            df_pliegues.columns = ['Pliegue', 'Valor (mm)']
            df_pliegues.to_excel(writer, sheet_name='Medidas_Pliegues', index=False)
            
            df_circs = pd.Series(pa.get('circunferencias', {})).reset_index()
            df_circs.columns = ['Circunferencia', 'Valor (cm)']
            df_circs.to_excel(writer, sheet_name='Medidas_Circunferencias', index=False)
            
            df_diams = pd.Series(pa.get('diametros', {})).reset_index()
            df_diams.columns = ['Diametro', 'Valor (cm)']
            df_diams.to_excel(writer, sheet_name='Medidas_Diametros', index=False)
            
            # --- NUEVA HOJA: COMPOSICIÓN 5C ---
            if 'modelo_5c' in comp: # LEER DESDE NUEVA ESTRUCTURA
                kerr = comp['modelo_5c']
                kerr_info = {
                    'Métrica': [
                        'Densidad Corporal (ISAK)', '% Grasa (Siri)', '% Grasa (Brozek)',
                        'Masa Grasa (MG) (kg)', 'Masa Muscular (MM) (kg)', 'Masa Ósea (MO) (kg)',
                        'Masa Residual (MR) (kg)', 'Masa de Piel (MP) (kg)', 'Error'
                    ],
                    'Valor': [
                        f"{kerr.get('dc', 0):.4f}", f"{kerr.get('porc_grasa_siri', 0):.1f}%", f"{kerr.get('porc_grasa_brozek', 0):.1f}%",
                        f"{kerr.get('mg', 0):.2f}", f"{kerr.get('mm', 0):.2f}", f"{kerr.get('mo', 0):.2f}",
                        f"{kerr.get('mr', 0):.2f}", f"{kerr.get('mp', 0):.2f}", str(kerr.get('error', 'N/A'))
                    ]
                }
                df_kerr = pd.DataFrame(kerr_info)
                df_kerr.to_excel(writer, sheet_name='Composicion_5C_Kerr', index=False)

    except ImportError:
        st.error("Se necesita la librería 'openpyxl'. Por favor, instálala con: pip install openpyxl")
        return None
    except Exception as e:
        st.error(f"Error al generar el Excel de composición: {e}")
        return None
            
    return output.getvalue()
# --- FIN DE FUNCIÓN DE EXPORTACIÓN ---


# --- Funciones de las Páginas de la App ---

# --- PÁGINA DE INICIO (MODIFICADA) ---
def mostrar_pagina_inicio():
    """Página principal para cargar, seleccionar y registrar pacientes."""
    st.title(f"Gestión de Pacientes 🧑‍⚕️ (Usuario: {st.session_state.usuario})")
    
    st.session_state.db_alimentos = cargar_base_de_datos_alimentos()

    # --- Sección de Carga/Selección de Paciente ---
    st.header("Seleccionar Paciente Existente")
    
    # --- MODIFICADO: Lista pacientes del usuario logueado ---
    pacientes = listar_pacientes(st.session_state.usuario)
    
    if not pacientes:
        st.info("No hay pacientes registrados para este usuario. Registre uno nuevo a continuación.")
    
    paciente_seleccionado = st.selectbox("Pacientes Registrados", options=pacientes, index=None, placeholder="Seleccione un paciente...")
    
    col1, col2, col3 = st.columns(3)
    if col1.button("Nuevo Paciente (Limpiar Formulario)", use_container_width=True):
        st.session_state.paciente_actual = None
        st.session_state.dieta_temporal = []
        st.rerun()

    if col2.button("Cargar Paciente", use_container_width=True) and paciente_seleccionado:
        # --- MODIFICADO: Carga paciente del usuario logueado ---
        st.session_state.paciente_actual = cargar_paciente(st.session_state.usuario, paciente_seleccionado)
        if st.session_state.paciente_actual:
            # Cargar la dieta del paciente a la sesión temporal
            st.session_state.dieta_temporal = st.session_state.paciente_actual.get('dieta_actual', [])
            st.success(f"Paciente '{st.session_state.paciente_actual['nombre']}' cargado.")
            st.rerun() # Recargar para reflejar el paciente cargado
    
    if col3.button("Eliminar Paciente", type="primary", use_container_width=True) and paciente_seleccionado:
        # --- MODIFICADO: Elimina paciente del usuario logueado ---
        if eliminar_paciente(st.session_state.usuario, paciente_seleccionado):
            st.success(f"Paciente '{paciente_seleccionado}' ha sido eliminado.")
            # Limpiar estado
            st.session_state.paciente_actual = None
            st.session_state.dieta_temporal = []
            st.rerun() # Recargar para actualizar el selectbox
        else:
            st.error(f"No se pudo eliminar al paciente '{paciente_seleccionado}'.")
            
    if st.session_state.paciente_actual:
        st.success(f"Paciente activo: **{st.session_state.paciente_actual['nombre']}**")

    # --- Sección de Registro/Actualización ---
    st.header("Datos del Paciente")
    
    # Si hay un paciente cargado, pre-llenamos el formulario
    pa = st.session_state.paciente_actual if st.session_state.paciente_actual else {}
    
    with st.form("form_paciente"):
        nombre = st.text_input("Nombre Completo", value=pa.get('nombre', ''))
        
        col1, col2, col3 = st.columns(3)
        edad = col1.number_input("Edad", min_value=1, max_value=120, value=pa.get('edad', 25), step=1)
        sexo = col2.selectbox("Sexo", ["Masculino", "Femenino"], index=0 if pa.get('sexo', 'Masculino') == 'Masculino' else 1)
        
        # --- NUEVO CAMPO: Raza ---
        raza_options = ["Caucásico", "Asiático", "Africano"]
        raza_default = pa.get('raza', 'Caucásico')
        raza_index = raza_options.index(raza_default) if raza_default in raza_options else 0
        raza = col3.selectbox(
            "Raza (para fórmula Lee)", 
            raza_options, 
            index=raza_index
        )
        # --- FIN NUEVO CAMPO ---

        col1, col2, col3 = st.columns(3)
        peso = col1.number_input("Peso (kg)", min_value=1.0, value=pa.get('peso', 70.0), step=0.1, format="%.1f")
        talla_cm = col2.number_input("Talla (cm)", min_value=1.0, value=pa.get('talla_cm', 170.0), step=0.1, format="%.1f")
        actividad = col3.selectbox("Actividad Física", ["Ligera", "Moderada", "Intensa"], index=1 if pa.get('actividad', 'Moderada') == 'Moderada' else 0 if pa.get('actividad', 'Ligera') == 'Ligera' else 2)
        
        opciones_formula = ["Mifflin-St Jeor", "Harris-Benedict", "Cunningham"]
        default_formula = pa.get('formula_get', 'Mifflin-St Jeor')
        default_index = opciones_formula.index(default_formula) if default_formula in opciones_formula else 0
        formula_get = st.selectbox(
            "Fórmula GET", 
            opciones_formula, 
            index=default_index,
            help="Cunningham requiere datos de composición corporal (Masa Magra) de la pestaña 'Antropometría'."
        )

        historia_clinica = st.text_area(
            "Historia Clínica Nutricional (Recordatorio 24h, alergias, etc.)", 
            value=pa.get('historia_clinica', ''),
            height=150
        )
        
        submitted = st.form_submit_button("Guardar Paciente")

    if submitted:
        if not nombre:
            st.error("El nombre es obligatorio para guardar al paciente.")
        else:
            # Calcular IMC y GET
            imc, diagnostico_imc = calcular_imc(peso, talla_cm)
            
            # --- Lógica de GET actualizada (lee desde la nueva estructura) ---
            masa_magra = pa.get('composicion', {}).get('modelo_2c', {}).get('masa_magra', 0) # Usa masa magra de 2C
            get = calcular_get(sexo, peso, talla_cm, edad, actividad, formula_get, masa_magra)
            
            if formula_get == "Cunningham" and masa_magra == 0:
                st.warning("Se seleccionó Cunningham pero no hay datos de composición corporal. El GET será 0. Por favor, vaya a 'Antropometría', calcule la composición, y vuelva a guardar aquí.")
            # --- Fin lógica actualizada ---

            # Crear diccionario del paciente
            datos_paciente = {
                'nombre': nombre,
                'edad': edad,
                'sexo': sexo,
                'peso': peso,
                'talla_cm': talla_cm,
                'actividad': actividad,
                'raza': raza, # --- GUARDAR RAZA ---
                'historia_clinica': historia_clinica,
                'imc': imc,
                'diagnostico_imc': diagnostico_imc,
                'get': get,
                'formula_get': formula_get, # Guardar la fórmula elegida
                # Mantener datos existentes si se está actualizando
                'pliegues': pa.get('pliegues', {}),
                'circunferencias': pa.get('circunferencias', {}), # Agregado
                'diametros': pa.get('diametros', {}), # Agregado
                'composicion': pa.get('composicion', {}),
                'dieta_actual': st.session_state.dieta_temporal # Usar la dieta en sesión
            }
            
            # --- MODIFICADO: Guarda el paciente para el usuario logueado ---
            nombre_archivo = guardar_paciente(st.session_state.usuario, datos_paciente)
            
            if nombre_archivo:
                st.success(f"Paciente '{nombre}' guardado/actualizado exitosamente.")
                st.session_state.paciente_actual = datos_paciente
                
                # Mostrar resultados
                st.subheader("Resultados de Evaluación Inicial")
                col1, col2 = st.columns(2)
                col1.metric("IMC", f"{imc:.2f}", diagnostico_imc)
                col2.metric("GET (Gasto Energético Total)", f"{get:.0f} kcal/día")

# --- PÁGINA DE ANTROPOMETRÍA (MODIFICADA) ---
def mostrar_pagina_antropometria():
    """Página para registrar pliegues, circunferencias, diámetros y ver composición corporal."""
    st.title("Evaluación Antropométrica 📐")
    
    if not st.session_state.paciente_actual:
        st.warning("Por favor, cargue o registre un paciente en la página de 'Inicio' primero.")
        st.stop()
        
    pa = st.session_state.paciente_actual
    st.subheader(f"Paciente: {pa['nombre']}")
    
    pliegues = pa.get('pliegues', {})
    circs = pa.get('circunferencias', {})
    diams = pa.get('diametros', {})
    
    with st.form("form_medidas"):
        st.markdown("##### Registro de Pliegues Cutáneos (mm)")
        col1, col2, col3 = st.columns(3)
        pliegues['Tricipital'] = col1.number_input("Tricipital", min_value=0.0, value=pliegues.get('Tricipital', 0.0), step=0.1, format="%.1f")
        pliegues['Bicipital'] = col2.number_input("Bicipital", min_value=0.0, value=pliegues.get('Bicipital', 0.0), step=0.1, format="%.1f")
        pliegues['Subescapular'] = col3.number_input("Subescapular", min_value=0.0, value=pliegues.get('Subescapular', 0.0), step=0.1, format="%.1f")
        
        col1, col2, col3 = st.columns(3)
        pliegues['Suprailíaco'] = col1.number_input("Suprailíaco", min_value=0.0, value=pliegues.get('Suprailíaco', 0.0), step=0.1, format="%.1f")
        pliegues['Abdominal'] = col2.number_input("Abdominal", min_value=0.0, value=pliegues.get('Abdominal', 0.0), step=0.1, format="%.1f")
        pliegues['Pantorrilla Medial'] = col3.number_input("Pantorrilla Medial", min_value=0.0, value=pliegues.get('Pantorrilla Medial', 0.0), step=0.1, format="%.1f")
        
        # --- NUEVO PLIEGUE ---
        col1, col2, col3 = st.columns(3)
        pliegues['Muslo (frontal)'] = col1.number_input("Muslo (frontal)", min_value=0.0, value=pliegues.get('Muslo (frontal)', 0.0), step=0.1, format="%.1f")


        st.divider()
        st.markdown("##### Registro de Circunferencias (cm)")
        col1, col2, col3 = st.columns(3)
        circs['Brazo (relajado)'] = col1.number_input("Brazo (relajado)", min_value=0.0, value=circs.get('Brazo (relajado)', 0.0), step=0.1, format="%.1f")
        circs['Pantorrilla (máxima)'] = col2.number_input("Pantorrilla (máxima)", min_value=0.0, value=circs.get('Pantorrilla (máxima)', 0.0), step=0.1, format="%.1f")
        # --- NUEVA CIRCUNFERENCIA ---
        circs['Muslo (medial)'] = col3.number_input("Muslo (medial)", min_value=0.0, value=circs.get('Muslo (medial)', 0.0), step=0.1, format="%.1f")
        
        st.divider()
        st.markdown("##### Registro de Diámetros Óseos (cm)")
        col1, col2, col3 = st.columns(3)
        diams['Húmero (bi-epicondilar)'] = col1.number_input("Húmero (bi-epicondilar)", min_value=0.0, value=diams.get('Húmero (bi-epicondilar)', 0.0), step=0.1, format="%.1f")
        diams['Fémur (bi-condilar)'] = col2.number_input("Fémur (bi-condilar)", min_value=0.0, value=diams.get('Fémur (bi-condilar)', 0.0), step=0.1, format="%.1f")
        # --- NUEVO DIÁMETRO ---
        diams['Muñeca (bi-estiloideo)'] = col3.number_input("Muñeca (bi-estiloideo)", min_value=0.0, value=diams.get('Muñeca (bi-estiloideo)', 0.0), step=0.1, format="%.1f")
        
        st.divider()
        
        # --- SE QUITÓ EL SELECTBOX ---
        
        submitted = st.form_submit_button("Calcular Composición y Somatotipo")

    if submitted:
        # --- CÁLCULO ACTUALIZADO (INCONDICIONAL) ---
        
        # 1. Calcular Somatotipo
        endo, meso, ecto = calcular_somatotipo(
            pa['peso'], pa['talla_cm'], pliegues, circs, diams
        )
        clasificacion_somato = clasificar_somatotipo(endo, meso, ecto)
        
        # 2. Calcular Modelo 2C
        comp_2c = calcular_composicion_2c_durnin_siri(
            pa['peso'], pa['sexo'], pa['edad'], pliegues
        )
        st.success("Modelo 2C (Durnin-Edad/Siri) calculado.")

        # 3. Calcular Modelo 5C
        comp_5c = calcular_composicion_5c_kerr(
            pa['peso'], pa['talla_cm'], pa['sexo'], pa['edad'], pa.get('raza', 'Caucásico'),
            pliegues, circs, diams
        )
        if comp_5c['error']:
            st.error(f"Error en cálculo 5C: {comp_5c['error']}")
        else:
            st.success("Modelo 5C (ISAK/Kerr) calculado.")

        # Guardar todos los datos en el paciente (Nueva estructura)
        pa['pliegues'] = pliegues
        pa['circunferencias'] = circs
        pa['diametros'] = diams
        
        pa['composicion'] = {
            'modelo_2c': comp_2c,
            'modelo_5c': comp_5c,
            'somatotipo': {
                'endo': endo, 
                'meso': meso, 
                'ecto': ecto,
                'clasificacion': clasificacion_somato
            }
        }
        
        # Recalcular GET si es Cunningham (usando masa magra del 2C)
        if pa.get('formula_get') == 'Cunningham':
            masa_magra_2c = comp_2c.get('masa_magra', 0)
            if masa_magra_2c > 0:
                nuevo_get = calcular_get(
                    pa['sexo'], pa['peso'], pa['talla_cm'], pa['edad'], 
                    pa['actividad'], 'Cunningham', masa_magra_2c
                )
                pa['get'] = nuevo_get
                st.success(f"GET (Cunningham) recalculado con nueva masa magra: {nuevo_get:.0f} kcal")

        # --- MODIFICADO: Guarda el paciente para el usuario logueado ---
        guardar_paciente(st.session_state.usuario, pa)
        st.session_state.paciente_actual = pa 
        
        st.success("Cálculos de composición y somatotipo realizados y guardados.")
        st.rerun() # Recargar para mostrar los nuevos datos

    # --- SECCIÓN DE RESULTADOS (ACTUALIZADA CON TABS) ---
    st.subheader("Diagnóstico Nutricional")
    
    # Obtener los diccionarios de resultados para fácil acceso
    pa = st.session_state.paciente_actual
    comp = pa.get('composicion', {})
    comp_2c = comp.get('modelo_2c', {})
    comp_5c = comp.get('modelo_5c', {})
    som = comp.get('somatotipo', {})

    # Crear las pestañas (Tabs)
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚖️ Evaluación General", 
        "🧍 Modelo 2C (Durnin-Edad/Siri)", 
        "🔬 Modelo 5C (ISAK/Kerr)", 
        "📊 Somatotipo (Heath-Carter)"
    ])

    # --- Tab 1: Evaluación General (IMC y GET) ---
    with tab1:
        st.markdown("##### Resumen de Indicadores Clave")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(f"IMC: {pa.get('imc', 0):.2f}", pa.get('diagnostico_imc', 'Sin datos'))
        
        with col2:
            st.metric(f"GET (Gasto Energético Total)", f"{pa.get('get', 0):.0f} kcal/día", help=f"Fórmula: {pa.get('formula_get', 'N/A')}")
        
        st.markdown(
            """
            - **IMC (Índice de Masa Corporal):** Una relación entre peso y altura, usada como indicador de riesgo.
            - **GET (Gasto Energético Total):** La estimación de calorías que el paciente quema al día.
            """
        )

    # --- Tab 2: Modelo 2C (Durnin-Edad/Siri) ---
    with tab2:
        if comp_2c and comp_2c.get('porc_grasa', 0) > 0:
            st.markdown("##### Composición de 2 Componentes (Grasa vs. Magra)")
            st.metric(f"% Grasa Corporal: {comp_2c.get('porc_grasa', 0):.1f}%", comp_2c.get('diag_grasa', 'Sin datos'))
            
            st.divider()
            
            col1, col2 = st.columns(2)
            col1.metric("Masa Grasa", f"{comp_2c.get('masa_grasa', 0):.1f} kg")
            col2.metric("Masa Magra", f"{comp_2c.get('masa_magra', 0):.1f} kg")
            
            st.caption("Cálculo basado en pliegues usando las fórmulas de Durnin & Womersley (con edad) y Siri.")
        else:
            st.info("No se han calculado datos para el Modelo 2C. Por favor, ingrese pliegues y presione 'Calcular'.")

    # --- Tab 3: Modelo 5C (ISAK/Kerr) ---
    with tab3:
        if comp_5c:
            if comp_5c.get('error'):
                st.warning(f"No se pudo calcular el modelo 5C: {comp_5c['error']}")
                st.info("Este modelo requiere todos los pliegues, circunferencias y diámetros solicitados en el formulario.")
            elif comp_5c.get('mg', 0) > 0 or comp_5c.get('mm', 0) > 0:
                st.markdown("##### Composición de 5 Componentes (Modelo Kerr)")
                st.caption("Análisis fraccionario de la masa corporal basado en protocolos ISAK.")
                
                # Sub-sección: % Grasa
                st.markdown("###### Porcentaje de Grasa (Durnin-ISAK)")
                col1, col2, col3 = st.columns(3)
                col1.metric("Densidad Corporal", f"{comp_5c['dc']:.4f} g/cm³")
                col2.metric("% Grasa (Siri)", f"{comp_5c['porc_grasa_siri']:.1f}%")
                col3.metric("% Grasa (Brozek)", f"{comp_5c['porc_grasa_brozek']:.1f}%")
                
                st.divider()
                
                # Sub-sección: Masas
                st.markdown("###### Masas Corporales Fraccionadas (kg)")
                col1, col2, col3 = st.columns(3)
                col1.metric("Masa Grasa (MG)", f"{comp_5c['mg']:.2f} kg", help="Fórmula Durnin-ISAK/Siri")
                col2.metric("Masa Muscular (MM)", f"{comp_5c['mm']:.2f} kg", help="Fórmula Lee et al.")
                col3.metric("Masa Ósea (MO)", f"{comp_5c['mo']:.2f} kg", help="Fórmula Rocha")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Masa Residual (MR)", f"{comp_5c['mr']:.2f} kg", help="Valor fijo por sexo")
                col2.metric("Masa de Piel (MP)", f"{comp_5c['mp']:.2f} kg", help="Calculada por diferencia")
                
                suma_comp = comp_5c['mg'] + comp_5c['mm'] + comp_5c['mo'] + comp_5c['mr'] + comp_5c['mp']
                col3.metric("Suma Componentes", f"{suma_comp:.2f} kg", 
                             f"Diferencia vs Peso: {pa['peso'] - suma_comp:.2f} kg")
            else:
                 st.info("No se han calculado datos para el Modelo 5C. Por favor, ingrese todas las medidas y presione 'Calcular'.")
        else:
            st.info("No se han calculado datos para el Modelo 5C. Por favor, ingrese todas las medidas y presione 'Calcular'.")

    # --- Tab 4: Somatotipo (Heath-Carter) ---
    with tab4:
        if som and som.get('endo', 0) > 0:
            st.markdown("##### Clasificación de la Forma Corporal (Heath-Carter)")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Endomorfia", f"{som.get('endo', 0):.1f}", help="Adiposidad relativa")
                st.metric("Mesomorfia", f"{som.get('meso', 0):.1f}", help="Robustez músculo-esquelética")
                st.metric("Ectomorfia", f"{som.get('ecto', 0):.1f}", help="Linealidad relativa")
                
                st.divider()
                
                clasificacion_str = som.get('clasificacion', 'N/A')
                st.metric(
                    "Clasificación",
                    f"{clasificacion_str}"
                )
                
                # --- NUEVA SECCIÓN DE EXPLICACIÓN ---
                st.divider()
                st.markdown("###### Diagnóstico Predominante")
                explicacion = obtener_explicacion_somatotipo(clasificacion_str)
                st.write(explicacion)
                # --- FIN DE NUEVA SECCIÓN ---
            
            with col2:
                fig_somato = crear_grafico_somatotipo(
                    som.get('endo', 0), 
                    som.get('meso', 0), 
                    som.get('ecto', 0)
                )
                st.plotly_chart(fig_somato, use_container_width=True)
        else:
            st.info("No se han calculado datos para el Somatotipo. Por favor, ingrese las medidas necesarias y presione 'Calcular'.")

# --- PÁGINA DE CREAR DIETA (MODIFICADA) ---
def mostrar_pagina_crear_dieta():
    """Página para buscar alimentos y agregarlos a la dieta del paciente."""
    st.title("Creación de Dieta 🍲")
    
    if not st.session_state.paciente_actual:
        st.warning("Por favor, cargue o registre un paciente en la página de 'Inicio' primero.")
        st.stop()
        
    if st.session_state.db_alimentos is None or st.session_state.db_alimentos.empty:
        st.error("La base de datos de alimentos no se ha cargado correctamente.")
        st.stop()

    # --- MODIFICACIÓN BÚSQUEDA: Crear columna de búsqueda ---
    db_alimentos = st.session_state.db_alimentos.copy()
    db_alimentos['busqueda_display'] = "[" + db_alimentos['CÓDIGO'].astype(str) + "] " + db_alimentos['NOMBRE DEL ALIMENTO']
    # --- FIN MODIFICACIÓN ---
    
    pa = st.session_state.paciente_actual
    
    st.info(f"Paciente: **{pa['nombre']}** | GET Objetivo: **{pa.get('get', 0):.0f} kcal** (Usando: {pa.get('formula_get', 'N/A')})")
    
    # --- Formulario para agregar alimento (con búsqueda integrada) ---
    st.subheader("Agregar Alimento")
    
    with st.form("form_agregar_alimento"):
        
        # --- MODIFICACIÓN BÚSQUEDA: Usar 'busqueda_display' ---
        alimento_busqueda_sel = st.selectbox(
            "Buscar y seleccionar alimento (por Código o Nombre):", 
            options=db_alimentos['busqueda_display'],
            index=None,
            placeholder="Escriba el código o nombre del alimento..."
        )
        # --- FIN MODIFICACIÓN ---
        
        col1, col2 = st.columns(2)
        gramos = col1.number_input("Cantidad (gramos)", min_value=0, value=100, step=1)
        tiempo_comida = col2.selectbox(
            "Tiempo de Comida", 
            ["Desayuno", "Colación Mañana", "Almuerzo", "Colación Tarde", "Cena", "Colación Noche"]
        )
        
        submitted = st.form_submit_button("Agregar a la Dieta")

    # --- MODIFICACIÓN BÚSQUEDA: Lógica de guardado ---
    if submitted and alimento_busqueda_sel and gramos > 0:
        # Obtener datos del alimento seleccionado usando la columna de búsqueda
        alimento_data = db_alimentos[db_alimentos['busqueda_display'] == alimento_busqueda_sel].iloc[0]
        alimento_nombre_sel = alimento_data['NOMBRE DEL ALIMENTO'] # Obtener el nombre real
        
        # Calcular nutrientes según gramos (datos están por 100g)
        factor = gramos / 100.0
        
        # --- AÑADIR TODOS LOS MICRONUTRIENTES ---
        item_dieta = {
            'id': f"{alimento_data['CÓDIGO']}_{pd.Timestamp.now().isoformat()}", # ID único
            'Tiempo Comida': tiempo_comida,
            'Código': alimento_data['CÓDIGO'],
            'Alimento': alimento_data['NOMBRE DEL ALIMENTO'],
            'Gramos': gramos,
            'Kcal': alimento_data['Kcal'] * factor,
            'Proteínas': alimento_data['Proteínas'] * factor,
            'Grasas': alimento_data['Grasas'] * factor,
            'Carbohidratos': alimento_data['Carbohidratos'] * factor,
            'Fibra': alimento_data['Fibra'] * factor,
            'Agua': alimento_data['Agua'] * factor,
            'Calcio': alimento_data['Calcio'] * factor,
            'Fósforo': alimento_data['Fósforo'] * factor,
            'Zinc': alimento_data['Zinc'] * factor,
            'Hierro': alimento_data['Hierro'] * factor,
            'Vitamina C': alimento_data['Vitamina C'] * factor,
            'Sodio': alimento_data['Sodio'] * factor,
            'Potasio': alimento_data['Potasio'] * factor,
            'Beta-Caroteno': alimento_data['Beta-Caroteno'] * factor,
            'Vitamina A': alimento_data['Vitamina A'] * factor,
            'Tiamina': alimento_data['Tiamina'] * factor,
            'Riboflavina': alimento_data['Riboflavina'] * factor,
            'Niacina': alimento_data['Niacina'] * factor,
            'Acido Folico': alimento_data['Acido Folico'] * factor
        }
        
        st.session_state.dieta_temporal.append(item_dieta)
        st.success(f"{gramos}g de '{alimento_nombre_sel}' agregados a '{tiempo_comida}'.")
        
        # --- MODIFICADO: Guarda el paciente para el usuario logueado ---
        st.session_state.paciente_actual['dieta_actual'] = st.session_state.dieta_temporal
        guardar_paciente(st.session_state.usuario, st.session_state.paciente_actual)
        st.rerun() # Recargar para que el item aparezca en la tabla correcta

    # --- MODIFICACIÓN BÚSQUEDA: Expander ---
    with st.expander("Ver detalles del alimento seleccionado en la búsqueda"):
        if alimento_busqueda_sel:
            alimento_detalles = db_alimentos[db_alimentos['busqueda_display'] == alimento_busqueda_sel].iloc[0]
            st.dataframe(alimento_detalles)
    # --- FIN MODIFICACIÓN ---

    # --- Dieta Actual ---
    st.subheader("Plan de Dieta Actual")
    if not st.session_state.dieta_temporal:
        st.info("Aún no se han agregado alimentos a la dieta.")
    else:
        # --- MODIFICACIÓN: Mostrar tablas por tiempo de comida ---
        df_dieta = pd.DataFrame(st.session_state.dieta_temporal)
        
        # Columnas a mostrar
        columnas_display = ['Alimento', 'Gramos', 'Kcal', 'Proteínas', 'Grasas', 'Carbohidratos', 'Fibra']
        
        # Asegurarse que todas las columnas existan
        columnas_finales = [col for col in columnas_display if col in df_dieta.columns]

        tiempos_de_comida_orden = [
            "Desayuno", "Colación Mañana", "Almuerzo", 
            "Colación Tarde", "Cena", "Colación Noche"
        ]
        
        # Asegurar que el DataFrame tenga la columna 'Tiempo Comida' si hay datos
        if 'Tiempo Comida' in df_dieta.columns:
            grupos = df_dieta.groupby('Tiempo Comida')
            
            for tiempo in tiempos_de_comida_orden:
                if tiempo in grupos.groups:
                    st.markdown(f"##### {tiempo}")
                    df_tiempo = grupos.get_group(tiempo)
                    # Mostrar el dataframe sin el índice
                    st.dataframe(df_tiempo[columnas_finales].round(1).reset_index(drop=True))
        else:
            st.dataframe(df_dieta[columnas_finales].round(1).reset_index(drop=True))
        
        # --- FIN MODIFICACIÓN ---
        
        
        # --- MODIFICACIÓN: Lógica de eliminación ---
        st.markdown("##### Eliminar Alimentos")
        
        # Crear un diccionario de opciones para el selectbox
        options_dict = {
            item['id']: f"{item['Tiempo Comida']}: {item['Alimento']} ({item['Gramos']}g)" 
            for item in st.session_state.dieta_temporal
        }
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            item_id_to_delete = st.selectbox(
                "Seleccionar alimento para eliminar", 
                options=list(options_dict.keys()), 
                format_func=lambda x: options_dict[x], 
                index=None, 
                placeholder="Seleccione un item..."
            )

        with col2:
            if st.button("Eliminar Alimento", use_container_width=True):
                if item_id_to_delete:
                    # Filtrar la lista temporal
                    new_dieta = [item for item in st.session_state.dieta_temporal if item['id'] != item_id_to_delete]
                    st.session_state.dieta_temporal = new_dieta
                    
                    # --- MODIFICADO: Guarda el paciente para el usuario logueado ---
                    st.session_state.paciente_actual['dieta_actual'] = new_dieta
                    guardar_paciente(st.session_state.usuario, st.session_state.paciente_actual)
                    
                    st.success("Alimento eliminado de la dieta.")
                    st.rerun()
                else:
                    st.warning("Por favor, seleccione un alimento de la lista para eliminar.")

        with col3:
            # Botón existente de limpiar
            if st.button("Limpiar toda la dieta", type="primary", use_container_width=True):
                st.session_state.dieta_temporal = []
                st.session_state.paciente_actual['dieta_actual'] = []
                # --- MODIFICADO: Guarda el paciente para el usuario logueado ---
                guardar_paciente(st.session_state.usuario, st.session_state.paciente_actual)
                st.rerun()
        # --- FIN MODIFICACIÓN ---

def mostrar_pagina_resumen_dieta():
    """Página para ver los totales de la dieta, gráficos y adecuación."""
    st.title("Resumen de Dieta y Adecuación 📊")
    
    if not st.session_state.paciente_actual:
        st.warning("Por favor, cargue o registre un paciente en la página de 'Inicio' primero.")
        st.stop()
        
    pa = st.session_state.paciente_actual
    
    # Manejar el caso de que la dieta esté vacía
    if not st.session_state.dieta_temporal:
        st.info("No hay alimentos en la dieta actual para mostrar un resumen.")
        
        # --- NUEVO: Permitir descargar evaluación aunque la dieta esté vacía ---
        st.subheader("Exportar Evaluación")
        st.write("Aunque no hay dieta, puede descargar la evaluación corporal del paciente.")
        excel_data_composicion = generar_excel_composicion(pa)
        st.download_button(
            label="📥 Descargar Evaluación Corporal (.xlsx)",
            data=excel_data_composicion,
            file_name=f"evaluacion_{pa['nombre'].replace(' ','_').lower()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        st.stop()
        # --- FIN NUEVO ---

    
    df_dieta = pd.DataFrame(st.session_state.dieta_temporal)
    
    # --- 1. Totales y Adecuación ---
    st.subheader("Totales Diarios y Adecuación al GET")
    
    get_paciente = pa.get('get', 0)
    
    total_kcal = df_dieta['Kcal'].sum()
    total_prot = df_dieta['Proteínas'].sum()
    total_fat = df_dieta['Grasas'].sum()
    total_cho = df_dieta['Carbohidratos'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Kcal Totales", f"{total_kcal:.0f} kcal")
    col2.metric("GET Objetivo", f"{get_paciente:.0f} kcal", help=f"Calculado con: {pa.get('formula_get', 'N/A')}")
    
    adecuacion = 0.0
    if get_paciente > 0:
        adecuacion = (total_kcal / get_paciente) * 100
        
    col3.metric("Adecuación GET", f"{adecuacion:.1f} %")
    
    # --- 2. Distribución de Macronutrientes ---
    st.subheader("Distribución de Macronutrientes")
    
    kcal_prot = total_prot * 4
    kcal_fat = total_fat * 9
    kcal_cho = total_cho * 4
    kcal_total_macros = kcal_prot + kcal_fat + kcal_cho
    
    if kcal_total_macros == 0:
        st.info("Agregue alimentos para ver la distribución de macronutrientes.")
        return

    porc_prot = (kcal_prot / kcal_total_macros) * 100
    porc_fat = (kcal_fat / kcal_total_macros) * 100
    porc_cho = (kcal_cho / kcal_total_macros) * 100
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("##### Distribución Actual")
        labels = ['Proteínas', 'Grasas', 'Carbohidratos']
        values = [porc_prot, porc_fat, porc_cho]

        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            textinfo='label+percent', 
            insidetextorientation='radial',
            pull=[0.05, 0.05, 0.05],
            marker_colors=['#007bff', '#dc3545', '#ffc107']
        )])
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### Ajuste de Distribución Deseada")
        # Usamos session state para mantener los sliders
        if 'dist_macros' not in st.session_state:
            st.session_state.dist_macros = {'cho': 50, 'prot': 20, 'fat': 30}
            
        porc_cho_deseado = st.slider("% Carbohidratos", 0, 100, st.session_state.dist_macros['cho'], key='slider_cho')
        porc_prot_deseado = st.slider("% Proteínas", 0, 100, st.session_state.dist_macros['prot'], key='slider_prot')
        
        # Calcular grasas automáticamente para sumar 100%
        porc_fat_deseado = 100 - porc_cho_deseado - porc_prot_deseado
        if porc_fat_deseado < 0: porc_fat_deseado = 0
        
        st.slider("% Grasas (auto)", 0, 100, porc_fat_deseado, disabled=True)
        
        st.session_state.dist_macros = {'cho': porc_cho_deseado, 'prot': porc_prot_deseado, 'fat': porc_fat_deseado}
        
        if porc_cho_deseado + porc_prot_deseado + porc_fat_deseado != 100:
            st.warning("Los porcentajes deben sumar 100%. Ajuste los sliders.")

    # --- 3. Tabla de Adecuación (Gramos) ---
    st.markdown("##### Comparativa de Gramos (Actual vs Objetivo)")
    
    gramos_prot_obj = (get_paciente * (porc_prot_deseado / 100)) / 4
    gramos_cho_obj = (get_paciente * (porc_cho_deseado / 100)) / 4
    gramos_fat_obj = (get_paciente * (porc_fat_deseado / 100)) / 9

    df_macros = pd.DataFrame({
        '': ['Actual (g)', 'Objetivo (g)', 'Diferencia (g)'],
        'Proteínas': [total_prot, gramos_prot_obj, total_prot - gramos_prot_obj],
        'Grasas': [total_fat, gramos_fat_obj, total_fat - gramos_fat_obj],
        'Carbohidratos': [total_cho, gramos_cho_obj, total_cho - gramos_cho_obj]
    }).set_index('')
    
    st.dataframe(df_macros.style.format("{:.1f}"))
    
    # --- 4. Resumen por Tiempo de Comida ---
    st.subheader("Resumen por Tiempo de Comida")
    
    df_resumen_comidas = df_dieta.groupby('Tiempo Comida')[['Kcal', 'Proteínas', 'Grasas', 'Carbohidratos']].sum()
    df_resumen_comidas = df_resumen_comidas.sort_values('Tiempo Comida')
    
    st.dataframe(df_resumen_comidas.style.format("{:.1f}", subset=pd.IndexSlice[:, ['Kcal', 'Proteínas', 'Grasas', 'Carbohidratos']]))

    # --- 5. NUEVO: Resumen de Micronutrientes ---
    st.subheader("Resumen de Micronutrientes y Minerales (Totales)")
    
    # Lista de todas las columnas de micros que queremos sumar
    micros_cols = [
        'Fibra', 'Agua', 'Calcio', 'Fósforo', 'Zinc', 'Hierro', 
        'Vitamina C', 'Sodio', 'Potasio', 'Beta-Caroteno', 
        'Vitamina A', 'Tiamina', 'Riboflavina', 'Niacina', 'Acido Folico'
    ]
    
    # Filtramos solo las columnas que realmente existen en el DataFrame (por si acaso)
    cols_presentes = [col for col in micros_cols if col in df_dieta.columns]
    
    if cols_presentes:
        df_micros = df_dieta[cols_presentes].sum().reset_index()
        df_micros.columns = ['Nutriente', 'Total']
        
        # Filtrar los que son 0 para no saturar la vista
        df_micros = df_micros[df_micros['Total'] > 0]
        
        st.dataframe(df_micros.style.format({'Total': "{:.1f}"}), use_container_width=True)
    else:
        st.info("No hay datos de micronutrientes para mostrar.")

    # --- 6. MODIFICACIÓN: Exportación (Dos botones) ---
    st.subheader("Exportar Archivos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Botón de Dieta (existente)
        excel_data_dieta = generar_excel_dieta(df_dieta, df_resumen_comidas, df_macros)
        st.download_button(
            label="📥 Descargar Dieta (.xlsx)",
            data=excel_data_dieta,
            file_name=f"dieta_{pa['nombre'].replace(' ','_').lower()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col2:
        # NUEVO Botón de Composición
        excel_data_composicion = generar_excel_composicion(pa)
        st.download_button(
            label="📥 Descargar Evaluación Corporal (.xlsx)",
            data=excel_data_composicion,
            file_name=f"evaluacion_{pa['nombre'].replace(' ','_').lower()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    # --- FIN MODIFICACIÓN ---

# --- NUEVA PÁGINA: PANEL DE ADMINISTRADOR ---
def mostrar_pagina_admin():
    """Página para gestionar usuarios (solo visible para 'admin')."""
    st.title("Panel de Administración 👑")
    
    usuarios = cargar_usuarios()
    
    st.subheader("Usuarios Existentes")
    # Mostrar usuarios pero ocultar la contraseña
    usuarios_display = []
    for user, data in usuarios.items():
        usuarios_display.append({"usuario": user, "rol": data.get("rol", "usuario")})
    
    st.dataframe(pd.DataFrame(usuarios_display), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    # --- Formulario para agregar nuevo usuario ---
    with col1:
        st.subheader("Agregar Nuevo Usuario")
        with st.form("form_agregar_usuario", clear_on_submit=True):
            nuevo_usuario = st.text_input("Nombre de Usuario")
            nueva_password = st.text_input("Contraseña", type="password")
            rol_usuario = st.selectbox("Rol", ["usuario", "admin"])
            
            submit_agregar = st.form_submit_button("Agregar Usuario")
            
            if submit_agregar:
                if not nuevo_usuario or not nueva_password:
                    st.error("Por favor, complete todos los campos.")
                elif nuevo_usuario in usuarios:
                    st.error("Ese nombre de usuario ya existe.")
                else:
                    hashed_pass = hash_password(nueva_password)
                    usuarios[nuevo_usuario] = {"password": hashed_pass, "rol": rol_usuario}
                    guardar_usuarios(usuarios)
                    st.success(f"Usuario '{nuevo_usuario}' agregado con rol '{rol_usuario}'.")
                    st.rerun()

    # --- Sección para eliminar usuario ---
    with col2:
        st.subheader("Eliminar Usuario")
        # Opciones para eliminar (no se puede eliminar al 'admin' principal)
        opciones_eliminar = [user for user in usuarios.keys() if user != 'admin']
        
        if not opciones_eliminar:
            st.info("No hay otros usuarios para eliminar.")
        else:
            usuario_a_eliminar = st.selectbox("Seleccionar Usuario a Eliminar", options=opciones_eliminar)
            
            if st.button("Eliminar Usuario", type="primary"):
                if usuario_a_eliminar in usuarios:
                    del usuarios[usuario_a_eliminar]
                    guardar_usuarios(usuarios)
                    st.success(f"Usuario '{usuario_a_eliminar}' eliminado.")
                    st.rerun()
                else:
                    st.error("El usuario seleccionado no existe.")

# --- NUEVA PÁGINA: INICIO DE SESIÓN ---
def mostrar_pagina_login():
    """Muestra la página de inicio de sesión centrada."""
    
    # Centrar el formulario de inicio de sesión
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        # Mostrar el logo si existe, si no, el título
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
            st.title("Bienvenido a ComVida")
        else:
            st.title("ComVida 🍏")
            st.caption("Bienvenido a su asistente nutricional")

        st.markdown("---")
        
        with st.form("login_form"):
            usuario = st.text_input("Usuario")
            password = st.text_input("Contraseña", type="password")
            submitted = st.form_submit_button("Ingresar")

            if submitted:
                usuarios = cargar_usuarios()
                
                # Verificar si el usuario existe y la contraseña es correcta
                if usuario in usuarios and check_password(password, usuarios[usuario]['password']):
                    st.session_state.autenticado = True
                    st.session_state.usuario = usuario
                    st.session_state.rol = usuarios[usuario].get('rol', 'usuario') # Asignar rol
                    st.success("Inicio de sesión exitoso. Redirigiendo...")
                    st.rerun()
                else:
                    st.error("Usuario o contraseña incorrectos.")
                    st.session_state.autenticado = False


# --- Lógica Principal (Main App Router) ---
def mostrar_app_principal():
    """Muestra la aplicación principal (barra lateral y páginas) después de iniciar sesión."""
    
    # --- Barra Lateral (Sidebar) ---
    
    # --- MODIFICACIÓN: Lógica del Logo ---
    if os.path.exists(LOGO_PATH):
        st.sidebar.image(LOGO_PATH, use_container_width=True)
    else:
        # Mostrar un placeholder si logo.png no se encuentra
        st.sidebar.image("https://placehold.co/400x100/007bff/FFFFFF?text=ComVida&font=inter", use_container_width=True)
        st.sidebar.caption("Reemplaza esta imagen creando un archivo 'logo.png'.")
    # --- FIN MODIFICACIÓN ---

    st.sidebar.title("Navegación Principal")
    
    # Definir las páginas
    menu = {
        "🏠 Inicio": mostrar_pagina_inicio,
        "📐 Antropometría": mostrar_pagina_antropometria,
        "🍲 Crear Dieta": mostrar_pagina_crear_dieta,
        "📊 Resumen de Dieta": mostrar_pagina_resumen_dieta
    }
    
    # --- NUEVO: Añadir Panel de Admin solo si el rol es 'admin' ---
    if st.session_state.rol == 'admin':
        menu["👑 Panel de Admin"] = mostrar_pagina_admin
    # --- FIN NUEVO ---

    # --- MODIFICACIÓN: Navegación con Botones ---
    
    # Función helper para cambiar la página en el estado
    def set_pagina(pagina):
        st.session_state.pagina_activa = pagina

    for pagina_nombre in menu.keys():
        # Usar botón "primary" para la página activa, "secondary" para las demás
        tipo_boton = "primary" if st.session_state.pagina_activa == pagina_nombre else "secondary"
        
        st.sidebar.button(
            pagina_nombre, 
            on_click=set_pagina, 
            args=(pagina_nombre,), 
            use_container_width=True,
            type=tipo_boton
        )
    # --- FIN MODIFICACIÓN ---
    
    st.sidebar.divider()
    
    # Mostrar paciente activo en la barra lateral
    if st.session_state.paciente_actual:
        pa = st.session_state.paciente_actual
        st.sidebar.subheader("Paciente Activo")
        st.sidebar.markdown(
            f"**Nombre:** {pa['nombre']}\n\n"
            f"**Edad:** {pa.get('edad', 0)} años\n\n"
            f"**Peso:** {pa.get('peso', 0):.1f} kg\n\n"
            f"**IMC:** {pa.get('imc', 0):.2f}\n\n"
            f"**GET:** {pa.get('get', 0):.0f} kcal (*{pa.get('formula_get', 'N/A')}*)"
        )
    else:
        st.sidebar.info("No hay ningún paciente cargado.")

    st.sidebar.divider()
    
    # --- NUEVO: Botón de Cerrar Sesión ---
    st.sidebar.info(f"Usuario: {st.session_state.usuario} ({st.session_state.rol})")
    if st.sidebar.button("Cerrar Sesión", use_container_width=True, type="primary"):
        st.session_state.autenticado = False
        st.session_state.usuario = None
        st.session_state.rol = None
        st.session_state.pagina_activa = "🏠 Inicio" # Resetear página
        st.session_state.paciente_actual = None # Limpiar paciente
        st.session_state.dieta_temporal = []
        st.rerun()
    # --- FIN NUEVO ---

    # Llamar a la función de la página seleccionada desde el session_state
    pagina_a_mostrar = menu[st.session_state.pagina_activa]
    pagina_a_mostrar()


def main():
    # Inicializar Session State para autenticación
    if 'autenticado' not in st.session_state:
        st.session_state.autenticado = False
        st.session_state.usuario = None
        st.session_state.rol = None

    # Inicializar otros estados de la app
    if 'paciente_actual' not in st.session_state:
        st.session_state.paciente_actual = None
    if 'db_alimentos' not in st.session_state:
        st.session_state.db_alimentos = None
    if 'dieta_temporal' not in st.session_state:
        st.session_state.dieta_temporal = []
    if 'pagina_activa' not in st.session_state:
        st.session_state.pagina_activa = "🏠 Inicio"
        
    # Inicializar directorios
    inicializar_pacientes()

    # --- Lógica de Enrutamiento Principal (Login vs App) ---
    if not st.session_state.autenticado:
        mostrar_pagina_login()
    else:
        mostrar_app_principal()

if __name__ == "__main__":

    main()

