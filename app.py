import streamlit as st
import pandas as pd
import os
import json
import tempfile
import plotly.graph_objects as go
from io import BytesIO
import numpy as np # Importado para cálculos matemáticos
import bcrypt # Importado para hashear contrpresas
import sys # Importado para compatibilidad de rutas de archivos

# --- NUEVAS IMPORTACIONES PARA IA ---
import google.generativeai as genai
import traceback # Para un mejor manejo de errores
# Importación específica para manejar los filtros de seguridad
from google.generativeai.types import HarmCategory, HarmBlockThreshold
# --- FIN NUEVAS IMPORTACIONES ---

# --- NUEVAS IMPORTACIONES PARA PDF ---
from datetime import datetime # Para la fecha en el PDF
from fpdf import FPDF # Para generar el PDF
import plotly.io as pio


# --- FIN NUEVAS IMPORTACIONES ---


# --- Configuración de la Página ---
st.set_page_config(
    page_title="ComVida",
    page_icon="🥕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constantes y Variables Globales ---

# Determinar la ruta base para los archivos (funciona en Streamlit Cloud y local)
BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0] if hasattr(sys, 'argv') else __file__))

BASE_DIRECTORIO_PACIENTES = os.path.join(BASE_DIR, "pacientes") 
DB_ALIMENTOS_PATH = os.path.join(BASE_DIR, "alimentos.csv")
DB_USUARIOS_PATH = os.path.join(BASE_DIR, "usuarios.json")
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")

# --- NUEVO: Configuración de API Key de Gemini ---
try:
    # Lee la API key desde los secretos de Streamlit (archivo .streamlit/secrets.toml)
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("Error: No se encontró la API Key de Google. Asegúrate de crear el archivo .streamlit/secrets.toml")
    GOOGLE_API_KEY = None
# --- FIN NUEVO ---


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
        df = pd.read_csv(filepath, delimiter=';', header=0, skiprows=[1])
        
        columnas_limpias = []
        for col in df.columns:
            col_limpia = str(col).replace('\r\n', ' ').replace('\n', ' ').replace('  ', ' ').strip()
            columnas_limpias.append(col_limpia)
        df.columns = columnas_limpias

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
            'β caroteno equivalentes totais <CARTBQ>': 'Beta-Caroteno',
            'Vitamina A equivalentes totais <VITA>': 'Vitamina A',
            'Tiamina <THIA>': 'Tiamina',
            'Riboflavina <RIBF>': 'Riboflavina',
            'Niacina <NIA>': 'Niacina',
            'Ácido fólico': 'Acido Folico'
        }
        df = df.rename(columns=columnas_renombrar)

        cols_numericas = [
            'Kcal', 'Proteínas', 'Grasas', 'Carbohidratos', 'Fibra', 'Agua',
            'Calcio', 'Fósforo', 'Zinc', 'Hierro', 'Vitamina C', 'Sodio', 'Potasio',
            'Beta-Caroteno', 'Vitamina A', 'Tiamina', 'Riboflavina', 'Niacina', 'Acido Folico'
        ]
        
        for col in cols_numericas:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', '.'), 
                    errors='coerce'
                ).fillna(0)
            else:
                df[col] = 0
                
        return df

    except FileNotFoundError:
        st.error(f"Error Crítico: No se encontró el archivo '{filepath}'. Asegúrese de que 'alimentos.csv' esté en el mismo directorio.")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar la base de datos de alimentos: {e}")
        st.stop()


# --- NUEVO: Funciones de Inteligencia Artificial (Gemini) ---

@st.cache_resource
def configurar_modelo_gemini():
    """
    Configura y retorna el modelo generativo de Gemini.
    Usa @st.cache_resource para no reiniciarlo en cada rerun.
    """
    if not GOOGLE_API_KEY:
        st.error("API Key de Google no configurada. La IA no funcionará.")
        return None
        
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        modelo = genai.GenerativeModel('gemini-2.5-pro') 
        return modelo
    except Exception as e:
        st.error(f"Error al configurar el modelo Gemini: {e}")
        return None

def generar_respuesta_gemini(modelo, prompt):
    """
    Envía un prompt al modelo Gemini y maneja la respuesta y los errores,
    ajustando la configuración de seguridad y el límite de tokens.
    """
    if modelo is None:
        return "Error: El modelo de IA no está configurado."
        
    try:
        # Configuración de seguridad ajustada
        safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

        configuracion_generacion = {
            "temperature": 0.7,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 8192, # Límite aumentado
        }
        
        respuesta = modelo.generate_content(
            prompt, 
            generation_config=configuracion_generacion,
            safety_settings=safety_settings 
        )
        
        # Manejo de respuesta bloqueada (más robusto)
        if not respuesta.parts:
            razon_bloqueo = "Razón desconocida"
            if respuesta.candidates and respuesta.candidates[0].finish_reason:
                razon_bloqueo = f"Razón: {respuesta.candidates[0].finish_reason.name}"
                
            st.error(f"Respuesta bloqueada. {razon_bloqueo}")
            st.info("Esto puede ser por un filtro de seguridad o porque la respuesta superó el límite de tokens.")
            return f"La respuesta fue bloqueada ({razon_bloqueo})."

        return respuesta.text
        
    except Exception as e:
        st.error(f"Error al generar la respuesta de la IA: {e}")
        traceback.print_exc() 
        return f"Error de la API: {e}"
# --- FIN DE FUNCIONES DE IA ---


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
            geb = 0 
    
    else: # Fallback a Mifflin por si acaso
        if sexo == 'Masculino':
            geb = (10 * peso) + (6.25 * talla_cm) - (5 * edad) + 5
        else: # Femenino
            geb = (10 * peso) + (6.25 * talla_cm) - (5 * edad) - 161
            
    if actividad == 'Ligera':
        factor = 1.375
    elif actividad == 'Moderada':
        factor = 1.55
    else: # Intensa
        factor = 1.725
        
    get = geb * factor
    return get

# --- Funciones de Composición Corporal y Somatotipo (ISAK) ---

def get_densidad_durnin(sexo, edad, L):
    """Calcula la Densidad Corporal (D) según Durnin & Womersley (1974)."""
    
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

def calcular_porcentaje_grasa_siri(densidad):
    """
    Calcula el % de grasa usando la fórmula de Siri (1961).
    """
    if densidad <= 0:
        return 0
    porc_grasa = ((4.95 / densidad) - 4.5) * 100
    
    if porc_grasa < 0: porc_grasa = 0
    if porc_grasa > 60: porc_grasa = 60 # Límite superior razonable
    return porc_grasa


def calcular_composicion_personalizada(formula_nombre, sexo, edad, peso, pliegues, diams, circs, talla_cm):
    """
    Calcula la Densidad Corporal (DC) y % Grasa (Siri) usando una fórmula seleccionada.
    """
    dc = 0.0
    p_tri = pliegues.get('Tricipital', 0.0)
    p_bic = pliegues.get('Bicipital', 0.0)
    p_sub = pliegues.get('Subescapular', 0.0)
    p_sup = pliegues.get('Suprailíaco', 0.0) 
    p_abd = pliegues.get('Abdominal', 0.0)
    p_mus = pliegues.get('Muslo (frontal)', 0.0)
    
    try:
        if formula_nombre == "Sloan (1967) - Varones" and sexo == 'Masculino':
            if p_mus == 0 or p_sub == 0:
                return None, "Faltan pliegues de Muslo Frontal o Subescapular."
            dc = 1.1043 - (0.001327 * p_mus) - (0.001310 * p_sub)
        
        elif formula_nombre == "Wilmore & Behnke (1969) - Varones" and sexo == 'Masculino':
            if p_abd == 0 or p_mus == 0:
                return None, "Faltan pliegues Abdominal o Muslo Frontal."
            dc = 1.08543 - (0.000886 * p_abd) - (0.00040 * p_mus)

        elif formula_nombre == "Katch & McArdle (1973) - Varones" and sexo == 'Masculino':
            if p_tri == 0 or p_sub == 0 or p_abd == 0:
                return None, "Faltan pliegues Triccipital, Subescapular o Abdominal."
            dc = 1.09655 - 0.00049 - (0.00103 * p_tri) - (0.00056 * p_sub) + (0.00054 * p_abd)

        elif formula_nombre == "Sloan, Burt, & Blyth (1962) - Mujeres" and sexo == 'Femenino':
            if p_sup == 0 or p_tri == 0:
                return None, "Faltan pliegues Suprailíaco (Cresta Iliaca) o Triccipital."
            dc = 1.0764 - (0.00081 * p_sup) - (0.00088 * p_tri)

        elif formula_nombre == "Wilmore & Behnke (1970) - Mujeres" and sexo == 'Femenino':
            if p_sub == 0 or p_tri == 0 or p_mus == 0:
                return None, "Faltan pliegues Subescapular, Triccipital o Muslo Frontal."
            dc = 1.06234 - (0.00068 * p_sub) - (0.00039 * p_tri) - (0.00025 * p_mus)
        
        elif formula_nombre == "Jackson, Pollock, & Ward (1980) - Mujeres" and sexo == 'Femenino':
            if p_tri == 0 or p_mus == 0 or p_sup == 0:
                return None, "Faltan pliegues Triccipital, Muslo Frontal o Suprailíaco."
            suma_3_pliegues = p_tri + p_mus + p_sup
            if suma_3_pliegues <= 0:
                return None, "La suma de pliegues no puede ser cero."
            log_suma = np.log10(suma_3_pliegues)
            dc = 1.221389 - (0.04057 * log_suma) - (0.00016 * edad)

        else:
            return None, "La fórmula seleccionada no es compatible con el sexo del paciente o no es válida."

        if dc <= 0:
            return None, "Cálculo de Densidad inválido (<= 0)."

        porc_grasa = calcular_porcentaje_grasa_siri(dc)
        masa_grasa = peso * (porc_grasa / 100)
        masa_magra = peso - masa_grasa
        
        return {
            'formula_usada': formula_nombre,
            'dc': dc,
            'porc_grasa': porc_grasa,
            'masa_grasa': masa_grasa,
            'masa_magra': masa_magra
        }, None

    except Exception as e:
        return None, f"Error matemático en el cálculo: {e}. Revise los pliegues."


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
    densidad = get_densidad_durnin(sexo, edad, L) 
    porc_grasa = calcular_porcentaje_grasa_siri(densidad)
    
    if porc_grasa == 0 and densidad <= 0:
         return {'masa_grasa': 0, 'masa_magra': 0, 'porc_grasa': 0, 'diag_grasa': "Error en cálculo de densidad"}
    
    masa_grasa = peso * (porc_grasa / 100)
    masa_magra = peso - masa_grasa
    
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

def obtener_diagnostico_5c(componente, porc_valor, sexo):
    """
    Proporciona un diagnóstico simple para un componente del modelo 5C 
    basado en su porcentaje y el sexo.
    """
    diagnostico = "N/A"
    
    if componente == 'MG': # Masa Grasa
        if sexo == 'Masculino':
            if porc_valor < 8: diagnostico = "Muy Bajo (Esencial)"
            elif porc_valor <= 15: diagnostico = "Bajo (Atleta)"
            elif porc_valor <= 22: diagnostico = "Saludable"
            elif porc_valor <= 28: diagnostico = "Elevado"
            else: diagnostico = "Muy Elevado"
        else: # Femenino
            if porc_valor < 15: diagnostico = "Muy Bajo (Esencial)"
            elif porc_valor <= 22: diagnostico = "Bajo (Atleta)"
            elif porc_valor <= 30: diagnostico = "Saludable"
            elif porc_valor <= 38: diagnostico = "Elevado"
            else: diagnostico = "Muy Elevado"
            
    elif componente == 'MM': # Masa Muscular
        if sexo == 'Masculino':
            if porc_valor < 38: diagnostico = "Bajo"
            elif porc_valor <= 44: diagnostico = "Promedio"
            elif porc_valor <= 50: diagnostico = "Alto (Atlético)"
            else: diagnostico = "Muy Alto (Hipertrofia)"
        else: # Femenino
            if porc_valor < 30: diagnostico = "Bajo"
            elif porc_valor <= 36: diagnostico = "Promedio"
            elif porc_valor <= 42: diagnostico = "Alto (Atlético)"
            else: diagnostico = "Muy Alto (Hipertrofia)"

    elif componente == 'MO': # Masa Ósea
        if 12 <= porc_valor <= 15: diagnostico = "Promedio (Robusto)"
        elif porc_valor < 12: diagnostico = "Ligero"
        else: diagnostico = "Muy Robusto"
        
    elif componente == 'MR': # Masa Residual
        diagnostico = "Componente fijo (Órganos)"
    elif componente == 'MP': # Masa Piel
        diagnostico = "Componente fijo"
        
    return diagnostico


def calcular_composicion_5c_kerr(peso, talla_cm, sexo, pliegues, diams):
    """
    Calcula el modelo de 5 componentes (ISAK - Kerr, 1988)
    """
    H_m = talla_cm / 100.0 
    resultados = {
        'mg_kg': 0, 'mg_porc': 0, 'mg_diag': 'N/A',
        'mo_kg': 0, 'mo_porc': 0, 'mo_diag': 'N/A',
        'mr_kg': 0, 'mr_porc': 0, 'mr_diag': 'N/A',
        'mp_kg': 0, 'mp_porc': 0, 'mp_diag': 'N/A',
        'mm_kg': 0, 'mm_porc': 0, 'mm_diag': 'N/A',
        'dc': 0, 'error': None, 'suma_total': 0
    }

    if H_m <= 0 or peso <= 0:
        resultados['error'] = "Peso o Talla deben ser mayores a 0."
        return resultados

    # --- 1. Masa Grasa (MG) ---
    p_tri = pliegues.get('Tricipital', 0.0)
    p_sub = pliegues.get('Subescapular', 0.0)
    p_bic = pliegues.get('Bicipital', 0.0)
    p_sup = pliegues.get('Suprailíaco', 0.0)
    suma_4_pliegues = p_tri + p_sub + p_bic + p_sup
    
    if suma_4_pliegues <= 0:
        resultados['error'] = "Pliegues para DC (Durnin) no ingresados."
        return resultados

    log_suma = np.log10(suma_4_pliegues)
    
    if sexo == 'Masculino':
        DC = 1.1765 - (0.0744 * log_suma) 
    else: # Femenino
        DC = 1.1567 - (0.0717 * log_suma) 

    if DC <= 0:
        resultados['error'] = "Error en cálculo de Densidad Corporal."
        return resultados

    porc_grasa_siri = calcular_porcentaje_grasa_siri(DC)
    masa_grasa = (porc_grasa_siri / 100.0) * peso
    
    resultados['dc'] = DC
    resultados['mg_kg'] = masa_grasa
    resultados['mg_porc'] = porc_grasa_siri
    resultados['mg_diag'] = obtener_diagnostico_5c('MG', porc_grasa_siri, sexo)

    # --- 2. Masa Ósea (MO) - Rocha (1975) ---
    d_muñeca_cm = diams.get('Muñeca (bi-estiloideo)', 0.0)
    d_femur_cm = diams.get('Fémur (bi-condilar)', 0.0)
    
    if d_muñeca_cm <= 0 or d_femur_cm <= 0:
        resultados['error'] = "Diámetros de Muñeca y Fémur (Rocha) no ingresados."
        return resultados
        
    d_muñeca_m = d_muñeca_cm / 100.0
    d_femur_m = d_femur_cm / 100.0
    
    termino_base = (H_m ** 2) * d_muñeca_m * d_femur_m * 400
    if termino_base <= 0: 
        resultados['error'] = "Error en cálculo de MO (base negativa)."
        return resultados
        
    masa_osea = 3.02 * (termino_base ** 0.712)
    porc_oseo = (masa_osea / peso) * 100.0
    
    resultados['mo_kg'] = masa_osea
    resultados['mo_porc'] = porc_oseo
    resultados['mo_diag'] = obtener_diagnostico_5c('MO', porc_oseo, sexo)

    # --- 3. Masa Residual (MR) ---
    if sexo == 'Masculino':
        porc_residual = 24.0
    else: # Femenino
        porc_residual = 21.0
    masa_residual = (porc_residual / 100.0) * peso
    
    resultados['mr_kg'] = masa_residual
    resultados['mr_porc'] = porc_residual
    resultados['mr_diag'] = obtener_diagnostico_5c('MR', porc_residual, sexo)

    # --- 4. Masa de Piel (MP) ---
    porc_piel = 3.5
    masa_piel = (porc_piel / 100.0) * peso
    
    resultados['mp_kg'] = masa_piel
    resultados['mp_porc'] = porc_piel
    resultados['mp_diag'] = obtener_diagnostico_5c('MP', porc_piel, sexo)

    # --- 5. Masa Muscular (MM) (Por diferencia) ---
    suma_componentes_fijos = masa_grasa + masa_osea + masa_residual + masa_piel
    masa_muscular = peso - suma_componentes_fijos
    
    if masa_muscular < 0:
        masa_muscular = 0 
        resultados['error'] = "Error: La suma de MG, MO, MR y MP supera el peso total. Revise las mediciones."

    porc_muscular = (masa_muscular / peso) * 100.0
    
    resultados['mm_kg'] = masa_muscular
    resultados['mm_porc'] = porc_muscular
    resultados['mm_diag'] = obtener_diagnostico_5c('MM', porc_muscular, sexo)

    resultados['suma_total'] = suma_componentes_fijos + masa_muscular

    return resultados


def calcular_somatotipo(peso, talla_cm, pliegues, circs, diams):
    """
    Calcula el Somatotipo de Heath-Carter.
    """
    talla_m = talla_cm / 100
    if talla_m == 0 or peso == 0:
        return 0, 0, 0
        
    triceps = pliegues.get('Tricipital', 0.0)
    subescapular = pliegues.get('Subescapular', 0.0)
    suprailiaco = pliegues.get('Suprailíaco', 0.0)
    pantorrilla = pliegues.get('Pantorrilla Medial', 0.0)
    humero_diam = diams.get('Húmero (bi-epicondilar)', 0.0)
    femur_diam = diams.get('Fémur (bi-condilar)', 0.0)
    brazo_circ = circs.get('Brazo (relajado)', 0.0)
    pantorrilla_circ = circs.get('Pantorrilla (máxima)', 0.0)

    # 1. ENDOMORFIA
    if talla_cm == 0: 
        X = 0
    else:
        X = (triceps + subescapular + suprailiaco) * (170.18 / talla_cm)
        
    if X <= 0:
        endo = 0
    else:
        endo = -0.7182 + (0.1451 * X) - (0.00068 * (X**2)) + (0.0000014 * (X**3))
        if endo < 0.1: endo = 0.1
        
    # 2. MESOMORFIA
    brazo_circ_corr = brazo_circ - (triceps / 10)
    pantorrilla_circ_corr = pantorrilla_circ - (pantorrilla / 10)
    
    if humero_diam <= 0 or femur_diam <= 0 or brazo_circ_corr <= 0 or pantorrilla_circ_corr <= 0 or talla_cm <= 0:
        meso = 0
    else:
        meso = (0.858 * humero_diam) + \
               (0.601 * femur_diam) + \
               (0.188 * brazo_circ_corr) + \
               (0.161 * pantorrilla_circ_corr) - \
               (0.131 * talla_cm) + 4.5
        if meso < 0.1: meso = 0.1

    # 3. ECTOMORFIA
    if peso <= 0: 
        HWR = 0
        ecto = 0.1
    else:
        HWR = talla_cm / (peso ** (1/3))
        if HWR > 40.75:
            ecto = (0.732 * HWR) - 28.58
        elif HWR > 38.25:
            ecto = (0.463 * HWR) - 17.63
        else:
            ecto = 0.1
        
    return round(endo, 1), round(meso, 1), round(ecto, 1)


def clasificar_somatotipo(endo, meso, ecto):
    """
    Clasifica el somatotipo en una categoría diagnóstica simple 
    basada en el componente dominante.
    """
    componentes = {'Endomorfo': endo, 'Mesomorfo': meso, 'Ectomorfo': ecto}
    ordenados = sorted(componentes.items(), key=lambda item: item[1], reverse=True)
    
    d1_nombre, d1_val = ordenados[0]
    d2_nombre, d2_val = ordenados[1]
    d3_nombre, d3_val = ordenados[2]

    if (d1_val - d2_val) < 1.0:
        if (d2_val - d3_val) < 1.0:
            return "Central" 
        else:
            return f"{d1_nombre}-{d2_nombre}" 
    else:
        if (d2_val - d3_val) < 1.0:
            return f"{d1_nombre} balanceado" 
        else:
            return f"{d1_nombre} (dominante)" 

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
    return explicaciones.get(clasificacion, "No se pudo generar una explicación para esta clasificación.")

def crear_grafico_somatotipo(endo, meso, ecto):
    """
    Genera un gráfico de Plotly (Somatocarta) para el somatotipo dado.
    """
    x = ecto - endo
    y = (2 * meso) - (endo + ecto)

    boundary_x = [
        0.0, -1.0, -2.0, -3.0, -4.0, -4.0, -3.0, -2.0, -1.0, 
        0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 0.0
    ]
    boundary_y = [
        1.0, 2.0, 4.0, 7.0, 11.0, 10.0, 9.0, 8.0, 7.0, 
        6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 7.0, 4.0, 2.0, 1.0
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=boundary_x, y=boundary_y, fill="toself",
        fillcolor='rgba(230, 230, 230, 0.5)',
        line=dict(color='black', width=1), name='Somatocarta'
    ))
    fig.add_trace(go.Scatter(
        x=[-8, 8], y=[0, 0], mode='lines', 
        line=dict(color='grey', width=1, dash='dot'), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[-10, 18], mode='lines', 
        line=dict(color='grey', width=1, dash='dot'), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[x], y=[y], mode='markers',
        marker=dict(color='red', size=12, line=dict(color='black', width=1)),
        name='Paciente'
    ))

    fig.update_layout(
        title='Somatocarta',
        xaxis=dict(range=[-8, 8], title='X (Ecto - Endo)'),
        yaxis=dict(range=[-10, 18], title='Y (2*Meso - (Endo+Ecto))'),
        width=500, height=500, showlegend=False,
        yaxis_scaleanchor="x", yaxis_scaleratio=1
    )
    return fig


# --- Funciones de Manejo de Pacientes ---

def get_directorio_pacientes_usuario(username):
    """Retorna la ruta al directorio de pacientes para un usuario específico."""
    if not username:
        return None
    safe_username = "".join(c for c in username if c.isalnum() or c in ('_', '-')).rstrip()
    user_dir = os.path.join(BASE_DIRECTORIO_PACIENTES, safe_username)
    return user_dir

def inicializar_pacientes():
    """Crea el directorio BASE de pacientes si no existe."""
    if not os.path.exists(BASE_DIRECTORIO_PACIENTES):
        os.makedirs(BASE_DIRECTORIO_PACIENTES)

def listar_pacientes(username):
    """Retorna una lista de pacientes (sin .json) para el usuario especificado."""
    user_dir = get_directorio_pacientes_usuario(username)
    if not user_dir or not os.path.exists(user_dir):
        return [] 

    pacientes = []
    if os.path.exists(user_dir):
        for f in os.listdir(user_dir):
            if f.endswith('.json'):
                pacientes.append(f.replace('.json', ''))
    return pacientes

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
            # Asegurar que las claves principales existan
            datos_paciente.setdefault('pliegues', {})
            datos_paciente.setdefault('circunferencias', {}) 
            datos_paciente.setdefault('diametros', {}) 
            datos_paciente.setdefault('composicion', {})
            datos_paciente.setdefault('dieta_actual', [])
            datos_paciente.setdefault('plan_semanal', {}) # <-- NUEVA LÍNEA
            return datos_paciente
    except Exception as e:
        st.error(f"Error al cargar el paciente {nombre_archivo}: {e}")
        return None

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
        
    if not os.path.exists(user_dir):
        try:
            os.makedirs(user_dir)
        except OSError as e:
            st.error(f"Error crítico al crear directorio de paciente: {e}")
            return None
        
    nombre_archivo = nombre.replace(' ', '_').replace('.', '').lower()
    filepath = os.path.join(user_dir, f"{nombre_archivo}.json")
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(datos_paciente, f, indent=4, ensure_ascii=False)
        return nombre_archivo
    except Exception as e:
        st.error(f"Error al guardar el paciente: {e}")
        return None

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

# --- Funciones de Utilidad (Exportación) ---

@st.cache_data
def generar_excel_dieta(df_dieta, df_resumen_comidas, df_macros):
    """
    Genera un archivo Excel en memoria con varias pestañas.
    """
    output = BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            
            df_dieta_export = df_dieta.copy()
            for col in df_dieta_export.columns:
                if df_dieta_export[col].dtype == 'object':
                    df_dieta_export[col] = pd.to_numeric(df_dieta_export[col], errors='ignore')
            df_dieta_export.to_excel(writer, sheet_name='Dieta_Detallada', index=False)
            
            # Asegurarse de que el índice (Tiempo Comida) se escriba en el Excel
            df_resumen_comidas.to_excel(writer, sheet_name='Resumen_Comidas', index=True)
            
            df_macros.to_excel(writer, sheet_name='Adecuacion_Macros')

            # --- Generar pestañas individuales por tiempo de comida ---
            tiempos_de_comida_orden = [
                "Desayuno", "Colación Mañana", "Almuerzo", 
                "Colación Tarde", "Cena", "Colación Noche"
            ]
            
            columnas_exportar = [
                'Código', 'Alimento', 'Gramos', 'Kcal', 'Proteínas', 'Grasas', 'Carbohidratos', 'Fibra',
                'Agua', 'Calcio', 'Fósforo', 'Zinc', 'Hierro', 'Vitamina C', 'Sodio', 'Potasio',
                'Beta-Caroteno', 'Vitamina A', 'Tiamina', 'Riboflavina', 'Niacina', 'Acido Folico'
            ]
            
            columnas_presentes = [col for col in columnas_exportar if col in df_dieta.columns]
            
            if 'Tiempo Comida' in df_dieta_export.columns:
                grupos = df_dieta_export.groupby('Tiempo Comida')
                
                for tiempo in tiempos_de_comida_orden:
                    if tiempo in grupos.groups: 
                        df_tiempo = grupos.get_group(tiempo)
                        df_tiempo_export = df_tiempo[columnas_presentes]
                        
                        nombre_pestaña = tiempo.replace(' ', '_').replace('ñ', 'n')
                        df_tiempo_export.to_excel(writer, sheet_name=nombre_pestaña, index=False)
            
    except ImportError:
        st.error("Se necesita la librería 'openpyxl'. Por favor, instálala con: pip install openpyxl")
        return None
    except Exception as e:
        st.error(f"Error al generar el Excel: {e}")
        return None
        
    return output.getvalue()

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
            comp_2c = comp.get('modelo_2c', {}) 
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
            
            # --- MODIFICACIÓN HOJA: COMPOSICIÓN 5C ---
            if 'modelo_5c' in comp: 
                kerr = comp['modelo_5c']
                
                if kerr.get('error'):
                    kerr_info = {'Error': [kerr['error']]}
                else:
                    kerr_info = {
                        'Componente': [
                            "Masa Grasa (MG)", "Masa Muscular (MM)", "Masa Ósea (MO)",
                            "Masa Residual (MR)", "Masa de Piel (MP)", "SUMA TOTAL"
                        ],
                        'Masa (kg)': [
                            f"{kerr.get('mg_kg', 0):.2f}", f"{kerr.get('mm_kg', 0):.2f}", 
                            f"{kerr.get('mo_kg', 0):.2f}", f"{kerr.get('mr_kg', 0):.2f}",
                            f"{kerr.get('mp_kg', 0):.2f}", f"{kerr.get('suma_total', 0):.2f}"
                        ],
                        '% Corporal': [
                            f"{kerr.get('mg_porc', 0):.1f}%", f"{kerr.get('mm_porc', 0):.1f}%",
                            f"{kerr.get('mo_porc', 0):.1f}%", f"{kerr.get('mr_porc', 0):.1f}%",
                            f"{kerr.get('mp_porc', 0):.1f}%", "100.0%"
                        ],
                        'Diagnóstico': [
                            kerr.get('mg_diag', 'N/A'), kerr.get('mm_diag', 'N/A'),
                            kerr.get('mo_diag', 'N/A'), kerr.get('mr_diag', 'N/A'),
                            kerr.get('mp_diag', 'N/A'), "---"
                        ],
                        'Otros Datos': [
                            f"Densidad (Durnin): {kerr.get('dc', 0):.4f}",
                            f"Peso Paciente: {pa.get('peso', 0):.2f} kg",
                            f"Error (Suma vs Peso): {kerr.get('suma_total', 0) - pa.get('peso', 0):.2f} kg",
                            "", "", ""
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

# --- NUEVAS FUNCIONES PARA EXPORTAR PLAN SEMANAL ---

class PDFPlan(FPDF):
    """
    Clase personalizada para generar el PDF con encabezado y pie de página.
    """
    def __init__(self, paciente_nombre, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paciente_nombre = paciente_nombre
        self.fecha_hoy = datetime.now().strftime("%d/%m/%Y")
        
    def header(self):
        # Logo (si existe)
        if os.path.exists(LOGO_PATH):
            self.image(LOGO_PATH, 10, 8, 33)
        
        # Título
        self.set_font('Arial', 'B', 15)
        self.cell(80) # Mover a la derecha
        self.cell(30, 10, 'Plan de Alimentación Semanal', 0, 0, 'C')
        
        # Info del Paciente (Subtítulo)
        self.set_font('Arial', '', 12)
        self.ln(10) # Salto de línea
        self.cell(80)
        self.cell(30, 10, f"Paciente: {self.paciente_nombre}", 0, 0, 'C')
        
        # Fecha
        self.set_font('Arial', '', 10)
        self.ln(5)
        self.cell(80)
        self.cell(30, 10, f"Fecha de Generación: {self.fecha_hoy}", 0, 0, 'C')
        
        # Salto de línea final del encabezado
        self.ln(20)

    def footer(self):
        self.set_y(-15) # Posición a 1.5 cm del final
        self.set_font('Arial', 'I', 8)
        # Número de página
        self.cell(0, 10, 'Página ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

def generar_pdf_composicion(paciente_data):
    """
    Genera un informe PDF completo de Composición Corporal,
    incluyendo el gráfico de Somatocarta.
    (Versión robusta con Archivo Temporal)
    """
    pa = paciente_data
    pdf = PDFComposicion(pa.get('nombre', 'N/A'))
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # --- PÁGINA 1: RESUMEN GENERAL ---
    pdf.draw_section_title("Resumen General")
    pdf.draw_metric("Nombre", pa.get('nombre', 'N/A'))
    pdf.draw_metric("Edad", pa.get('edad', 0), "años")
    pdf.draw_metric("Sexo", pa.get('sexo', 'N/A'))
    pdf.draw_metric("Peso", f"{pa.get('peso', 0):.1f}", "kg")
    pdf.draw_metric("Talla", f"{pa.get('talla_cm', 0):.1f}", "cm")
    pdf.ln(5)

    pdf.draw_section_title("Indicadores Clave")
    imc = pa.get('imc', 0)
    pdf.draw_metric("IMC", f"{imc:.2f}", f"({pa.get('diagnostico_imc', 'N/A')})")
    pdf.draw_metric("GET (Gasto Energético)", f"{pa.get('get', 0):.0f}", f"kcal/día ({pa.get('formula_get', 'N/A')})")
    pdf.ln(5)
    
    # Gráfico de IMC
    if imc > 0:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Clasificación Visual de IMC', 0, 1, 'L')
        pdf.draw_imc_gauge(imc)
        pdf.ln(5)

    # --- PÁGINA 2: COMPOSICIÓN CORPORAL ---
    pdf.add_page()
    pdf.draw_section_title("Análisis de Composición Corporal")
    
    comp = pa.get('composicion', {})
    
    # Modelo 2C (Durnin-Edad/Siri)
    comp_2c = comp.get('modelo_2c', {})
    if comp_2c and comp_2c.get('porc_grasa', 0) > 0:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Modelo 2 Componentes (Durnin-Edad/Siri)", 0, 1, 'L')
        pdf.draw_metric("% Grasa Corporal", f"{comp_2c.get('porc_grasa', 0):.1f}%", f"({comp_2c.get('diag_grasa', 'N/A')})")
        pdf.draw_metric("Masa Grasa", f"{comp_2c.get('masa_grasa', 0):.1f}", "kg")
        pdf.draw_metric("Masa Magra", f"{comp_2c.get('masa_magra', 0):.1f}", "kg")
        
        # Gráfico 2C
        porc_magra = 100.0 - comp_2c.get('porc_grasa', 0)
        pdf.draw_composition_bar({
            'MG': comp_2c.get('porc_grasa', 0),
            'Magra': porc_magra
        }, pa.get('peso', 0))
    else:
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 10, "Datos del Modelo 2C no calculados.", 0, 1, 'L')
    
    pdf.ln(5)

    # Modelo 5C (Kerr)
    comp_5c = comp.get('modelo_5c', {})
    if comp_5c and not comp_5c.get('error'):
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Modelo 5 Componentes (Kerr - Modificado)", 0, 1, 'L')
        
        # Gráfico 5C
        percentages_5c = {
            'MG': comp_5c.get('mg_porc', 0),
            'MM': comp_5c.get('mm_porc', 0),
            'MO': comp_5c.get('mo_porc', 0),
            'MR': comp_5c.get('mr_porc', 0),
            'MP': comp_5c.get('mp_porc', 0)
        }
        pdf.draw_composition_bar(percentages_5c, pa.get('peso', 0))
        pdf.ln(3)

        # Tabla 5C (Encabezado)
        pdf.set_font('Arial', 'B', 10)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(40, 7, "Componente", 1, 0, 'C', fill=True)
        pdf.cell(30, 7, "Masa (kg)", 1, 0, 'C', fill=True)
        pdf.cell(30, 7, "% Corporal", 1, 0, 'C', fill=True)
        pdf.cell(0, 7, "Diagnóstico", 1, 1, 'C', fill=True)
        
        pdf.set_font('Arial', '', 10)
        
        # Filas de la tabla
        def add_row(comp, kg, porc, diag):
            pdf.cell(40, 7, comp, 1, 0, 'L')
            pdf.cell(30, 7, f"{kg:.2f}", 1, 0, 'R')
            pdf.cell(30, 7, f"{porc:.1f}%", 1, 0, 'R')
            pdf.cell(0, 7, diag, 1, 1, 'L')

        add_row("Masa Grasa (MG)", comp_5c.get('mg_kg', 0), comp_5c.get('mg_porc', 0), comp_5c.get('mg_diag', 'N/A'))
        add_row("Masa Muscular (MM)", comp_5c.get('mm_kg', 0), comp_5c.get('mm_porc', 0), comp_5c.get('mm_diag', 'N/A'))
        add_row("Masa Ósea (MO)", comp_5c.get('mo_kg', 0), comp_5c.get('mo_porc', 0), comp_5c.get('mo_diag', 'N/A'))
        add_row("Masa Residual (MR)", comp_5c.get('mr_kg', 0), comp_5c.get('mr_porc', 0), comp_5c.get('mr_diag', 'N/A'))
        add_row("Masa de Piel (MP)", comp_5c.get('mp_kg', 0), comp_5c.get('mp_porc', 0), comp_5c.get('mp_diag', 'N/A'))
        
        # Fila Total
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(40, 7, "SUMA TOTAL", 1, 0, 'L')
        pdf.cell(30, 7, f"{comp_5c.get('suma_total', 0):.2f}", 1, 0, 'R')
        pdf.cell(30, 7, "100.0%", 1, 0, 'R')
        pdf.cell(0, 7, f"Dif. vs Peso: {comp_5c.get('suma_total', 0) - pa.get('peso', 0):.2f} kg", 1, 1, 'L')
        
    else:
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 10, f"Datos del Modelo 5C no calculados. ({comp_5c.get('error', '')})", 0, 1, 'L')

    # --- PÁGINA 3: SOMATOTIPO (CON GRÁFICO) ---
    pdf.add_page()
    pdf.draw_section_title("Somatotipo (Heath-Carter)")
    
    # Recargamos 'comp' y 'som' por si acaso, aunque ya debería estar en 'pa'
    comp = pa.get('composicion', {})
    som = comp.get('somatotipo', {})
    
    # --- ¡ESTA LÍNEA ES DIFERENTE! ---
    if som and som.get('endo', 0) > 0:
        pdf.draw_metric("Endomorfia", f"{som.get('endo', 0):.1f}", "(Adiposidad relativa)")
        pdf.draw_metric("Mesomorfia", f"{som.get('meso', 0):.1f}", "(Robustez músculo-esquelética)")
        pdf.draw_metric("Ectomorfia", f"{som.get('ecto', 0):.1f}", "(Linealidad relativa)")
        pdf.ln(3)
        
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f"Clasificación: {som.get('clasificacion', 'N/A')}", 0, 1, 'L')
        
        pdf.set_font('Arial', '', 11)
        explicacion = obtener_explicacion_somatotipo(som.get('clasificacion', 'N/A'))
        pdf.multi_cell(0, 6, f"{explicacion}")
        pdf.ln(5)
        
        # --- INICIO: ESTRATEGIA DE ARCHIVO TEMPORAL ---
        temp_file_path = None
        try:
            fig_somato = crear_grafico_somatotipo(
                som.get('endo', 0), 
                som.get('meso', 0), 
                som.get('ecto', 0)
            )
            
            # 1. Crear un archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                temp_file_path = temp_file.name
                # 2. Guardar la imagen de Plotly en el archivo
                pio.write_image(fig_somato, temp_file_path, width=500, height=500, scale=1.5)

            # 3. Insertar la imagen en el PDF desde la RUTA DEL ARCHIVO
            if temp_file_path and os.path.exists(temp_file_path):
                img_width_mm = 120 
                img_x_pos = (210 - img_width_mm) / 2
                current_y = pdf.get_y()
                # Leemos la imagen desde la ruta del archivo
                pdf.image(temp_file_path, x=img_x_pos, y=current_y, w=img_width_mm, type='PNG')
                pdf.ln(img_width_mm + 5)
            else:
                raise Exception("No se pudo crear el archivo temporal de la imagen.")
        
        except Exception as e:
            # Si esto falla, AHORA SÍ veremos el error
            print("--- ¡ERROR AL GENERAR GRÁFICO PDF (Estrategia TempFile)! ---")
            traceback.print_exc()
            print("--------------------------------------")
            pdf.set_font('Arial', 'I', 9)
            pdf.set_text_color(128)
            pdf.multi_cell(0, 5, f"Nota: No se pudo generar el gráfico. (Error: {e}).\n"
                                  "Revise la terminal.")
        finally:
            # 4. Limpiar el archivo temporal sin importar lo que pase
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        # --- FIN: ESTRATEGIA DE ARCHIVO TEMPORAL ---
            
    else:
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 10, "Datos de Somatotipo no calculados.", 0, 1, 'L')

    return bytes(pdf.output(dest='S'))

class PDFDietaDetallada(FPDF):
    """
    Clase personalizada para generar el PDF de la Dieta Detallada (1 Día).
    """
    def __init__(self, paciente_nombre, get_objetivo, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paciente_nombre = paciente_nombre
        self.get_objetivo = get_objetivo
        self.fecha_hoy = datetime.now().strftime("%d/%m/%Y")
        self.colores = { 'gris_claro': (236, 240, 241), 'gris_oscuro': (44, 62, 80) }

    def header(self):
        if os.path.exists(LOGO_PATH):
            self.image(LOGO_PATH, 10, 8, 33)
        self.set_font('Arial', 'B', 15)
        self.cell(80)
        self.cell(30, 10, 'Dieta Detallada (1 Día)', 0, 0, 'C')
        self.ln(10)
        self.set_font('Arial', '', 12)
        self.cell(80)
        self.cell(30, 10, f"Paciente: {self.paciente_nombre}", 0, 0, 'C')
        self.ln(5)
        self.set_font('Arial', '', 10)
        self.cell(80)
        self.cell(30, 10, f"GET Objetivo: {self.get_objetivo:.0f} kcal", 0, 0, 'C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Página ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def draw_section_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(*self.colores['gris_claro'])
        self.set_text_color(*self.colores['gris_oscuro'])
        self.cell(0, 10, f" {title}", 0, 1, 'L', fill=True)
        self.ln(4)
    
    def draw_macro_table(self, df_macros):
        self.set_font('Arial', 'B', 10)
        self.set_fill_color(240, 240, 240)
        
        # Encabezado
        self.cell(40, 7, "Macro", 1, 0, 'C', fill=True)
        self.cell(50, 7, "Actual (g)", 1, 0, 'C', fill=True)
        self.cell(50, 7, "Objetivo (g)", 1, 0, 'C', fill=True)
        self.cell(50, 7, "Diferencia (g)", 1, 1, 'C', fill=True)
        
        self.set_font('Arial', '', 10)
        
        # Transponer el dataframe para iterar
        df_macros_t = df_macros.transpose()
        
        for macro, row in df_macros_t.iterrows():
            self.cell(40, 7, macro, 1, 0, 'L')
            self.cell(50, 7, f"{row['Actual (g)']:.1f}", 1, 0, 'R')
            self.cell(50, 7, f"{row['Objetivo (g)']:.1f}", 1, 0, 'R')
            self.cell(50, 7, f"{row['Diferencia (g)']:.1f}", 1, 1, 'R')
        self.ln(5)

    def draw_detailed_diet(self, df_dieta):
        tiempos_comida_orden = ["Desayuno", "Colación Mañana", "Almuerzo", "Colación Tarde", "Cena", "Colación Noche"]
        cols_display = ['Alimento', 'Gramos', 'Kcal', 'Proteínas', 'Grasas', 'Carbohidratos']
        
        if 'Tiempo Comida' not in df_dieta.columns:
            self.cell(0, 10, "No hay datos de dieta.", 0, 1)
            return
        
        df_dieta['Tiempo Comida'] = pd.Categorical(df_dieta['Tiempo Comida'], categories=tiempos_comida_orden, ordered=True)
        df_dieta = df_dieta.sort_values('Tiempo Comida')
        grupos = df_dieta.groupby('Tiempo Comida', observed=True)
        
        for tiempo in tiempos_comida_orden:
            if tiempo in grupos.groups:
                df_tiempo = grupos.get_group(tiempo)
                
                self.set_font('Arial', 'B', 12)
                self.cell(0, 8, f"--- {tiempo} ---", 0, 1, 'L')
                
                # Encabezado de la mini-tabla
                self.set_font('Arial', 'B', 9)
                self.set_fill_color(245, 245, 245)
                self.cell(80, 6, "Alimento", 1, 0, 'L', fill=True) # Más ancho para Alimento
                self.cell(20, 6, "Gramos", 1, 0, 'R', fill=True)
                self.cell(20, 6, "Kcal", 1, 0, 'R', fill=True)
                self.cell(20, 6, "Prot(g)", 1, 0, 'R', fill=True)
                self.cell(20, 6, "Gras(g)", 1, 0, 'R', fill=True)
                self.cell(20, 6, "Carb(g)", 1, 1, 'R', fill=True)
                
                self.set_font('Arial', '', 9)
                
                for _, item in df_tiempo.iterrows():
                    # MultiCell para Alimento por si es largo
                    y_before = self.get_y()
                    self.multi_cell(80, 6, str(item['Alimento']), 1, 'L')
                    y_after = self.get_y()
                    x_after = self.get_x()
                    # Volver a la misma línea
                    self.set_xy(x_after + 80, y_before)
                    
                    cell_height = y_after - y_before
                    self.cell(20, cell_height, f"{item['Gramos']:.0f}", 1, 0, 'R')
                    self.cell(20, cell_height, f"{item['Kcal']:.0f}", 1, 0, 'R')
                    self.cell(20, cell_height, f"{item['Proteínas']:.1f}", 1, 0, 'R')
                    self.cell(20, cell_height, f"{item['Grasas']:.1f}", 1, 0, 'R')
                    self.cell(20, cell_height, f"{item['Carbohidratos']:.1f}", 1, 1, 'R')

                # Totales del tiempo de comida
                self.set_font('Arial', 'B', 9)
                self.cell(80, 6, "Total", 1, 0, 'R')
                self.cell(20, 6, f"{df_tiempo['Gramos'].sum():.0f}", 1, 0, 'R')
                self.cell(20, 6, f"{df_tiempo['Kcal'].sum():.0f}", 1, 0, 'R')
                self.cell(20, 6, f"{df_tiempo['Proteínas'].sum():.1f}", 1, 0, 'R')
                self.cell(20, 6, f"{df_tiempo['Grasas'].sum():.1f}", 1, 0, 'R')
                self.cell(20, 6, f"{df_tiempo['Carbohidratos'].sum():.1f}", 1, 1, 'R')
                
                self.ln(5)

def generar_pdf_dieta_detallada(paciente, df_dieta, df_macros, df_resumen_comidas, total_kcal):
    """
    Genera un informe PDF completo de la Dieta Detallada.
    """
    pdf = PDFDietaDetallada(paciente.get('nombre', 'N/A'), paciente.get('get', 0))
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # --- Totales y Adecuación ---
    pdf.draw_section_title("Resumen de Macronutrientes")
    
    # Totales Kcal
    adecuacion = 0.0
    if paciente.get('get', 0) > 0:
        adecuacion = (total_kcal / paciente.get('get', 0)) * 100
        
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(95, 10, f"Kcal Totales: {total_kcal:.0f} kcal", 1, 0, 'C')
    pdf.cell(95, 10, f"Adecuación GET: {adecuacion:.1f} %", 1, 1, 'C')
    pdf.ln(5)
    
    # Tabla de Macros
    pdf.draw_macro_table(df_macros)
    
    # --- Dieta Detallada por Tiempo ---
    pdf.draw_section_title("Alimentos por Tiempo de Comida")
    pdf.draw_detailed_diet(df_dieta)
    
    # --- Página 2: Resumen por Comidas y Micros ---
    if pdf.get_y() > 200: # Evitar que el resumen de micros quede cortado
        pdf.add_page()
    else:
        pdf.ln(10)

    pdf.draw_section_title("Resumen por Tiempo de Comida")
    
    # Encabezado
    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(50, 7, "Tiempo Comida", 1, 0, 'L', fill=True)
    pdf.cell(35, 7, "Kcal", 1, 0, 'R', fill=True)
    pdf.cell(35, 7, "Proteínas (g)", 1, 0, 'R', fill=True)
    pdf.cell(35, 7, "Grasas (g)", 1, 0, 'R', fill=True)
    pdf.cell(35, 7, "Carbs (g)", 1, 1, 'R', fill=True)
    
    pdf.set_font('Arial', '', 10)
    for tiempo, row in df_resumen_comidas.iterrows():
        pdf.cell(50, 7, tiempo, 1, 0, 'L')
        pdf.cell(35, 7, f"{row['Kcal']:.0f}", 1, 0, 'R')
        pdf.cell(35, 7, f"{row['Proteínas']:.1f}", 1, 0, 'R')
        pdf.cell(35, 7, f"{row['Grasas']:.1f}", 1, 0, 'R')
        pdf.cell(35, 7, f"{row['Carbohidratos']:.1f}", 1, 1, 'R')
    
    if pdf.get_y() > 200: # Evitar que el resumen de micros quede cortado
        pdf.add_page()
    else:
        pdf.ln(10)
    
    # --- Micros ---
    pdf.draw_section_title("Resumen de Micronutrientes")
    
    micros_cols = [
        'Fibra', 'Agua', 'Calcio', 'Fósforo', 'Zinc', 'Hierro', 'Vitamina C', 
        'Sodio', 'Potasio', 'Beta-Caroteno', 'Vitamina A', 'Tiamina', 
        'Riboflavina', 'Niacina', 'Acido Folico'
    ]
    cols_presentes = [col for col in micros_cols if col in df_dieta.columns]
    
    if cols_presentes:
        df_micros = df_dieta[cols_presentes].sum().reset_index()
        df_micros.columns = ['Nutriente', 'Total']
        df_micros = df_micros[df_micros['Total'] > 0]
        
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(60, 7, "Nutriente", 1, 0, 'L', fill=True)
        pdf.cell(40, 7, "Total", 1, 1, 'R', fill=True)
        pdf.set_font('Arial', '', 10)
        
        for _, row in df_micros.iterrows():
            if pdf.get_y() > 270:
                pdf.add_page()
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(60, 7, "Nutriente", 1, 0, 'L', fill=True)
                pdf.cell(40, 7, "Total", 1, 1, 'R', fill=True)
                pdf.set_font('Arial', '', 10)
                
            pdf.cell(60, 7, str(row['Nutriente']), 1, 0, 'L')
            pdf.cell(40, 7, f"{row['Total']:.1f}", 1, 1, 'R')
    
    return bytes(pdf.output(dest='S'))

# --- CLASES Y FUNCIONES NUEVAS PARA EXPORTAR EL PLAN SEMANAL ---

class PDFPlanSemanal(FPDF):
    """Clase auxiliar para generar el PDF del Plan Semanal."""
    def __init__(self, paciente_nombre, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paciente_nombre = paciente_nombre
        self.fecha_hoy = datetime.now().strftime("%d/%m/%Y")

    def header(self):
        # Intenta poner el logo si existe
        if os.path.exists(LOGO_PATH):
            self.image(LOGO_PATH, 10, 8, 33)
        
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Plan de Alimentación Semanal', 0, 1, 'C')
        
        self.set_font('Arial', '', 12)
        self.cell(0, 10, f"Paciente: {self.paciente_nombre}", 0, 1, 'C')
        self.ln(5)
        
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f"Fecha: {self.fecha_hoy}", 0, 1, 'C')
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}}', 0, 0, 'C')

def generar_pdf_plan_semanal(paciente_data, plan_semanal):
    """
    Genera el PDF del plan semanal (formato lista por día).
    CORREGIDO: Cálculo explícito de ancho para evitar error FPDFException.
    """
    if not plan_semanal:
        return None
    
    nombre = paciente_data.get('nombre', 'N/A')
    pdf = PDFPlanSemanal(nombre)
    pdf.alias_nb_pages()
    pdf.add_page()
    
    dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    tiempos_comida = ["Desayuno", "Colación Mañana", "Almuerzo", "Colación Tarde", "Cena", "Colación Noche"]
    
    for dia in dias_semana:
        # Encabezado del día
        pdf.set_font('Arial', 'B', 12)
        pdf.set_fill_color(230, 230, 230) # Gris claro
        pdf.cell(0, 8, f" {dia}", 1, 1, 'L', fill=True)
        
        dia_data = plan_semanal.get(dia, {})
        hay_info = False
        
        pdf.set_font('Arial', '', 10)
        
        for tiempo in tiempos_comida:
            preparacion = dia_data.get(tiempo, "").strip()
            if preparacion:
                hay_info = True
                
                # --- CORRECCIÓN AQUÍ ---
                # 1. Definir ancho de la etiqueta (ej. "Desayuno:")
                ancho_etiqueta = 40
                
                # 2. Calcular ancho disponible para el texto
                # Ancho total página - Margen Izq - Margen Der - Ancho Etiqueta
                ancho_disponible = pdf.w - pdf.l_margin - pdf.r_margin - ancho_etiqueta
                
                # 3. Imprimir etiqueta (Negrita)
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(ancho_etiqueta, 6, f"{tiempo}:", 0, 0) # El 0 final deja el cursor a la derecha
                
                # 4. Imprimir contenido (Normal) con ancho explícito
                pdf.set_font('Arial', '', 10)
                pdf.multi_cell(ancho_disponible, 6, preparacion)
        
        if not hay_info:
            pdf.set_font('Arial', 'I', 9)
            pdf.cell(0, 6, " (Sin registro)", 0, 1)
            
        pdf.ln(4) # Espacio entre días
        
    return bytes(pdf.output(dest='S'))

def generar_excel_plan_semanal(plan_semanal):
    """
    Genera el archivo Excel del plan semanal.
    """
    output = BytesIO()
    dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    tiempos_comida = ["Desayuno", "Colación Mañana", "Almuerzo", "Colación Tarde", "Cena", "Colación Noche"]
    
    data_filas = []
    
    for dia in dias_semana:
        row = {'Día': dia}
        dia_data = plan_semanal.get(dia, {})
        for tiempo in tiempos_comida:
            row[tiempo] = dia_data.get(tiempo, "")
        data_filas.append(row)
        
    df = pd.DataFrame(data_filas)
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
        
    return output.getvalue()
# --- (REEMPLAZA ESTA CLASE COMPLETA) ---

class PDFComposicion(FPDF):
    """
    Clase personalizada para generar el PDF de Composición Corporal
    con gráficos y tablas.
    """
    def __init__(self, paciente_nombre, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paciente_nombre = paciente_nombre
        self.fecha_hoy = datetime.now().strftime("%d/%m/%Y")
        # Colores (R, G, B)
        self.colores = {
            'azul': (52, 152, 219),
            'verde': (46, 204, 113),
            'amarillo': (241, 196, 15),
            'naranja': (230, 126, 34),
            'rojo': (231, 76, 60),
            'gris_claro': (236, 240, 241),
            'gris_oscuro': (44, 62, 80),
            'gris_medio': (149, 165, 166)
        }
        
    def header(self):
        # Logo (si existe)
        if os.path.exists(LOGO_PATH):
            self.image(LOGO_PATH, 10, 8, 33)
        
        # Título
        self.set_font('Arial', 'B', 16)
        self.set_text_color(*self.colores['gris_oscuro'])
        self.cell(80) # Mover a la derecha
        self.cell(30, 10, 'Evaluación de Composición Corporal', 0, 0, 'C')
        
        # Info del Paciente (Subtítulo)
        self.set_font('Arial', '', 12)
        self.ln(10)
        self.cell(80)
        self.cell(30, 10, f"Paciente: {self.paciente_nombre}", 0, 0, 'C')
        
        # Fecha
        self.set_font('Arial', '', 10)
        self.ln(5)
        self.cell(80)
        self.cell(30, 10, f"Fecha de Generación: {self.fecha_hoy}", 0, 0, 'C')
        
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, 'Página ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def draw_section_title(self, title):
        """Dibuja un título de sección destacado."""
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(*self.colores['gris_claro'])
        self.set_text_color(*self.colores['gris_oscuro'])
        self.cell(0, 10, f" {title}", 0, 1, 'L', fill=True)
        self.ln(5)

    def draw_metric(self, label, value, unit=""):
        """Dibuja un par de etiqueta-valor (como una tabla de 2 columnas)."""
        self.set_font('Arial', 'B', 11)
        self.set_text_color(0)
        self.cell(50, 8, label, 0, 0, 'L')
        self.set_font('Arial', '', 11)
        self.cell(0, 8, f": {value} {unit}", 0, 1, 'L')
        self.ln(1)

    def draw_imc_gauge(self, imc):
        """Dibuja un medidor de IMC simple con FPDF."""
        start_x = self.get_x() + 10
        y = self.get_y()
        max_width = 170 # Ancho total del medidor
        
        # Definir rangos y colores
        rangos = [
            (18.5, self.colores['azul']),   # Bajo Peso
            (25.0, self.colores['verde']),  # Normal
            (30.0, self.colores['amarillo']), # Sobrepeso
            (40.0, self.colores['rojo'])    # Obesidad (límite 40)
        ]
        
        # Dibujar las barras de color
        self.set_line_width(0.5)
        current_x = start_x
        last_limit = 10 # Empezar a dibujar desde 10
        
        for limit, color in rangos:
            self.set_fill_color(*color)
            ancho_rango = ((limit - last_limit) / (40 - 10)) * max_width
            self.rect(current_x, y, ancho_rango, 10, 'F')
            current_x += ancho_rango
            last_limit = limit

        # Dibujar el marcador del paciente
        if imc > 0 and imc <= 40:
            imc_pos = start_x + (((imc - 10) / (40 - 10)) * max_width)
            self.set_fill_color(*self.colores['gris_oscuro'])
            
            # --- ESTA ES LA SECCIÓN CORREGIDA ---
            # Definir los 3 puntos del triángulo
            points = [
                (imc_pos - 3, y + 12),  # Punto inferior izquierdo
                (imc_pos + 3, y + 12),  # Punto inferior derecho
                (imc_pos, y + 17)       # Punto pico (hacia abajo)
            ]
            self.polygon(points, 'F') # Usar polygon() en lugar de triangle()
            # --- FIN DE LA CORRECCIÓN ---

            self.set_font('Arial', 'B', 10)
            self.set_xy(imc_pos - 10, y + 18)
            self.cell(20, 5, f"{imc:.1f}", 0, 0, 'C')

        # Etiquetas
        self.set_font('Arial', '', 9)
        self.set_xy(start_x, y + 10)
        self.cell(0, 5, "18.5", 0, 0, 'L')
        self.set_xy(start_x + ((15 / 30) * max_width) - 5, y + 10)
        self.cell(10, 5, "25.0", 0, 0, 'C')
        self.set_xy(start_x + ((20 / 30) * max_width) - 5, y + 10)
        self.cell(10, 5, "30.0", 0, 0, 'C')
        self.set_xy(start_x + max_width - 10, y + 10)
        self.cell(10, 5, "40.0+", 0, 0, 'R')
        self.ln(18)

    def draw_composition_bar(self, percentages_dict, total_kg):
        """Dibuja una barra apilada de 100% para la composición."""
        start_x = self.get_x() + 10
        y = self.get_y()
        max_width = 170
        height = 12
        
        colores_comp = {
            'MG': self.colores['rojo'],
            'MM': self.colores['naranja'],
            'MO': self.colores['gris_claro'],
            'MR': self.colores['gris_medio'],
            'MP': self.colores['azul'],
            'Magra': self.colores['verde'] # Añadido para el gráfico 2C
        }
        
        current_x = start_x
        self.set_line_width(0.2)
        
        # Dibujar barra
        for comp, porc in percentages_dict.items():
            if porc > 0:
                ancho = (porc / 100) * max_width
                self.set_fill_color(*colores_comp.get(comp, (0,0,0)))
                self.rect(current_x, y, ancho, height, 'F')
                current_x += ancho
        
        # Borde negro
        self.set_draw_color(0)
        self.rect(start_x, y, max_width, height, 'D')
        
        # Leyenda
        self.ln(height + 3)
        self.set_font('Arial', '', 9)
        legend_x = start_x
        
        for comp, porc in percentages_dict.items():
            if porc > 0:
                self.set_fill_color(*colores_comp.get(comp, (0,0,0)))
                self.rect(legend_x, self.get_y(), 3, 3, 'F')
                self.set_x(legend_x + 5)
                self.cell(30, 4, f"{comp} ({porc:.1f}%)", 0, 0, 'L')
                legend_x += 35
                if legend_x > start_x + 140: # Siguiente línea de leyenda
                    self.ln(4)
                    legend_x = start_x
        self.ln(5)

# --- (FIN DE LA CLASE) ---


def generar_pdf_composicion(paciente_data):
    """
    Genera un informe PDF completo de Composición Corporal.
    """
    pa = paciente_data
    pdf = PDFComposicion(pa.get('nombre', 'N/A'))
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # --- PÁGINA 1: RESUMEN GENERAL ---
    pdf.draw_section_title("Resumen General")
    pdf.draw_metric("Nombre", pa.get('nombre', 'N/A'))
    pdf.draw_metric("Edad", pa.get('edad', 0), "años")
    pdf.draw_metric("Sexo", pa.get('sexo', 'N/A'))
    pdf.draw_metric("Peso", f"{pa.get('peso', 0):.1f}", "kg")
    pdf.draw_metric("Talla", f"{pa.get('talla_cm', 0):.1f}", "cm")
    pdf.ln(5)

    pdf.draw_section_title("Indicadores Clave")
    imc = pa.get('imc', 0)
    pdf.draw_metric("IMC", f"{imc:.2f}", f"({pa.get('diagnostico_imc', 'N/A')})")
    pdf.draw_metric("GET (Gasto Energético)", f"{pa.get('get', 0):.0f}", f"kcal/día ({pa.get('formula_get', 'N/A')})")
    pdf.ln(5)
    
    # Gráfico de IMC
    if imc > 0:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Clasificación Visual de IMC', 0, 1, 'L')
        pdf.draw_imc_gauge(imc)
        pdf.ln(5)

    # --- PÁGINA 2: COMPOSICIÓN CORPORAL ---
    pdf.add_page()
    pdf.draw_section_title("Análisis de Composición Corporal")
    
    comp = pa.get('composicion', {})
    
    # Modelo 2C (Durnin-Edad/Siri)
    comp_2c = comp.get('modelo_2c', {})
    if comp_2c and comp_2c.get('porc_grasa', 0) > 0:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Modelo 2 Componentes (Durnin-Edad/Siri)", 0, 1, 'L')
        pdf.draw_metric("% Grasa Corporal", f"{comp_2c.get('porc_grasa', 0):.1f}%", f"({comp_2c.get('diag_grasa', 'N/A')})")
        pdf.draw_metric("Masa Grasa", f"{comp_2c.get('masa_grasa', 0):.1f}", "kg")
        pdf.draw_metric("Masa Magra", f"{comp_2c.get('masa_magra', 0):.1f}", "kg")
        
        # Gráfico 2C
        porc_magra = 100.0 - comp_2c.get('porc_grasa', 0)
        pdf.draw_composition_bar({
            'MG': comp_2c.get('porc_grasa', 0),
            'Magra': porc_magra
        }, pa.get('peso', 0))
    else:
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 10, "Datos del Modelo 2C no calculados.", 0, 1, 'L')
    
    pdf.ln(5)

    # Modelo 5C (Kerr)
    comp_5c = comp.get('modelo_5c', {})
    if comp_5c and not comp_5c.get('error'):
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Modelo 5 Componentes (Kerr - Modificado)", 0, 1, 'L')
        
        # Gráfico 5C
        percentages_5c = {
            'MG': comp_5c.get('mg_porc', 0),
            'MM': comp_5c.get('mm_porc', 0),
            'MO': comp_5c.get('mo_porc', 0),
            'MR': comp_5c.get('mr_porc', 0),
            'MP': comp_5c.get('mp_porc', 0)
        }
        pdf.draw_composition_bar(percentages_5c, pa.get('peso', 0))
        pdf.ln(3)

        # Tabla 5C (Encabezado)
        pdf.set_font('Arial', 'B', 10)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(40, 7, "Componente", 1, 0, 'C', fill=True)
        pdf.cell(30, 7, "Masa (kg)", 1, 0, 'C', fill=True)
        pdf.cell(30, 7, "% Corporal", 1, 0, 'C', fill=True)
        pdf.cell(0, 7, "Diagnóstico", 1, 1, 'C', fill=True)
        
        pdf.set_font('Arial', '', 10)
        
        # Filas de la tabla
        def add_row(comp, kg, porc, diag):
            pdf.cell(40, 7, comp, 1, 0, 'L')
            pdf.cell(30, 7, f"{kg:.2f}", 1, 0, 'R')
            pdf.cell(30, 7, f"{porc:.1f}%", 1, 0, 'R')
            pdf.cell(0, 7, diag, 1, 1, 'L')

        add_row("Masa Grasa (MG)", comp_5c.get('mg_kg', 0), comp_5c.get('mg_porc', 0), comp_5c.get('mg_diag', 'N/A'))
        add_row("Masa Muscular (MM)", comp_5c.get('mm_kg', 0), comp_5c.get('mm_porc', 0), comp_5c.get('mm_diag', 'N/A'))
        add_row("Masa Ósea (MO)", comp_5c.get('mo_kg', 0), comp_5c.get('mo_porc', 0), comp_5c.get('mo_diag', 'N/A'))
        add_row("Masa Residual (MR)", comp_5c.get('mr_kg', 0), comp_5c.get('mr_porc', 0), comp_5c.get('mr_diag', 'N/A'))
        add_row("Masa de Piel (MP)", comp_5c.get('mp_kg', 0), comp_5c.get('mp_porc', 0), comp_5c.get('mp_diag', 'N/A'))
        
        # Fila Total
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(40, 7, "SUMA TOTAL", 1, 0, 'L')
        pdf.cell(30, 7, f"{comp_5c.get('suma_total', 0):.2f}", 1, 0, 'R')
        pdf.cell(30, 7, "100.0%", 1, 0, 'R')
        pdf.cell(0, 7, f"Dif. vs Peso: {comp_5c.get('suma_total', 0) - pa.get('peso', 0):.2f} kg", 1, 1, 'L')
        
    else:
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 10, f"Datos del Modelo 5C no calculados. ({comp_5c.get('error', '')})", 0, 1, 'L')

    # --- PÁGINA 3: SOMATOTIPO ---
    pdf.add_page()
    pdf.draw_section_title("Somatotipo (Heath-Carter)")
    
    som = comp.get('somatotipo', {})
    if som:
        pdf.draw_metric("Endomorfia", f"{som.get('endo', 0):.1f}", "(Adiposidad relativa)")
        pdf.draw_metric("Mesomorfia", f"{som.get('meso', 0):.1f}", "(Robustez músculo-esquelética)")
        pdf.draw_metric("Ectomorfia", f"{som.get('ecto', 0):.1f}", "(Linealidad relativa)")
        pdf.ln(3)
        
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f"Clasificación: {som.get('clasificacion', 'N/A')}", 0, 1, 'L')
        
        pdf.set_font('Arial', '', 11)
        explicacion = obtener_explicacion_somatotipo(som.get('clasificacion', 'N/A'))
        pdf.multi_cell(0, 6, f"{explicacion}")
        pdf.ln(5)
        
        pdf.set_font('Arial', 'I', 9)
        pdf.set_text_color(128)
        pdf.cell(0, 10, "Nota: El gráfico visual de Somatocarta está disponible en la pestaña 'Antropometría' de la aplicación web.", 0, 1, 'C')
        
    else:
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 10, "Datos de Somatotipo no calculados.", 0, 1, 'L')

    return bytes(pdf.output(dest='S'))

# --- FUNCIONES HELPER PARA EDICIÓN DE DIETA ---

def eliminar_item_dieta(item_id_to_delete):
    """
    Encuentra un item por su ID único en la dieta_temporal y lo elimina.
    """
    if not item_id_to_delete:
        return

    new_dieta = [
        item for item in st.session_state.dieta_temporal 
        if item['id'] != item_id_to_delete
    ]
    
    st.session_state.dieta_temporal = new_dieta
    st.session_state.paciente_actual['dieta_actual'] = new_dieta
    guardar_paciente(st.session_state.usuario, st.session_state.paciente_actual)
    st.success("Alimento eliminado.")

def actualizar_gramos_item(item_id):
    """
    Recalcula macros y micros. 
    CORREGIDO: Lee el valor directamente del estado usando la key, 
    no espera argumento de valor.
    """
    # 1. Reconstruimos la 'key' del widget para leer el valor nuevo
    key_widget = f"g_in_{item_id}"
    
    # Verificamos si existe en sesión (debería, porque se acaba de editar)
    if key_widget in st.session_state:
        nuevos_gramos = st.session_state[key_widget]
        
        if nuevos_gramos <= 0: return

        # 2. Buscar el item en la lista temporal
        for item in st.session_state.dieta_temporal:
            if item['id'] == item_id:
                # 3. Buscar datos originales en la BD
                db = st.session_state.db_alimentos
                # Usamos iloc[0] sobre el filtro para obtener la fila
                row_ref = db[db['CÓDIGO'] == item['Código']].iloc[0]
                
                # 4. Recalcular todo con el nuevo factor
                f = nuevos_gramos / 100.0
                
                item['Gramos'] = nuevos_gramos
                item['Kcal'] = row_ref['Kcal'] * f
                item['Proteínas'] = row_ref['Proteínas'] * f
                item['Grasas'] = row_ref['Grasas'] * f
                item['Carbohidratos'] = row_ref['Carbohidratos'] * f
                
                # Micros
                item['Fibra'] = row_ref['Fibra'] * f
                item['Agua'] = row_ref['Agua'] * f
                item['Calcio'] = row_ref['Calcio'] * f
                item['Fósforo'] = row_ref['Fósforo'] * f
                item['Zinc'] = row_ref['Zinc'] * f
                item['Hierro'] = row_ref['Hierro'] * f
                item['Vitamina C'] = row_ref['Vitamina C'] * f
                item['Sodio'] = row_ref['Sodio'] * f
                item['Potasio'] = row_ref['Potasio'] * f
                item['Vitamina A'] = row_ref['Vitamina A'] * f
                item['Acido Folico'] = row_ref['Acido Folico'] * f
                break
        
        # 5. Guardar cambios
        st.session_state.paciente_actual['dieta_actual'] = st.session_state.dieta_temporal
        guardar_paciente(st.session_state.usuario, st.session_state.paciente_actual)
        
def asignar_item_a_plan_semanal(item_id, dia_seleccionado_key):
    """
    Toma un item de la dieta detallada y lo asigna al plan semanal
    en el día y tiempo de comida correspondiente.
    """
    try:
        # 1. Obtener el día seleccionado del widget que llamó a esta función
        dia_seleccionado = st.session_state[dia_seleccionado_key]
        
        # Si la opción es la por defecto ("-- Asignar a... --"), no hacer nada
        if dia_seleccionado == "-- Asignar a... --":
            return

        # 2. Encontrar el ítem en la dieta_temporal usando su ID
        item_a_asignar = None
        for item in st.session_state.dieta_temporal:
            if item['id'] == item_id:
                item_a_asignar = item
                break
        
        if not item_a_asignar:
            st.error(f"Error: No se encontró el ítem {item_id} para asignar.")
            return

        # 3. Obtener los detalles del ítem
        alimento_nombre = item_a_asignar['Alimento']
        alimento_gramos = item_a_asignar['Gramos']
        tiempo_comida = item_a_asignar['Tiempo Comida'] # ej. "Almuerzo"
        
        # 4. Crear el string a añadir
        item_str = f"{alimento_nombre} ({alimento_gramos:.0f}g)"

        # 5. Cargar el plan semanal actual del paciente
        plan_semanal_actual = st.session_state.paciente_actual.get('plan_semanal', {})
        
        # 6. Asegurar que la estructura del diccionario exista
        if dia_seleccionado not in plan_semanal_actual:
            plan_semanal_actual[dia_seleccionado] = {}
        if tiempo_comida not in plan_semanal_actual[dia_seleccionado]:
            plan_semanal_actual[dia_seleccionado][tiempo_comida] = ""

        # 7. Añadir el string al plan semanal
        texto_existente = plan_semanal_actual[dia_seleccionado][tiempo_comida]
        
        # Evitar duplicados (si ya se asignó)
        if item_str in texto_existente:
            st.toast(f"'{item_str}' ya estaba en {dia_seleccionado} - {tiempo_comida}.", icon="ℹ️")
        else:
            if texto_existente == "":
                plan_semanal_actual[dia_seleccionado][tiempo_comida] = item_str
            else:
                # Añadir con una coma si ya había algo
                plan_semanal_actual[dia_seleccionado][tiempo_comida] += f", {item_str}"
            
            # 8. Guardar los cambios en el paciente
            st.session_state.paciente_actual['plan_semanal'] = plan_semanal_actual
            guardar_paciente(st.session_state.usuario, st.session_state.paciente_actual)
            
            # 9. Mostrar confirmación
            st.toast(f"'{item_str}' asignado a {dia_seleccionado} - {tiempo_comida}.", icon="🗓️")

        # 10. Resetear el selectbox a su estado inicial
        st.session_state[dia_seleccionado_key] = "-- Asignar a... --"

    except Exception as e:
        st.error(f"Error al asignar al plan semanal: {e}")
        traceback.print_exc()

# --- FIN DE NUEVAS FUNCIONES HELPER ---
    

# --- Funciones de las Páginas de la App ---

# --- PÁGINA DE INICIO ---
def mostrar_pagina_inicio():
    """Página principal para cargar, seleccionar y registrar pacientes."""
    st.title(f"Gestión de Pacientes 🧑‍⚕️ ({st.session_state.usuario})")
    
    # Cargar la base de datos aquí para asegurar que esté disponible
    st.session_state.db_alimentos = cargar_base_de_datos_alimentos()

    st.header("Seleccionar Paciente Existente")
    
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
        st.session_state.paciente_actual = cargar_paciente(st.session_state.usuario, paciente_seleccionado)
        if st.session_state.paciente_actual:
            st.session_state.dieta_temporal = st.session_state.paciente_actual.get('dieta_actual', [])
            st.success(f"Paciente '{st.session_state.paciente_actual['nombre']}' cargado.")
            st.rerun() 
    
    if col3.button("Eliminar Paciente", type="primary", use_container_width=True) and paciente_seleccionado:
        if eliminar_paciente(st.session_state.usuario, paciente_seleccionado):
            st.success(f"Paciente '{paciente_seleccionado}' ha sido eliminado.")
            st.session_state.paciente_actual = None
            st.session_state.dieta_temporal = []
            st.rerun() 
        else:
            st.error(f"No se pudo eliminar al paciente '{paciente_seleccionado}'.")
            
    if st.session_state.paciente_actual:
        st.success(f"Paciente activo: **{st.session_state.paciente_actual['nombre']}**")

    st.header("Datos del Paciente")
    
    pa = st.session_state.paciente_actual if st.session_state.paciente_actual else {}
    
    with st.form("form_paciente"):
        nombre = st.text_input("Nombre Completo", value=pa.get('nombre', ''))
        
        col1, col2, col3 = st.columns(3)
        edad = col1.number_input("Edad", min_value=1, max_value=120, value=pa.get('edad', 25), step=1)
        sexo = col2.selectbox("Sexo", ["Masculino", "Femenino"], index=0 if pa.get('sexo', 'Masculino') == 'Masculino' else 1)
        
        raza_options = ["Caucásico", "Asiático", "Africano"]
        raza_default = pa.get('raza', 'Caucásico')
        raza_index = raza_options.index(raza_default) if raza_default in raza_options else 0
        raza = col3.selectbox(
            "Raza (para fórmula Lee)", 
            raza_options, 
            index=raza_index
        )

        col1, col2, col3 = st.columns(3)
        peso = col1.number_input("Peso (kg)", min_value=1.0, value=pa.get('peso', 70.0), step=0.1, format="%.1f")
        talla_cm = col2.number_input("Talla (cm)", min_value=1.0, value=pa.get('talla_cm', 170.0), step=0.1, format="%.1f")
        actividad = col3.selectbox("Actividad Física", ["Ligera", "Moderada", "Intensa"], index=1 if pa.get('actividad', 'Moderada') == 'Moderada' else 0 if pa.get('actividad', 'Ligera') == 'Ligera' else 2)
        
        opciones_formula = ["Mifflin-St Jeor", "Harris-Benedict", "Cunningham"]
        default_formula = pa.get('formula_get', 'Mifflin-St Jeor')
        default_index = opciones_formula.index(default_formula) if default_formula in opciones_formula else 0
        formula_get = st.selectbox(
            "Fórmula GET (Mantenimiento)", 
            opciones_formula, 
            index=default_index,
            help="Cunningham requiere datos de composición corporal (Masa Magra) de la pestaña 'Antropometría'."
        )

        # --- NUEVA SECCIÓN: OBJETIVOS Y PROTEÍNA ---
        st.divider()
        st.markdown("#### 🎯 Definición de Objetivos Nutricionales")
        st.caption("Define si el paciente debe subir o bajar de peso y cuánta proteína requiere.")
        
        c_obj1, c_obj2 = st.columns(2)
        
        # Selector de Objetivo
        tipo_objetivo_guardado = pa.get('tipo_objetivo', 'Mantenimiento')
        idx_obj = 0
        if tipo_objetivo_guardado == 'Déficit (Bajar Peso)': idx_obj = 1
        elif tipo_objetivo_guardado == 'Superávit (Subir Peso)': idx_obj = 2

        tipo_objetivo = c_obj1.selectbox(
            "Objetivo de Peso", 
            ["Mantenimiento", "Déficit (Bajar Peso)", "Superávit (Subir Peso)"],
            index=idx_obj
        )
        
        ajuste_kcal = c_obj2.number_input(
            "Kcal de Ajuste (Restar o Sumar)", 
            min_value=0, value=pa.get('ajuste_kcal', 300), step=50,
            help="Si es Déficit, estas Kcal se restarán. Si es Superávit, se sumarán."
        )
        
        # Selector de Proteína
        st.markdown("**Requerimiento Proteico**")
        c_prot1, c_prot2 = st.columns(2)
        
        proteina_g_kg = c_prot1.number_input(
            "Proteína (g/kg de peso)", 
            min_value=0.5, max_value=4.0, 
            value=pa.get('proteina_g_kg', 1.6), step=0.1, format="%.1f",
            help="Ejemplo: 1.6 a 2.2 g/kg para ganancia muscular o pérdida de grasa."
        )
        
        # Cálculo visual inmediato
        prot_total_calc = peso * proteina_g_kg
        c_prot2.info(f"Total Proteínas objetivo: **{prot_total_calc:.1f} g/día**")
        # --- FIN NUEVA SECCIÓN ---

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
            imc, diagnostico_imc = calcular_imc(peso, talla_cm)
            
            # Usar la masa magra guardada (si existe) para Cunningham
            masa_magra = pa.get('composicion', {}).get('modelo_2c', {}).get('masa_magra', 0) 
            
            # 1. Calcular GET BASE (Mantenimiento)
            get_mantenimiento = calcular_get(sexo, peso, talla_cm, edad, actividad, formula_get, masa_magra)
            
            if formula_get == "Cunningham" and masa_magra == 0:
                st.warning("Se seleccionó Cunningham pero no hay datos de composición corporal. El GET será 0.")

            # 2. Calcular GET OBJETIVO (Final)
            get_objetivo = get_mantenimiento
            if tipo_objetivo == "Déficit (Bajar Peso)":
                get_objetivo = get_mantenimiento - ajuste_kcal
            elif tipo_objetivo == "Superávit (Subir Peso)":
                get_objetivo = get_mantenimiento + ajuste_kcal
            
            if get_objetivo < 0: get_objetivo = 0

            # 3. Guardar diccionario completo
            datos_paciente = {
                'nombre': nombre,
                'edad': edad,
                'sexo': sexo,
                'peso': peso,
                'talla_cm': talla_cm,
                'actividad': actividad,
                'raza': raza,
                'historia_clinica': historia_clinica,
                'imc': imc,
                'diagnostico_imc': diagnostico_imc,
                
                'get': get_mantenimiento,         # Guardamos el mantenimiento
                'formula_get': formula_get, 
                
                # Nuevos Campos Guardados
                'tipo_objetivo': tipo_objetivo,
                'ajuste_kcal': ajuste_kcal,
                'get_objetivo': get_objetivo,     # Guardamos el objetivo final
                'proteina_g_kg': proteina_g_kg,
                'proteina_total_objetivo': peso * proteina_g_kg, # Guardamos los gramos totales

                'pliegues': pa.get('pliegues', {}),
                'circunferencias': pa.get('circunferencias', {}), 
                'diametros': pa.get('diametros', {}), 
                'composicion': pa.get('composicion', {}),
                'dieta_actual': st.session_state.dieta_temporal,
                'plan_semanal': pa.get('plan_semanal', {})
            }
            
            nombre_archivo = guardar_paciente(st.session_state.usuario, datos_paciente)
            
            if nombre_archivo:
                st.success(f"Paciente '{nombre}' guardado/actualizado exitosamente.")
                st.session_state.paciente_actual = datos_paciente
                
                st.subheader("Resultados de Evaluación Inicial")
                col1, col2, col3 = st.columns(3)
                col1.metric("IMC", f"{imc:.2f}", diagnostico_imc)
                col2.metric("GET Mantenimiento", f"{get_mantenimiento:.0f} kcal")
                
                # Mostrar métrica con color según objetivo
                delta_color = "normal" # Gris/Negro
                if tipo_objetivo == "Déficit (Bajar Peso)": delta_color = "inverse" # Rojo
                elif tipo_objetivo == "Superávit (Subir Peso)": delta_color = "normal" # Verde (normalmente)

                col3.metric(
                    "GET Objetivo (Meta)", 
                    f"{get_objetivo:.0f} kcal", 
                    f"{tipo_objetivo}",
                    delta_color=delta_color
                )

# --- PÁGINA DE ANTROPOMETRÍA ---
# --- PÁGINA DE ANTROPOMETRÍA ---
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
        pliegues['Suprailíaco'] = col1.number_input("Suprailíaco (Cresta Iliaca)", min_value=0.0, value=pliegues.get('Suprailíaco', 0.0), step=0.1, format="%.1f")
        pliegues['Abdominal'] = col2.number_input("Abdominal", min_value=0.0, value=pliegues.get('Abdominal', 0.0), step=0.1, format="%.1f")
        pliegues['Pantorrilla Medial'] = col3.number_input("Pantorrilla Medial", min_value=0.0, value=pliegues.get('Pantorrilla Medial', 0.0), step=0.1, format="%.1f")
        
        col1, col2, col3 = st.columns(3)
        pliegues['Muslo (frontal)'] = col1.number_input("Muslo (frontal)", min_value=0.0, value=pliegues.get('Muslo (frontal)', 0.0), step=0.1, format="%.1f")


        st.divider()
        st.markdown("##### Registro de Circunferencias (cm)")
        col1, col2, col3 = st.columns(3)
        circs['Brazo (relajado)'] = col1.number_input("Brazo (relajado)", min_value=0.0, value=circs.get('Brazo (relajado)', 0.0), step=0.1, format="%.1f")
        circs['Pantorrilla (máxima)'] = col2.number_input("Pantorrilla (máxima)", min_value=0.0, value=circs.get('Pantorrilla (máxima)', 0.0), step=0.1, format="%.1f")
        circs['Muslo (medial)'] = col3.number_input("Muslo (medial)", min_value=0.0, value=circs.get('Muslo (medial)', 0.0), step=0.1, format="%.1f")
        
        st.divider()
        st.markdown("##### Registro de Diámetros Óseos (cm)")
        col1, col2, col3 = st.columns(3)
        diams['Húmero (bi-epicondilar)'] = col1.number_input("Húmero (bi-epicondilar)", min_value=0.0, value=diams.get('Húmero (bi-epicondilar)', 0.0), step=0.1, format="%.1f")
        diams['Fémur (bi-condilar)'] = col2.number_input("Fémur (bi-condilar)", min_value=0.0, value=diams.get('Fémur (bi-condilar)', 0.0), step=0.1, format="%.1f")
        diams['Muñeca (bi-estiloideo)'] = col3.number_input("Muñeca (bi-estiloideo)", min_value=0.0, value=diams.get('Muñeca (bi-estiloideo)', 0.0), step=0.1, format="%.1f")
        
        st.divider()
        
        submitted = st.form_submit_button("Calcular Composición y Somatotipo")

    if submitted:
        endo, meso, ecto = calcular_somatotipo(
            pa['peso'], pa['talla_cm'], pliegues, circs, diams
        )
        clasificacion_somato = clasificar_somatotipo(endo, meso, ecto)
        
        comp_2c = calcular_composicion_2c_durnin_siri(
            pa['peso'], pa['sexo'], pa['edad'], pliegues
        )
        st.success("Modelo 2C (Durnin-Edad/Siri) calculado.")

        comp_5c = calcular_composicion_5c_kerr(
            pa['peso'], pa['talla_cm'], pa['sexo'],
            pliegues, diams 
        )
        if comp_5c['error']:
            st.error(f"Error en cálculo 5C: {comp_5c['error']}")
        else:
            st.success("Modelo 5C (Kerr - Fórmulas solicitadas) calculado.")

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
        
        # Recalcular GET si la fórmula es Cunningham y ahora tenemos masa magra
        if pa.get('formula_get') == 'Cunningham':
            masa_magra_2c = comp_2c.get('masa_magra', 0)
            if masa_magra_2c > 0:
                nuevo_get = calcular_get(
                    pa['sexo'], pa['peso'], pa['talla_cm'], pa['edad'], 
                    pa['actividad'], 'Cunningham', masa_magra_2c
                )
                pa['get'] = nuevo_get
                st.success(f"GET (Cunningham) recalculado con nueva masa magra: {nuevo_get:.0f} kcal")

        guardar_paciente(st.session_state.usuario, pa)
        st.session_state.paciente_actual = pa 
        
        st.success("Cálculos de composición y somatotipo realizados y guardados.")
        st.rerun() 

    # --- SECCIÓN DE RESULTADOS (ACTUALIZADA CON TABS) ---
    st.subheader("Diagnóstico Nutricional")
    
    # --- NUEVO: SECCIÓN DE DESCARGAS DE COMPOSICIÓN ---
    st.markdown("##### Exportar Evaluación Corporal")
    st.caption("Descargue el informe completo de la evaluación antropométrica, composición y somatotipo.")
    
    col_ex_1, col_ex_2 = st.columns(2)
    
    with col_ex_1:
        excel_data_composicion = generar_excel_composicion(pa)
        st.download_button(
            label="📥 Descargar Evaluación (.xlsx)",
            data=excel_data_composicion,
            file_name=f"evaluacion_{pa.get('nombre', 'paciente').replace(' ','_').lower()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="export_eval_xlsx_antropo"
        )
    with col_ex_2:
        # Esta función ahora genera el PDF con el gráfico de somatotipo
        pdf_data_composicion = generar_pdf_composicion(pa)
        st.download_button(
            label="📄 Descargar Evaluación (PDF)",
            data=pdf_data_composicion,
            file_name=f"evaluacion_{pa.get('nombre', 'paciente').replace(' ','_').lower()}.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary",
            key="export_eval_pdf_antropo"
        )
    
    st.divider()
    # --- FIN DE SECCIÓN DE DESCARGAS ---

    
    pa = st.session_state.paciente_actual
    comp = pa.get('composicion', {})
    comp_2c = comp.get('modelo_2c', {})
    comp_5c = comp.get('modelo_5c', {}) 
    som = comp.get('somatotipo', {})
    paciente_sexo = pa.get('sexo', 'Masculino')

    tab_nombres = [
        "⚖️ Evaluación General",
        "🧍 Modelo 2C (Durnin-Edad/Siri)",
        "🔬 Modelo 5C (Kerr - Modificado)",
    ]
    
    # Fórmulas específicas por sexo
    formulas_masculinas = {
        "🧬 Sloan (1967)": "Sloan (1967) - Varones",
        "🧫 Wilmore & Behnke (1969)": "Wilmore & Behnke (1969) - Varones",
        "🔍 Katch & McArdle (1973)": "Katch & McArdle (1973) - Varones"
    }
    
    formulas_femeninas = {
        "🧬 Sloan et al. (1962)": "Sloan, Burt, & Blyth (1962) - Mujeres",
        "🧫 Wilmore & Behnke (1970)": "Wilmore & Behnke (1970) - Mujeres",
        "🔍 Jackson et al. (1980)": "Jackson, Pollock, & Ward (1980) - Mujeres"
    }

    if paciente_sexo == 'Masculino':
        formulas_a_mostrar = formulas_masculinas
    else:
        formulas_a_mostrar = formulas_femeninas

    tab_nombres.extend(list(formulas_a_mostrar.keys()))
    tab_nombres.append("📊 Somatotipo (Heath-Carter)")
    
    try:
        tab_general, tab_2c, tab_5c, *tabs_formulas, tab_somatotipo = st.tabs(tab_nombres)
    except st.errors.StreamlitAPIException:
        st.error("Error al crear las pestañas de diagnóstico.")
        st.stop()
    
    with tab_general:
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

    with tab_2c:
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

    with tab_5c:
        if comp_5c:
            if comp_5c.get('error'):
                st.warning(f"No se pudo calcular el modelo 5C: {comp_5c['error']}")
                st.info("Este modelo requiere los pliegues de Durnin (4) y los diámetros de Húmero y Fémur.")
            
            elif comp_5c.get('mg_kg', 0) > 0 or comp_5c.get('mm_kg', 0) > 0 or comp_5c.get('mo_kg', 0) > 0:
                st.markdown("##### Composición de 5 Componentes (Modelo Kerr - Solicitado)")
                st.caption("Análisis fraccionario basado en De Rose & Kerr (MO), % fijos (MR, MP) y Durnin (MG).")
                
                data_5c = {
                    "Componente": ["Masa Grasa (MG)", "Masa Muscular (MM)", "Masa Ósea (MO)", "Masa Residual (MR)", "Masa de Piel (MP)"],
                    "Masa (kg)": [
                        comp_5c.get('mg_kg', 0), comp_5c.get('mm_kg', 0), comp_5c.get('mo_kg', 0), 
                        comp_5c.get('mr_kg', 0), comp_5c.get('mp_kg', 0)
                    ],
                    "% Corporal": [
                        comp_5c.get('mg_porc', 0), comp_5c.get('mm_porc', 0), comp_5c.get('mo_porc', 0), 
                        comp_5c.get('mr_porc', 0), comp_5c.get('mp_porc', 0)
                    ],
                    "Diagnóstico": [
                        comp_5c.get('mg_diag', 'N/A'), comp_5c.get('mm_diag', 'N/A'), comp_5c.get('mo_diag', 'N/A'), 
                        comp_5c.get('mr_diag', 'N/A'), comp_5c.get('mp_diag', 'N/A')
                    ]
                }
                df_5c = pd.DataFrame(data_5c)
                
                st.dataframe(df_5c.style.format({
                    'Masa (kg)': '{:.2f}',
                    '% Corporal': '{:.1f}%'
                }), use_container_width=True)

                st.divider()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Peso Total (Paciente)", f"{pa.get('peso', 0):.2f} kg")
                col2.metric("Suma de Componentes", f"{comp_5c.get('suma_total', 0):.2f} kg", 
                            delta=f"{comp_5c.get('suma_total', 0) - pa.get('peso', 0):.2f} kg de diferencia")
                col3.metric("Densidad Corporal (Durnin)", f"{comp_5c.get('dc', 0):.4f} g/cm³")
                
            else:
                 st.info("No se han calculado datos para el Modelo 5C. Por favor, ingrese todas las medidas y presione 'Calcular'.")
        else:
            st.info("No se han calculado datos para el Modelo 5C. Por favor, ingrese todas las medidas y presione 'Calcular'.")
    
    # Iterar sobre las pestañas de fórmulas personalizadas
    for tab, (nombre_corto_con_emoji, nombre_largo_funcion) in zip(tabs_formulas, formulas_a_mostrar.items()):
        with tab:
            st.markdown(f"##### Cálculo de Composición: {nombre_corto_con_emoji.split(' ', 1)[-1]}")
            st.caption(f"Referencia: {nombre_largo_funcion}")
            
            resultado, error = calcular_composicion_personalizada(
                nombre_largo_funcion,
                pa['sexo'], pa['edad'], pa['peso'],
                pa.get('pliegues', {}),
                pa.get('diametros', {}),
                pa.get('circunferencias', {}),
                pa.get('talla_cm', 0)
            )
            
            if error:
                st.error(f"Error al calcular: {error}")
                st.info("Asegúrese de haber ingresado los pliegues necesarios para esta fórmula en el formulario de arriba y haber presionado 'Calcular Composición'.")
            elif resultado:
                st.success(f"Resultados para: {nombre_corto_con_emoji.split(' ', 1)[-1]}")
                
                col1, col2 = st.columns(2)
                col1.metric("Densidad Corporal (DC)", f"{resultado['dc']:.4f} g/cm³")
                col2.metric("% Grasa (Siri)", f"{resultado['porc_grasa']:.1f} %")
                
                col1, col2 = st.columns(2)
                col1.metric("Masa Grasa", f"{resultado['masa_grasa']:.1f} kg")
                col2.metric("Masa Magra", f"{resultado['masa_magra']:.1f} kg")

                with st.expander("Detalles de la fórmula y pliegues utilizados"):
                    if "Sloan (1967)" in nombre_largo_funcion:
                        st.write("`DC = 1.1043 - 0.001327(Muslo Frontal) - 0.001310(Subescapular)`")
                        st.write(f"- Pliegue Muslo (frontal): {pa.get('pliegues', {}).get('Muslo (frontal)', 0.0)} mm")
                        st.write(f"- Pliegue Subescapular: {pa.get('pliegues', {}).get('Subescapular', 0.0)} mm")
                    elif "Wilmore & Behnke (1969)" in nombre_largo_funcion:
                        st.write("`DC = 1.08543 - 0.000886(Abdominal) - 0.00040(Muslo Frontal)`")
                        st.write(f"- Pliegue Abdominal: {pa.get('pliegues', {}).get('Abdominal', 0.0)} mm")
                        st.write(f"- Pliegue Muslo (frontal): {pa.get('pliegues', {}).get('Muslo (frontal)', 0.0)} mm")
                    elif "Katch & McArdle (1973)" in nombre_largo_funcion:
                        st.write("`DC = 1.09655 - 0.00049 - 0.00103(Triccipital) - 0.00056(Subescapular) + 0.00054(Abdominal)`")
                        st.write(f"- Pliegue Tricipital: {pa.get('pliegues', {}).get('Tricipital', 0.0)} mm")
                        st.write(f"- Pliegue Subescapular: {pa.get('pliegues', {}).get('Subescapular', 0.0)} mm")
                        st.write(f"- Pliegue Abdominal: {pa.get('pliegues', {}).get('Abdominal', 0.0)} mm")
                    elif "Sloan, Burt, & Blyth (1962)" in nombre_largo_funcion:
                        st.write("`DC = 1.0764 - 0.00081(Cresta Iliaca) - 0.00088(Triccipital)`")
                        st.write(f"- Pliegue Suprailíaco (Cresta Iliaca): {pa.get('pliegues', {}).get('Suprailíaco', 0.0)} mm")
                        st.write(f"- Pliegue Tricipital: {pa.get('pliegues', {}).get('Tricipital', 0.0)} mm")
                    elif "Wilmore & Behnke (1970)" in nombre_largo_funcion:
                        st.write("`DC = 1.06234 - 0.00068(Subescapular) - 0.00039(Triccipital) - 0.00025(Muslo Frontal)`")
                        st.write(f"- Pliegue Subescapular: {pa.get('pliegues', {}).get('Subescapular', 0.0)} mm")
                        st.write(f"- Pliegue Tricipital: {pa.get('pliegues', {}).get('Tricipital', 0.0)} mm")
                        st.write(f"- Pliegue Muslo (frontal): {pa.get('pliegues', {}).get('Muslo (frontal)', 0.0)} mm")
                    elif "Jackson, Pollock, & Ward (1980)" in nombre_largo_funcion:
                        st.write("`DC = 1.221389 - 0.04057(log10(Suma 3 Pliegues)) - 0.00016(Edad)`")
                        st.write(f"- Pliegue Tricipital: {pa.get('pliegues', {}).get('Tricipital', 0.0)} mm")
                        st.write(f"- Pliegue Muslo (frontal): {pa.get('pliegues', {}).get('Muslo (frontal)', 0.0)} mm")
                        st.write(f"- Pliegue Suprailíaco (Cresta Iliaca): {pa.get('pliegues', {}).get('Suprailíaco', 0.0)} mm")
                        st.write(f"- Edad: {pa.get('edad', 0)} años")


    with tab_somatotipo:
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
                
                st.divider()
                st.markdown("###### Diagnóstico Predominante")
                explicacion = obtener_explicacion_somatotipo(clasificacion_str)
                st.write(explicacion)
            
            with col2:
                fig_somato = crear_grafico_somatotipo(
                    som.get('endo', 0), 
                    som.get('meso', 0), 
                    som.get('ecto', 0)
                )
                st.plotly_chart(fig_somato, use_container_width=True)
        else:
            st.info("No se han calculado datos para el Somatotipo. Por favor, ingrese las medidas necesarias y presione 'Calcular'.")


# --- PÁGINA: ASISTENTE DE IA ---
def mostrar_pagina_asistente_ia():
    """
    Página dedicada para interactuar con el Asistente de IA (Gemini).
    """
    st.title("🤖 Asistente de IA (Gemini)")
    st.markdown("Usa la inteligencia artificial para generar ideas de planes y recetas.")

    # --- Verificar si hay un paciente cargado ---
    if not st.session_state.paciente_actual:
        st.warning("Por favor, cargue o registre un paciente en la página de 'Inicio' primero.")
        st.stop()
        
    # Cargar modelo de IA
    modelo_gemini = configurar_modelo_gemini()
    pa = st.session_state.paciente_actual

    # Obtener lista de muestra de alimentos (necesaria para el prompt de recetas)
    db_alimentos = st.session_state.db_alimentos.copy()
    try:
        lista_alimentos_muestra = db_alimentos['NOMBRE DEL ALIMENTO'].sample(20).tolist()
        lista_alimentos_str = ", ".join(lista_alimentos_muestra)
    except ValueError:
        # En caso de que la BD tenga menos de 20 alimentos
        lista_alimentos_muestra = db_alimentos['NOMBRE DEL ALIMENTO'].tolist()
        lista_alimentos_str = ", ".join(lista_alimentos_muestra)
    except Exception as e:
        # Fallback por si la base de datos no está cargada
        st.error(f"Error al cargar la lista de alimentos: {e}")
        lista_alimentos_str = "quinua, pollo, arroz, papa"


    # --- Tarjeta 1: Generador de Plan de Alimentación ---
    with st.container(border=True):
        st.markdown("### 💡 Generar Ideas de Plan de Alimentación")
        st.markdown(
            """
            Genera un **ejemplo de menú de 1 día** basado en el GET
            y la historia clínica del paciente.
            """
        )
        
        macros_deseados = st.session_state.get('dist_macros', {'cho': 50, 'prot': 20, 'fat': 30})
        
        st.info(
            f"Se usará el GET de **{pa.get('get', 0):.0f} kcal** y la distribución "
            f"(P/C/G): **{macros_deseados['prot']}% / {macros_deseados['cho']}% / {macros_deseados['fat']}%**"
        )
        
        if st.button("Generar Plan de Ejemplo", use_container_width=True, type="primary"):
            if modelo_gemini is None:
                st.error("El modelo de IA no está disponible.")
            else:
                prompt_plan = f"""
                Actúa como un **asistente culinario** o **planificador de menús**. Tu tarea es generar **ideas de menú de ejemplo** (1 día) 
                para **una persona** con las siguientes características:
                
                - Objetivo Calórico: {pa.get('get', 0):.0f} kcal
                - Distribución de Macros Deseada:
                    - Proteínas: {macros_deseados['prot']}%
                    - Carbohidratos: {macros_deseados['cho']}%
                    - Grasas: {macros_deseados['fat']}%
                - Notas Adicionales (preferencias o alergias):
                  "{pa.get('historia_clinica', 'Sin notas')}"

                Instrucciones para la respuesta:
                1.  Crea un menú de ejemplo para Desayuno, Colación Mañana, Almuerzo, Colación Tarde y Cena.
                2.  Incluye ideas de alimentos y cantidades aproximadas (en gramos o medidas caseras).
                3.  Asegúrate de que la suma total se aproxime a los objetivos.
                4.  Si se mencionan alergias o restricciones en las notas, EVITA esos alimentos.
                5.  Formatea la respuesta usando Markdown (títulos, listas).
                """
                
                with st.spinner("🧠 Pensando..."):
                    respuesta_ia = generar_respuesta_gemini(modelo_gemini, prompt_plan)
                    # Usamos una clave de sesión específica para esta página
                    st.session_state.respuesta_plan_ia = respuesta_ia
    
    # --- RESULTADO MINIMIZABLE 1 ---
    if 'respuesta_plan_ia' in st.session_state:
        with st.expander("Ver/Ocultar Plan de Ejemplo Generado", expanded=True):
            st.success("🤖 Respuesta de la IA:")
            st.markdown(st.session_state.respuesta_plan_ia)


    # Espacio entre tarjetas
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Tarjeta 2: Generador de Recetas Peruanas (MODIFICADA) ---
    with st.container(border=True):
        st.markdown("### 🇵🇪 Generar Ideas de Recetas Peruanas (Adaptadas)")
        st.markdown("Escribe un ingrediente y la IA buscará recetas **adecuadas al GET y macros del paciente**.")
        
        with st.form("form_recetas_ia"):
            ingrediente = st.text_input("Ingrediente principal (ej. Pollo, Lentejas, Quinua, Pescado)")
            submit_receta = st.form_submit_button("Buscar Recetas Peruanas Adaptadas")
            
            if submit_receta and ingrediente:
                if modelo_gemini is None:
                    st.error("El modelo de IA no está disponible.")
                else:
                    pa = st.session_state.paciente_actual
                    macros_deseados = st.session_state.get('dist_macros', {'cho': 50, 'prot': 20, 'fat': 30})
                    total_get = pa.get('get', 0)
                    
                    calorias_comida_objetivo = total_get * 0.35
                    
                    if calorias_comida_objetivo <= 0:
                        calorias_comida_objetivo = 600 
                        
                    historia_clinica = pa.get('historia_clinica', 'Sin notas')

                    prompt_receta = f"""
                    Actúa como un chef experto en **cocina peruana** y un **asistente de nutrición**.
                    Quiero 3 ideas de recetas peruanas que usen el ingrediente principal: "{ingrediente}".

                    **REQUISITOS IMPORTANTES DEL PACIENTE:**
                    Estas recetas deben ser adecuadas para un paciente con las siguientes necesidades. 
                    Cada receta (para 1 porción/plato) debe apuntar a:

                    1.  **Objetivo Calórico por Plato:** Aprox. **{calorias_comida_objetivo:.0f} kcal**.
                    2.  **Distribución de Macros (General del Paciente):** - Proteínas: {macros_deseados['prot']}%
                        - Carbohidratos: {macros_deseados['cho']}%
                        - Grasas: {macros_deseados['fat']}%
                    3.  **Restricciones/Alergias:** "{historia_clinica}". (Si dice 'Sin notas', ignora esto).
                    4.  **Ingrediente Principal Obligatorio:** "{ingrediente}".
                    5.  **Inspiración (opcional):** {lista_alimentos_str}

                    **Instrucciones para la respuesta (Formato Markdown):**
                    Para CADA UNA de las 3 recetas:
                    1.  **Título de la Receta (con emoji peruano 🇵🇪)**.
                    2.  **Estimación Nutricional (1 porción):**
                        - Kcal: (El valor aproximado a {calorias_comida_objetivo:.0f})
                        - Proteínas: (g)
                        - Grasas: (g)
                        - Carbohidratos: (g)
                    3.  **Ingredientes Principales (para 1 porción):** (lista de ingredientes CON GRAMOS APROXIMADOS).
                    4.  **Preparación:** (pasos claros y sencillos).
                    5.  **Justificación de Adecuación:** (Breve explicación de por qué esta receta cumple los requisitos calóricos y/o de macros).

                    ---
                    **Recuerda:** Para agregar esta receta a tu dieta, debes buscar y añadir los ingredientes (con sus gramos) uno por uno usando el **buscador manual** en la página 'Crear Dieta'.
                    """
                    
                    with st.spinner(f"Buscando recetas peruanas adaptadas con {ingrediente}..."):
                        respuesta_ia = generar_respuesta_gemini(modelo_gemini, prompt_receta)
                        st.session_state.respuesta_receta_ia = respuesta_ia

    # --- RESULTADO MINIMIZABLE 2 ---
    if 'respuesta_receta_ia' in st.session_state:
        with st.expander("Ver/Ocultar Recetas Generadas", expanded=True):
            st.success("🤖 Respuesta de la IA:")
            st.markdown(st.session_state.respuesta_receta_ia)
# --- FIN DE LA NUEVA PÁGINA ---


# --- PÁGINA DE CREAR DIETA (CON ASIGNADOR SEMANAL) ---
def mostrar_pagina_crear_dieta():
    """Página para buscar alimentos y agregarlos a la dieta del paciente."""
    st.title("Creación de Dieta 🍲")
    
    if not st.session_state.paciente_actual:
        st.warning("Por favor, cargue o registre un paciente en la página de 'Inicio' primero.")
        st.stop()
        
    if st.session_state.db_alimentos is None or st.session_state.db_alimentos.empty:
        st.error("La base de datos de alimentos no se ha cargado correctamente.")
        st.stop()
    
    pa = st.session_state.paciente_actual
    
    # --- 1. Obtener Metas Diarias ---
    get_diario = pa.get('get_objetivo', 2000)
    prot_diaria = pa.get('proteina_total_objetivo', (get_diario * 0.20)/4)
    
    if 'k_cho' in st.session_state:
        cho_diario = (get_diario * (st.session_state.k_cho/100))/4
        fat_diario = (get_diario * (st.session_state.k_fat/100))/9
    else:
        cho_diario = (get_diario * 0.50)/4
        fat_diario = (get_diario * 0.30)/9

    st.info(f"Paciente: **{pa['nombre']}** | Meta Diaria: **{get_diario:.0f} kcal** (P: {prot_diaria:.0f}g | C: {cho_diario:.0f}g | G: {fat_diario:.0f}g)")
    
    tab_diaria, tab_semanal = st.tabs(["🥣 Dieta Detallada & Asignación", "🗓️ Plan Semanal (Vista Completa)"])

    with tab_diaria:
        
        # ============================================================
        # LÓGICA DE REDISTRIBUCIÓN PROPORCIONAL
        # ============================================================
        tiempos_orden = ["Desayuno", "Colación Mañana", "Almuerzo", "Colación Tarde", "Cena", "Colación Noche"]

        def recalcular_porcentajes(tiempo_trigger):
            key_trigger = f"pct_{tiempo_trigger}"
            if key_trigger not in st.session_state: return
            
            nuevo_valor = st.session_state[key_trigger]
            resto_disponible = 100 - nuevo_valor
            
            otros_tiempos = [t for t in tiempos_orden if t != tiempo_trigger]
            vals_otros = {t: st.session_state.get(f"pct_{t}", 0) for t in otros_tiempos}
            suma_actual_otros = sum(vals_otros.values())
            
            acumulado = 0
            for i, t in enumerate(otros_tiempos):
                key_t = f"pct_{t}"
                val_antiguo = vals_otros[t]
                
                if suma_actual_otros > 0:
                    ratio = val_antiguo / suma_actual_otros
                    if i == len(otros_tiempos) - 1:
                        nuevo_otro = max(0, resto_disponible - acumulado)
                    else:
                        nuevo_otro = int(round(ratio * resto_disponible))
                else:
                    nuevo_otro = 0 
                
                st.session_state[key_t] = nuevo_otro
                st.session_state.dist_porc_comidas[t] = nuevo_otro
                acumulado += nuevo_otro
            
            st.session_state.dist_porc_comidas[tiempo_trigger] = nuevo_valor

        # ============================================================
        # CONFIGURACIÓN DE PORCENTAJES
        # ============================================================
        with st.expander("⚙️ Configurar Distribución (%) Automática", expanded=False):
            
            if 'dist_porc_comidas' not in st.session_state:
                defaults = [20, 10, 35, 10, 25, 0]
                st.session_state.dist_porc_comidas = dict(zip(tiempos_orden, defaults))
            
            for t in tiempos_orden:
                key_t = f"pct_{t}"
                if key_t not in st.session_state:
                    st.session_state[key_t] = st.session_state.dist_porc_comidas.get(t, 0)

            if st.button("🔄 Restablecer estándar"):
                defaults = [20, 10, 35, 10, 25, 0]
                st.session_state.dist_porc_comidas = dict(zip(tiempos_orden, defaults))
                for i, t in enumerate(tiempos_orden):
                    st.session_state[f"pct_{t}"] = defaults[i]
                st.rerun()

            st.markdown("##### Modifica un valor y los demás se ajustarán:")
            
            cols_cfg = st.columns(6)
            for i, t in enumerate(tiempos_orden):
                with cols_cfg[i]:
                    st.number_input(
                        f"% {t.split()[0]}", 
                        min_value=0, max_value=100, step=1,
                        key=f"pct_{t}",
                        on_change=recalcular_porcentajes,
                        args=(t,)
                    )
            
            suma_pct = sum([st.session_state.get(f"pct_{t}", 0) for t in tiempos_orden])
            if suma_pct == 100:
                st.progress(1.0)
            else:
                st.progress(min(suma_pct/100, 1.0))
                st.caption(f"Suma: {suma_pct}%")

        # ============================================================
        
        db_alimentos = st.session_state.db_alimentos.copy()
        db_alimentos['busqueda_display'] = "[" + db_alimentos['CÓDIGO'].astype(str) + "] " + db_alimentos['NOMBRE DEL ALIMENTO']
        
        st.divider()
        with st.expander("➕ Agregar Nuevo Ingrediente", expanded=False):
            with st.form("form_agregar_alimento"):
                alimento_busqueda_sel = st.selectbox("Buscar:", options=db_alimentos['busqueda_display'], index=None)
                c1, c2, c3 = st.columns([1, 1, 2])
                gramos = c1.number_input("Gramos", min_value=1, value=100, step=10)
                tiempo = c2.selectbox("Tiempo", ["Desayuno", "Colación Mañana", "Almuerzo", "Colación Tarde", "Cena", "Colación Noche"])
                dias = c3.multiselect("Copiar a:", ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"])
                
                if st.form_submit_button("Agregar"):
                    if alimento_busqueda_sel:
                        row = db_alimentos[db_alimentos['busqueda_display'] == alimento_busqueda_sel].iloc[0]
                        f = gramos / 100.0
                        item = {
                            'id': f"{row['CÓDIGO']}_{pd.Timestamp.now().isoformat()}",
                            'Tiempo Comida': tiempo, 'Código': row['CÓDIGO'], 'Alimento': row['NOMBRE DEL ALIMENTO'], 'Gramos': gramos,
                            'Kcal': row['Kcal']*f, 'Proteínas': row['Proteínas']*f, 'Grasas': row['Grasas']*f, 'Carbohidratos': row['Carbohidratos']*f,
                            'Fibra': row['Fibra']*f, 'Agua': row['Agua']*f, 'Calcio': row['Calcio']*f, 'Fósforo': row['Fósforo']*f,
                            'Zinc': row['Zinc']*f, 'Hierro': row['Hierro']*f, 'Vitamina C': row['Vitamina C']*f, 'Sodio': row['Sodio']*f,
                            'Potasio': row['Potasio']*f, 'Vitamina A': row['Vitamina A']*f, 'Acido Folico': row['Acido Folico']*f
                        }
                        st.session_state.dieta_temporal.append(item)
                        st.session_state.paciente_actual['dieta_actual'] = st.session_state.dieta_temporal
                        
                        if dias:
                            plan = st.session_state.paciente_actual.get('plan_semanal', {})
                            txt = f"{item['Alimento']} ({gramos}g)"
                            for d in dias:
                                if d not in plan: plan[d] = {}
                                plan[d][tiempo] = (plan[d].get(tiempo, "") + ", " + txt).strip(", ")
                            st.session_state.paciente_actual['plan_semanal'] = plan
                        
                        guardar_paciente(st.session_state.usuario, st.session_state.paciente_actual)
                        st.success("✅ Agregado"); st.rerun()
                    else: st.error("Seleccione alimento")

        # --- LISTA TIPO TABLA CON EDICIÓN ---
        st.divider()
        
        # --- INTERRUPTOR GLOBAL PARA OCULTAR MICROS ---
        c_tit, c_toggle, c_clean = st.columns([3, 2, 1])
        c_tit.subheader("Menú del Día")
        # Aquí está la magia:
        
        ver_micros = c_toggle.toggle("🔬 Mostrar Micronutrientes", value=False)
        
        if c_clean.button("🗑️ Borrar Todo"):
            st.session_state.dieta_temporal = []; guardar_paciente(st.session_state.usuario, st.session_state.paciente_actual); st.rerun()

        if st.session_state.dieta_temporal:
            df = pd.DataFrame(st.session_state.dieta_temporal)
            df['Tiempo Comida'] = pd.Categorical(df['Tiempo Comida'], categories=tiempos_orden, ordered=True)
            df = df.sort_values('Tiempo Comida')
            
            for t in tiempos_orden:
                df_t = df[df['Tiempo Comida'] == t]
                
                pct_comida = st.session_state.dist_porc_comidas.get(t, 0)
                ratio = pct_comida / 100.0
                
                meta_kcal_t = get_diario * ratio
                meta_prot_t = prot_diaria * ratio
                meta_cho_t = cho_diario * ratio
                meta_fat_t = fat_diario * ratio

                if not df_t.empty or pct_comida > 0:
                    st.markdown(f"#### 🍽️ {t} <span style='background-color:#0A0B23; padding:2px 6px; border-radius:4px; font-size:0.7em'>{pct_comida}%</span>", unsafe_allow_html=True)
                    
                    if not df_t.empty:
                        c1, c2, c3, c4, c5, c6, c7 = st.columns([2.5, 1.2, 1, 1, 1, 1, 0.5])
                        c1.markdown("✏️ **Alimento**"); c2.markdown("**Gramos**"); c3.markdown("🔥 **Kcal**"); c4.markdown("🥩 **P(g)**"); c5.markdown("🍞 **C(g)**"); c6.markdown("🥑 **G(g)**"); c7.markdown("")

                        for _, row in df_t.iterrows():
                            c1, c2, c3, c4, c5, c6, c7 = st.columns([2.5, 1.2, 1, 1, 1, 1, 0.5])
                            c1.caption(f"{row['Alimento']}")
                            key_gramos = f"g_in_{row['id']}"
                            c2.number_input("g", min_value=1, max_value=5000, step=10, value=int(row['Gramos']), key=key_gramos, label_visibility="collapsed", on_change=actualizar_gramos_item, args=(row['id'],))
                            c3.write(f"{row['Kcal']:.0f}"); c4.write(f"{row['Proteínas']:.1f}"); c5.write(f"{row['Carbohidratos']:.1f}"); c6.write(f"{row['Grasas']:.1f}")
                            if c7.button("✕", key=f"d_{row['id']}"): eliminar_item_dieta(row['id']); st.rerun()
                        st.markdown("---")
                    else:
                        st.caption(f"⚠️ *Asignado {meta_kcal_t:.0f} kcal, pero sin alimentos.*")

                    # TOTALES VS META
                    sum_kcal = df_t['Kcal'].sum() if not df_t.empty else 0
                    sum_prot = df_t['Proteínas'].sum() if not df_t.empty else 0
                    sum_cho = df_t['Carbohidratos'].sum() if not df_t.empty else 0
                    sum_fat = df_t['Grasas'].sum() if not df_t.empty else 0

                    t1, t2, t3, t4, t5, t6, t7 = st.columns([2.5, 1.2, 1, 1, 1, 1, 0.5])
                    t1.markdown("**TOTAL ACTUAL:**"); t2.markdown(""); 
                    color_kcal = "green" if abs(sum_kcal - meta_kcal_t) < 50 else "orange"
                    t3.markdown(f":{color_kcal}[**{sum_kcal:.0f}**]"); t4.markdown(f"**{sum_prot:.1f}**"); t5.markdown(f"**{sum_cho:.1f}**"); t6.markdown(f"**{sum_fat:.1f}**")
                    
                    r1, r2, r3, r4, r5, r6, r7 = st.columns([2.5, 1.2, 1, 1, 1, 1, 0.5])
                    r1.markdown(f"<span style='color:gray'><i>Meta ({pct_comida}%):</i></span>", unsafe_allow_html=True); r2.markdown("")
                    r3.markdown(f"<span style='color:gray'><i>{meta_kcal_t:.0f}</i></span>", unsafe_allow_html=True)
                    r4.markdown(f"<span style='color:gray'><i>{meta_prot_t:.1f}</i></span>", unsafe_allow_html=True)
                    r5.markdown(f"<span style='color:gray'><i>{meta_cho_t:.1f}</i></span>", unsafe_allow_html=True)
                    r6.markdown(f"<span style='color:gray'><i>{meta_fat_t:.1f}</i></span>", unsafe_allow_html=True)

                    # --- AQUÍ APLICAMOS EL INTERRUPTOR ---
                    if ver_micros and not df_t.empty:
                        st.markdown("") # Espacio
                        st.markdown("**🔬 Micronutrientes**")
                        cols_micros = ['Alimento', 'Fibra', 'Calcio', 'Hierro', 'Sodio', 'Potasio', 'Vitamina C', 'Vitamina A']
                        df_view = df_t[cols_micros].copy()
                        sumas = df_view.drop(columns=['Alimento']).sum()
                        fila_sum = pd.DataFrame([['TOTAL'] + sumas.tolist()], columns=cols_micros)
                        df_final = pd.concat([df_view, fila_sum], ignore_index=True)
                        st.dataframe(df_final.style.format(precision=1), use_container_width=True, hide_index=True)
                    
                    st.write("") 

        else:
            st.info("Lista vacía. Agrega alimentos arriba.")

    with tab_semanal:
        st.subheader("Plan Semanal Escrito")
        dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        tiempos = ["Desayuno", "Colación Mañana", "Almuerzo", "Colación Tarde", "Cena", "Colación Noche"]
        plan = st.session_state.paciente_actual.get('plan_semanal', {})

        with st.form("plan_edit"):
            cols = st.columns(2)
            for i, d in enumerate(dias):
                with cols[i%2]:
                    with st.expander(f"📅 {d}"):
                        if d not in plan: plan[d] = {}
                        for t in tiempos:
                            plan[d][t] = st.text_area(t, plan[d].get(t,""), height=68, key=f"t_{d}_{t}")
            if st.form_submit_button("💾 Guardar Plan"):
                st.session_state.paciente_actual['plan_semanal'] = plan
                guardar_paciente(st.session_state.usuario, st.session_state.paciente_actual)
                st.success("Guardado")
# --- FIN DE PÁGINA CREAR DIETA ---


# --- PÁGINA DE RESUMEN DE DIETA (MODIFICADA CON PLAN SEMANAL) ---
def mostrar_pagina_resumen_dieta():
    """Página para ver los totales de la dieta, gráficos y adecuación."""
    st.title("Resumen de Dieta y Exportación 📊")
    
    if not st.session_state.paciente_actual:
        st.warning("Por favor, cargue o registre un paciente en la página de 'Inicio' primero.")
        st.stop()
        
    pa = st.session_state.paciente_actual
    
    # --- Pestañas ---
    tab_diario, tab_semanal = st.tabs(["📊 Resumen del Día (Detallado)", "🗓️ Resumen Semanal (Vista Menú)"])

    # ========================================================
    # PESTAÑA 1: RESUMEN DEL DÍA
    # ========================================================
    with tab_diario:
        if not st.session_state.dieta_temporal:
            st.info("No hay alimentos en la dieta detallada para mostrar un resumen.")
            st.info("Agregue alimentos en la pestaña 'Crear Dieta'.")
            
        else:
            df_dieta = pd.DataFrame(st.session_state.dieta_temporal)
            
            # --- 1. Totales Consumidos (ACTUAL) ---
            total_kcal = df_dieta['Kcal'].sum()
            total_prot = df_dieta['Proteínas'].sum()
            total_fat = df_dieta['Grasas'].sum()
            total_cho = df_dieta['Carbohidratos'].sum()
            
            # --- 2. Obtener Datos Base del Paciente (Inicio) ---
            get_target = pa.get('get_objetivo', pa.get('get', 2000))
            gramos_prot_iniciales = pa.get('proteina_total_objetivo', 0)
            
            # --- 3. CÁLCULO INTELIGENTE DE PORCENTAJES INICIALES ---
            # Calculamos qué porcentaje representa la proteína que definiste en "Inicio"
            if get_target > 0 and gramos_prot_iniciales > 0:
                kcal_prot_inicial = gramos_prot_iniciales * 4
                pct_prot_calculado = int((kcal_prot_inicial / get_target) * 100)
                
                # Ajustamos límites lógicos (entre 10% y 60%)
                if pct_prot_calculado < 10: pct_prot_calculado = 10
                if pct_prot_calculado > 60: pct_prot_calculado = 60
            else:
                pct_prot_calculado = 20 # Valor por defecto si no hay datos

            # Calculamos el resto para repartir entre Carbos y Grasas (aprox 60/40 del resto)
            resto_pct = 100 - pct_prot_calculado
            pct_cho_calculado = int(resto_pct * 0.6) # Le damos prioridad a los carbos por defecto
            pct_fat_calculado = 100 - pct_prot_calculado - pct_cho_calculado

            # ==================================================
            # CONFIGURACIÓN DE MACROS (SLIDERS ENLAZADOS)
            # ==================================================
            st.subheader("🎯 Ajuste de Metas (Distribución)")
            
            # Inicializar variables de estado CON LOS DATOS DEL PACIENTE
            if 'k_prot' not in st.session_state: st.session_state.k_prot = pct_prot_calculado
            if 'k_cho' not in st.session_state: st.session_state.k_cho = pct_cho_calculado
            if 'k_fat' not in st.session_state: st.session_state.k_fat = pct_fat_calculado

            # 2. Contenedor de Configuración
            with st.container(border=True):
                col_head, col_reset = st.columns([4, 1])
                with col_head:
                    st.markdown(f"**Meta Calórica:** `{get_target:.0f} kcal`")
                    st.caption(f"Base Clínica: **{gramos_prot_iniciales:.1f}g Proteína** ({pct_prot_calculado}%) definidos en Inicio.")
                
                with col_reset:
                    # 3. BOTÓN RESET INTELIGENTE
                    # Resetea a los valores CLÍNICOS CALCULADOS, no a un estándar genérico
                    if st.button("🔄 Resetear", use_container_width=True, help="Vuelve a la proteína definida en Inicio"):
                        st.session_state.k_prot = pct_prot_calculado
                        st.session_state.k_cho = pct_cho_calculado
                        st.session_state.k_fat = pct_fat_calculado
                        st.rerun() 

                # 4. TRES BARRAS DESLIZANTES
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.slider("🥩 % Proteínas", 0, 100, key="k_prot")
                with c2:
                    st.slider("🍞 % Carbohidratos", 0, 100, key="k_cho")
                with c3:
                    st.slider("🥑 % Grasas", 0, 100, key="k_fat")

                # Validación visual
                suma = st.session_state.k_prot + st.session_state.k_cho + st.session_state.k_fat
                if suma != 100:
                    st.warning(f"⚠️ La suma es **{suma}%**. Ajusta para que sume 100%.", icon="⚠️")

            # --- CÁLCULO DE GRAMOS META (Enlazado a los Sliders) ---
            meta_g_prot = (get_target * (st.session_state.k_prot / 100)) / 4
            meta_g_cho = (get_target * (st.session_state.k_cho / 100)) / 4
            meta_g_fat = (get_target * (st.session_state.k_fat / 100)) / 9

            st.markdown("---")

            # ==================================================
            # DASHBOARD VISUAL (PASTELES SÓLIDOS)
            # ==================================================
            st.markdown("### 📊 Panel de Control Nutricional")
            
            col_dist, col_prog = st.columns([1, 1.5])
            
            # --- A. DISTRIBUCIÓN REAL (LO QUE COMISTE) ---
            with col_dist:
                st.markdown("##### Distribución Real")
                st.caption("Proporción de tus comidas de hoy.")
                
                labels = ['Carbos', 'Proteínas', 'Grasas']
                values_kcal = [total_cho * 4, total_prot * 4, total_fat * 9]
                values_g = [total_cho, total_prot, total_fat]
                colors = ['#1ABC9C', '#FF6B6B', '#FDCB6E']
                
                textos = [f"{g:.0f}g" for g in values_g]

                fig_pie = go.Figure(data=[go.Pie(
                    labels=labels, values=values_kcal,
                    text=textos,
                    textinfo='percent+text',
                    texttemplate='<b>%{percent}</b><br>%{text}', 
                    textposition='inside',
                    hole=0, # Pastel Sólido
                    marker=dict(colors=colors, line=dict(color='#ffffff', width=1)),
                    sort=False,
                    showlegend=True
                )])
                fig_pie.update_layout(
                    margin=dict(t=0, b=0, l=0, r=0),
                    height=300,
                    legend=dict(orientation="h", y=-0.1)
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # --- B. GRÁFICOS DE PROGRESO (PASTELES META VS ACTUAL) ---
            with col_prog:
                st.markdown("##### Progreso vs. Metas")
                st.caption("Visualiza cuánto te falta para llegar al objetivo.")

                def crear_pastel_progreso(titulo, actual, meta, color_hex):
                    restante = max(0, meta - actual)
                    if actual >= meta:
                        valores = [1, 0]
                        colores = [color_hex, "#eee"]
                    else:
                        valores = [actual, restante]
                        colores = [color_hex, "#E0E0E0"]
                        
                    pct = (actual/meta)*100 if meta>0 else 0
                    
                    fig = go.Figure(data=[go.Pie(
                        values=valores,
                        hole=0, # Pastel Sólido
                        marker=dict(colors=colores, line=dict(color='#ffffff', width=1)),
                        sort=False,
                        textinfo='none', 
                        hoverinfo='label+value+percent',
                        direction='clockwise'
                    )])
                    
                    fig.update_layout(
                        title=dict(
                            text=f"<b>{titulo}</b><br><span style='font-size:14px; color:#555'>{actual:.0f} / {meta:.0f} g</span>",
                            x=0.5, y=0.95, xanchor='center', yanchor='top'
                        ),
                        margin=dict(t=50, b=10, l=10, r=10),
                        height=180,
                        showlegend=False
                    )
                    return fig

                cp1, cp2, cp3 = st.columns(3)
                with cp1:
                    st.plotly_chart(crear_pastel_progreso("Proteína", total_prot, meta_g_prot, "#FF6B6B"), use_container_width=True)
                with cp2:
                    st.plotly_chart(crear_pastel_progreso("Carbos", total_cho, meta_g_cho, "#1ABC9C"), use_container_width=True)
                with cp3:
                    st.plotly_chart(crear_pastel_progreso("Grasas", total_fat, meta_g_fat, "#FDCB6E"), use_container_width=True)

            # ==================================================
            # TABLAS Y EXPORTACIÓN
            # ==================================================
            st.divider()
            with st.expander("🔍 Ver Datos Detallados"):
                df_macros_export = pd.DataFrame({
                    '': ['Actual (g)', 'Objetivo (g)', 'Diferencia (g)'],
                    'Proteínas': [total_prot, meta_g_prot, total_prot - meta_g_prot],
                    'Grasas': [total_fat, meta_g_fat, total_fat - meta_g_fat],
                    'Carbohidratos': [total_cho, meta_g_cho, total_cho - meta_g_cho]
                }).set_index('')
                st.dataframe(df_macros_export.style.format("{:.1f}"), use_container_width=True)
                
                cols_resumen = ['Kcal', 'Proteínas', 'Grasas', 'Carbohidratos']
                df_resumen_final = pd.DataFrame(columns=cols_resumen)
                if 'Tiempo Comida' in df_dieta.columns:
                    grupos = df_dieta.groupby('Tiempo Comida', observed=True)
                    for t in ["Desayuno", "Colación Mañana", "Almuerzo", "Colación Tarde", "Cena", "Colación Noche"]:
                        if t in grupos.groups:
                            df_resumen_final.loc[t] = grupos.get_group(t)[cols_resumen].sum()
                st.dataframe(df_resumen_final.style.format("{:.1f}"), use_container_width=True)

            st.subheader("📥 Descargar Informe")
            c_pdf, c_xls = st.columns(2)
            with c_xls:
                st.download_button("Descargar Excel", generar_excel_dieta(df_dieta, df_resumen_final, df_macros_export), f"dieta_{pa.get('nombre')}.xlsx", use_container_width=True)
            with c_pdf:
                st.download_button("Descargar PDF", generar_pdf_dieta_detallada(pa, df_dieta, df_macros_export, df_resumen_final, total_kcal), f"dieta_{pa.get('nombre')}.pdf", use_container_width=True, type='primary')

    # ========================================================
    # PESTAÑA 2: RESUMEN SEMANAL
    # ========================================================
    with tab_semanal:
        st.markdown("## 🗓️ Menú Semanal")
        plan = pa.get('plan_semanal', {})
        hay_datos = False
        if plan:
            for d in plan.values():
                if any(v.strip() for v in d.values()): hay_datos = True; break
        
        if not hay_datos:
            st.info("⚠️ No hay plan semanal. Ve a 'Crear Dieta' > 'Plan Semanal' para escribirlo.")
        else:
            dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
            tiempos = ["Desayuno", "Colación Mañana", "Almuerzo", "Colación Tarde", "Cena", "Colación Noche"]
            cols = st.columns(3)
            for i, dia in enumerate(dias):
                with cols[i % 3]:
                    with st.container(border=True):
                        st.markdown(f"<div style='text-align:center; color:#444; font-weight:bold; font-size:1.1em'>{dia}</div>", unsafe_allow_html=True)
                        st.markdown("---")
                        d_data = plan.get(dia, {})
                        vacio = True
                        for t in tiempos:
                            prep = d_data.get(t, "").strip()
                            if prep:
                                vacio = False
                                emoji = "☕" if "Desayuno" in t else "🍎" if "Mañana" in t else "🥗" if "Almuerzo" in t else "🥜" if "Tarde" in t else "🌙" if "Cena" in t else "🥛"
                                st.markdown(f"**{emoji} {t}**")
                                st.caption(prep)
                                st.write("")
                        if vacio:
                            st.markdown("<div style='text-align:center; color:#ccc'><i>Libre</i></div>", unsafe_allow_html=True)

            st.divider()
            st.subheader("Exportar Menú Semanal")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("📄 PDF Menú", generar_pdf_plan_semanal(pa, plan), f"menu_{pa.get('nombre')}.pdf", use_container_width=True, type='primary')
            with c2:
                st.download_button("📥 Excel Menú", generar_excel_plan_semanal(plan), f"menu_{pa.get('nombre')}.xlsx", use_container_width=True)

# --- NUEVA PÁGINA: PANEL DE ADMINISTRADOR ---
def mostrar_pagina_admin():
    """Página para gestionar usuarios (solo visible para 'admin')."""
    st.title("Panel de Administración 👑")
    
    usuarios = cargar_usuarios()
    
    st.subheader("Usuarios Existentes")
    usuarios_display = []
    for user, data in usuarios.items():
        usuarios_display.append({"usuario": user, "rol": data.get("rol", "usuario")})
    
    st.dataframe(pd.DataFrame(usuarios_display), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
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

    with col2:
        st.subheader("Eliminar Usuario")
        # Se puede eliminar a cualquiera EXCEPTO al 'admin' principal
        opciones_eliminar = [user for user in usuarios.keys() if user != 'admin']
        
        if not opciones_eliminar:
            st.info("No hay otros usuarios para eliminar.")
        else:
            usuario_a_eliminar = st.selectbox("Seleccionar Usuario a Eliminar", options=opciones_eliminar, index=None, placeholder="Seleccione un usuario...")
            
            if st.button("Eliminar Usuario", type="primary"):
                if not usuario_a_eliminar:
                    st.warning("Por favor, seleccione un usuario para eliminar.")
                elif usuario_a_eliminar in usuarios:
                    del usuarios[usuario_a_eliminar]
                    guardar_usuarios(usuarios)
                    st.success(f"Usuario '{usuario_a_eliminar}' eliminado.")
                    st.rerun()
                else:
                    st.error("El usuario seleccionado no existe.")
    
    st.divider()

    # --- INICIO: NUEVA SECCIÓN PARA CAMBIAR CONTRASEÑA ---
    st.subheader("Cambiar Contraseña de Usuario")
    
    # Obtenemos la lista de todos los usuarios
    todos_los_usuarios = list(usuarios.keys())
    
    if not todos_los_usuarios:
        st.info("No hay usuarios creados.")
    else:
        with st.form("form_cambiar_password", clear_on_submit=True):
            
            usuario_a_modificar = st.selectbox(
                "Seleccionar Usuario", 
                options=todos_los_usuarios,
                index=None,
                placeholder="Elija un usuario..."
            )
            
            nueva_password_admin = st.text_input(
                "Nueva Contraseña", 
                type="password",
                key="nueva_pass_admin_input" # Clave única para el widget
            )
            
            submit_cambiar = st.form_submit_button("Forzar Cambio de Contraseña")

            if submit_cambiar:
                if not usuario_a_modificar:
                    st.error("Por favor, seleccione un usuario.")
                elif not nueva_password_admin:
                    st.error("Por favor, ingrese una nueva contraseña.")
                else:
                    # Cargamos los usuarios de nuevo por si acaso
                    usuarios_actualizados = cargar_usuarios()
                    
                    # Hasheamos la nueva contraseña
                    nuevo_hash = hash_password(nueva_password_admin)
                    
                    # Actualizamos el diccionario y guardamos
                    usuarios_actualizados[usuario_a_modificar]['password'] = nuevo_hash
                    guardar_usuarios(usuarios_actualizados)
                    
                    st.success(f"¡Contraseña del usuario '{usuario_a_modificar}' actualizada exitosamente!")
    # --- FIN: NUEVA SECCIÓN ---

# --- NUEVA PÁGINA: INICIO DE SESIÓN ---
def mostrar_pagina_login():
    """Muestra la página de inicio de sesión centrada."""
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
            st.title("Bienvenido")
        else:
            st.title("ComVida 🍋‍🟩")
            st.caption("Bienvenido a su asistente nutricional")

        st.markdown("---")
        
        with st.form("login_form"):
            usuario = st.text_input("Usuario")
            password = st.text_input("Contraseña", type="password")
            submitted = st.form_submit_button("Ingresar")

            if submitted:
                usuarios = cargar_usuarios()
                
                if usuario in usuarios and check_password(password, usuarios[usuario]['password']):
                    st.session_state.autenticado = True
                    st.session_state.usuario = usuario
                    st.session_state.rol = usuarios[usuario].get('rol', 'usuario') 
                    st.success("Inicio de sesión exitoso. Redirigiendo...")
                    st.rerun()
                else:
                    st.error("Usuario o contraseña incorrectos.")
                    st.session_state.autenticado = False


# --- Lógica Principal (Main App Router) ---
# --- Lógica Principal (Main App Router) ---
def mostrar_app_principal():
    """Muestra la aplicación principal (barra lateral y páginas) después de iniciar sesión."""
    
    if os.path.exists(LOGO_PATH):
        st.sidebar.image(LOGO_PATH, use_container_width=True)
    else:
        st.sidebar.image("https://placehold.co/400x100/007bff/FFFFFF?text=ComVida&font=inter", use_container_width=True)
        st.sidebar.caption("Reemplaza esta imagen creando un archivo 'logo.png'.")

    st.sidebar.title("Navegación Principal")
    
    # --- ORDEN MODIFICADO AQUÍ ---
    menu = {
        "🏠 Inicio": mostrar_pagina_inicio,
        "📐 Antropometría": mostrar_pagina_antropometria,
        "🍲 Crear Dieta": mostrar_pagina_crear_dieta,
        "📊 Resumen de Dieta": mostrar_pagina_resumen_dieta,
        "🤖 Asistente de IA": mostrar_pagina_asistente_ia
    }
    # --- FIN DE LA MODIFICACIÓN ---
    
    if st.session_state.rol == 'admin':
        menu["👑 Administrador"] = mostrar_pagina_admin

    def set_pagina(pagina):
        st.session_state.pagina_activa = pagina

    for pagina_nombre in menu.keys():
        tipo_boton = "primary" if st.session_state.pagina_activa == pagina_nombre else "secondary"
        
        st.sidebar.button(
            pagina_nombre, 
            on_click=set_pagina, 
            args=(pagina_nombre,), 
            use_container_width=True,
            type=tipo_boton
        )
    
    st.sidebar.divider()
    
    if st.session_state.paciente_actual:
        pa = st.session_state.paciente_actual
        st.sidebar.subheader("Paciente Activo")
        st.sidebar.markdown(
            f"**Nombre:** {pa.get('nombre', 'N/A')}\n\n"
            f"**Edad:** {pa.get('edad', 0)} años\n\n"
            f"**Peso:** {pa.get('peso', 0):.1f} kg\n\n"
            f"**IMC:** {pa.get('imc', 0):.2f}\n\n"
            f"**GET:** {pa.get('get', 0):.0f} kcal (*{pa.get('formula_get', 'N/A')}*)"
        )
    else:
        st.sidebar.info("No hay ningún paciente cargado.")

    st.sidebar.divider()
    
    st.sidebar.info(f"Usuario: {st.session_state.usuario} ({st.session_state.rol})")
    if st.sidebar.button("Cerrar Sesión", use_container_width=True, type="primary"):
        st.session_state.autenticado = False
        st.session_state.usuario = None
        st.session_state.rol = None
        st.session_state.pagina_activa = "🏠 Inicio" 
        st.session_state.paciente_actual = None 
        st.session_state.dieta_temporal = []
        # Limpiar estados de IA
        if 'respuesta_plan_ia' in st.session_state:
            del st.session_state['respuesta_plan_ia']
        if 'respuesta_receta_ia' in st.session_state:
            del st.session_state['respuesta_receta_ia']
        st.rerun()

    # Mostrar la página activa
    pagina_a_mostrar = menu[st.session_state.pagina_activa]
    pagina_a_mostrar()


def main():
    # Inicializar Session State
    if 'autenticado' not in st.session_state:
        st.session_state.autenticado = False
        st.session_state.usuario = None
        st.session_state.rol = None
    if 'paciente_actual' not in st.session_state:
        st.session_state.paciente_actual = None
    if 'db_alimentos' not in st.session_state:
        st.session_state.db_alimentos = None
    if 'dieta_temporal' not in st.session_state:
        st.session_state.dieta_temporal = []
    if 'pagina_activa' not in st.session_state:
        st.session_state.pagina_activa = "🏠 Inicio"
        
    inicializar_pacientes()

    if not st.session_state.autenticado:
        mostrar_pagina_login()
    else:
        mostrar_app_principal()

    st.divider() 
    st.caption("© 2025 - Creado por IDLB. Todos los derechos reservados.") 

if __name__ == "__main__":
    main()