# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 20:18:47 2026

@author: dell_
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
import pandas as pd

# --- FUNCIONES DE IMPORTACION ---
def importar_arta(ruta_archivo):
    # Omitimos las 6 líneas de encabezado
    datos = np.loadtxt(ruta_archivo, skiprows=6)
    
    # Ordenamos por la primera columna (Frecuencia)
    datos = datos[datos[:, 0].argsort()]
    
    # Retornamos Frecuencia, Magnitud y Fase
    return datos[:, 0], datos[:, 1], datos[:, 2]


def importar_basta(ruta_archivo):
    datos = np.loadtxt(ruta_archivo, skiprows=1)
    datos = datos[datos[:, 0].argsort()]
    return datos[:, 0], datos[:, 1], datos[:, 2]


def importar_datos_limp(archivo_path):
    """
    Lee el archivo CSV de LIMP, limpia el formato regional de decimales,
    ignora metadatos y devuelve 3 vectores numéricos limpios.
    """
    # Usamos decimal=',' por si LIMP exportó con comas europeas/latinas.
    df = pd.read_csv(archivo_path, comment='*', header=None, names=['freq', 'mag', 'phase'], decimal=',')
    
    # Forzamos la conversión a numérico por si quedó algún texto huérfano (los errores se vuelven NaN y se limpian)
    df['freq'] = pd.to_numeric(df['freq'], errors='coerce')
    df['mag'] = pd.to_numeric(df['mag'], errors='coerce')
    df['phase'] = pd.to_numeric(df['phase'], errors='coerce')
    
    # Eliminamos filas vacías o con errores si las hubiera
    df = df.dropna()
    
    # Convertimos a vectores numéricos puros
    frecuencias = df['freq'].to_numpy()
    magnitud_imp = df['mag'].to_numpy()
    fase_imp = df['phase'].to_numpy()
    
    return frecuencias, magnitud_imp, fase_imp


# --- FUNCIONES PARA USAR EN OTRAS FUNCIONES ---
def desacoplar_fase(frec, fase, umbral=180):
    frec = np.array(frec, dtype=float)
    fase = np.array(fase, dtype=float)
    saltos = np.abs(np.diff(fase)) > umbral
    indices_saltos = np.where(saltos)[0] + 1
    frec_proc = np.insert(frec, indices_saltos, np.nan)
    fase_proc = np.insert(fase, indices_saltos, np.nan)
    return frec_proc, fase_proc


def aplicar_formato_ejes(ax, es_inferior=True):
    ticks_audio = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    labels_audio = ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k']

    # Forzar ticks principales y sus etiquetas de texto
    ax.xaxis.set_major_locator(FixedLocator(ticks_audio))
    ax.xaxis.set_major_formatter(FixedFormatter(labels_audio))
    ax.xaxis.set_minor_formatter(FixedFormatter([]))

    ax.set_xlim(20, 20000)

    # Ahora SIEMPRE muestra los ticks y números en el eje X
    ax.tick_params(labelbottom=True)
    
    # Solo ponemos la etiqueta textual "Frecuencia [Hz]" en el gráfico inferior para no amontonar
    if es_inferior:
        ax.set_xlabel("Frecuencia [Hz]", fontsize=10)
    else:
        ax.set_xlabel("")
 
        
def detectar_resonancia(frecuencias, magnitudes):
    mascara_graves = frecuencias < 300
    f_res = None
    if any(mascara_graves):
        idx_max = magnitudes[mascara_graves].argmax()
        f_res = frecuencias[mascara_graves][idx_max]
        mag_res = magnitudes[mascara_graves][idx_max]
         
    return f_res, mag_res
 
    
# --- FUNCIONES DE GRAFICACIÓN ELÉCTRICA ---
def comparar_impedancias(frecuencias_1, mags_1, frecuencias_2, mags_2, 
                         title="Comparación de Impedancias", 
                         label_1="BASTA (Simulación)", label_2="LIMP (Medición)"):
    """
    Grafica DOS curvas de impedancia independientes con sus respectivas frecuencias.
    """
    plt.figure(figsize=(12, 6.5))
    
    # --- GRAFICADO DE AMBAS CURVAS (Cada una con su vector de frecuencia) ---
    plt.semilogx(frecuencias_1, mags_1, linewidth=2.5, color='darkblue', label=label_1, zorder=3)
    plt.semilogx(frecuencias_2, mags_2, linewidth=2.5, color='red', label=label_2, zorder=3)
    
    # --- CONFIGURACIÓN DEL EJE X (Frecuencias de Audio) ---
    ticks_audio = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    labels_audio = ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k']
    plt.xlim(20, 20000)
    plt.xticks(ticks_audio, labels_audio)
    plt.xlabel('Frecuencia (Hz)', fontsize=11)
    
    # --- CONFIGURACIÓN DEL EJE Y (Máximo 40 Ohms, saltos de a 4) ---
    limite_y = 40
    ticks_y = list(range(0, limite_y + 4, 4))
    labels_y = [f'{val} $\\Omega$' for val in ticks_y]
    
    plt.ylim(0, limite_y)
    plt.yticks(ticks_y, labels_y)
    plt.ylabel('Impedancia Magnitud (|Z|)', fontsize=11)

    plt.title(title, fontsize=13, pad=15)
    plt.legend(loc="upper right", fontsize=10, framealpha=0.9)
    plt.grid(True, which="both", ls="-", alpha=0.15, color='gray', zorder=0)
    
    plt.tight_layout()
    plt.show()
# --- FUNCIONES DE GRAFICACIÓN ACUSTICA ---
def graficar_respuesta_db_20k(frec, mag_db, fase=None, titulo="Respuesta en Frecuencia (20 Hz - 20 kHz)", tipo="relativo"):
    frec_arr = np.array(frec)
    mag_db_arr = np.array(mag_db)

    # Normalización según el tipo
    if tipo == "relativo":
        idx_200hz = np.argmin(np.abs(frec_arr - 200))
        mag_norm = mag_db_arr - mag_db_arr[idx_200hz]
        etiqueta_y = "Transferencia [dB]"

    elif tipo == "absoluto":
        mag_norm = mag_db_arr
        etiqueta_y = "Transferencia [dB V/V]"

    else:
        raise ValueError("El parámetro 'tipo' debe ser 'relativo' o 'absoluto'")

    tiene_fase = fase is not None and len(fase) > 0

    if tiene_fase:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6.5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 4.5))
        ax2 = None

    fig.suptitle(titulo, fontsize=12, fontweight="bold")

    # --- SUBPLOT MAGNITUD ---
    ax1.semilogx(frec_arr, mag_norm, label="Nivel de Presión Sonora (SPL)")
    ax1.set_ylabel(etiqueta_y)
    ax1.grid(True, which="both", linestyle="--", alpha=0.6)

    mascara_20k = (frec_arr >= 20) & (frec_arr <= 20000)
    
    if tipo == "absoluto":
        ax1.set_ylim(-60, 20)
        
    elif np.any(mascara_20k):
        mag_filtrada = mag_norm[mascara_20k]
        margen = (np.max(mag_filtrada) - np.min(mag_filtrada)) * 0.08
        ax1.set_ylim(np.min(mag_filtrada) - margen, np.max(mag_filtrada) + margen)

    # --- SUBPLOT FASE ---
    if tiene_fase:
        frec_fase, fase_proc = desacoplar_fase(frec, fase)
        ax2.semilogx(frec_fase, fase_proc, color="orange", label="Fase")
        ax2.set_ylabel("Fase [°]")
        ax2.grid(True, which="both", linestyle="--", alpha=0.6)
        ax2.set_ylim(-180, 180)
        ax2.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])

    # Formato de ejes según corresponda
    if tiene_fase:
        aplicar_formato_ejes(ax1, es_inferior=False)
        aplicar_formato_ejes(ax2, es_inferior=True)
    else:
        aplicar_formato_ejes(ax1, es_inferior=True)

    plt.tight_layout()
    plt.show()
    
    
def comparar_respuesta_db(listas_frec, listas_mag_db, listas_fase=None, etiquetas=None, titulo="Comparativa de Respuesta en Frecuencia y Fase", tipo="absoluto"):
    num_curvas = len(listas_frec)
    if etiquetas is None:
        etiquetas = [f"Curva {i+1}" for i in range(num_curvas)]

    # Verificamos si hay datos de fase disponibles
    tiene_fase = listas_fase is not None and len(listas_fase) == num_curvas and listas_fase[0] is not None

    if tiene_fase:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(9, 4.5))
        ax2 = None

    fig.suptitle(titulo, fontsize=12, fontweight="bold")

    colores = plt.cm.tab10(np.linspace(0, 1, max(num_curvas, 2)))
    y1_min, y1_max = float("inf"), float("-inf")

    for i in range(num_curvas):
        frec = np.array(listas_frec[i])
        mag_db = np.array(listas_mag_db[i])
        color = colores[i]
        label = etiquetas[i]

        # --- Lógica según tipo (absoluto vs relativo) ---
        if tipo == "relativo":
            idx_200hz = np.argmin(np.abs(frec - 200))
            mag_a_graficar = mag_db - mag_db[idx_200hz]
        else:
            mag_a_graficar = mag_db

        # Subplot Magnitud
        ax1.semilogx(frec, mag_a_graficar, color=color, label=label)

        # Subplot Fase (si está disponible)
        if tiene_fase:
            fase = np.array(listas_fase[i])
            frec_fase, fase_proc = desacoplar_fase(frec, fase)
            ax2.semilogx(frec_fase, fase_proc, color=color, linestyle="--", alpha=0.8, label=label)

        # Escala Y dinámica considerando la ventana de 20 Hz a 20 kHz
        mascara_res = (frec >= 20) & (frec <= 20000)
        if np.any(mascara_res):
            mag_filtrada = mag_a_graficar[mascara_res]
            y1_min = min(y1_min, np.min(mag_filtrada))
            y1_max = max(y1_max, np.max(mag_filtrada))

    # Etiqueta del eje Y según el modo elegido
    if tipo == "relativo":
        ax1.set_ylabel("Transferencia [dB]")
    else:
        ax1.set_ylabel("Transferencia [dB V/V]")

    ax1.grid(True, which="both", linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9)

    if y1_min < y1_max:
        margen = (y1_max - y1_min) * 0.08
        ax1.set_ylim(y1_min - margen, y1_max + margen)

    # Configuración de fase
    if tiene_fase:
        ax2.set_ylabel("Fase [°]")
        ax2.grid(True, which="both", linestyle="--", alpha=0.6)
        ax2.set_ylim(-180, 180)
        ax2.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
        aplicar_formato_ejes(ax1, es_inferior=False)
        aplicar_formato_ejes(ax2, es_inferior=True)
    else:
        aplicar_formato_ejes(ax1, es_inferior=True)

    plt.tight_layout()
    plt.show()
    

def diferencia_maxima_db(frec1, mag1_db, frec2, mag2_db, f_min=20, f_max=50, f_norm=200):
    """
    Calcula la mayor diferencia en dB entre dos respuestas en frecuencia en un rango dado.
    Aplica normalización a 0 dB en f_norm (por defecto 200 Hz).
    
    Parámetros:
    - frec1, mag1_db: Frecuencias y magnitudes en dB de la Curva 1.
    - frec2, mag2_db: Frecuencias y magnitudes en dB de la Curva 2.
    - f_min, f_max: Rango de búsqueda en Hz (ej. 20 y 50 Hz).
    - f_norm: Frecuencia de referencia para normalizar a 0 dB (ej. 200 Hz).
    
    Devuelve:
    - f_max_diff: Frecuencia (Hz) donde ocurre la mayor diferencia.
    - max_diff: Valor absoluto de la diferencia máxima en dB.
    """
    frec1 = np.array(frec1)
    mag1_db = np.array(mag1_db)
    frec2 = np.array(frec2)
    mag2_db = np.array(mag2_db)
    
    # 1. Normalización a 0 dB en 200 Hz para ambas curvas
    idx_norm1 = np.argmin(np.abs(frec1 - f_norm))
    idx_norm2 = np.argmin(np.abs(frec2 - f_norm))
    
    mag1_norm = mag1_db - mag1_db[idx_norm1]
    mag2_norm = mag2_db - mag2_db[idx_norm2]
    
    # 2. Filtrar la curva 1 dentro del rango [f_min, f_max]
    mask_rango = (frec1 >= f_min) & (frec1 <= f_max)
    
    if not np.any(mask_rango):
        raise ValueError(f"No hay puntos de la Curva 1 en el rango especificado ({f_min} Hz - {f_max} Hz).")
        
    frec1_eval = frec1[mask_rango]
    mag1_eval = mag1_norm[mask_rango]
    
    # 3. Interpolar la curva 2 sobre la grilla de frecuencias de la curva 1 en ese rango
    mag2_interp = np.interp(frec1_eval, frec2, mag2_norm)
    
    # 4. Diferencia punto a punto
    diferencias = np.abs(mag1_eval - mag2_interp)
    
    # 5. Buscar el máximo
    idx_max = np.argmax(diferencias)
    
    f_max_diff = frec1_eval[idx_max]
    max_diff = diferencias[idx_max]
    
    print("--- Análisis de Diferencia Máxima ---")
    print(f"Rango analizado: {f_min} Hz a {f_max} Hz")
    print(f"Máxima diferencia: {max_diff:.2f} dB en {f_max_diff:.2f} Hz")
    
    return f_max_diff, max_diff


# ----- IMPORTES ------
frec_port, mag_port, fas_port = importar_arta(r"stf\stf_port.txt")
frec_driver, mag_driver, fas_driver = importar_arta(r"stf\stf_driver.txt")
comparar_respuesta_db([frec_port, frec_driver], 
                      [mag_port, mag_driver], 
                      [fas_port, fas_driver],
                      etiquetas=["Tubo de sintonía", "Altoparlante"])

frec_far, mag_far, fas_far = importar_arta(r"stf\stf_far.txt")
graficar_respuesta_db_20k(frec_far, mag_far, fas_far, titulo="FAR", tipo="absoluto")

frec_port_escalado, mag_port_escalado, fas_port_escalado = importar_arta(r"stf\stf_port_escalado.txt")
frec_suma_near, mag_suma_near, fas_suma_near = importar_arta(r"stf\stf_suma_near.txt")

comparar_respuesta_db([frec_port, frec_port_escalado], 
                      [mag_port, mag_port_escalado], 
                      [fas_port, fas_port_escalado],
                      etiquetas=["Tubo de sintonía", "escalado"])

comparar_respuesta_db([frec_port_escalado, frec_driver, frec_suma_near], 
                      [mag_port_escalado, mag_driver, mag_suma_near], 
                      [fas_port_escalado, fas_driver, fas_suma_near],
                      etiquetas=["Tubo de sintonía (escalado)", "Altoparlante", "Suma en campo cercano"])

frec_suma_near_reescalado, mag_suma_near_reescalado, fas_suma_near_reescalado = importar_arta(r"stf\stf_suma_near_reescalado.txt")
comparar_respuesta_db([frec_far, frec_suma_near, frec_suma_near_reescalado], 
                      [mag_far, mag_suma_near, mag_suma_near_reescalado], 
                      [fas_far, fas_suma_near, fas_suma_near_reescalado],
                      etiquetas=["Campo lejano", "Campo cercano", "Campo cercano escalado"])

frec_suma_near_reescalado_BS, mag_suma_near_reescalado_BS, fas_suma_near_reescalado_BS = importar_arta(r"stf\stf_suma_near_reescalado_BS.txt")
comparar_respuesta_db([frec_suma_near_reescalado, frec_suma_near_reescalado_BS], 
                      [mag_suma_near_reescalado, mag_suma_near_reescalado_BS], 
                      [fas_suma_near_reescalado, fas_suma_near_reescalado_BS],
                      etiquetas=["Campo cercano escalado", "Corrección por bafle"])

comparar_respuesta_db([frec_far, frec_suma_near_reescalado_BS], 
                      [mag_far, mag_suma_near_reescalado_BS], 
                      [fas_far, fas_suma_near_reescalado_BS],
                      etiquetas=["Campo lejano", "Campo cercano escalado + bafle"])

frec_respuesta_final, mag_respuesta_final, fas_respuesta_final = importar_arta(r"stf\stf_respuesta_final_BIEN_HECHO.txt")
graficar_respuesta_db_20k(frec_respuesta_final, mag_respuesta_final, fas_respuesta_final, titulo="RESPUESTA FINAL", tipo="relativo")

frec_limp, mag_limp, fas_limp = importar_datos_limp(r"basta y limp\MEDICION IMPEDANCIA C7 VENTILADO.csv")
frec_imp_basta, mag_imp_basta, fas_imp_basta =  importar_basta(r"basta y limp\electrical_impedance.txt")
frec_system_response, mag_system_response, fas_system_response = importar_basta(r"basta y limp\system_response.txt")

comparar_respuesta_db([frec_system_response, frec_respuesta_final], 
                      [mag_system_response, mag_respuesta_final], 
                      [fas_system_response, fas_respuesta_final],
                      etiquetas=["Simulación en BASTA!", "Medición cuasianecóica"], tipo = "relativo")

comparar_impedancias(
    frecuencias_1=frec_imp_basta, mags_1=mag_imp_basta,
    frecuencias_2=frec_limp, mags_2=mag_limp,
    title="Comparación BASTA vs LIMP",
    label_1="Simulación BASTA",
    label_2="Medición LIMP")
    
freq_dif1, db_dif1 = diferencia_maxima_db(
    frec1=frec_respuesta_final, 
    mag1_db=mag_respuesta_final, 
    frec2=frec_system_response, 
    mag2_db=mag_system_response, 
    f_min=20, 
    f_max=30
)

freq_dif2, db_dif2 = diferencia_maxima_db(
    frec1=frec_respuesta_final, 
    mag1_db=mag_respuesta_final, 
    frec2=frec_system_response, 
    mag2_db=mag_system_response, 
    f_min=30, 
    f_max=50
)

freq_dif3, db_dif3 = diferencia_maxima_db(
    frec1=frec_respuesta_final, 
    mag1_db=mag_respuesta_final, 
    frec2=frec_system_response, 
    mag2_db=mag_system_response, 
    f_min=50, 
    f_max=300
)

freq_dif4, db_dif4 = diferencia_maxima_db(
    frec1=frec_respuesta_final, 
    mag1_db=mag_respuesta_final, 
    frec2=frec_system_response, 
    mag2_db=mag_system_response, 
    f_min=300, 
    f_max=1000
)
