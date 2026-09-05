import numpy as np
import matplotlib.pyplot as plt

def importar_basta(ruta_archivo):
    datos = np.loadtxt(ruta_archivo, skiprows=1)
    datos = datos[datos[:, 0].argsort()]
    return datos[:, 0], datos[:, 1], datos[:, 2]

def desacoplar_fase(frec, fase, umbral=180):
    frec = np.array(frec, dtype=float)
    fase = np.array(fase, dtype=float)
    saltos = np.abs(np.diff(fase)) > umbral
    indices_saltos = np.where(saltos)[0] + 1
    frec_proc = np.insert(frec, indices_saltos, np.nan)
    fase_proc = np.insert(fase, indices_saltos, np.nan)
    return frec_proc, fase_proc

def aplicar_formato_ejes(ax, es_inferior=True):
    ticks_audio = [20, 50, 100, 200, 500, 1000, 2000]
    labels_audio = ['20', '50', '100', '200', '500', '1k', '2k']
    
    ax.set_xlim(20, 2000)
    ax.set_xticks(ticks_audio)
    ax.set_xticklabels(labels_audio)
    ax.tick_params(labelbottom=True)  # Muestra ticks y labels en ambos cuadros
    ax.set_xlabel("Frecuencia [Hz]", fontsize=10)
    
    # Línea vertical y texto en posición relativa (70% de altura)
    ax.axvline(x=994, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(994, 0.70, " ka=1", color="gray", alpha=0.8, 
            verticalalignment="top", fontsize=9, transform=ax.get_xaxis_transform())

def graficar_impedancia(frec, mag, fase, titulo="Respuesta de Impedancia y Fase"):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(8, 6.5))
    fig.suptitle(titulo, fontsize=12, fontweight="bold")

    # --- SUBPLOT MAGNITUD ---
    ax1.semilogx(frec, mag, label="Respuesta en magnitud")
    ax1.set_ylabel(r"Impedancia [$\Omega$]")
    ax1.grid(True, which="both", linestyle="--", alpha=0.6)

    # --- SUBPLOT FASE ---
    frec_fase, fase_proc = desacoplar_fase(frec, fase)
    ax2.semilogx(frec_fase, fase_proc, label="Fase")
    ax2.set_ylabel("Fase [°]")
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)
    ax2.set_ylim(-180, 180)
    ax2.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])

    # b) Escala Y enfocada en zona de resonancia hasta ka=1 (994 Hz)
    frec_arr = np.array(frec)
    mag_arr = np.array(mag)
    mascara_res = (frec_arr >= 20) & (frec_arr <= 994)
    if np.any(mascara_res):
        mag_filtrada = mag_arr[mascara_res]
        margen = (np.max(mag_filtrada) - np.min(mag_filtrada)) * 0.08
        ax1.set_ylim(np.min(mag_filtrada) - margen, np.max(mag_filtrada) + margen)

    # a) Ticks, etiquetas y ka=1 en AMBOS gráficos
    aplicar_formato_ejes(ax1)
    aplicar_formato_ejes(ax2)

    plt.tight_layout()
    plt.show()

def graficar_respuesta_db(frec, mag_db, fase, titulo="Respuesta en Frecuencia y Fase"):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(8, 6.5))
    fig.suptitle(titulo, fontsize=12, fontweight="bold")

    # --- SUBPLOT MAGNITUD ---
    ax1.semilogx(frec, mag_db, label="Nivel de Presión Sonora (SPL)")
    ax1.set_ylabel("Magnitud [dB]")
    ax1.grid(True, which="both", linestyle="--", alpha=0.6)

    # --- SUBPLOT FASE ---
    frec_fase, fase_proc = desacoplar_fase(frec, fase)
    ax2.semilogx(frec_fase, fase_proc, color="orange", label="Fase")
    ax2.set_ylabel("Fase [°]")
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)
    ax2.set_ylim(-180, 180)
    ax2.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])

    # b) Escala Y enfocada en zona de interés hasta ka=1 (994 Hz)
    frec_arr = np.array(frec)
    mag_db_arr = np.array(mag_db)
    mascara_res = (frec_arr >= 20) & (frec_arr <= 994)
    if np.any(mascara_res):
        mag_filtrada = mag_db_arr[mascara_res]
        margen = (np.max(mag_filtrada) - np.min(mag_filtrada)) * 0.08
        ax1.set_ylim(np.min(mag_filtrada) - margen, np.max(mag_filtrada) + margen)

    # a) Ticks, etiquetas y ka=1 en AMBOS gráficos
    aplicar_formato_ejes(ax1)
    aplicar_formato_ejes(ax2)

    plt.tight_layout()
    plt.show()

def graficar_respuesta(frec, mag_db, fase, titulo="Máximo nivel de salida"):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(8, 6.5))
    fig.suptitle(titulo, fontsize=12, fontweight="bold")

    # --- SUBPLOT MAGNITUD ---
    ax1.semilogx(frec, mag_db, label="Nivel de Presión Sonora (SPL)", color="#1f77b4")
    ax1.set_ylabel("Magnitud [dB]")
    ax1.grid(True, which="both", linestyle="--", alpha=0.6)

    # --- SUBPLOT FASE ---
    frec_fase, fase_proc = desacoplar_fase(frec, fase)
    ax2.semilogx(frec_fase, fase_proc, color="orange", label="Fase")
    ax2.set_ylabel("Fase [°]")
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)
    ax2.set_ylim(-180, 180)
    ax2.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])

    # b) Escala Y enfocada en zona de interés hasta ka=1 (994 Hz)
    frec_arr = np.array(frec)
    mag_db_arr = np.array(mag_db)
    mascara_res = (frec_arr >= 20) & (frec_arr <= 994)
    if np.any(mascara_res):
        mag_filtrada = mag_db_arr[mascara_res]
        margen = (np.max(mag_filtrada) - np.min(mag_filtrada)) * 0.08
        ax1.set_ylim(np.min(mag_filtrada) - margen, np.max(mag_filtrada) + margen)

    # a) Ticks, etiquetas y ka=1 en AMBOS gráficos
    aplicar_formato_ejes(ax1)
    aplicar_formato_ejes(ax2)

    plt.tight_layout()
    plt.show()

def comparar_respuesta_db(listas_frec, listas_mag_db, listas_fase, etiquetas=None, titulo="Comparativa de Respuesta en Frecuencia y Fase"):
    num_curvas = len(listas_frec)
    if etiquetas is None:
        etiquetas = [f"Curva {i+1}" for i in range(num_curvas)]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(9, 7))
    fig.suptitle(titulo, fontsize=12, fontweight="bold")

    colores = plt.cm.tab10(np.linspace(0, 1, max(num_curvas, 2)))
    y1_min, y1_max = float("inf"), float("-inf")

    for i in range(num_curvas):
        frec = np.array(listas_frec[i])
        mag_db = np.array(listas_mag_db[i])
        fase = np.array(listas_fase[i])
        color = colores[i]
        label = etiquetas[i]

        ax1.semilogx(frec, mag_db, color=color, label=label)

        frec_fase, fase_proc = desacoplar_fase(frec, fase)
        ax2.semilogx(frec_fase, fase_proc, color=color, linestyle="--", alpha=0.8, label=label)

        # b) Escala Y ajustada al pico máximo dentro de la banda 20-994 Hz
        mascara_res = (frec >= 20) & (frec <= 994)
        if np.any(mascara_res):
            mag_filtrada = mag_db[mascara_res]
            y1_min = min(y1_min, np.min(mag_filtrada))
            y1_max = max(y1_max, np.max(mag_filtrada))

    ax1.set_ylabel("Magnitud [dB]")
    ax1.grid(True, which="both", linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9)

    if y1_min < y1_max:
        margen = (y1_max - y1_min) * 0.08
        ax1.set_ylim(y1_min - margen, y1_max + margen)

    ax2.set_ylabel("Fase [°]")
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)
    ax2.set_ylim(-180, 180)
    ax2.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])

    # a) Ticks, etiquetas y ka=1 en AMBOS gráficos
    aplicar_formato_ejes(ax1)
    aplicar_formato_ejes(ax2)

    plt.tight_layout()
    plt.show()
    
# ----- IMPORTS ------
frec_system_response, mag_system_response, fase_system_respones = importar_basta(r"ELECTRO II\TP 1\exports\system_response.txt")
frec_driver_response, mag_driver_response, fase_driver_respones = importar_basta(r"ELECTRO II\TP 1\exports\driver_response.txt")
frec_vent_response, mag_vent_response, fase_vent_respones = importar_basta(r"ELECTRO II\TP 1\exports\vent_response.txt")
frec_electrical_impedance, mag_electrical_impedance, fase_electrical_impedance = importar_basta(r"ELECTRO II\TP 1\exports\electrical_impedance.txt")
frec_system_response_CB, mag_system_response_CB, fase_system_respones_CB = importar_basta(r"ELECTRO II\TP 1\exports\system_response_CB.txt")
frec_system_response_MODIFICADO, mag_system_response_MODIFICADO, fase_system_respones_MODIFICADO = importar_basta(r"ELECTRO II\TP 1\exports\system_response_MODIFICADO.txt")
frec_max_level, mag_max_level, fase_max_level = importar_basta(r"ELECTRO II\TP 1\exports\level_MOL.txt")

"""graficar_impedancia(frec_electrical_impedance, mag_electrical_impedance, fase_electrical_impedance)
comparar_respuesta_db(
    listas_frec=[frec_system_response,frec_driver_response,frec_vent_response],
    listas_mag_db=[mag_system_response, mag_driver_response, mag_vent_response],
    listas_fase=[fase_system_respones,fase_driver_respones,fase_vent_respones],
    etiquetas=["Sistema", "Altoparlante", "Tubo de sintonía"],
    titulo="Comparativa Respuesta en frecuencia"
)

comparar_respuesta_db(
    listas_frec=[frec_system_response_CB,frec_system_response],
    listas_mag_db=[mag_system_response_CB, mag_system_response],
    listas_fase=[fase_system_respones_CB,fase_system_respones],
    etiquetas=["Gabinete cerrado", "Gabinete ventilado"],
    titulo="Comparativa Respuesta en frecuencia"
)

comparar_respuesta_db(
    listas_frec=[frec_system_response,frec_system_response_MODIFICADO],
    listas_mag_db=[mag_system_response, mag_system_response_MODIFICADO],
    listas_fase=[fase_system_respones,fase_system_respones_MODIFICADO],
    etiquetas=["Gabinete original", "Gabinete modificado"],
    titulo="Comparativa Respuesta en frecuencia"
)"""

# COSAS LULA FORMATO DISTINTO PERDON!!!
import numpy as np
import matplotlib.pyplot as plt

def aplicar_formato_ejes(ax):
    # Reemplaza esto con el contenido de tu función original
    pass

def graficar_mol_tres_curvas(frec, mag_db, excursion, vent_v, titulo="Análisis de Máximo Nivel de Salida (MOL)"):
    # Creamos 3 subplots compartiendo el eje X para comparar fácilmente
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
    fig.suptitle(titulo, fontsize=12, fontweight="bold")

    # --- SUBPLOT 1: MAGNITUD (SPL) ---
    ax1.semilogx(frec, mag_db, label="Nivel de Presión Sonora (SPL)", color="#1f77b4")
    ax1.set_ylabel("Magnitud [dB]")
    ax1.grid(True, which="both", linestyle="--", alpha=0.6)

    # Escala Y enfocada en zona de interés hasta ka=1 (994 Hz) para Magnitud
    frec_arr = np.array(frec)
    mag_db_arr = np.array(mag_db)
    mascara_res = (frec_arr >= 20) & (frec_arr <= 994)
    if np.any(mascara_res):
        mag_filtrada = mag_db_arr[mascara_res]
        margen = (np.max(mag_filtrada) - np.min(mag_filtrada)) * 0.08
        ax1.set_ylim(np.min(mag_filtrada) - margen, np.max(mag_filtrada) + margen)

    # --- SUBPLOT 2: EXCURSIÓN ---
    # Multiplicamos por 1000 para graficar en milímetros en lugar de metros
    excursion_mm = np.array(excursion) * 1000
    ax2.semilogx(frec, excursion_mm, color="#1f77b4")
    ax2.set_ylabel("Excursión [mm]")
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)

    # --- SUBPLOT 3: VELOCIDAD DEL PUERTO ---
    ax3.semilogx(frec, vent_v, color="#1f77b4")
    ax3.set_ylabel("Vel. Puerto [m/s]")
    ax3.set_xlabel("Frecuencia [Hz]")
    ax3.grid(True, which="both", linestyle="--", alpha=0.6)

    # Aplicar formato a los tres ejes
    aplicar_formato_ejes(ax1)
    aplicar_formato_ejes(ax2)
    aplicar_formato_ejes(ax3)

    plt.tight_layout()
    plt.show()

# --- CARGA DE DATOS Y EJECUCIÓN ---
# Utilizamos skiprows=1 para ignorar el encabezado de los archivos .txt
try:
    frec, excursion = np.loadtxt('excursion_MOL.txt', skiprows=1, unpack=True)
    _, mag_db = np.loadtxt('level_MOL.txt', skiprows=1, unpack=True)
    _, vent_v = np.loadtxt('vent_v_MOL.txt', skiprows=1, unpack=True)
    
    graficar_mol_tres_curvas(frec, mag_db, excursion, vent_v)
except FileNotFoundError as e:
    print(f"Error: Asegúrate de que los archivos .txt estén en la misma carpeta que este script. {e}")