import matplotlib.pyplot as plt
import numpy as np

def importar_polar_arta(filepath):
    """
    Lee un archivo de datos polares exportado desde ARTA (.txt) donde 
    las frecuencias y los datos de cada ángulo se encuentran en líneas completas únicas.
    """
    frecuencias = []
    angulos = []
    filas_db = []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lineas = [linea.strip() for linea in f.readlines() if linea.strip()]
    
    # 1. Extraer la línea de frecuencias (empieza con 'deg/freq')
    idx_header = -1
    for i, linea in enumerate(lineas):
        if 'deg/freq' in linea:
            idx_header = i
            break
            
    if idx_header == -1:
        raise ValueError("No se encontró la cabecera 'deg/freq' en el archivo de ARTA.")
    
    # Los tokens de frecuencia están todos en esa misma línea después de 'deg/freq'
    tokens_freq = lineas[idx_header].replace('deg/freq', '').split()
    frecuencias = np.array([float(x) for x in tokens_freq])
    num_freqs = len(frecuencias)
    
    # 2. Procesar las líneas siguientes (cada una es un ángulo con todos sus valores en dB)
    for i in range(idx_header + 1, len(lineas)):
        tokens = lineas[i].split()
        if not tokens:
            continue
            
        try:
            angulo = float(tokens[0])
        except ValueError:
            continue  # Por si hay alguna línea que no sea un número válido al inicio
            
        valores_angulo = [float(x) for x in tokens[1:]]
        
        if len(valores_angulo) == num_freqs:
            angulos.append(angulo)
            filas_db.append(valores_angulo)
            
    angulos = np.array(angulos)
    matriz_db = np.array(filas_db)
    
    return frecuencias, angulos, matriz_db


def comparar_patron_polar_arta(frecuencias, angulos, matriz_db, lista_f_centro, title=None):
    """
    Grafica MÚLTIPLES patrones polares divididos en el mismo gráfico desde datos de ARTA.
    - Lado Derecho (Der): Frecuencias bajas/medias (0° a 180° graficados a la derecha).
    - Lado Izquierdo (Izq): Frecuencias altas (0° a 180° espejados a la izquierda).
    - Normalizado a 0 dB respecto a 0°.
    - Escala radial extendida de -25 dB a +10 dB.
    """
    frecuencias = np.asarray(frecuencias)
    angulos_arr = np.asarray(angulos)
    
    # Tomar valor absoluto para trabajar en el rango [0°, 180°]
    angulos_abs = np.abs(angulos_arr)
    
    # Filtrar puntos en el rango 0° a 180°
    mascara_0_180 = (angulos_abs >= 0) & (angulos_abs <= 180)
    angulos_base = angulos_abs[mascara_0_180]
    matriz_db_filtrada = matriz_db[mascara_0_180, :]
    
    # Índice de 0° para la normalización
    idx_0deg = np.argmin(angulos_base)

    # Inicializar gráfico polar
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    
    # Convención acústica (0° arriba, sentido horario)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    mitad = len(lista_f_centro) // 2
    
    # --- BUCLE PRINCIPAL POR FRECUENCIA ---
    for idx_bucle, f_centro in enumerate(lista_f_centro):
        
        # Encontrar la frecuencia más cercana en el dataset
        idx_f = np.argmin(np.abs(frecuencias - f_centro))
        f_real = frecuencias[idx_f]
        
        # Extraer vector de dB para [0..180]°
        valores_absolutos = matriz_db_filtrada[:, idx_f]
        
        # Normalización a 0 dB respecto al valor en 0°
        valor_ref_0deg = valores_absolutos[idx_0deg]
        valores_medidos = valores_absolutos - valor_ref_0deg
        
        # --- ASIGNACIÓN DE HEMISFERIO (0° a 180°) ---
        if idx_bucle < mitad:
            # Lado Derecho: Ángulos de 0° a 180°
            angulos_grafico = angulos_base
            lado_str = "Der"
        else:
            # Lado Izquierdo: Mapear a 360° - ángulo
            angulos_grafico = 360 - angulos_base
            lado_str = "Izq"
            
        # Ordenar vectores para trazo continuo
        indices_ordenados = np.argsort(angulos_grafico)
        angulos_ordenados = angulos_grafico[indices_ordenados]
        valores_ordenados = valores_medidos[indices_ordenados]
        
        # Recortar valores entre el límite inferior (-25 dB) e superior (+10 dB)
        valores_ordenados = np.clip(valores_ordenados, -25, 5)
        angulos_rad = np.radians(angulos_ordenados)
        
        # Etiqueta para la leyenda
        label_f = f"{int(f_real)} Hz ({lado_str})" if f_real < 1000 else f"{f_real/1000:.1f} kHz ({lado_str})".replace(".0", "")
        
        # Graficar el semicírculo
        ax.plot(angulos_rad, valores_ordenados, linewidth=2, label=label_f, zorder=3)

    # --- CONFIGURACIÓN ESCALA RADIAL (dB) EXTENDIDA A +10 dB ---
    bottom_limit = -25 
    top_limit = 5
    ticks_radial = [-25, -20, -15, -10, -5, 0, 5]
    
    ax.set_rlim(bottom=bottom_limit, top=top_limit)
    ax.set_rticks(ticks_radial)
    
    labels_radial = [f"{t:+d} dB" if t != 0 else "0 dB" for t in ticks_radial]
    ax.set_yticklabels(labels_radial, fontsize=8.5, fontweight='bold', color='dimgray')
    ax.set_rlabel_position(15)  # Rota la posición de las etiquetas de dB a 70 grados
    
    # --- CONFIGURACIÓN DE ETIQUETAS ANGULARES SIMÉTRICAS ---
    ticks_angulares = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    ax.set_xticks(np.radians(ticks_angulares))
    
    labels_angulares = [f"{a}°" if a <= 180 else f"{360 - a}°" for a in ticks_angulares]
    ax.set_xticklabels(labels_angulares, fontsize=9.5)
    
    ax.grid(True, ls='-', alpha=0.15, color='gray', zorder=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1.0), fontsize=10, framealpha=0.9)
    
    if title is None:
        title = "Patrón Polar Comparativo (ARTA)\n(Bajas a la Derecha | Altas a la Izquierda)"
    ax.set_title(title, fontsize=12, pad=25, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    
def comparar_db(frecuencias, lista_db, labels=None, title="Comparación de Respuestas en Frecuencia", ylabel="Nivel (dB)"):
    """
    Grafica MÚLTIPLES curvas de dB en el mismo gráfico para compararlas.
    - Eje X: Logarítmico con ticks estándar de audio (20Hz a 20kHz).
    - Eje Y: Lineal y con etiqueta personalizable.
    """
    plt.figure(figsize=(12, 6.5))
    
    if labels is None:
        labels = [f"Medición {i+1}" for i in range(len(lista_db))]
    elif len(labels) != len(lista_db):
        print("Advertencia: La cantidad de labels no coincide con la cantidad de curvas. Usando nombres automáticos.")
        labels = [f"Medición {i+1}" for i in range(len(lista_db))]

    # Graficado de todas las curvas
    for db_curva, label_curva in zip(lista_db, labels):
        plt.semilogx(frecuencias, db_curva, linewidth=2, label=label_curva, zorder=3)
            
    # --- CONFIGURACIÓN DEL EJE X ---
    ticks_audio = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    labels_audio = ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k']
    plt.xlim(20, 20000)
    plt.xticks(ticks_audio, labels_audio)
    plt.xlabel('Frecuencia (Hz)', fontsize=11)
    
    # --- CONFIGURACIÓN DEL EJE Y ---
    plt.ylabel(ylabel, fontsize=11) # <--- Ahora es una variable editable
    
    plt.title(title, fontsize=13, pad=15)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10, framealpha=0.9)
    plt.grid(True, which="both", ls="-", alpha=0.15, color='gray', zorder=0)
    
    plt.tight_layout()
    plt.show()
    
    
# --- 3. EJEMPLO DE EJECUCIÓN ---
if __name__ == "__main__":
    # Ruta a tu archivo exportado de ARTA
    archivo = "MEDICIONES DIRECTIVIDAD\HF hermanito 5g\polar3 hermanito 5g.txt"    
    # Importar los datos
    frecuencias, angulos, matriz_db = importar_polar_arta(archivo)
    
    # Definir las frecuencias de interés a comparar (ejemplo: 2 bandas bajas/medias y 2 altas)
    frecuencias_deseadas = [1000, 2000, 4000, 8000, 16000]
    
    # Ejecutar la función de ploteo
    comparar_patron_polar_arta(
        frecuencias=frecuencias,
        angulos=angulos,
        matriz_db=matriz_db,
        lista_f_centro=frecuencias_deseadas,
        title="HERMANITO 5g"
    )
    
if __name__ == "__main__":
    # Ruta a tu archivo exportado de ARTA
    archivo = "MEDICIONES DIRECTIVIDAD\HF hermanito 15g\polar3 hermanito 15g.txt"    
    # Importar los datos
    frecuencias, angulos, matriz_db = importar_polar_arta(archivo)
    
    # Definir las frecuencias de interés a comparar (ejemplo: 2 bandas bajas/medias y 2 altas)
    frecuencias_deseadas = [1000, 2000, 4000, 8000, 16000]
    
    # Ejecutar la función de ploteo
    comparar_patron_polar_arta(
        frecuencias=frecuencias,
        angulos=angulos,
        matriz_db=matriz_db,
        lista_f_centro=frecuencias_deseadas,
        title="HERMANITO 15g"
    )
    
    
if __name__ == "__main__":
    # Ruta a tu archivo exportado de ARTA
    archivo = "MEDICIONES DIRECTIVIDAD\DIRECTIVIDAD LM C7 5g\polar3 c7 5g.txt"
    # Importar los datos
    frecuencias, angulos, matriz_db = importar_polar_arta(archivo)
    
    # Definir las frecuencias de interés a comparar (ejemplo: 2 bandas bajas/medias y 2 altas)
    frecuencias_deseadas = [500, 1000, 2000, 4000, 8000]
    
    # Ejecutar la función de ploteo
    comparar_patron_polar_arta(
        frecuencias=frecuencias,
        angulos=angulos,
        matriz_db=matriz_db,
        lista_f_centro=frecuencias_deseadas,
        title="C7 5g"
    )
    
    
if __name__ == "__main__":
    # Ruta a tu archivo exportado de ARTA
    archivo = "MEDICIONES DIRECTIVIDAD\DIRECTIVIDAD LM C7 15g\polar3 c7 15g.txt"    
    # Importar los datos
    frecuencias, angulos, matriz_db = importar_polar_arta(archivo)
    
    # Definir las frecuencias de interés a comparar (ejemplo: 2 bandas bajas/medias y 2 altas)
    frecuencias_deseadas = [500, 1000, 2000, 4000, 8000]
    
    # Ejecutar la función de ploteo
    comparar_patron_polar_arta(
        frecuencias=frecuencias,
        angulos=angulos,
        matriz_db=matriz_db,
        lista_f_centro=frecuencias_deseadas,
        title="C7 15g"
    )
    

# --- SPINORAMA ---
# 1. Cargar datos del archivo ARTA
archivo = 'MEDICIONES DIRECTIVIDAD\DIRECTIVIDAD LM C7 15g\polar3 c7 15g.txt'  # Reemplazá por el nombre/ruta de tu archivo
frecuencias, angulos, matriz_db = importar_polar_arta(archivo)

# 2. Definir los ángulos que querés comparar
angulos_deseados = [0, 15, 30, 45]

# 3. Extraer las curvas correspondientes a esos ángulos
lista_db = []
labels = []

# Normalizamos los ángulos del archivo a [0, 360) o valor absoluto si hiciera falta
angulos_abs = np.abs(angulos)

for ang in angulos_deseados:
    # Buscar el índice del ángulo más cercano en las mediciones
    idx = np.argmin(np.abs(angulos_abs - ang))
    ang_real = angulos[idx]
    
    # Extraer la fila correspondiente a ese ángulo
    curva_db = matriz_db[idx, :]
    
    lista_db.append(curva_db)
    labels.append(f"{abs(int(ang_real))}°")

# 4. Graficar usando tu función comparar_db
comparar_db(
    frecuencias=frecuencias,
    lista_db=lista_db,
    labels=labels,
    title="Respuesta en Frecuencia Fuera de Eje (0°, 15°, 30°, 45°)",
    ylabel="Nivel (dB SPL)"
)