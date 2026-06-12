import numpy as np
import matplotlib.pyplot as plt

def graficar_respuesta_polar(N, b, lam):
    """
    Genera un gráfico polar de 360° para la función D(theta).
    
    Parámetros:
    N   : Número de elementos (entero)
    b   : Separación entre elementos
    lam : Longitud de onda (lambda)
    """
    
    # Crear un vector de ángulos theta de 0 a 2*pi (360 grados)
    # Usamos muchos puntos para que la curva sea suave
    theta = np.linspace(0, 2 * np.pi, 2000)
    
    # Calculamos el argumento interno del seno en el denominador
    # arg = (pi * b / lambda) * sin(theta)
    arg = (np.pi * b / lam) * np.sin(theta)
    
    # Suprimimos las advertencias de división por cero de numpy temporalmente
    with np.errstate(divide='ignore', invalid='ignore'):
        D = np.sin(N * arg) / (N * np.sin(arg))
        
        # En los puntos donde ocurre la división por cero (0/0 resultando en NaN),
        # el límite teórico de la magnitud de la función es 1.
        D[np.isnan(D)] = 1.0
        
    # Tomamos el valor absoluto para graficar la magnitud del patrón de radiación
    D_mag = np.abs(D)
    
    # Configuración del gráfico polar
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    
    # Orientación del gráfico (0 grados a la derecha, que es el estándar)
    # Puedes usar ax.set_theta_zero_location("N") si quieres el 0 arriba.
    
    ax.plot(theta, D_mag, color='blue', linewidth=1.5)
    
    # Configuración de estilo y etiquetas
    ax.set_title(rf"Patrón de Radiación\n$N={N}$, $b={b}$, $\lambda={lam}$", pad=20)
    ax.set_rmax(1.0) # El valor máximo normalizado siempre es 1
    ax.grid(True)
    
    # Mostrar el gráfico
    plt.show()

graficar_respuesta_polar(N=3, b=0.114, lam=0.02143)