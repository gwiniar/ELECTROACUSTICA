import numpy as np


def suavizado(f: np.ndarray, amp_db: np.ndarray, oct: int) -> np.ndarray:
    """

    Devuelve una array de amplitudes suavizada en una fracción de octava.

    Parameters
    ----------
    f : numpy.ndarray
        Frecuencias de menor a mayor
    amp_db : numpy.ndarray
        Amplitudes en dB
    oct: int
        Fracción o subdivisión de octava para suavizar
        (ej. OCT=3 -> Suavizado por tercios de octava)
        oct=0 es un caso especial donde no se realiza el suavizado
        y se devuelve la curva intacta

    Returns
    -------
    out : numpy.ndarray
        Amplitudes suavizadas

    """

    if oct == 0:
        return amp_db

    if len(f) != len(amp_db):
        raise Exception("f y amp_db deben ser del mismo largo.")

    ampsmooth_db = np.zeros(np.size(amp_db))
    for i, freq in enumerate(f):
        finf = freq / pow(2, 1 / (2 * oct))  # calcula el corte inferior del promedio
        fsup = freq * pow(2, 1 / (2 * oct))  # calcula el corte superior del promedio

        idx = np.logical_and(
            f >= finf, f <= fsup
        )  # busca los elementos dentro del rango de frecuencias

        # Suma energetica del rango de frecuencias
        amp = pow(10, amp_db[idx] / 10)
        ampsmooth_db[i] = 10 * np.log10(sum(amp) / len(amp))

    return ampsmooth_db