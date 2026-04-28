import os
import numpy as np
from PIL import Image
import csv

CARPETA_IMAGENES = r"C:\Users\Valery Carranza\Desktop\IA en IE\proyecto1\muestras-procesadas"    # Acceso a la carpeta donde están los .png
ARCHIVO_CSV      = "dataset.csv"       # nombre del CSV de salida
UMBRAL           = 128                 # píxeles >= umbral → 1 (blanco)
                                       # píxeles <  umbral → 0 (objeto)



def imagen_a_vector(ruta_imagen: str, umbral: int = 128) -> np.ndarray:
    
    img = Image.open(ruta_imagen).convert("L")   # asegura escala de grises

    # Verificación de tamaño
    if img.size != (128, 128):
        raise ValueError(
            f"La imagen '{ruta_imagen}' tiene tamaño {img.size}, "
            "se esperaba (128, 128). Revisa el preprocesamiento."
        )

    arr = np.array(img)                          # shape (128, 128), valores 0-255
    binario = (arr >= umbral).astype(int)        # 1=blanco, 0=objeto
    return binario.flatten()                     # vector de 16 384 elementos


def obtener_etiqueta(nombre_archivo: str) -> int:
    """
    Devuelve 1 si el nombre empieza con 'positivo', 0 si empieza con 'negativo'.
    Lanza un error si el nombre no coincide con ninguno.
    """
    nombre = nombre_archivo.lower()
    if nombre.startswith("positivo"):
        return 1
    elif nombre.startswith("negativo"):
        return 0
    else:
        raise ValueError(
            f"No se pudo determinar la clase del archivo '{nombre_archivo}'. "
            "El nombre debe comenzar con 'positivo' o 'negativo'."
        )


def main():
    # Listar archivos .png ordenados
    archivos = sorted([
        f for f in os.listdir(CARPETA_IMAGENES)
        if f.lower().endswith(".png")
    ])

    if not archivos:
        print(f"[ERROR] No se encontraron archivos .png en '{CARPETA_IMAGENES}'.")
        return

    print(f"Se encontraron {len(archivos)} imágenes. Procesando...\n")

    filas = []
    errores = []

    for nombre in archivos:
        ruta = os.path.join(CARPETA_IMAGENES, nombre)
        try:
            vector   = imagen_a_vector(ruta, UMBRAL)
            etiqueta = obtener_etiqueta(nombre)
            fila     = np.append(vector, etiqueta)   # 16 384 píxeles + 1 etiqueta
            filas.append(fila)
            print(f"  ✓ {nombre:30s}  etiqueta={etiqueta}  "
                  f"píxeles_objeto={int((vector == 0).sum())}")
        except Exception as e:
            errores.append(nombre)
            print(f"  x  {nombre}: {e}")

    if not filas:
        print("\n[ERROR] No se procesó ninguna imagen correctamente.")
        return

    #  Construir encabezado 
    n_pixeles  = 128 * 128
    encabezado = [f"p{i}" for i in range(n_pixeles)] + ["etiqueta"]

    # Guardar CSV 
    with open(ARCHIVO_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(encabezado)
        writer.writerows([fila.tolist() for fila in filas])

    etiquetas  = [int(fila[-1]) for fila in filas]
    n_pos      = etiquetas.count(1)
    n_neg      = etiquetas.count(0)
    n_cols     = len(encabezado)

    print(f"""

  CSV generado : {ARCHIVO_CSV}
  Filas        : {len(filas)}  (positivos={n_pos}, negativos={n_neg})
  Columnas     : {n_cols}  (16 384 píxeles + 1 etiqueta)
  Errores      : {len(errores)}

""")

    if errores:
        print("Archivos con error:")
        for e in errores:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
