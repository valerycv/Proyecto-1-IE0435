import os
import pandas as pd
import numpy as np
 
  
CARPETA_COMPANEROS = "datasets_grupo"   
MI_DATASET         = "dataset.csv"         
SALIDA             = "dataset_grupo.csv"    

# funciones
 
def cargar_y_normalizar(ruta_csv, etiquetas_manuales=None):
   
    nombre = os.path.basename(ruta_csv)
    try:
        df = pd.read_csv(ruta_csv)
    except Exception as e:
        return None, f"No se pudo leer: {e}"
 
    n_filas, n_cols = df.shape
 
    # Caso 1: tiene 16385 columnas (pixeles + etiqueta) ---------------
    if n_cols == 16385:
        # Buscar columna de etiqueta (debe ser la ultima)
        ultima = df.columns[-1]
        # Validar que la ultima columna sean solo 0 y 1
        valores_unicos = set(df[ultima].unique())
        if not valores_unicos.issubset({0, 1}):
            return None, f"La ultima columna tiene valores raros: {valores_unicos}"
 
        # Renombrar columnas a formato estandar
        nuevas_columnas = [f"p{i}" for i in range(16384)] + ["etiqueta"]
        df.columns = nuevas_columnas
        return df, "OK (con etiqueta)"
 
    # Caso 2: tiene 16384 columnas (solo pixeles, sin etiqueta) -------
    if n_cols == 16384:
        if etiquetas_manuales is None:
            return None, (
                f"Tiene {n_cols} columnas pero NO tiene columna de etiqueta. "
                f"Aniade el orden a ETIQUETAS_CONOCIDAS en el script."
            )
        if len(etiquetas_manuales) != n_filas:
            return None, (
                f"Tiene {n_filas} filas pero las etiquetas manuales son "
                f"{len(etiquetas_manuales)}. No coinciden."
            )
        # Renombrar columnas y aniadir etiqueta
        df.columns = [f"p{i}" for i in range(16384)]
        df["etiqueta"] = etiquetas_manuales
        return df, "OK (etiquetas asignadas manualmente)"
 
    # Caso 3: numero de columnas no esperado --------------------------
    return None, f"Numero de columnas inesperado: {n_cols} (se esperaba 16384 o 16385)"
 
 
def main():
    print("=" * 70)
    print("COMBINANDO DATASETS DEL GRUPO")
    print("=" * 70)
 
    dfs        = []
    aceptados  = []
    rechazados = []
 
    if os.path.exists(MI_DATASET):
        df_mio, msg = cargar_y_normalizar(MI_DATASET)
        if df_mio is not None:
            dfs.append(df_mio)
            aceptados.append((MI_DATASET, len(df_mio), msg))
            print(f"  OK  {MI_DATASET:40s} {len(df_mio):3d} filas  ({msg})")
        else:
            rechazados.append((MI_DATASET, msg))
            print(f"  XX  {MI_DATASET:40s} RECHAZADO: {msg}")
    else:
        print(f"  !!  No se encontro tu dataset propio: {MI_DATASET}")
 
    if not os.path.isdir(CARPETA_COMPANEROS):
        print(f"\n  !!  No existe la carpeta '{CARPETA_COMPANEROS}'.")
        print(f"      Creala y mete ahi los CSV de tus companeros.")
    else:
        archivos = sorted([
            f for f in os.listdir(CARPETA_COMPANEROS)
            if f.lower().endswith(".csv")
        ])
        print(f"\nArchivos en '{CARPETA_COMPANEROS}/': {len(archivos)}")
        for nombre in archivos:
            ruta = os.path.join(CARPETA_COMPANEROS, nombre)
            etiq = ETIQUETAS_CONOCIDAS.get(nombre)
            df, msg = cargar_y_normalizar(ruta, etiquetas_manuales=etiq)
            if df is not None:
                dfs.append(df)
                aceptados.append((nombre, len(df), msg))
                print(f"  OK  {nombre:40s} {len(df):3d} filas  ({msg})")
            else:
                rechazados.append((nombre, msg))
                print(f"  XX  {nombre:40s} RECHAZADO: {msg}")
 
    if not dfs:
        print("\n[ERROR] No hay datasets validos para combinar.")
        return
 
    df_total = pd.concat(dfs, ignore_index=True)
    df_total.to_csv(SALIDA, index=False)
 
    n_pos = int((df_total["etiqueta"] == 1).sum())
    n_neg = int((df_total["etiqueta"] == 0).sum())
 
    print()
    print("=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"  Archivos aceptados : {len(aceptados)}")
    print(f"  Archivos rechazados: {len(rechazados)}")
    print(f"  Total de filas     : {len(df_total)}  (positivos={n_pos}, negativos={n_neg})")
    print(f"  Total de columnas  : {df_total.shape[1]}  (16,384 pixeles + 1 etiqueta)")
    print(f"  Archivo generado   : {SALIDA}")
    print()
 
    if rechazados:
        print("Archivos rechazados (revisa estos):")
        for nombre, motivo in rechazados:
            print(f"  - {nombre}: {motivo}")
        print()
 
 
if __name__ == "__main__":
    main()
 