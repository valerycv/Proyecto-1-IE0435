
# Proyecto 1 - IE0435 - Valery Carranza

import os
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


ARCHIVO_CSV   = "dataset_grupo.csv"
CARNE         = "B91660"
NOMBRE        = "valery"
APELLIDO      = "carranza"

SEMILLA       = 42                      # para reproducibilidad
N_FOLDS       = 5                       # folds de validacion cruzada
SCORING       = "f1"                    # metrica para optimizar 


# Se cargan y severifican los datos

def cargar_datos(ruta_csv: str):
    print(f"Cargando dataset desde: {ruta_csv}")
    df = pd.read_csv(ruta_csv)
    print(f"  Dimensiones: {df.shape}")
    print(f"  Distribucion de clases:")
    print(df["etiqueta"].value_counts().to_string())

    X = df.drop(columns=["etiqueta"]).values
    y = df["etiqueta"].values

    print(f"  X: {X.shape}, y: {y.shape}")
    print(f"  Valores unicos en X: {np.unique(X)}")
    print()
    return X, y


# Se definen los modelos

def definir_modelos_y_grids():
    """
    Cada entrada: nombre -> (estimador base, grid de hiperparametros)
    """
    modelos = {
        "Arbol de Decision": (
            DecisionTreeClassifier(random_state=SEMILLA),
            {
                "max_depth":         [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "criterion":         ["gini", "entropy"],
            }
        ),
        "Naive Bayes (Bernoulli)": (
            BernoulliNB(),
            {
                "alpha": [0.01, 0.1, 0.5, 1.0, 2.0],
            }
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "n_neighbors": [1, 3, 5, 7],
                "weights":     ["uniform", "distance"],
                "metric":      ["euclidean", "manhattan", "hamming"],
            }
        ),
        "SVM": (
            SVC(random_state=SEMILLA, probability=False),
            {
                "C":      [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
                "gamma":  ["scale", "auto"],
            }
        ),
    }
    return modelos


# Entrenamiento

def entrenar_y_evaluar(X, y):

    # Split adicional para una metrica de prueba "fuera" del grid
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEMILLA
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Distribucion en train: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Distribucion en test:  {dict(zip(*np.unique(y_test,  return_counts=True)))}")
    print()

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)
    modelos = definir_modelos_y_grids()
    resultados = []

    for nombre, (estimador, grid) in modelos.items():
       # print("=" * 70)
        print(f"Entrenando: {nombre}")
       # print("=" * 70)
        t0 = time.time()

        gs = GridSearchCV(
            estimator   = estimador,
            param_grid  = grid,
            scoring     = SCORING,
            cv          = cv,
            n_jobs      = -1,
            refit       = True,
        )
        gs.fit(X_train, y_train)
        t = time.time() - t0

        # Evaluacion en el conjunto de prueba
        y_pred  = gs.predict(X_test)
        acc     = accuracy_score (y_test, y_pred)
        prec    = precision_score(y_test, y_pred, zero_division=0)
        rec     = recall_score   (y_test, y_pred, zero_division=0)
        f1      = f1_score       (y_test, y_pred, zero_division=0)

        print(f"  Mejor CV {SCORING}: {gs.best_score_:.4f}")
        print(f"  Mejores hiperparametros: {gs.best_params_}")
        print(f"  --- Metricas en test (25%) ---")
        print(f"    Accuracy : {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall   : {rec:.4f}")
        print(f"    F1       : {f1:.4f}")
        print(f"  Matriz de confusion (test):")
        print(confusion_matrix(y_test, y_pred))
        print(f"  Tiempo: {t:.2f}s\n")

        resultados.append({
            "modelo":         nombre,
            "mejor_estimador": gs.best_estimator_,
            "mejores_params": gs.best_params_,
            "cv_score":       gs.best_score_,
            "test_accuracy":  acc,
            "test_precision": prec,
            "test_recall":    rec,
            "test_f1":        f1,
            "tiempo_s":       t,
        })

    return resultados, (X_test, y_test)


# Se exportan los resultados

def seleccionar_mejor(resultados):
    """
    Criterio principal: F1 en CV (mas robusto que test al ser dataset pequeno).
    Desempate: F1 en test.
    """
    ordenados = sorted(
        resultados,
        key=lambda r: (r["cv_score"], r["test_f1"]),
        reverse=True
    )
    return ordenados[0], ordenados


def imprimir_resumen(resultados_ordenados):
    # print("=" * 70)
    print("RESUMEN COMPARATIVO (ordenado por CV F1)")
    #print("=" * 70)
    df = pd.DataFrame([
        {
            "Modelo":     r["modelo"],
            "CV F1":      round(r["cv_score"],       4),
            "Test Acc":   round(r["test_accuracy"],  4),
            "Test Prec":  round(r["test_precision"], 4),
            "Test Rec":   round(r["test_recall"],    4),
            "Test F1":    round(r["test_f1"],        4),
            "Tiempo (s)": round(r["tiempo_s"],       2),
        }
        for r in resultados_ordenados
    ])
    print(df.to_string(index=False))
    print()

def exportar_mejor(mejor, carne, nombre, apellido):
    nombre_archivo = f"{carne}_{nombre}_{apellido}.joblib"
    joblib.dump(mejor["mejor_estimador"], nombre_archivo)
    print(f"Modelo exportado a: {nombre_archivo}")
    print(f"  Modelo: {mejor['modelo']}")
    print(f"  Hiperparametros: {mejor['mejores_params']}")
    return nombre_archivo

def main():
    if not os.path.exists(ARCHIVO_CSV):
        print(f"[ERROR] No se encontro el archivo '{ARCHIVO_CSV}'.")
        print("Asegurate de correr este script desde la carpeta donde esta el CSV.")
        return

    X, y = cargar_datos(ARCHIVO_CSV)
    resultados, _ = entrenar_y_evaluar(X, y)
    mejor, ordenados = seleccionar_mejor(resultados)
    imprimir_resumen(ordenados)

    # print("=" * 70)
    print(f"MEJOR MODELO: {mejor['modelo']}")
    print(f"  CV F1 : {mejor['cv_score']:.4f}")
    print(f"  Test F1: {mejor['test_f1']:.4f}")
    #print("=" * 70)

    exportar_mejor(mejor, CARNE, NOMBRE, APELLIDO)


if __name__ == "__main__":
    main()
