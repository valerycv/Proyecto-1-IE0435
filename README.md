# Proyecto 1 — Clasificación de contaminaciones en una línea de producción simulada

**Curso:** IE0435 — Inteligencia Artificial en Ingeniería Eléctrica
**Universidad de Costa Rica**
I-2026
**Estudiante:** Valery Carranza Vásquez
**Carné:** B91660

---

## 1. Descripción

Este proyecto implementa un sistema de clasificación binaria para detectar la presencia de granos de arroz (contaminaciones positivas) en imágenes de una línea de producción simulada sobre una hoja blanca. Otros objetos (clips, aros) o imágenes vacías se consideran ejemplos negativos.

El pipeline incluye:

1. **Preprocesamiento** de imágenes: conversión a escala de grises y redimensión a 128×128 píxeles.
2. **Vectorización binaria**: cada imagen se convierte en un vector de 16,384 valores (0 = objeto, 1 = blanco) más una etiqueta.
3. **Consolidación colaborativa**: combinación de los datasets de varios estudiantes del curso en un único CSV.
4. **Entrenamiento** y evaluación de cuatro modelos clásicos: Árbol de Decisión, Naive Bayes (Bernoulli), KNN y SVM, con búsqueda de hiperparámetros mediante `GridSearchCV` y validación cruzada estratificada.
5. **Exportación** del mejor modelo en formato `.joblib`.
---

## 2. Requisitos

- Python 3.10 o superior (probado en Python 3.13)
- Bibliotecas listadas en `requirements.txt`

Instalación:

```bash
pip install -r requirements.txt
```

---

## 3. Cómo correr el entrenamiento

1. Colocar las imágenes preprocesadas (128×128, escala de grises, formato `.png`) en una carpeta llamada `muestras-procesadas/`.
   - Los nombres deben comenzar con `positivo` o `negativo` para asignar la etiqueta automáticamente.

2. Generar el dataset CSV individual:

   ```bash
   python convertir_imagenes.py
   ```

   Esto produce `dataset.csv` con 16,385 columnas (16,384 píxeles + 1 etiqueta).

3. Combinar con los CSV de varios compañeros:

   - Colocar los CSV externos en la carpeta `datasets_companeros/`.
   - Correr:

     ```bash
     python combinar_datasets.py
     ```

   - Esto genera `dataset_grupo.csv`.

4. Entrenar los modelos y exportar el mejor:

   ```bash
   python entrenar_modelos.py
   ```

   El script:
   - Carga `dataset_grupo.csv` (o `dataset.csv` si solo se trabaja con datos propios).
   - Realiza `GridSearchCV` con `StratifiedKFold` (5 folds) sobre los 4 modelos.
   - Muestra una tabla comparativa.
   - Exporta el mejor modelo a `B91660_valery_carranza.joblib`.

---

## 4. Cómo hacer inferencia con el modelo

```python
import joblib
import numpy as np
from PIL import Image

# 1. Cargar el modelo
modelo = joblib.load("B91660_valery_carranza.joblib")

# 2. Cargar y preprocesar una imagen nueva
img = Image.open("imagen_nueva.png").convert("L").resize((128, 128))
arr = np.array(img)
binario = (arr >= 128).astype(int).flatten()  # vector de 16384

# 3. Predecir
prediccion = modelo.predict([binario])[0]
print("Contaminación detectada" if prediccion == 1 else "Sin contaminación")
```

---

## 5. Reproducibilidad

Todos los scripts usan `random_state = 42` para garantizar resultados reproducibles. Cualquier persona que clone este repositorio y siga los pasos anteriores obtendrá las mismas métricas.

---

## 6. Licencia

Ver archivo [`LICENSE`](LICENSE).
