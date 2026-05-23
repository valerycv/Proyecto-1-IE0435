# MODEL_CARD.md — Ficha técnica del modelo

## 1. Identificación del modelo

- **Nombre del modelo**: Clasificador de contaminaciones por granos de arroz
- **Versión**: 1.0
- **Tipo**: KNN (K-Nearest Neighbors) — clasificador binario
- **Autora**: Valery Carranza Vásquez
- **Carné**: B91660
- **Fecha de entrenamiento**: [Fecha en que corriste el script, ej. 22/05/2026]
- **Curso**: IE0435 — Inteligencia Artificial en Ingeniería Eléctrica, UCR
- **Archivo del modelo**: `B91660_valery_carranza.joblib`

---

## 2. Uso previsto (Intended use)

### Uso pretendido

Clasificar imágenes de 128×128 píxeles binarizadas (0/1) de una línea de producción simulada, indicando si contienen granos de arroz (contaminación positiva) o no.

### Usuarios objetivo

- Estudiantes y profesores del curso IE0435.
- Demostración académica de un pipeline completo de clasificación clásica con scikit-learn.

### Fuera de alcance (Out of scope)

- **No usar en producción real**: el modelo fue entrenado con un conjunto pequeño de datos académicos (119 imágenes) y condiciones controladas.
- **No es robusto** a cambios drásticos de iluminación, fondo, cámara o resolución.
- **No detecta otros tipos de contaminación** distintos a granos de arroz crudo.
- **No es un sistema de inspección industrial certificado**.

---

## 3. Resumen de datos (Data summary)

- **Origen**: imágenes recolectadas colaborativamente por varios estudiantes del curso, consolidadas en un único CSV.
- **Cantidad total**: 119 imágenes (59 positivas con arroz, 60 negativas).
- **Resolución de entrada al modelo**: 128×128 píxeles binarizados (16,384 características por imagen).
- **Variaciones presentes**: múltiples dispositivos (varias cámaras de teléfono, incluida iPhone 13 para el aporte propio), distintas iluminaciones y entornos de captura entre estudiantes.

Ver [`DATASET.md`](DATASET.md) para más detalles sobre la recolección y limitaciones.

---

## 4. Proceso de etiquetado

- **Etiquetado manual** por cada estudiante al momento de la captura.
- Las etiquetas se codifican en el nombre del archivo (`positivo_*.png` / `negativo_*.png`) y luego se vuelven una columna `etiqueta` en el CSV.
- **Consistencia**: variable entre estudiantes (criterios individuales).
- **Sesgo posible**: cada persona puede tener un criterio distinto sobre qué cuenta como "contaminación visible".

---

## 5. Métricas

### Métrica primaria

- **F1-score binario** (clase positiva = "contiene arroz").
- Métrica elegida porque las clases están balanceadas pero se busca priorizar la detección correcta de positivos sin sacrificar precisión.

### Estrategia de evaluación

- **Validación cruzada estratificada**: 5 folds (`StratifiedKFold` con `random_state=42`).
- **Split adicional de prueba**: 75% entrenamiento / 25% prueba estratificado (`random_state=42`).
- **Búsqueda de hiperparámetros**: `GridSearchCV` sobre el conjunto de entrenamiento.

### Resultados — comparación de modelos

| Modelo                  | CV F1   | Test Acc | Test Prec | Test Rec | Test F1 |
|-------------------------|---------|----------|-----------|----------|---------|
| **KNN**                 | **0.6768** | 0.6000   | 0.5556    | 1.0000   | 0.7143  |
| Naive Bayes (Bernoulli) | 0.6724  | 0.5333   | 0.5238    | 0.7333   | 0.6111  |
| SVM                     | 0.6675  | 0.6000   | 0.5714    | 0.8000   | 0.6667  |
| Árbol de Decisión       | 0.6098  | 0.6667   | 0.6087    | 0.9333   | 0.7368  |

### Modelo seleccionado

- **KNN** con hiperparámetros: `n_neighbors = 1`, `weights = "uniform"`, `metric = "euclidean"`.
- Razón de selección: mejor F1 en validación cruzada (0.6768). Aunque el árbol obtuvo un F1 más alto en el conjunto de prueba (0.7368), KNN mostró el mejor desempeño promedio en validación cruzada (5-fold), lo cual es una estimación más robusta del desempeño esperado en datos nuevos.

---

## 6. Notas éticas y de seguridad

- **Sesgo por iluminación**: aunque el dataset incluye varias condiciones, todas son ambientes domésticos/académicos. El modelo puede fallar en condiciones industriales.
- **Sesgo por cámara**: las cámaras de celulares de estudiantes difieren de cámaras industriales de inspección.
- **Sesgo por fondo**: solo se entrenó sobre hoja blanca. Cualquier otro fondo invalida el modelo.
- **Sesgo por etiquetado**: cada estudiante etiquetó con su propio criterio sin revisión cruzada.
- **Uso responsable**: no debe utilizarse como única medida de control de calidad en aplicaciones reales sin validación exhaustiva adicional.

---

## 7. Limitaciones

- **Objetos pequeños**: granos de arroz muy pequeños o desenfocados pueden no detectarse.
- **Oclusión**: el modelo no fue entrenado con casos de granos parcialmente cubiertos.
- **Borrosidad (blur)**: imágenes desenfocadas no se incluyeron sistemáticamente.
- **Generalización limitada**: 119 imágenes sigue siendo un dataset pequeño para visión por computadora.
- **Pérdida de información en binarización**: el umbral fijo de 128 elimina información de intensidad y textura.
- **KNN con n=1 es sensible al ruido**: cualquier imagen mal etiquetada afecta directamente las predicciones.

---

## 8. Reproducibilidad

### Hardware usado

- **Equipo**: Laptop personal con Windows 10.
- **Sistema operativo**: Windows 10.

### Software

- Python 3.13.
- scikit-learn, pandas, numpy, joblib, Pillow (ver `requirements.txt`).

### Comandos exactos para reproducir

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Generar dataset desde imágenes propias (carpeta muestras-procesadas/)
python convertir_imagenes.py

# 3. Combinar con datasets de compañeros (carpeta datasets_companeros/)
python combinar_datasets.py

# 4. Entrenar y exportar el mejor modelo
python entrenar_modelos.py
```

### Semillas

- Todas las operaciones aleatorias usan `random_state = 42`.

---

## 9. Contacto

- **Autora**: Valery Carranza Vásquez
- **Correo**: valery.carranza@ucr.ac.cr
- **Repositorio**: [URL del repositorio en GitHub] ← edita esto después de subirlo
