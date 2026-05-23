# Informe final — Proyecto 1

**Curso:** IE0435 — Inteligencia Artificial en Ingeniería Eléctrica
**Semestre:** I-2026
**Estudiante:** Valery Carranza Vásquez
**Carné:** B91660
**Fecha:** 20 de mayo de 2026

---

## 1. Introducción

En procesos de manufactura, la inspección visual automatizada permite detectar y cuantificar anomalías para asegurar la calidad del producto y reducir el error humano. Este proyecto aplica técnicas clásicas de aprendizaje automático para clasificar imágenes provenientes de una línea de producción simulada, identificando la presencia o ausencia de contaminaciones específicas (granos de arroz) sobre una hoja blanca.

El objetivo es construir un sistema completo que abarque la recolección de datos, el preprocesamiento, el entrenamiento de modelos, la evaluación y la exportación del mejor modelo para uso posterior.

---

## 2. Metodología

### 2.1 Recolección de datos

Se realizó una recolección colaborativa entre estudiantes del curso. El dataset consolidado quedó conformado por:

- **Aporte individual**: 30 imágenes propias (15 positivas + 15 negativas) capturadas con un iPhone 13 sobre una hoja blanca, con iluminación controlada.
- **Aportes de compañeros**: 89 imágenes adicionales obtenidas mediante consolidación con los CSV de tres compañeros del grupo que sí incluyeron la columna `etiqueta` en su formato.

**Total consolidado**: 119 imágenes (59 positivas + 60 negativas).

Ver [`DATASET.md`](../DATASET.md) para más detalles.

### 2.2 Preprocesamiento

Cada imagen fue procesada de la siguiente forma:

1. Conversión a escala de grises.
2. Redimensión a 128×128 píxeles.
3. Binarización con umbral 128: píxel ≥ 128 → 1 (blanco), píxel < 128 → 0 (objeto).
4. Aplanado a vector fila de 16,384 elementos.
5. Adición de la etiqueta (0 o 1) como columna 16,385.

Resultado: `dataset_grupo.csv` con dimensiones 119 × 16,385.

### 2.3 Modelos entrenados

Se entrenaron cuatro modelos clásicos de clasificación binaria:

| Modelo                  | Hiperparámetros explorados |
|-------------------------|----------------------------|
| Árbol de Decisión       | `max_depth`, `min_samples_split`, `criterion` |
| Naive Bayes             | `alpha` |
| KNN                     | `n_neighbors`, `weights`, `metric` |
| SVM                     | `C`, `kernel`, `gamma` |

Para cada modelo se utilizó `GridSearchCV` con `StratifiedKFold` (5 folds, `random_state=42`) sobre el 75% de los datos para entrenamiento. El 25% restante se usó como conjunto de prueba independiente.

La métrica de optimización fue F1-score binario para la clase positiva.

---

## 3. Resultados

### 3.1 Comparación de modelos

| Modelo                  | CV F1   | Test Acc | Test Prec | Test Rec | Test F1 |
|-------------------------|---------|----------|-----------|----------|---------|
| KNN                 | 0.6768 | 0.6000   | 0.5556    | 1.0000   | 0.7143  |
| Naive Bayes (Bernoulli) | 0.6724  | 0.5333   | 0.5238    | 0.7333   | 0.6111  |
| SVM                     | 0.6675  | 0.6000   | 0.5714    | 0.8000   | 0.6667  |
| Árbol de Decisión       | 0.6098  | 0.6667   | 0.6087    | 0.9333   | 0.7368  |

### 3.2 Modelo seleccionado

**KNN (K-Nearest Neighbors)** con hiperparámetros `n_neighbors=1, weights="uniform", metric="euclidean"` fue seleccionado por obtener el mejor F1-score en validación cruzada (0.6768).

Aunque el Árbol de Decisión obtuvo el mejor F1-score en el conjunto de prueba (0.7368), su F1 en validación cruzada (0.6098) fue notablemente menor que el de KNN. Como la validación cruzada es una estimación más robusta del desempeño esperado en datos nuevos, se priorizó este criterio.

El modelo se exportó en formato `.joblib` (paso 3.5 del enunciado):

```
B91660_valery_carranza.joblib
```

---

## 4. Análisis y discusión

### 4.1 Observaciones

- **Las métricas son consistentes pero modestas**. Con 119 imágenes y 16,384 features (píxeles binarios), el problema es genuinamente difícil. Esto se conoce como el "curse of dimensionality": hay muchas más dimensiones que ejemplos.
- **KNN ganó por poco**: la diferencia entre los cuatro modelos en CV F1 es pequeña (0.61 – 0.68), lo cual sugiere que ninguno encuentra un patrón decisivamente superior con la representación actual.
- **El árbol de decisión obtuvo el mejor F1 en test (0.7368) pero el peor en CV (0.6098)**: indicio probable de sobreajuste al split particular de prueba; por eso se priorizó la métrica de CV.
- **KNN con n=1 funciona razonablemente**: con n=1, el modelo simplemente busca la imagen más similar (en distancia euclidiana sobre los 16,384 píxeles) y copia su etiqueta. Que esto sea competitivo con métodos más sofisticados sugiere que las imágenes del mismo tipo son visualmente similares entre sí (al menos en términos de píxeles).
- **El KNN tiene recall=1.0 en test**: predice positivo para todos los ejemplos positivos, aunque a costa de precisión (0.5556). Esto puede ser ventajoso en un contexto de inspección de calidad, donde es preferible un falso positivo (revisar de más) que un falso negativo (dejar pasar contaminación).

### 4.2 Comparación con el dataset individual

En una iteración previa con solo las 30 imágenes propias, el modelo ganador fue Naive Bayes (Bernoulli) con CV F1 = 0.7648. Al ampliar el dataset a 119 imágenes del grupo, las métricas absolutas bajaron porque el problema se volvió más realista: las imágenes ahora provienen de múltiples cámaras, iluminaciones y técnicas de captura. Esta heterogeneidad reduce las métricas pero produce un modelo más generalizable, que es lo deseable.

### 4.3 Limitaciones

- **Heterogeneidad no controlada**: las imágenes del grupo provienen de distintos teléfonos y entornos, lo cual añade ruido.
- **Algunos compañeros omitieron la columna de etiqueta** en sus CSV, lo que redujo la cantidad de datos disponibles. Sería recomendable que el grupo acuerde un formato estándar antes de iniciar la recolección.
- **Binarización agresiva**: el umbral fijo de 128 descarta información de intensidad y textura.
- **Sin características derivadas**: los modelos trabajan directamente sobre píxeles, sin aprovechar descriptores más informativos como histogramas de gradientes o momentos.

### 4.4 Recomendaciones para trabajo futuro

- Estandarizar el formato de CSV entre todos los miembros del grupo antes de la recolección.
- Aplicar técnicas de preprocesamiento alternativas.
- Reducción de dimensionalidad (PCA) antes de entrenar modelos sensibles a la dimensionalidad como SVM y KNN.
- Extracción de características clásicas de visión por computadora que podrían ser más informativas que los píxeles crudos.

---

## 5. Conclusiones

Se construyó un pipeline completo de clasificación binaria que abarca recolección colaborativa, preprocesamiento, entrenamiento, evaluación y exportación del modelo. **KNN** resultó ser el mejor modelo bajo la métrica F1 en validación cruzada con el dataset consolidado del grupo.


---

## 6. Referencias

- Documentación oficial de scikit-learn: https://scikit-learn.org/stable/
- Enunciado del Proyecto 1, curso IE0435, I-2026.
- Pedregosa et al., "Scikit-learn: Machine Learning in Python", JMLR 12, pp. 2825-2830, 2011.
