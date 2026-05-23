# DATASET.md — Documentación del conjunto de datos

## 1. Descripción general

Este dataset simula imágenes de una línea de producción donde puede haber contaminaciones (granos de arroz). Cada imagen está codificada como un vector binario de 16,384 valores, donde:

- **1** = píxel completamente blanco (fondo sin objeto)
- **0** = píxel con presencia de objeto (arroz u otro)

Cada fila del CSV representa una imagen de 128×128 píxeles aplanada, seguida de una etiqueta binaria:

- **1** = imagen contiene granos de arroz (contaminación positiva)
- **0** = imagen no contiene arroz (puede estar vacía o contener clips )

---

## 2. Recolección de datos

### Datos consolidados del grupo

- **Cantidad total**: 119 imágenes provenientes de varios estudiantes del curso.
- **Distribución**: 59 positivas (con arroz) + 60 negativas (sin arroz).
- **Composición**: 4 datasets individuales combinados mediante el script `combinar_datasets.py`.
- **Datasets descartados**: 3 archivos compartidos no pudieron incluirse porque no contenían la columna `etiqueta` (solo 16,384 columnas de píxeles, sin la columna 16,385 que indica la clase).

### Aporte individual

- 30 imágenes propias capturadas con cámara de celular
- 15 positivas (hoja blanca con uno o varios granos de arroz crudo) + 15 negativas (hoja blanca con clips).

### Condiciones de iluminación

- Iluminación variable entre estudiantes: algunos usaron luz natural, otros luz artificial de techo o lámpara directa.
- Esto introduce diversidad en el dataset, lo cual es positivo para la generalización pero también añade ruido.

### Aporte propio

- **Cámara**: iPhone 13.
- **Tipo de luz**: artificial, lámpara de luz blanca dirigida hacia la hoja. Esta configuración minimizó la formación de sombras y permitió que la mayoría de las imágenes se obtuvieran correctamente desde la primera toma.
- **Hora aproximada**: noche.
- **Ubicación**: escritorio personal.

---

## 3. Proceso de etiquetado

El etiquetado se realizó **manualmente al momento de la captura** por cada estudiante:

- Las imágenes con granos de arroz se nombraron con el prefijo `positivo` (`positivo_01.png`).
- Las imágenes sin arroz se nombraron con el prefijo `negativo` (`negativo_01.png`).

El script `convertir_imagenes.py` extrae la etiqueta automáticamente del nombre del archivo.

**Calidad y consistencia del etiquetado**:

- Cada estudiante etiquetó sus propias imágenes (consistencia interna por persona, pero criterios potencialmente diferentes entre estudiantes).
- No hubo proceso de revisión cruzada del etiquetado entre estudiantes.

---

## 4. Preprocesamiento

Cada imagen pasó por las siguientes transformaciones (ver `convertir_imagenes.py`):

1. **Conversión a escala de grises** (modo `L` de PIL).
2. **Redimensión a 128×128 píxeles**.
3. **Binarización** con umbral = 128: píxeles ≥ 128 se codifican como 1, el resto como 0.
4. **Aplanado** del arreglo 2D a un vector fila de 16,384 elementos.
5. **Adición de la columna `etiqueta`** al final.

Resultado individual: `dataset.csv` con dimensiones 30 × 16,385.
Resultado consolidado: `dataset_grupo.csv` con dimensiones 119 × 16,385.

---

## 5. Limitaciones del dataset

- **Tamaño aún modesto**: 119 imágenes es mejor que un dataset individual, pero sigue siendo pequeño para tareas de visión por computadora.
- **Heterogeneidad no controlada**: las diferencias entre cámaras, iluminaciones y técnicas de captura entre estudiantes introducen ruido que el modelo debe manejar.
- **Iluminación**: condiciones académicas no representan escenarios industriales reales.
- **Fondo único**: solo se usó hoja blanca.
- **Resolución binaria**: la binarización con umbral fijo pierde información de intensidad y textura.
- **Tamaño y orientación de objetos**: el dataset no cubre exhaustivamente todas las variaciones de granos de arroz.
- **Oclusión y borrosidad**: no se exploran sistemáticamente.
- **Etiquetado entre personas**: no hubo proceso de validación cruzada del etiquetado.

---

## 6. Acceso y compartición

- El dataset individual (`dataset.csv`) está en la raíz del repositorio.
- Los CSV de los compañeros se guardan en `datasets_companeros/`.
- El dataset consolidado (`dataset_grupo.csv`) se genera al correr `combinar_datasets.py`.

**Nota sobre la consolidación del grupo**: el script `combinar_datasets.py` filtra automáticamente los CSV que no tienen el formato correcto (deben tener 16,385 columnas con la columna `etiqueta` al final). Tres archivos de compañeros no se pudieron incluir porque solo contenían los 16,384 píxeles, sin la columna de etiquetas.
