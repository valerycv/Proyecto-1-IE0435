# Primer avance – Proyecto 1

**Curso:** IE0435 – Inteligencia Artificial en Ingeniería Eléctrica
**Estudiante:** Valery Carranza Vásquez - B91660  
**Fecha:** 28 de abril de 2026  

---

## Introducción

Se documenta el avance de las etapas de preprocesamiento de imágenes y conformación del conjunto de datos (dataset). El objetivo de estas etapas es convertir las imágenes recolectadas en una representación numérica adecuada para el entrenamiento de modelos de clasificación clásica.

---

## Preprocesamiento de Imágenes 

Las imágenes fueron recolectadas de forma individual, siguiendo la metodología establecida para la simulación de la línea de producción. Cada imagen fue tomada sobre una hoja blanca, donde los elementos presentes podían ser granos de arroz (contaminación positiva), otros objetos como clips (contaminación negativa).

El preprocesamiento consistió en los siguientes pasos:

- En Canva se recortaron las imágenes y se convirtieron a blanco y negro.
- Conversión de cada imagen a escala de grises.
- Redimensionamiento a un tamaño uniforme de 128 × 128 píxeles.
- Almacenamiento en formato .png.
  
El conjunto de imágenes recolectadas consta de 30 imágenes en total:

| Clase | Cantidad |
|---|---|
| Positivas (granos de arroz) | 15 |
| Negativas (clips) | 15 |

---
## Conformación del Conjunto de Datos

### Conversión a matriz binaria

Utilizando las bibliotecas Pillow y NumPy de Python, cada imagen fue procesada de la siguiente manera:

1. Se cargó la imagen en escala de grises.
2. Se verificó que el tamaño fuera exactamente 128 × 128 píxeles.
3. Se aplicó un umbral de binarización de 128, los píxeles con valor mayor o igual al umbral fueron codificados como 1, y los píxeles con valor inferior fueron codificados como 0
4. La matriz resultante de 128 × 128 valores binarios fue aplanada en un vector fila de 16,384 elementos

### Generación del CSV

Cada vector fila fue almacenado como una fila en un archivo CSV, con una columna adicional al final denominada etiqueta, cuyo valor indica la clase de la imagen:

- 1 → imagen positiva (contiene arroz)
- 0 → imagen negativa (no contiene arroz)

El dataset.csv presenta las siguientes características:

| Característica | Valor |
|---|---|
| Número de filas | 30 |
| Número de columnas | 16,385 |
| Columnas de píxeles | 16,384 (p0 a p16383) |
| Columna de etiqueta | 1 (etiqueta) |
| Valores en píxeles | Binarios (0 o 1) |
| Distribución de clases | 15 positivas, 15 negativas |

### Implementación

El script desarrollado en Python (convertir_imagenes.py) automatiza el proceso completo de conversión para todas las imágenes contenidas en la carpeta de muestras procesadas. El código detecta la clase de cada imagen a partir del nombre del archivo (positivo*.png o negativo*.png) y genera el archivo CSV con el encabezado y las etiquetas correspondientes.

```python
# Fragmento principal del proceso de binarización
arr    = np.array(img)                    # matriz de valores 0-255
binario = (arr >= UMBRAL).astype(int)     # 1=blanco, 0=objeto
vector  = binario.flatten()              # vector de 16,384 elementos
```

---

## Herramientas Utilizadas

| Herramienta | Uso |
|---|---|
| Canva | Formato para las fotos |
| Python | Lenguaje de programación principal |
| Pillow (PIL) | Carga y conversión de imágenes |
| NumPy | Operaciones matriciales y binarización |
| CSV (stdlib) | Escritura del archivo de salida |

---
