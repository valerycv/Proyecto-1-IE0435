import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

ruta = r"C:\Users\Valery Carranza\Desktop\IA en IE\proyecto1\muestras-procesadas\positivo2.png"
UMBRAL = 128

img = Image.open(ruta).convert("L")
arr = np.array(img)
binario = (arr >= UMBRAL).astype(int)

# Visualización como imagen
plt.figure(figsize=(6, 6))
plt.imshow(binario, cmap="gray", vmin=0, vmax=1)
plt.title("Matriz binaria - positivo2.png")
plt.axis("off")
plt.tight_layout()
plt.savefig("matriz_visual.png", dpi=150)
plt.show()

print("Imagen guardada como matriz_visual.png")