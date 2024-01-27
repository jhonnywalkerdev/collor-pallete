from PIL import Image
from PIL import ImageColor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from sklearn.cluster import KMeans

# Caminho para a imagem enviada
image_path = 'example.jpg'

# Função para encontrar as cores principais de uma imagem
def get_dominant_colors(image_path, n_colors=6):
    # Carrega a imagem
    image = Image.open(image_path)
    # Converte a imagem para RGB
    image = image.convert('RGB')
    # Reduz a imagem para acelerar o processo de clustering
    image = image.resize((50, 50))
    # Converte a imagem para um array numpy
    np_image = np.array(image)
    # Reformata a imagem para ser uma longa lista de cores
    np_image = np_image.reshape((np_image.shape[0] * np_image.shape[1], 3))
    # Utiliza KMeans para encontrar as cores principais
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(np_image)
    # Cores principais
    dominant_colors = kmeans.cluster_centers_.astype(int)
    return dominant_colors


# Obtenha as cores dominantes
dominant_colors = get_dominant_colors(image_path)

# Exibir as cores dominantes
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.imshow(Image.open(image_path))
plt.title('Original Image')
plt.axis('off')

# Plot das cores dominantes
plt.subplot(1, 2, 2)
plt.title('Dominant Colors')
for i, color in enumerate(dominant_colors):
    if i == 5:
        base_color_hex = to_hex(color/255)
    plt.bar(i, 1, color=color/255)  # normaliza as cores para matplotlib
plt.axis('off')
plt.show()

dominant_colors

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

# Função para ajustar a saturação e o valor (brilho) de uma cor no espaço HSV
def adjust_hsv_saturation_value(rgb_color, saturation_factor, value_factor):
    # Converter de RGB para HSV
    hsv_color = rgb_to_hsv(rgb_color/255.0)
    # Ajustar saturação e valor (brilho)
    new_hsv = (hsv_color[0], max(min(hsv_color[1] * saturation_factor, 1), 0), max(min(hsv_color[2] * value_factor, 1), 0))
    # Converter de volta para RGB e então para Hex
    return to_hex(hsv_to_rgb(new_hsv))

# Gerar variações de cores para a paleta
def generate_color_palette_variations(dominant_colors, base_color_hex):
    palette_variations = {}
    base_color_rgb = np.array(ImageColor.getrgb(base_color_hex))
    
    # Para cada cor dominante, gerar variações
    for i, color in enumerate(dominant_colors):
        variations = []
        # Ajustar a saturação e o valor (brilho) em passos
        for saturation_factor in [0.75, 1, 1.25]:
            for value_factor in [0.75, 1, 1.25]:
                # Se a cor for muito próxima da cor base, ajustar apenas o valor (brilho)
                if np.allclose(color, base_color_rgb, atol=30):
                    if value_factor != 1:  # Evitar duplicar a cor base
                        adjusted_color = adjust_hsv_saturation_value(color, 1, value_factor)
                        variations.append(adjusted_color)
                else:
                    adjusted_color = adjust_hsv_saturation_value(color, saturation_factor, value_factor)
                    variations.append(adjusted_color)
        palette_variations[to_hex(color/255)] = variations
    
    return palette_variations

# Aplicar a função para gerar variações
palette_variations = generate_color_palette_variations(dominant_colors, base_color_hex)

# Mostrar a paleta de variações
fig, axs = plt.subplots(len(palette_variations), len(max(palette_variations.values(), key=len)), figsize=(12, 6))

for i, (base_color, variations) in enumerate(palette_variations.items()):
    for j, color_hex in enumerate(variations):
        axs[i, j].bar(0, 1, color=color_hex)
        axs[i, j].axis('off')
        axs[i, j].set_title(f"{color_hex.upper()}")
    # Esconder eixos extras se houver
    for j in range(len(variations), len(axs[i])):
        axs[i, j].axis('off')

plt.tight_layout()
plt.show()

palette_variations