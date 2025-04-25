import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- Máscaras ---
MASCARAS = {
    "Floyd-Steinberg": np.array([
        (0, 1, 7/16),
        (1, -1, 3/16),
        (1, 0, 5/16),
        (1, 1, 1/16)
    ]),
    "Stevenson-Arce": np.array([
        (0, 2, 32/200),
        (1, -3, 12/200), (1, -1, 26/200), (1, 1, 30/200), (1, 3, 16/200),
        (2, -2, 12/200), (2, 0, 26/200), (2, 2, 1/200),
        (3, -3, 5/200), (3, -1, 12/200), (3, 1, 12/200), (3, 3, 5/200)
    ]),
    "Burkes": np.array([
        (0, 1, 8/32), (0, 2, 4/32),
        (1, -2, 2/32), (1, -1, 4/32), (1, 0, 8/32), (1, 1, 4/32), (1, 2, 2/32)
    ]),
    "Sierra": np.array([
        (0, 1, 5/32), (0, 2, 3/32),
        (1, -2, 2/32), (1, -1, 4/32), (1, 0, 5/32), (1, 1, 4/32), (1, 2, 2/32),
        (2, -1, 2/32), (2, 0, 3/32), (2, 1, 2/32)
    ]),
    "Stucki": np.array([
        (0, 1, 8/42), (0, 2, 4/42),
        (1, -2, 2/42), (1, -1, 4/42), (1, 0, 8/42), (1, 1, 4/42), (1, 2, 2/42),
        (2, -2, 1/42), (2, -1, 2/42), (2, 0, 4/42), (2, 1, 2/42), (2, 2, 1/42)
    ]),
    "Jarvis-Judice-Ninke": np.array([
        (0, 1, 7/48), (0, 2, 5/48),
        (1, -2, 3/48), (1, -1, 5/48), (1, 0, 7/48), (1, 1, 5/48), (1, 2, 3/48),
        (2, -2, 1/48), (2, -1, 3/48), (2, 0, 5/48), (2, 1, 3/48), (2, 2, 1/48)
    ])
}

def carregar_imagem_cinza(caminho):
    return np.array(Image.open(caminho).convert('L'), dtype=np.float32)

def dithering_difusao_erro(img, mascara, serpentina=False):
    altura, largura = img.shape

    if serpentina:
        sentido_normal = True
        # precisamos de uma mascara com dy invertido, ou seja mascara(dx, dy, peso) -> mascara(dx, -dy, peso), pois estamos percorrendo no outro sentido
        mascara_flipped = mascara.copy()
        mascara_flipped[:, 1] *= -1
        for x in range(altura):
            if sentido_normal:
                for y in range(largura):
                    apply_dithering(img, mascara, altura, largura, x, y)
            else:
                for y in range(largura - 1, -1, -1):
                    apply_dithering(img, mascara_flipped, altura, largura, x, y)
            sentido_normal = not sentido_normal
    else:
        for x in range(altura):
            for y in range(largura):
                apply_dithering(img, mascara, altura, largura, x, y)

    return img

def apply_dithering(img, mascara, altura, largura, x, y):
    pixel = img[x, y]
    if pixel < 128:
        img[x, y] = 0
    else:
        img[x, y] = 255

    erro = pixel - img[x, y]

    for dx, dy, peso in mascara:
        x_novo = x + int(dx)
        y_novo = y + int(dy)
        if 0 <= x_novo < altura and 0 <= y_novo < largura:
            img[x_novo, y_novo] += erro * peso

# --- Main ---
caminho_imagem = "imgs/baboon_monocromatica.png"
imagem = carregar_imagem_cinza(caminho_imagem).astype(np.float32)

def comparar_diferencas_rgb(img1, img2):
    """
    Combina duas imagens binárias em uma RGB:
    - img1 vai para o canal vermelho.
    - img2 vai para o canal azul.
    """
    r = 255 - img1
    g = np.zeros_like(img1)
    b = 255 -img2

    # Cria imagem RGB
    rgb = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = r
    rgb[..., 1] = g
    rgb[..., 2] = b
    return Image.fromarray(rgb, 'RGB')

# Para cada máscara, cria imagem de comparação
for nome, mascara in MASCARAS.items():
    serp_off = dithering_difusao_erro(imagem.copy(), mascara, serpentina=False)
    serp_on = dithering_difusao_erro(imagem.copy(), mascara, serpentina=True)

    # Cria imagem RGB com comparação visual
    comparativa = comparar_diferencas_rgb(serp_on, serp_off)

    # Salva
    comparativa.save(f"out1/COMP_{nome.replace(' ', '_')}.png")
    print(f"Comparação salva: out1/COMP_{nome.replace(' ', '_')}.png")
