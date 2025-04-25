import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- Máscaras ---
MASCARAS = {
    "Floyd-Steinberg": [
        (0, 1, 7/16),
        (1, -1, 3/16),
        (1, 0, 5/16),
        (1, 1, 1/16)
    ],
    "Stevenson-Arce": [
        (0, 2, 32/200),
        (1, -3, 12/200), (1, -1, 26/200), (1, 1, 30/200), (1, 3, 16/200),
        (2, -2, 12/200), (2, 0, 26/200), (2, 2, 1/200),
        (3, -3, 5/200), (3, -1, 12/200), (3, 1, 12/200), (3, 3, 5/200)
    ],
    "Burkes": [
        (0, 1, 8/32), (0, 2, 4/32),
        (1, -2, 2/32), (1, -1, 4/32), (1, 0, 8/32), (1, 1, 4/32), (1, 2, 2/32)
    ],
    "Sierra": [
        (0, 1, 5/32), (0, 2, 3/32),
        (1, -2, 2/32), (1, -1, 4/32), (1, 0, 5/32), (1, 1, 4/32), (1, 2, 2/32),
        (2, -1, 2/32), (2, 0, 3/32), (2, 1, 2/32)
    ],
    "Stucki": [
        (0, 1, 8/42), (0, 2, 4/42),
        (1, -2, 2/42), (1, -1, 4/42), (1, 0, 8/42), (1, 1, 4/42), (1, 2, 2/42),
        (2, -2, 1/42), (2, -1, 2/42), (2, 0, 4/42), (2, 1, 2/42), (2, 2, 1/42)
    ],
    "Jarvis-Judice-Ninke": [
        (0, 1, 7/48), (0, 2, 5/48),
        (1, -2, 3/48), (1, -1, 5/48), (1, 0, 7/48), (1, 1, 5/48), (1, 2, 3/48),
        (2, -2, 1/48), (2, -1, 3/48), (2, 0, 5/48), (2, 1, 3/48), (2, 2, 1/48)
    ]
}

def carregar_imagem_cinza(caminho):
    return np.array(Image.open(caminho).convert('L'), dtype=np.float32)

def dithering_difusao_erro(img, mascara, serpentina=False):
    altura, largura = img.shape
    img_flip = np.flip(img, axis=1) # Inverte a imagem horizontalmente

    if serpentina:
        # nas linhas pares considera a imagem invertida para aplicar a máscara que deve ser invertida também
        # criamos uma mascara invertida
        mascara_invertida = [(dx, largura - 1 - dy, peso) for dx, dy, peso in mascara]
        for x in range(altura):
            if x % 2 == 0: # linhas pares (invertido)
                for y in range(largura):
                    apply_dithering(img_flip, mascara, altura, largura, x, y)
            else:
                for y in range(largura-1, -1, -1):
                    apply_dithering(img, mascara, altura, largura, x, y)
    else:
        for x in range(altura):
            for y in range(largura):
                apply_dithering(img, mascara, altura, largura, x, y)


    return

def apply_dithering(img, mascara, altura, largura, x, y):
    pixel = img[x, y]
    if pixel < 128:
        img[x, y] = 0
    else:
        img[x, y] = 255

    erro = pixel - img[x, y]

    for dx, dy, peso in mascara:
        x_novo = x + dx
        y_novo = y + dy
        if 0 <= x_novo < altura and 0 <= y_novo < largura:
            img[x_novo, y_novo] += erro * peso

# --- Main ---
caminho_imagem = "imgs/baboon_colorida.png"
imagem = carregar_imagem_cinza(caminho_imagem).astype(np.uint8)

# Cria pasta de saída
os.makedirs("out1", exist_ok=True)

# Para cada máscara, aplica e salva o resultado
for nome, mascara in MASCARAS.items():
    serp_off = dithering_difusao_erro(imagem, mascara, serpentina=False)
    serp_on = dithering_difusao_erro(imagem, mascara, serpentina=True)

    # Concatena horizontalmente
    combinado = np.hstack([imagem, serp_off, serp_on])

    # Converte e salva
    img_pil = Image.fromarray(combinado)
    img_pil.save(f"out1/{nome.replace(' ', '_')}.png")
    print(f"Salvo: out1/{nome.replace(' ', '_')}.png")