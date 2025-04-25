import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mascaras1 import MASCARAS
from utils import create_output_folder, salvar_imagem_concatenada

def dithering_difusao_erro(img, mascara, serpentina=False):
    altura, largura = img.shape

    if serpentina:
        sentido_normal = True
        # precisamos de uma mascara com dy invertido, ou seja mascara(dx, dy, peso) -> mascara(dx, -dy, peso), pois estamos percorrendo no outro sentido
        mascara_flipped = np.copy(mascara)
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
imagem = np.array(Image.open("imgs/baboon_monocromatica.png").convert('L'), dtype=np.float32)
create_output_folder("out1")

# Para cada máscara, aplica e salva o resultado
for nome, mascara in MASCARAS.items():
    serp_off = dithering_difusao_erro(imagem.copy(), mascara, serpentina=False)
    serp_on = dithering_difusao_erro(imagem.copy(), mascara, serpentina=True)

    # Concatena e salva usando a nova função
    salvar_imagem_concatenada(
        imagens=[imagem, serp_off, serp_on],
        diretorio="out1",
        nome_arquivo=f"{nome.replace(' ', '_')}.png",
        modo="L"
    )

