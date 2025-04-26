import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mascaras1 import MASCARAS
from utils import create_output_folder, salvar_imagem_concatenada, mostrar_imagem_concatenada, salvar_grid_imagens, plotar_histogramas_lado_a_lado

def dithering_difusao_erro(img, mascara, serpentina=False):
    altura, largura = img.shape
    output = np.zeros_like(img)

    if serpentina:
        sentido_normal = True
        for x in range(altura):
            if sentido_normal:
                for y in range(largura):
                    apply_dithering(img, output, mascara, altura, largura, x, y, direcao=1)
            else:
                for y in range(largura - 1, -1, -1):
                    apply_dithering(img, output, mascara, altura, largura, x, y, direcao=-1)
            sentido_normal = not sentido_normal
    else:
        for x in range(altura):
            for y in range(largura):
                apply_dithering(img, output, mascara, altura, largura, x, y, direcao=1)

    return output

def apply_dithering(img, output, mascara, altura, largura, x, y, direcao=1):
    pixel = img[x, y]
    novo_valor = 0 if pixel < 128 else 255
    erro = pixel - novo_valor
    output[x, y] = novo_valor  # imagem binarizada

    for dx, dy, peso in mascara:
        x_novo = x + int(dx)
        y_novo = y + int(dy * direcao)
        if 0 <= x_novo < altura and 0 <= y_novo < largura:
            img[x_novo, y_novo] += erro * peso

def dithering_difusao_erro_rgb(img_rgb, mascara, serpentina=False):
    # Inicializa a imagem de saída com mesma forma
    img_out = np.zeros_like(img_rgb)

    # Aplica dithering em cada canal separadamente
    for c in range(3):  # 0 = R, 1 = G, 2 = B
        canal = img_rgb[:, :, c].astype(np.float32).copy()
        canal_dither = dithering_difusao_erro(canal, mascara, serpentina)
        img_out[:, :, c] = np.clip(canal_dither, 0, 255)

    return img_out.astype(np.uint8)

# --- Main ---
imagem = np.array(Image.open("imgs/img1-mono.png").convert('L'), dtype=np.float32)
imagem_rgb = np.array(Image.open("imgs/img1-colored.png").convert('RGB'), dtype=np.float32)
create_output_folder("out1")

# lista de imagens para o grid (dois grids: com e sem serpentina)
# cada lista (list) é formada por tuplas (imagem, nome)
imgs_serpentina = []
imgs_sem_serpentina = []
imgs_serpentina_rgb = []
imgs_sem_serpentina_rgb = []

# Para cada máscara, aplica e salva o resultado
for nome, mascara in MASCARAS.items():
    # Processamento monocromático
    serp_off = dithering_difusao_erro(imagem.copy(), mascara, serpentina=False)
    serp_on = dithering_difusao_erro(imagem.copy(), mascara, serpentina=True)
    imgs_sem_serpentina.append((serp_off, nome))
    imgs_serpentina.append((serp_on, nome))
    salvar_imagem_concatenada([imagem, serp_off, serp_on], "out1/imgs", f"{nome.replace(' ', '_')}.png", "L")
    plotar_histogramas_lado_a_lado(
        [imagem, serp_off, serp_on],
        ["Original", "Serpentina Off", "Serpentina On"],
        f"out1/hists/{nome.replace(' ', '_')}_hist.png",
        "L",
        f"Histograma - {nome}"
    )

    # Processamento colorido
    serp_off_rgb = dithering_difusao_erro_rgb(imagem_rgb.copy(), mascara, serpentina=False)
    serp_on_rgb = dithering_difusao_erro_rgb(imagem_rgb.copy(), mascara, serpentina=True)
    imgs_sem_serpentina_rgb.append((serp_off_rgb, nome))
    imgs_serpentina_rgb.append((serp_on_rgb, nome))
    salvar_imagem_concatenada([imagem_rgb, serp_off_rgb, serp_on_rgb], "out1/imgs", f"{nome.replace(' ', '_')}_colorida.png", "RGB")
    plotar_histogramas_lado_a_lado(
        [imagem_rgb, serp_off_rgb, serp_on_rgb],
        ["Original", "Serpentina Off", "Serpentina On"],
        f"out1/hists/{nome.replace(' ', '_')}_colorido_hist.png",
        "RGB",
        f"Histograma - {nome} (Colorido)"
    )

for imgs, nome_arquivo, modo in [
    (imgs_sem_serpentina, "grid_sem_serpentina.png", "L"),
    (imgs_serpentina, "grid_com_serpentina.png", "L"),
    (imgs_sem_serpentina_rgb, "grid_sem_serpentina_colorida.png", "RGB"),
    (imgs_serpentina_rgb, "grid_com_serpentina_colorida.png", "RGB")
]:
    salvar_grid_imagens(imagens_nomes=imgs, diretorio="out1/grids", nome_arquivo=nome_arquivo, modo=modo)
