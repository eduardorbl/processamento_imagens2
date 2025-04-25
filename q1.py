import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mascaras1 import MASCARAS
from utils import create_output_folder, salvar_imagem_concatenada, mostrar_imagem_concatenada, salvar_grid_imagens

def dithering_difusao_erro(img, mascara, serpentina=False):
    altura, largura = img.shape

    if serpentina:
        sentido_normal = True
        for x in range(altura):
            if sentido_normal:
                # Sentido normal (esquerda para direita)
                for y in range(largura):
                    apply_dithering(img, mascara, altura, largura, x, y, direcao=1)
            else:
                # Sentido inverso (direita para esquerda)
                for y in range(largura - 1, -1, -1):
                    apply_dithering(img, mascara, altura, largura, x, y, direcao=-1)
            sentido_normal = not sentido_normal
    else:
        for x in range(altura):
            for y in range(largura):
                apply_dithering(img, mascara, altura, largura, x, y, direcao=1)

    return img

def apply_dithering(img, mascara, altura, largura, x, y, direcao=1):
    pixel = img[x, y]
    novo_valor = 0 if pixel < 128 else 255
    erro = pixel - novo_valor
    img[x, y] = novo_valor

    for dx, dy, peso in mascara:
        x_novo = x + int(dx)
        y_novo = y + int(dy * direcao)  # Aqui multiplicamos dy pela direção
        if 0 <= x_novo < altura and 0 <= y_novo < largura:
            img[x_novo, y_novo] += erro * peso

def dithering_difusao_erro_rgb(img_rgb, mascara, serpentina=False):
    # Inicializa a imagem de saída com mesma forma
    img_out = np.zeros_like(img_rgb)

    # Aplica dithering em cada canal separadamente
    for c in range(3):  # 0 = R, 1 = G, 2 = B
        canal = img_rgb[:, :, c].copy()
        canal_dither = dithering_difusao_erro(canal, mascara, serpentina)
        img_out[:, :, c] = np.clip(canal_dither, 0, 255)

    return img_out.astype(np.uint8)

# --- Main ---
imagem = np.array(Image.open("imgs/baboon_monocromatica.png").convert('L'), dtype=np.float32)
imagem_rgb = np.array(Image.open("imgs/baboon_colorida.png").convert('RGB'), dtype=np.float32)
create_output_folder("out1")

# lista de imagens para o grid (dois grids: com e sem serpentina)
# cada lista (list) é formada por tuplas (imagem, nome)
imgs_serpentina = []
imgs_sem_serpentina = []
imgs_serpentina_rgb = []
imgs_sem_serpentina_rgb = []

# Para cada máscara, aplica e salva o resultado
for nome, mascara in MASCARAS.items():
    serp_off = dithering_difusao_erro(imagem.copy(), mascara, serpentina=False)
    serp_on = dithering_difusao_erro(imagem.copy(), mascara, serpentina=True)

    # Adiciona as imagens à lista de grid
    imgs_sem_serpentina.append((serp_off, nome))
    imgs_serpentina.append((serp_on, nome))

    # Concatena e salva usando a nova função
    salvar_imagem_concatenada(
        imagens=[imagem, serp_off, serp_on],
        diretorio="out1",
        nome_arquivo=f"{nome.replace(' ', '_')}.png",
        modo="L"
    )

    # Exibe a imagem concatenada (basta descomentar para visualizar)
    #mostrar_imagem_concatenada(
    #    imagens=[imagem, serp_off, serp_on],
    #    modo="L"
    #)

    # Para a imagem colorida
    serp_off_rgb = dithering_difusao_erro_rgb(imagem_rgb.copy(), mascara, serpentina=False)
    serp_on_rgb = dithering_difusao_erro_rgb(imagem_rgb.copy(), mascara, serpentina=True)
    salvar_imagem_concatenada(
        imagens=[imagem_rgb, serp_off_rgb, serp_on_rgb],
        diretorio="out1",
        nome_arquivo=f"{nome.replace(' ', '_')}_colorida.png",
        modo="RGB"
    )

    # Adiciona as imagens coloridas à lista de grid
    imgs_sem_serpentina_rgb.append((serp_off_rgb, nome))
    imgs_serpentina_rgb.append((serp_on_rgb, nome))


    # Exibe a imagem colorida concatenada (basta descomentar para visualizar)
    #mostrar_imagem_concatenada(
    #    imagens=[imagem_rgb, serp_off_rgb, serp_on_rgb],
    #    modo="RGB"
    #)

salvar_grid_imagens(
    imagens_nomes=imgs_sem_serpentina, 
    diretorio="out1",
    nome_arquivo="grid_sem_serpentina.png",
    modo="L"
)

salvar_grid_imagens(
    imagens_nomes=imgs_serpentina,
    diretorio="out1",
    nome_arquivo="grid_com_serpentina.png",
    modo="L"
)

salvar_grid_imagens(
    imagens_nomes=imgs_sem_serpentina_rgb, 
    diretorio="out1",
    nome_arquivo="grid_sem_serpentina_colorida.png",
    modo="RGB"
)

salvar_grid_imagens(
    imagens_nomes=imgs_serpentina_rgb, 
    diretorio="out1",
    nome_arquivo="grid_com_serpentina_colorida.png",
    modo="RGB"
)