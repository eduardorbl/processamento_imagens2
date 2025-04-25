import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from PIL import Image

def create_output_folder(folder_name="output"):
    """
    Cria uma pasta de saída para armazenar os resultados.
    Se a pasta já existir, limpa seu conteúdo.
    Args:
        folder_name (str): Nome da pasta de saída.
    """
    if os.path.exists(folder_name):
        for file in os.listdir(folder_name):
            file_path = os.path.join(folder_name, file)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
    else:
        os.makedirs(folder_name)

def salvar_imagem_concatenada(imagens, diretorio, nome_arquivo, modo="L"):
    """
    Concatena horizontalmente uma lista de imagens e salva o resultado.

    Args:
        imagens (list): Lista de arrays numpy representando as imagens.
        diretorio (str): Nome do diretório onde a imagem será salva.
        nome_arquivo (str): Nome do arquivo de saída.
        modo (str): Modo da imagem ("L" para grayscale ou "RGB" para colorido).
    """
    # Garante que o diretório exista
    os.makedirs(diretorio, exist_ok=True)

    # Define a distância entre as imagens (em pixels)
    distancia = 10

    # Adiciona a distância entre as imagens
    altura_maxima = max(imagem.shape[0] for imagem in imagens)
    largura_total = sum(imagem.shape[1] for imagem in imagens) + distancia * (len(imagens) - 1)

    # Cria um array vazio com a altura máxima e largura total
    # Cria um array vazio com a altura máxima e largura total
    if modo == "RGB":
        combinado = np.zeros((altura_maxima, largura_total, 3), dtype=imagens[0].dtype)
    else:
        combinado = np.zeros((altura_maxima, largura_total), dtype=imagens[0].dtype)


    # Preenche o array com as imagens e a distância entre elas
    x_offset = 0
    for imagem in imagens:
        altura, largura = imagem.shape[:2]
        combinado[:altura, x_offset:x_offset + largura] = imagem
        x_offset += largura + distancia

    # Converte para uint8
    combinado = combinado.astype(np.uint8)

    # Converte para imagem PIL no modo especificado
    if modo == "RGB":
        img_pil = Image.fromarray(combinado, mode="RGB")
    else:  # Default para grayscale
        img_pil = Image.fromarray(combinado, mode="L")

    # Salva a imagem no diretório especificado
    caminho_completo = os.path.join(diretorio, nome_arquivo)
    img_pil.save(caminho_completo)
    print(f"Imagem salva: {caminho_completo}")

def mostrar_imagem_concatenada(imagens, modo="L"):
    """
    Concatena horizontalmente uma lista de imagens e exibe o resultado em uma janela.

    Args:
        imagens (list): Lista de arrays numpy representando as imagens.
        modo (str): Modo da imagem ("L" para grayscale ou "RGB" para colorido).
    """
    # Define a distância entre as imagens (em pixels)
    distancia = 10

    # Adiciona a distância entre as imagens
    altura_maxima = max(imagem.shape[0] for imagem in imagens)
    largura_total = sum(imagem.shape[1] for imagem in imagens) + distancia * (len(imagens) - 1)

    # Cria um array vazio com a altura máxima e largura total, preenchido com branco
    if modo == "RGB":
        combinado = np.ones((altura_maxima, largura_total, 3), dtype=np.uint8) * 255  # Branco para RGB
    else:
        combinado = np.ones((altura_maxima, largura_total), dtype=np.uint8) * 255  # Branco para grayscale

    # Preenche o array com as imagens e a distância entre elas
    x_offset = 0
    for imagem in imagens:
        altura, largura = imagem.shape[:2]
        if modo == "RGB":
            combinado[:altura, x_offset:x_offset + largura, :] = imagem
        else:
            combinado[:altura, x_offset:x_offset + largura] = imagem
        x_offset += largura + distancia

    # Converte para uint8
    combinado = combinado.astype(np.uint8)

    # Exibe a imagem concatenada
    plt.figure(figsize=(10, 5))
    if modo == "RGB":
        plt.imshow(combinado)
    else:  # Default para grayscale
        plt.imshow(combinado, cmap="gray")
    plt.axis("off")
    plt.title("Imagens Concatenadas")
    plt.show()

def salvar_grid_imagens(imagens_nomes, diretorio, nome_arquivo, modo="L"):
    """
    Cria um grid com 3 colunas e quantas linhas forem necessárias e salva as imagens.

    Args:
        imagens_nomes (list): Lista de tuplas no formato (imagem, nome_do_metodo).
        diretorio (str): Nome do diretório onde a imagem será salva.
        nome_arquivo (str): Nome do arquivo de saída.
        modo (str): Modo da imagem ("L" para grayscale ou "RGB" para colorido).
    """
    # Garante que o diretório exista
    os.makedirs(diretorio, exist_ok=True)

    # Configurações do grid
    colunas = 3
    largura_imagem = max(imagem.shape[1] for imagem, _ in imagens_nomes)
    altura_imagem = max(imagem.shape[0] for imagem, _ in imagens_nomes)
    linhas = (len(imagens_nomes) + colunas - 1) // colunas  # Calcula o número de linhas necessárias

    # Altura extra para a legenda
    altura_legenda = 30

    # Cria o canvas para o grid
    largura_total = colunas * largura_imagem
    altura_total = linhas * (altura_imagem + altura_legenda)
    if modo == "RGB":
        grid = np.ones((altura_total, largura_total, 3), dtype=np.uint8) * 255  # Branco para RGB
    else:
        grid = np.ones((altura_total, largura_total), dtype=np.uint8) * 255  # Branco para grayscale

    # Preenche o grid com as imagens e as legendas
    for idx, (imagem, nome_metodo) in enumerate(imagens_nomes):
        linha = idx // colunas
        coluna = idx % colunas
        x_offset = coluna * largura_imagem
        y_offset = linha * (altura_imagem + altura_legenda)

        # Adiciona a imagem ao grid
        if modo == "RGB" and len(imagem.shape) == 2:  # Converte grayscale para RGB
            imagem = np.stack([imagem] * 3, axis=-1)
        elif modo == "L" and len(imagem.shape) == 3:  # Converte RGB para grayscale
            imagem = np.mean(imagem, axis=-1).astype(np.uint8)

        grid[y_offset:y_offset + altura_imagem, x_offset:x_offset + largura_imagem] = imagem

        # Adiciona a legenda
        img_pil = Image.fromarray(grid)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.load_default()  # Usa a fonte padrão
        text_x = x_offset + largura_imagem // 2 - len(nome_metodo) * 3  # Centraliza o texto
        text_y = y_offset + altura_imagem + 5
        draw.text((text_x, text_y), nome_metodo, fill=(0 if modo == "L" else (0, 0, 0)), font=font)
        grid = np.array(img_pil)

    # Salva o grid como imagem
    img_pil = Image.fromarray(grid, mode="RGB" if modo == "RGB" else "L")
    caminho_completo = os.path.join(diretorio, nome_arquivo)
    img_pil.save(caminho_completo)
    print(f"Grid de imagens salvo: {caminho_completo}")