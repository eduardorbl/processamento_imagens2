import os
import numpy as np
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

    # Concatena as imagens horizontalmente
    combinado = np.hstack(imagens)

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