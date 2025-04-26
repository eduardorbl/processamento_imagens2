import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from utils import create_output_folder

# Criação da pasta de saída
create_output_folder("out2")

# (i) Abrir imagem e converter para escala de cinza
imagem = np.array(Image.open("imgs/img2-high.png").convert('L'), dtype=np.float32)

# (ii) Transformada de Fourier
f = fft2(imagem)

# (iii) Centralização do espectro
fshift = fftshift(f)

# (iv) Magnitude do espectro (usando log para melhor visualização)
magnitude_spectrum = np.log(1 + np.abs(fshift))

# (v) Reconstrução com inversa de Fourier
f_ishift = ifftshift(fshift)
imagem_reconstruida = np.abs(ifft2(f_ishift))

# Exibição em grid
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Imagem original
axs[0].imshow(imagem, cmap='gray', vmin=0, vmax=255)
axs[0].set_title('Imagem Original')
axs[0].axis('off')

# Espectro de Fourier
axs[1].imshow(magnitude_spectrum, cmap='gray', vmin=np.min(magnitude_spectrum), vmax=np.max(magnitude_spectrum))
axs[1].set_title('Espectro de Fourier (Magnitude)')
axs[1].axis('off')

# Imagem reconstruída
axs[2].imshow(imagem_reconstruida, cmap='gray', vmin=0, vmax=255)
axs[2].set_title('Imagem Reconstruída (Após IFFT)')
axs[2].axis('off')

# Salvar imagens separadamente
Image.fromarray(imagem.astype(np.uint8)).save("out2/imgs/imagem_original.png")
Image.fromarray(imagem_reconstruida.astype(np.uint8)).save("out2/imgs/imagem_reconstruida.png")

# Salvar o espectro de Fourier
magnitude_spectrum_normalized = 255 * (magnitude_spectrum - np.min(magnitude_spectrum)) / (np.max(magnitude_spectrum) - np.min(magnitude_spectrum))
Image.fromarray(magnitude_spectrum_normalized.astype(np.uint8)).save("out2/imgs/espectro_fourier.png")

# Salvar o grid de imagens
fig.savefig("out2/grids/grid_imagens.png", bbox_inches='tight')

plt.tight_layout()
plt.show()

# Função para criar o filtro
def criar_filtro(formato, tipo_filtro, raio1, raio2=None):
    linhas, colunas = formato
    centro_linhas, centro_colunas = linhas // 2, colunas // 2
    Y, X = np.ogrid[:linhas, :colunas]
    distancia = np.sqrt((X - centro_colunas)**2 + (Y - centro_linhas)**2)

    if tipo_filtro == 'passa-baixa':
        return distancia <= raio1
    elif tipo_filtro == 'passa-alta':
        return distancia > raio1
    elif tipo_filtro == 'passa-faixa':
        return (distancia >= raio1) & (distancia <= raio2)
    elif tipo_filtro == 'rejeita-faixa':
        return ~((distancia >= raio1) & (distancia <= raio2))
    else:
        raise ValueError("Tipo de filtro inválido.")

# Criação das máscaras
mascara_pb = criar_filtro(imagem.shape, 'passa-baixa', raio1=40)
mascara_pa = criar_filtro(imagem.shape, 'passa-alta', raio1=40)
mascara_pf = criar_filtro(imagem.shape, 'passa-faixa', raio1=40, raio2=60)
mascara_rf = criar_filtro(imagem.shape, 'rejeita-faixa', raio1=40, raio2=60)

# Aplicação das máscaras no espectro
espectro_pb = np.log(1 + np.abs(fshift * mascara_pb))
espectro_pa = np.log(1 + np.abs(fshift * mascara_pa))
espectro_pf = np.log(1 + np.abs(fshift * mascara_pf))
espectro_rf = np.log(1 + np.abs(fshift * mascara_rf))

# Determinação de vmin e vmax globais para normalização
vmin_global = min(espectro_pb.min(), espectro_pa.min(), espectro_pf.min(), espectro_rf.min())
vmax_global = max(espectro_pb.max(), espectro_pa.max(), espectro_pf.max(), espectro_rf.max())

# Plotagem dos espectros filtrados
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(espectro_pb, cmap='gray', vmin=vmin_global, vmax=vmax_global)
axs[0].set_title('Espectro Passa-Baixa')
axs[0].axis('off')

axs[1].imshow(espectro_pa, cmap='gray', vmin=vmin_global, vmax=vmax_global)
axs[1].set_title('Espectro Passa-Alta')
axs[1].axis('off')

axs[2].imshow(espectro_pf, cmap='gray', vmin=vmin_global, vmax=vmax_global)
axs[2].set_title('Espectro Passa-Faixa')
axs[2].axis('off')

axs[3].imshow(espectro_rf, cmap='gray', vmin=vmin_global, vmax=vmax_global)
axs[3].set_title('Espectro Rejeita-Faixa')
axs[3].axis('off')

# Salvamento das imagens espectrais separadamente
espectro_pb_normalized = 255 * (espectro_pb - vmin_global) / (vmax_global - vmin_global)
espectro_pb_pil = Image.fromarray(espectro_pb_normalized.astype(np.uint8))
espectro_pb_pil.save("out2/imgs/espectro_passa_baixa.png")

espectro_pa_normalized = 255 * (espectro_pa - vmin_global) / (vmax_global - vmin_global)
espectro_pa_pil = Image.fromarray(espectro_pa_normalized.astype(np.uint8))
espectro_pa_pil.save("out2/imgs/espectro_passa_alta.png")

espectro_pf_normalized = 255 * (espectro_pf - vmin_global) / (vmax_global - vmin_global)
espectro_pf_pil = Image.fromarray(espectro_pf_normalized.astype(np.uint8))
espectro_pf_pil.save("out2/imgs/espectro_passa_faixa.png")

espectro_rf_normalized = 255 * (espectro_rf - vmin_global) / (vmax_global - vmin_global)
espectro_rf_pil = Image.fromarray(espectro_rf_normalized.astype(np.uint8))
espectro_rf_pil.save("out2/imgs/espectro_rejeita_faixa.png")

# Salvamento do grid de espectros
fig.savefig("out2/grids/grid_espectros.png", bbox_inches='tight')

plt.tight_layout()
plt.show()

# Aplicar as máscaras diretamente no fshift (sem o log agora, para reconstruir corretamente)
fshift_pb = fshift * mascara_pb
fshift_pa = fshift * mascara_pa
fshift_pf = fshift * mascara_pf
fshift_rf = fshift * mascara_rf

# Voltar do domínio da frequência para o espacial (imagem)
img_pb = np.abs(ifft2(ifftshift(fshift_pb)))
img_pa = np.abs(ifft2(ifftshift(fshift_pa)))
img_pf = np.abs(ifft2(ifftshift(fshift_pf)))
img_rf = np.abs(ifft2(ifftshift(fshift_rf)))

# Mostrar as imagens filtradas
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(img_pb, cmap='gray', vmin=0, vmax=255)
axs[0].set_title('Imagem Passa-Baixa')
axs[0].axis('off')

axs[1].imshow(img_pa, cmap='gray', vmin=0, vmax=255)
axs[1].set_title('Imagem Passa-Alta')
axs[1].axis('off')

axs[2].imshow(img_pf, cmap='gray', vmin=0, vmax=255)
axs[2].set_title('Imagem Passa-Faixa')
axs[2].axis('off')

axs[3].imshow(img_rf, cmap='gray', vmin=0, vmax=255)
axs[3].set_title('Imagem Rejeita-Faixa')
axs[3].axis('off')

# Salva as imagens filtradas separadamente
img_pb_pil = Image.fromarray(img_pb.astype(np.uint8))
img_pb_pil.save("out2/imgs/imagem_passa_baixa.png")

img_pa_pil = Image.fromarray(img_pa.astype(np.uint8))
img_pa_pil.save("out2/imgs/imagem_passa_alta.png")

img_pf_pil = Image.fromarray(img_pf.astype(np.uint8))
img_pf_pil.save("out2/imgs/imagem_passa_faixa.png")

img_rf_pil = Image.fromarray(img_rf.astype(np.uint8))
img_rf_pil.save("out2/imgs/imagem_rejeita_faixa.png")

# Salva o grid de imagens filtradas
fig.savefig("out2/grids/grid_imagens_filtradas.png", bbox_inches='tight')

plt.tight_layout()
plt.show()

# Função para comprimir a imagem no domínio da frequência com base em uma porcentagem
def comprimir_imagem_porcentagem(imagem, porcentagem):
    # Passo 1: Transformada de Fourier
    f = fft2(imagem)
    fshift = fftshift(f)  # Centralização do espectro
    
    # Passo 2: Calculando a magnitude e o limiar com base na porcentagem
    magnitude = np.abs(fshift)
    limiar = np.max(magnitude) * (porcentagem / 100)
    
    # Passo 3: Zerar coeficientes abaixo do limiar
    mascara = magnitude > limiar
    fshift[~mascara] = 0  # Zera os coeficientes com magnitude abaixo do limiar
    
    # Passo 4: Reconstrução da imagem no domínio espacial (após a compressão)
    imagem_comprimida = np.abs(ifft2(ifftshift(fshift)))
    
    return imagem_comprimida

# Função para salvar histogramas
def salvar_histograma(imagem, titulo, caminho, cor):
    fig, ax = plt.subplots()
    ax.hist(imagem.flatten(), bins=256, range=(0, 255), color=cor, alpha=0.7)
    ax.set_title(titulo)
    ax.set_xlabel('Intensidade')
    ax.set_ylabel('Frequência')
    fig.savefig(caminho, bbox_inches='tight')
    plt.close(fig)

# Comprimir a imagem
imagem_comprimida = comprimir_imagem_porcentagem(imagem, 0.1)

# Salvar a imagem comprimida
imagem_comprimida_pil = Image.fromarray(imagem_comprimida.astype(np.uint8))
imagem_comprimida_pil.save("out2/imgs/imagem_comprimida.png")

# Salvar histogramas
salvar_histograma(imagem, 'Histograma da Imagem Original', "out2/hists/histograma_imagem_original.png", 'blue')
salvar_histograma(imagem_comprimida, 'Histograma da Imagem Comprimida', "out2/hists/histograma_imagem_comprimida.png", 'green')

# Plotar imagens: original, comprimida, e seus histogramas
fig, axs = plt.subplots(1, 3, figsize=(20, 5))

# Imagem comprimida
axs[0].imshow(imagem_comprimida, cmap='gray', vmin=0, vmax=255)
axs[0].set_title('Imagem Comprimida (FFT)')
axs[0].axis('off')

# Histograma da imagem original
axs[1].hist(imagem.flatten(), bins=256, range=(0, 255), color='blue', alpha=0.7)
axs[1].set_title('Histograma da Imagem Original')
axs[1].set_xlabel('Intensidade')
axs[1].set_ylabel('Frequência')

# Histograma da imagem comprimida
axs[2].hist(imagem_comprimida.flatten(), bins=256, range=(0, 255), color='green', alpha=0.7)
axs[2].set_title('Histograma da Imagem Comprimida')
axs[2].set_xlabel('Intensidade')
axs[2].set_ylabel('Frequência')

# Salva o grid de histogramas
fig.savefig("out2/grids/grid_histogramas.png", bbox_inches='tight')

plt.tight_layout()
plt.show()
