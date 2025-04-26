import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

# (i) Abrir imagem e converter para escala de cinza
imagem = np.array(Image.open("imgs/img2-high.png").convert('L'), dtype=np.float32)

# (ii) Transformada de Fourier
f = fft2(imagem)

# (iii) Centralização do espectro
fshift = fftshift(f)

# Magnitude do espectro (usando log para melhor visualização)
magnitude_spectrum = np.log(1 + np.abs(fshift))

# (vi) Reconstrução com inversa de Fourier
f_ishift = ifftshift(fshift)
imagem_reconstruida = np.abs(ifft2(f_ishift))

# Exibição em grid
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(imagem, cmap='gray', vmin=0, vmax=255)
axs[0].set_title('Imagem Original')
axs[0].axis('off')

axs[1].imshow(magnitude_spectrum, cmap='gray', vmin=np.min(magnitude_spectrum), vmax=np.max(magnitude_spectrum))
axs[1].set_title('Espectro de Fourier (Magnitude)')
axs[1].axis('off')

axs[2].imshow(imagem_reconstruida, cmap='gray', vmin=0, vmax=255)
axs[2].set_title('Imagem Reconstruída (Após IFFT)')
axs[2].axis('off')

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

# Primeiro, aplica todas as máscaras sem plotar
espectro_pb = np.log(1 + np.abs(fshift * mascara_pb))
espectro_pa = np.log(1 + np.abs(fshift * mascara_pa))
espectro_pf = np.log(1 + np.abs(fshift * mascara_pf))
espectro_rf = np.log(1 + np.abs(fshift * mascara_rf))

# Calcula vmin e vmax globais (pegando o valor mínimo e máximo entre todas imagens)
vmin_global = min(espectro_pb.min(), espectro_pa.min(), espectro_pf.min(), espectro_rf.min())
vmax_global = max(espectro_pb.max(), espectro_pa.max(), espectro_pf.max(), espectro_rf.max())

# Agora plota usando o mesmo vmin e vmax para todos
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

plt.tight_layout()
plt.show()
