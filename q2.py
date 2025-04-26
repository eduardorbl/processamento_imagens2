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

