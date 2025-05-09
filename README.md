# README - Trabalho 2: Processamento Digital de Imagens

## Universidade Estadual de Campinas  
**Instituto de Computação**  
**Disciplina:** Introdução ao Processamento Digital de Imagens (MC920 / MO443)  
**Professor:** Hélio Pedrini  

---

## **Descrição do Trabalho**

Este trabalho tem como objetivo explorar técnicas de **realce de imagens** por meio de duas abordagens principais:  
1. **Técnicas de Meios-Tons com Difusão de Erro**  
2. **Filtragem no Domínio de Frequência**  

O trabalho foi desenvolvido em Python e utiliza bibliotecas como `numpy`, `Pillow` e `matplotlib` para manipulação e visualização de imagens. A entrega inclui o código-fonte, resultados gerados e um relatório técnico detalhado.

---

## **Estrutura do Projeto**

O projeto está organizado da seguinte forma:

```plaintext
processamento_imagens2/ 
    ├── imgs/                      # Diretório com imagens de entrada 
    │   ├── img1-mono.png          # Imagem monocromática para meios-tons 
    │   ├── img1-colored.png       # Imagem colorida para meios-tons 
    │   ├── img2-high.png         # Imagem para filtragem no domínio de frequência 
    ├── out1/                      # Resultados da técnica de meios-tons 
    │   ├── imgs/                  # Imagens geradas 
    │   ├── grids/                 # Grids de imagens 
    │   └── hists/                 # Histogramas gerados 
    ├── out2/                      # Resultados da filtragem no domínio de frequência 
    │   ├── imgs/                  # Imagens geradas 
    │   ├── grids/                 # Grids de imagens 
    │   └── hists/                 # Histogramas gerados 
    ├── mascaras1.py               # Máscaras para difusão de erro 
    ├── q1.py                      # Código para meios-tons 
    ├── q2.py                      # Código para filtragem no domínio de frequência 
    ├── utils.py                   # Funções utilitárias 
    ├── run_all.sh                 # Script para executar o projeto 
    └── README.md                  # Este arquivo
```

---

## **Requisitos**

### **Dependências**
Certifique-se de ter as seguintes bibliotecas instaladas:

- `numpy`
- `Pillow`
- `matplotlib`
- `scipy`

### **Ambiente**
- O código foi desenvolvido e testado em **Python 3.8+**.
- O ambiente de execução é **Linux**, mas o código também funciona em sistemas Windows.

Para instalar as dependências, execute:

```bash
pip install -r requirements.txt
```

---

## **Execução**

### 1. **Configuração do Ambiente**
Antes de executar o projeto, certifique-se de que o ambiente virtual está configurado. Use o script `run_all.sh` para configurar e executar automaticamente:

```bash
bash run_all.sh
```

### 2. **Execução Manual**
Se preferir executar manualmente:

#### Técnicas de Meios-Tons:
```bash
python q1.py
```

#### Filtragem no Domínio de Frequência:
```bash
python q2.py
```

Os resultados serão salvos nos diretórios `out1/` e `out2/`.

---

## **Descrição das Técnicas Implementadas**

### 1. **Técnicas de Meios-Tons com Difusão de Erro**

**Objetivo**  
Reduzir a quantidade de cores de uma imagem (quantização) mantendo uma boa percepção visual. A técnica utiliza difusão de erro para distribuir a diferença entre o valor exato de um pixel e seu valor aproximado para os pixels adjacentes.

**Máscaras Implementadas**  
As seguintes máscaras foram utilizadas para difusão de erro:

- Floyd-Steinberg
- Stevenson-Arce
- Burkes
- Sierra
- Stucki
- Jarvis-Judice-Ninke

**Varredura**  
Duas varreduras foram implementadas:

- **Linear:** Da esquerda para a direita.
- **Serpentina:** Alterna a direção da varredura a cada linha para evitar padrões indesejados.

**Resultados**  
Imagens monocromáticas e coloridas foram processadas.  
Grids de imagens e histogramas foram gerados para comparação dos resultados.

**Exemplo de Uso**  
A função principal para meios-tons é `dithering_difusao_erro` (para imagens monocromáticas) e `dithering_difusao_erro_rgb` (para imagens coloridas).  
Os resultados são salvos em `out1/`.

### 2. **Filtragem no Domínio de Frequência**

**Objetivo**  
Aplicar a Transformada Rápida de Fourier (FFT) para manipular imagens no domínio de frequência, permitindo operações como:

- **Filtragem:** Passa-baixa, passa-alta, passa-faixa e rejeita-faixa.
- **Compressão:** Redução do tamanho da imagem eliminando coeficientes de baixa magnitude.

**Passos Implementados**  
- **Transformada de Fourier:** Converte a imagem do domínio espacial para o domínio de frequência.
- **Centralização do Espectro:** Translada a componente de frequência zero para o centro.
- **Criação de Máscaras:** Máscaras circulares para os diferentes filtros.
- **Filtragem:** Multiplicação do espectro pela máscara.
- **Transformada Inversa:** Reconstrói a imagem filtrada no domínio espacial.
- **Compressão:** Remove coeficientes de baixa magnitude para reduzir o tamanho da imagem.

**Filtros Implementados**  
- **Passa-Baixa:** Atenua altas frequências, suavizando a imagem.
- **Passa-Alta:** Atenua baixas frequências, destacando bordas.
- **Passa-Faixa:** Preserva frequências dentro de um intervalo.
- **Rejeita-Faixa:** Atenua frequências dentro de um intervalo.

**Resultados**  
Espectros de Fourier e imagens filtradas para cada tipo de filtro.  
Imagens comprimidas e seus histogramas.

**Exemplo de Uso**  
A função principal para criar filtros é `criar_filtro`.  
Os resultados são salvos em `out2/`.

---

## **Resultados**

### 1. **Meios-Tons**
Imagens processadas com diferentes máscaras e varreduras.  
Grids comparativos e histogramas foram gerados para análise.

### 2. **Filtragem no Domínio de Frequência**
Espectros de Fourier e imagens filtradas para cada tipo de filtro.  
Imagens comprimidas e seus histogramas.

---

## **Relatório Técnico**

### **Considerações**
- A implementação foi feita com foco na clareza e modularidade do código.
- As máscaras de difusão de erro foram implementadas conforme especificado no enunciado.
- Para filtragem no domínio de frequência, diferentes raios de máscaras foram testados.

### **Limitações**
- O desempenho pode ser impactado para imagens de alta resolução devido ao uso intensivo de operações matriciais.
- A compressão no domínio de frequência pode gerar artefatos visuais dependendo do limiar escolhido.

### **Testes**
- Foram utilizados exemplos de imagens monocromáticas e coloridas disponíveis no diretório `imgs/`.
- Os resultados foram validados visualmente e por meio de histogramas.

---

## **Conclusão**

Este trabalho explorou técnicas fundamentais de processamento digital de imagens, abordando tanto o domínio espacial (meios-tons) quanto o domínio de frequência (filtragem e compressão). Os resultados demonstram a eficácia das técnicas implementadas e sua aplicabilidade em diferentes cenários.

---

## **Data de Entrega:** 28/04/2025  
**Desenvolvido por:** J. Eduardo
