#!/bin/bash

# Verifica se o ambiente virtual existe
if [ ! -d "venv" ]; then
    echo "Ambiente virtual não encontrado. Criando um novo..."
    python3 -m venv venv
    echo "Ambiente virtual criado com sucesso!"
fi

# Ativar o ambiente virtual se não estiver ativado
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Ativando o ambiente virtual..."
    source venv/bin/activate
else
    echo "O ambiente virtual já está ativado."
fi

# Instalar as dependências do requirements.txt
echo "Instalando dependências..."
pip install --upgrade pip
pip install -r requirements.txt

# Executar o script q1.py
echo "Executando q1.py..."
python q1.py

# Verificar se q1.py foi executado com sucesso
if [ $? -eq 0 ]; then
    echo "q1.py executado com sucesso!"
else
    echo "Erro ao executar q1.py."
    deactivate
    exit 1
fi

# Executar o script q2.py
echo "Executando q2.py..."
python q2.py

# Verificar se q2.py foi executado com sucesso
if [ $? -eq 0 ]; then
    echo "q2.py executado com sucesso!"
else
    echo "Erro ao executar q2.py."
    deactivate
    exit 1
fi

echo "Todos os scripts foram executados com sucesso!"