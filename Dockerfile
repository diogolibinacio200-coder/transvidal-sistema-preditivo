# =============================================================================
# Sistema Preditivo de Gargalos Portuários — Vidal Transportes
# =============================================================================
FROM python:3.11-slim

# Instala dependências do sistema + Chromium (para Selenium / scraping)
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    libpq-dev \
    gcc \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Variáveis de ambiente para o Chromium rodar em container
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copia e instala dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todos os arquivos do projeto
COPY . .

# Cria pasta onde os modelos treinados serão salvos
RUN mkdir -p modelos_salvos

# Dá permissão de execução ao entrypoint
RUN chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
