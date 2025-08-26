# Dockerfile para Sistema de Download Automático do ComprasNet
# Base: Python 3.11 com suporte a GUI e OCR

FROM python:3.11-slim

# Metadados
LABEL maintainer="TJES PROJ AUTOMAÇÃO COMPRASNET"
LABEL description="Sistema automatizado para download de licitações do ComprasNet"
LABEL version="1.0"

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema necessárias para OpenCV, Tesseract e Playwright
RUN apt-get update && apt-get install -y \
    # Dependências básicas
    wget \
    curl \
    unzip \
    git \
    # Dependências do OpenCV
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    # Dependências do Tesseract OCR
    tesseract-ocr \
    tesseract-ocr-por \
    libtesseract-dev \
    libleptonica-dev \
    # Dependências do Playwright (navegador)
    libnss3 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libgtk-3-0 \
    libgbm1 \
    libasound2 \
    # Ferramentas de rede
    net-tools \
    iputils-ping \
    # Limpeza
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configurar timezone para Brasil
ENV TZ=America/Sao_Paulo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Configurar variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DISPLAY=:99
ENV PLAYWRIGHT_BROWSERS_PATH=/app/browsers

# Copiar requirements.txt primeiro (para cache do Docker)
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Instalar browsers do Playwright
RUN playwright install chromium && \
    playwright install-deps chromium

# Copiar arquivos do projeto
COPY . .

# Criar diretórios necessários
RUN mkdir -p \
    captchas_limpos \
    captchas_processados \
    debug_captcha_rgb \
    debug_sistema/captchas_resolvidos \
    debug_sistema/captchas_falhas \
    debug_sistema/opencv_processamento \
    debug_sistema/paginas_licitacao \
    debug_sistema/popups_digitacao \
    debug_sistema/screenshots_navegador \
    downloads_licitacoes \
    heuristica_resultados \
    models \
    models_cnn_ocr \
    temp_user_data

# Dar permissões de execução aos scripts
RUN chmod +x *.py

# Expor porta (caso precise de interface web futuramente)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Comando padrão
CMD ["python", "comprasnet_download_automatico.py"]

# ======================================================================
# INSTRUÇÕES DE USO:
# ======================================================================
#
# 1. BUILD DA IMAGEM:
#    docker build -t comprasnet-downloader .
#
# 2. EXECUTAR CONTAINER:
#    docker run -d --name comprasnet \
#      -v $(pwd)/downloads_licitacoes:/app/downloads_licitacoes \
#      -v $(pwd)/debug_sistema:/app/debug_sistema \
#      comprasnet-downloader
#
# 3. EXECUTAR COM DISPLAY (se precisar de GUI):
#    docker run -d --name comprasnet \
#      -v $(pwd)/downloads_licitacoes:/app/downloads_licitacoes \
#      -v /tmp/.X11-unix:/tmp/.X11-unix \
#      -e DISPLAY=$DISPLAY \
#      comprasnet-downloader
#
# 4. LOGS:
#    docker logs -f comprasnet
#
# 5. PARAR/INICIAR:
#    docker stop comprasnet
#    docker start comprasnet
#
# 6. REMOVER:
#    docker rm -f comprasnet
#    docker rmi comprasnet-downloader
#
# ======================================================================
