# 🤖 Automação ComprasNet - CAPTCHA Solver Free

Sistema automatizado para download de licitações do portal ComprasNet com resolução inteligente de CAPTCHAs usando OCR.

## 🎯 **O que faz este sistema?**

- ✅ **Download automático** de licitações do ComprasNet
- 🔍 **Resolução inteligente de CAPTCHAs** sem intervenção manual
- 📊 **Processamento sequencial** com controle de progresso
- 🐳 **Docker ready** para deploy em qualquer ambiente
- 📁 **Organização automática** de arquivos com sistema UUID

## ⚡ **Execução Rápida**

### Docker (Recomendado)
```bash
# Construir e executar
docker-compose up --build

# Ou usar o script de gerenciamento
.\docker-manager.ps1 build-and-run
```

### Execução Local
```bash
# Instalar dependências
pip install -r requirements.txt

# Executar sistema
python comprasnet_download_automatico.py
```

## 🛠️ **Tecnologias Utilizadas**

- **Python 3.11+** - Linguagem principal
- **Selenium** - Automação web
- **Tesseract OCR** - Reconhecimento de texto
- **OpenCV** - Processamento de imagens
- **Docker** - Containerização
- **Machine Learning** - Aprendizado adaptativo de CAPTCHAs

## 📋 **Estrutura do Projeto**

```
📦 auto-comprasnet-tic-open/
├── 🚀 comprasnet_download_automatico.py    # Sistema principal
├── 🔍 captcha_resolver_v2_clean.py         # Resolver de CAPTCHA avançado
├── ⚙️ heuristica_*.py                      # Módulos de heurística
├── 🐳 Dockerfile                           # Container Docker
├── 📋 docker-compose.yml                   # Orquestração
├── 📁 downloads_licitacoes/                # Arquivos baixados
├── 🖼️ captcha_estudos/                     # Debug de CAPTCHAs
└── 📊 models/                              # Modelos ML treinados
```

## 🎛️ **Configuração**

### Arquivos de Configuração
- `config_downloads.json` - URLs e parâmetros de download
- `config_otimizado_98.json` - Configurações de OCR otimizadas
- `captchas_treinados.json` - Base de conhecimento de CAPTCHAs

### Variáveis de Ambiente (Docker)
```env
DISPLAY=:99                    # Para modo headless
PYTHONUNBUFFERED=1            # Output em tempo real
```

## 🔧 **Funcionalidades Principais**

### 🤖 **Resolução de CAPTCHA**
- **Técnica OpenCV Avançada**: Resize 2x + GaussianBlur + Threshold adaptativo
- **Multiple PSM**: PSM 7 (linha única) + PSM 6 (recomendação Gemini)
- **Aprendizado Automático**: Sistema aprende novos CAPTCHAs
- **Fallback Inteligente**: 4 estratégias de processamento

### 📥 **Download Inteligente**
- **Detecção UUID**: Renomeia arquivos automaticamente
- **Processamento Sequencial**: Evita sobrecarga do servidor
- **Retry Automático**: Reprocessa falhas automaticamente
- **Progresso Persistente**: Continua de onde parou

### 🐛 **Debug e Monitoramento**
- **Screenshots Automáticos**: Captura processo completo
- **Logs Detalhados**: Rastreamento de todas as operações
- **Métricas de Performance**: Taxa de sucesso por estratégia
- **Debug Visual**: Salva imagens processadas para análise

## 📊 **Performance**

- **Taxa de Sucesso**: ~98% na resolução de CAPTCHAs
- **Velocidade**: ~30-45 segundos por licitação
- **Estratégias**: 4 técnicas de processamento OCR
- **Aprendizado**: Melhora automaticamente com uso

## 🔄 **Fluxo de Execução**

1. **Carrega configurações** e URLs alvo
2. **Acessa portal** ComprasNet via Selenium
3. **Detecta CAPTCHA** e aplica técnicas de processamento
4. **Resolve automaticamente** usando OCR + ML
5. **Baixa arquivos** e organiza com UUID
6. **Atualiza progresso** e continua próximo item

## 🚨 **Requisitos do Sistema**

### Local
- Python 3.11+
- Tesseract OCR instalado
- Chrome/Chromium browser
- 4GB RAM mínimo

### Docker
- Docker Engine 20.10+
- Docker Compose 2.0+
- 2GB RAM disponível

## 🤝 **Contribuindo**

1. Fork o projeto
2. Crie sua feature branch
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 **Licença**

Este projeto é open source e está disponível sob a licença MIT.

## 🆘 **Suporte**

- 📧 **Issues**: Use o GitHub Issues para reportar problemas
- 📚 **Wiki**: Documentação completa no repositório
- 🔧 **Debug**: Ative logs detalhados para troubleshooting

---

> **💡 Dica**: Para melhor performance, use o modo Docker que já vem com todas as dependências configuradas!