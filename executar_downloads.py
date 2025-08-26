#!/usr/bin/env python3
"""
🎯 EXECUTADOR DE DOWNLOADS AUTOMÁTICOS
=====================================
Script para executar downloads automáticos de licitações 2023-2025
"""

import asyncio
import logging
import sys
from pathlib import Path

# Configurar logging mais detalhado
logging.basicConfig(
    level=logging.INFO,
    format='🎯 [%(asctime)s] %(name)s - %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('downloads_automaticos.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

def verificar_dependencias():
    """Verifica se todas as dependências estão disponíveis"""
    try:
        import playwright
        from playwright.async_api import async_playwright
        logger.info("✅ Playwright disponível")
    except ImportError:
        logger.error("❌ Playwright não instalado. Execute: pip install playwright")
        logger.error("❌ E depois: playwright install chromium")
        return False
    
    try:
        import cv2
        import numpy as np
        import easyocr
        import pytesseract
        logger.info("✅ Bibliotecas de processamento de imagem disponíveis")
    except ImportError as e:
        logger.error(f"❌ Biblioteca de imagem não disponível: {e}")
        return False
    
    # Verificar arquivos do sistema heurístico
    arquivos_necessarios = [
        "comprasnet_download_monitor_open.py",
        "heuristica_analisador.py", 
        "heuristica_reconhecedor.py",
        "heuristica_integrador.py",
        "config_otimizado_98.json"
    ]
    
    for arquivo in arquivos_necessarios:
        if not Path(arquivo).exists():
            logger.error(f"❌ Arquivo necessário não encontrado: {arquivo}")
            return False
        logger.info(f"✅ {arquivo}")
    
    return True

def mostrar_configuracao():
    """Mostra a configuração atual do sistema"""
    logger.info("📋 CONFIGURAÇÃO DO SISTEMA")
    logger.info("=" * 50)
    logger.info("🎯 Período: 01/01/2023 até hoje")
    logger.info("🏢 UASG: 925968")
    logger.info("🌍 Estado: Espírito Santo (ES)")
    logger.info("🏙️ Município: 57053")
    logger.info("📦 Grupo Material: 70")
    logger.info("📋 Modalidades: 1,2,3,20,5,99")
    logger.info("📄 Concorrências: 31,32,41,42,49")
    logger.info("🛒 Pregões: 1,2,3,4")
    logger.info("📊 RDC: 1,2,3,4")
    logger.info("💾 Downloads: ./downloads_licitacoes/")
    logger.info("🔒 CAPTCHA: Sistema Heurístico 98%+")
    logger.info("=" * 50)

async def executar_sistema():
    """Executa o sistema de downloads automáticos"""
    try:
        # Importar sistema
        from comprasnet_download_automatico import ComprasNetDownloadAutomatico
        
        # Criar instância
        downloader = ComprasNetDownloadAutomatico()
        
        # Executar downloads
        await downloader.executar_downloads_automaticos()
        
    except ImportError as e:
        logger.error(f"❌ Erro ao importar sistema: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Erro durante execução: {e}")
        return False
    
    return True

async def main():
    """Função principal"""
    logger.info("🚀 INICIANDO SISTEMA DE DOWNLOADS AUTOMÁTICOS")
    logger.info("=" * 60)
    
    # Verificar dependências
    logger.info("🔍 Verificando dependências...")
    if not verificar_dependencias():
        logger.error("❌ Dependências não atendidas. Abortando.")
        return
    
    logger.info("✅ Todas as dependências estão OK!")
    
    # Mostrar configuração
    mostrar_configuracao()
    
    # Confirmar execução
    print("\n🎯 INICIANDO DOWNLOADS AUTOMÁTICOS IMEDIATAMENTE")
    print("📋 O sistema executará automaticamente:")
    print("   • Dividir o período 2023-hoje em intervalos de 15 dias")
    print("   • Acessar cada período automaticamente")
    print("   • Resolver CAPTCHAs com 98%+ de confiança")
    print("   • Fazer download de todas as licitações encontradas")
    print("   • Salvar progresso para retomar se necessário")
    print("\n⚠️  ATENÇÃO: Este processo pode levar várias horas!")
    print("🚀 Iniciando automaticamente em 3 segundos...")
    
    # Aguardar 3 segundos para dar tempo de ler
    import time
    time.sleep(3)
    
    # Executar sistema
    logger.info("🎯 Iniciando execução...")
    sucesso = await executar_sistema()
    
    if sucesso:
        logger.info("🎉 Sistema executado com sucesso!")
    else:
        logger.error("❌ Erro durante execução do sistema")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⏹️ Execução interrompida pelo usuário (Ctrl+C)")
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}")
        sys.exit(1)
