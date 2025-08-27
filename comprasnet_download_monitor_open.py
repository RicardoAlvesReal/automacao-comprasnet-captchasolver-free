#!/usr/bin/env python3
"""
🎯 ComprasNet Download Monitor com HEURÍSTICA DUAL - 98%+ de Confiança
====================================================================
Sistema avançado de resolução automática de CAPTCHA com análise inteligente
e reconhecimento adaptativo para atingir 98%+ de confiança
"""

import asyncio
import logging
import time
import random
import os
import cv2
import numpy as np
import easyocr
import pytesseract
import re
import json
import pickle
from datetime import datetime
from playwright.async_api import async_playwright
from pathlib import Path

# 🎯 IMPORTS DA HEURÍSTICA DUAL
try:
    from heuristica_analisador import AnalisadorInteligente, AnaliseCompleta, EstrategiaProcessamento
    from heuristica_reconhecedor import ReconhecedorAdaptativo, DecisaoFinal, MetodoOCR
    from heuristica_integrador import IntegradorHeuristico, ResultadoHeuristicaDual
    HEURISTICA_DISPONIVEL = True
    logger_heuristica = logging.getLogger('HEURISTICA_DUAL')
    logger_heuristica.info("✅ Heurística Dual carregada com sucesso!")
except ImportError as e:
    HEURISTICA_DISPONIVEL = False
    logger.warning(f"⚠️ Heurística Dual não disponível: {e}")

# Imports para CNN customizada
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    CNN_AVAILABLE = True
    logger_cnn = logging.getLogger('CNN_OCR')
    logger_cnn.info("✅ TensorFlow disponível para CNN customizada")
except ImportError as e:
    CNN_AVAILABLE = False
    logger.warning(f"⚠️ TensorFlow não disponível: {e}")


def convert_numpy_types(obj):
    """Converte tipos numpy para tipos Python padrão para serialização JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Configurar Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configurar logging
logging.basicConfig(level=logging.INFO, format='🤖 [%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class ComprasNetDownloadMonitor:
    def __init__(self):
        self.page = None
        self.browser = None
        self.downloads_dir = Path.home() / "Downloads"
        
        # 🎯 INICIALIZAR HEURÍSTICA DUAL
        if HEURISTICA_DISPONIVEL:
            logger.info("🎯 INICIALIZANDO SISTEMA HEURÍSTICO DUAL...")
            self.integrador_heuristico = IntegradorHeuristico(meta_confianca=0.98)
            self.usar_heuristica = True
            logger.info("✅ Sistema Heurístico Dual pronto para 98%+ de confiança!")
        else:
            logger.warning("⚠️ Heurística não disponível, usando sistema tradicional")
            self.integrador_heuristico = None
            self.usar_heuristica = False
        
        # Pasta para salvar CAPTCHAs para estudos
        self.captcha_estudos_dir = Path("captcha_estudos")
        self.captcha_estudos_dir.mkdir(exist_ok=True)
        
        # Pasta para debug RGB
        self.debug_rgb_dir = Path("debug_captcha_rgb")
        self.debug_rgb_dir.mkdir(exist_ok=True)
        
        # NOVO: Pastas para CAPTCHAs limpos e processados
        captchas_limpos_dir = Path("captchas_limpos")
        captchas_limpos_dir.mkdir(exist_ok=True)
        captchas_processados_dir = Path("captchas_processados")
        captchas_processados_dir.mkdir(exist_ok=True)
        
        # Log das melhorias implementadas
        logger.info("🚀 SISTEMA CAPTCHA OTIMIZADO ATIVADO!")
        logger.info("📁 Estrutura de pastas criada:")
        logger.info(f"   • {captchas_limpos_dir} - Imagens originais e limpas com metadados")
        logger.info(f"   • {captchas_processados_dir} - Imagens finais para OCR e resultados")
        logger.info(f"   • {self.debug_rgb_dir} - Imagens de debug RGB")
        logger.info("🎯 Método otimizado: Threshold ≤85")
        logger.info("✨ Salvamento automático de todas as etapas do processamento")
        
        # Inicializar EasyOCR
        logger.info("🤖 Inicializando EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=False)
        
        # Inicializar sistema de Deep Learning (prioridade máxima)
        logger.info("🧠 Inicializando sistema de Deep Learning...")
        try:
            from captcha_dl_analyzer import CaptchaDLAnalyzer
            self.ml_analyzer = CaptchaDLAnalyzer("models_dl")
            self.ml_analyzer.carregar_modelos()
            
            # Treinar DL se necessário
            if not self.ml_analyzer.complexity_model or not self.ml_analyzer.confidence_model:
                logger.info("🎓 Treinando modelos DL com dados existentes...")
                self.ml_analyzer.treinar_modelos()
                
            logger.info("✅ Sistema Deep Learning inicializado")
            self.ml_tipo = "deep_learning"
            
        except Exception as e:
            logger.warning(f"⚠️ Deep Learning falhou: {e}")
            logger.info("🔄 Tentando ML simplificado...")
            
            try:
                from captcha_ml_simple import SimpleCaptchaMLAnalyzer
                self.ml_analyzer = SimpleCaptchaMLAnalyzer("captcha_training.db")
                logger.info("✅ Sistema ML simplificado inicializado")
                self.ml_tipo = "simple_ml"
                
            except Exception as e2:
                logger.error(f"❌ Erro crítico no ML: {e2}")
                logger.warning("⚠️ Sistema funcionará sem ML (modo tradicional)")
                self.ml_analyzer = None
                self.ml_tipo = "none"
        
        # Análise de CAPTCHAs para treinamento
        self.analisar_captchas_existentes()
        
        # Inicializar sistema CNN OCR customizado
        logger.info("🧠 Inicializando CNN OCR Customizado...")
        try:
            self.cnn_ocr = CNNCaptchaOCR()
            logger.info("✅ CNN OCR Customizado inicializado")
        except Exception as e:
            logger.warning(f"⚠️ CNN OCR falhou: {e}")
            self.cnn_ocr = None

    def _resolver_tesseract_simples(self, captcha_buffer):
        """Método Tesseract simplificado para uso com ML/DL"""
        try:
            import pytesseract
            
            nparr = np.frombuffer(captcha_buffer, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return None
            
            # Preprocessamento básico
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR
            config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            texto = pytesseract.image_to_string(thresh, config=config).strip()
            
            # Limpar resultado
            texto_limpo = ''.join(c for c in texto if c.isalnum())
            
            return texto_limpo if len(texto_limpo) >= 3 else None
            
        except Exception as e:
            logger.debug(f"Erro Tesseract simples: {e}")
            return None
        
    def analisar_captchas_existentes(self):
        """Analisa CAPTCHAs já capturados para otimizar o reconhecimento"""
        try:
            logger.info("🔍 Analisando CAPTCHAs existentes para treinamento...")
            
            captchas_resolvidos = []
            captchas_nao_resolvidos = []
            
            # Listar todos os CAPTCHAs na pasta de estudos
            for arquivo in self.captcha_estudos_dir.glob("captcha_*.png"):
                nome = arquivo.stem
                if "_sem_resposta" in nome:
                    captchas_nao_resolvidos.append(arquivo)
                else:
                    # Extrair resposta do nome do arquivo
                    partes = nome.split("_")
                    if len(partes) >= 4:
                        resposta = partes[-1]  # Última parte é a resposta
                        captchas_resolvidos.append((arquivo, resposta))
            
            logger.info(f"📊 Encontrados: {len(captchas_resolvidos)} resolvidos, {len(captchas_nao_resolvidos)} não resolvidos")
            
            if len(captchas_resolvidos) > 0:
                self.treinar_com_captchas_conhecidos(captchas_resolvidos)
                
            return len(captchas_resolvidos), len(captchas_nao_resolvidos)
            
        except Exception as e:
            logger.error(f"❌ Erro na análise de CAPTCHAs: {e}")
            return 0, 0
    
    def treinar_com_captchas_conhecidos(self, captchas_resolvidos):
        """Treina e otimiza parâmetros baseado em CAPTCHAs com respostas conhecidas"""
        try:
            logger.info(f"🎓 Iniciando treinamento com {len(captchas_resolvidos)} CAPTCHAs conhecidos...")
            
            # Configurações para testar
            configuracoes_teste = [
                {'threshold': 0.1, 'width_ths': 0.7, 'paragraph': False},
                {'threshold': 0.15, 'width_ths': 0.5, 'paragraph': False},
                {'threshold': 0.2, 'width_ths': 0.8, 'paragraph': False},
                {'threshold': 0.25, 'width_ths': 0.6, 'paragraph': False},
            ]
            
            tesseract_configs = [
                r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
                r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
                r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
            ]
            
            resultados = []
            
            # Testar cada configuração
            for config in configuracoes_teste:
                acertos = 0
                for arquivo_captcha, resposta_correta in captchas_resolvidos[:5]:  # Testar apenas 5 para velocidade
                    resultado = self.testar_captcha_com_config(arquivo_captcha, resposta_correta, config)
                    if resultado:
                        acertos += 1
                
                taxa_acerto = acertos / min(5, len(captchas_resolvidos))
                resultados.append((config, taxa_acerto))
                logger.info(f"   📈 Config {config}: {taxa_acerto:.1%} de acerto")
            
            # Testar Tesseract
            for tesseract_config in tesseract_configs:
                acertos = 0
                for arquivo_captcha, resposta_correta in captchas_resolvidos[:5]:
                    resultado = self.testar_captcha_tesseract(arquivo_captcha, resposta_correta, tesseract_config)
                    if resultado:
                        acertos += 1
                
                taxa_acerto = acertos / min(5, len(captchas_resolvidos))
                resultados.append((f"tesseract_{tesseract_config[:20]}", taxa_acerto))
                logger.info(f"   📈 Tesseract {tesseract_config[:20]}: {taxa_acerto:.1%} de acerto")
            
            # Encontrar melhor configuração
            if resultados:
                melhor_config, melhor_taxa = max(resultados, key=lambda x: x[1])
                logger.info(f"🏆 Melhor configuração: {melhor_config} com {melhor_taxa:.1%} de acerto")
                self.melhor_config = melhor_config
            
        except Exception as e:
            logger.error(f"❌ Erro no treinamento: {e}")
    
    async def resolver_captcha_popup(self, popup_page):
        """Resolve o CAPTCHA em um popup de download e retorna o botão de confirmação"""
        try:
            logger.info("4️⃣ Monitor: resolvendo CAPTCHA no popup...")
            
            # 📸 CAPTURA DE TELA DO POPUP PARA DEBUG
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                screenshot_path = f"debug_popup_{timestamp}.png"
                await popup_page.screenshot(path=screenshot_path, full_page=True)
                logger.info(f"📸 Screenshot do popup salva: {screenshot_path}")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao salvar screenshot do popup: {e}")
            
            # Aguardar elemento da imagem do captcha - usando URL específica do ComprasNet
            seletores_captcha = [
                'img[src*="captcha.aspx?opt=image"]',  # URL específica do CAPTCHA visual
                'img[src*="captcha.aspx"]',  # URL específica informada pelo usuário
                'img[src*="captcha.gif"]',   # URL alternativa
                'img[src*="captcha"]',       # Seletor genérico
                'img[alt*="captcha"]',       # Por atributo alt
                'img[alt*="CAPTCHA"]'        # Por atributo alt maiúsculo
            ]
            
            img_el = None
            captcha_url = None
            
            for seletor in seletores_captcha:
                try:
                    img_el = await popup_page.wait_for_selector(seletor, timeout=5000)
                    if img_el:
                        captcha_url = await img_el.get_attribute('src')
                        logger.info(f"🔗 CAPTCHA encontrado com seletor '{seletor}': {captcha_url}")
                        break
                except:
                    continue
            
            if not img_el:
                logger.error("❌ Monitor: imagem CAPTCHA não encontrada no popup")
                return None
                
            # Capturar buffer da imagem
            await asyncio.sleep(1)
            buf = await img_el.screenshot()
            
            # Resolver com heurística dual
            # Converter buffer para numpy array se necessário
            if isinstance(buf, bytes):
                import cv2
                import numpy as np
                nparr = np.frombuffer(buf, np.uint8)
                img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img_array = buf
                
            # Usar métodos OCR do sistema
            metodos_ocr = {
                'tesseract': lambda img: self.analisador_inteligente.ocr_tesseract_melhorado(img),
                'easyocr': lambda img: self.analisador_inteligente.ocr_easyocr_otimizado(img)
            }
            resultado = await self.integrador_heuristico.resolver_captcha_inteligente(img_array, metodos_ocr)
            
            # Se confiança baixa, tentar sistema avançado primeiro, depois DetectorCaracteresZero
            if not resultado or not hasattr(resultado, 'confianca_final') or resultado.confianca_final < 0.30:
                try:
                    # 🎯 PRIMEIRA TENTATIVA: Resolver Avançado por Caractere
                    from captcha_resolver_avancado import CaptchaResolverAvancado
                    resolver_avancado = CaptchaResolverAvancado()
                    resultado_avancado = resolver_avancado.resolver_captcha_completo(img_array)
                    
                    if resultado_avancado.get('sucesso') and len(resultado_avancado.get('texto', '')) >= 6:
                        logger.info(f"🎉 ResolverAvançado: '{resultado_avancado['texto']}' (confiança: {resultado_avancado['confianca']:.2f}, chars: {len(resultado_avancado['texto'])})")
                        
                        # Criar resultado compatível
                        class ResultadoAvancado:
                            def __init__(self, texto, confianca):
                                self.texto_final = texto
                                self.confianca_final = confianca
                                
                        resultado = ResultadoAvancado(resultado_avancado['texto'], resultado_avancado['confianca'])
                        
                    else:
                        # 🔄 FALLBACK: DetectorCaracteresZero
                        logger.info("🔄 Fallback para DetectorCaracteresZero...")
                        from detector_caracteres_zero import DetectorCaracteresZero
                        detector_direto = DetectorCaracteresZero()
                        resultado_direto = detector_direto.processar_captcha_completo(img_array)
                        
                        if resultado_direto and resultado_direto.get('sucesso') and resultado_direto.get('texto_detectado'):
                            # ✅ VALIDAÇÃO: Mínimo 6 caracteres
                            if len(resultado_direto.get('texto_detectado', '')) >= 6:
                                logger.info(f"🎯 DetectorCaracteresZero: '{resultado_direto['texto_detectado']}' ({resultado_direto.get('confianca', 0):.2f})")
                                
                                # Criar resultado compatível
                                class ResultadoDireto:
                                    def __init__(self, texto, confianca):
                                        self.texto_final = texto
                                        self.confianca_final = confianca
                                        
                                resultado = ResultadoDireto(resultado_direto['texto_detectado'], resultado_direto.get('confianca', 0.5))
                            else:
                                logger.warning(f"⚠️ Texto muito curto: '{resultado_direto.get('texto_detectado', '')}' ({len(resultado_direto.get('texto_detectado', ''))} chars, mínimo: 6)")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Erro no sistema avançado/DetectorCaracteresZero: {e}")
            
            # Salvar para estudos
            self.salvar_captcha_para_estudos(
                buf,
                resultado.texto_final if resultado else None,
                metadados_extras={
                    'confianca': resultado.confianca_final if resultado else None,
                    'origem': 'monitor popup',
                    'data_hora': datetime.now().strftime('%d/%m/%Y %H:%M:%S')
                }
            )
            # Verificar confiança mínima - removido limite para permitir todas as tentativas
            if not resultado or not hasattr(resultado, 'confianca_final') or resultado.confianca_final < 0.0:
                logger.warning(f"⚠️ Monitor: confiança baixa ({resultado.confianca_final if resultado and hasattr(resultado, 'confianca_final') else 'N/A'})")
                return None
            
            logger.info(f"✅ CAPTCHA válido detectado: '{resultado.texto_final}' (confiança: {resultado.confianca_final:.2f})")
            # 🔍 BUSCA AVANÇADA DO CAMPO DE CAPTCHA
            logger.info(f"🔍 Iniciando busca avançada por campo de CAPTCHA no popup...")
            
            # Lista completa de seletores para campo de CAPTCHA
            seletores_campo = [
                'input[name="idLetra"]',     # 🎯 CAMPO ESPECÍFICO DO COMPRASNET
                'input[id="idLetra"]',       # 🎯 CAMPO ESPECÍFICO DO COMPRASNET
                'input[name*="captcha"]',
                'input[id*="captcha"]', 
                'input[name*="Captcha"]',
                'input[id*="Captcha"]',
                'input[name*="CAPTCHA"]',
                'input[id*="CAPTCHA"]',
                'input[name*="codigo"]',
                'input[id*="codigo"]',
                'input[name*="validacao"]',
                'input[id*="validacao"]',
                'input[type="text"][maxlength="6"]',  # 🎯 MAXLENGTH ESPECÍFICO
                'input[type="text"]',
                'input[maxlength="5"]',
                'input[maxlength="6"]',
                'input[size="5"]',
                'input[size="6"]',
                'input',
                'textarea'
            ]
            
            campo = None
            seletor_usado = None
            
            # Primeiro, listar todos os inputs para debug
            try:
                todos_inputs = await popup_page.query_selector_all('input, textarea')
                logger.info(f"📝 Total de campos encontrados no popup: {len(todos_inputs)}")
                
                for i, input_el in enumerate(todos_inputs):
                    try:
                        name = await input_el.get_attribute('name') or 'N/A'
                        id_attr = await input_el.get_attribute('id') or 'N/A'
                        type_attr = await input_el.get_attribute('type') or 'N/A'
                        maxlength = await input_el.get_attribute('maxlength') or 'N/A'
                        logger.info(f"  Input {i+1}: name='{name}', id='{id_attr}', type='{type_attr}', maxlength='{maxlength}'")
                    except:
                        logger.info(f"  Input {i+1}: erro ao obter atributos")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao listar inputs: {e}")
            
            # Tentar cada seletor
            for seletor in seletores_campo:
                try:
                    campo = await popup_page.query_selector(seletor)
                    if campo:
                        seletor_usado = seletor
                        logger.info(f"✅ Campo encontrado com seletor: '{seletor}'")
                        break
                except:
                    continue
                
            if not campo:
                logger.error("❌ Monitor: NENHUM campo de CAPTCHA encontrado no popup")
                return None
                
            # 🎯 PREENCHIMENTO ESPECÍFICO PARA SITES ASP
            logger.info(f"📝 Digitando '{resultado.texto_final}' no campo encontrado (modo ASP)...")
            try:
                # Focar no campo primeiro
                await campo.focus()
                await popup_page.wait_for_timeout(200)
                
                # Método 1: Limpeza e preenchimento tradicional
                logger.info("🔄 Tentativa 1: Preenchimento tradicional...")
                await campo.fill('')  # Limpar campo
                await asyncio.sleep(0.2)  # Pausa maior para ASP (em segundos)
                await campo.fill(resultado.texto_final)  # Preencher com o texto
                
                # Verificar se funcionou
                valor_digitado = await campo.input_value()
                logger.info(f"📋 Valor após tentativa 1: '{valor_digitado}'")
                
                # Se não funcionou, tentar método específico para ASP
                if valor_digitado != resultado.texto_final:
                    logger.info("🔄 Tentativa 2: Método ASP com JavaScript...")
                    
                    # Usar JavaScript para definir o valor diretamente
                    await popup_page.evaluate(f"""
                        (texto) => {{
                            const campo = document.querySelector('input[name="idLetra"]');
                            if (campo) {{
                                campo.value = '';
                                campo.focus();
                                campo.value = texto;
                                // Disparar eventos que o ASP pode estar escutando
                                campo.dispatchEvent(new Event('input', {{ bubbles: true }}));
                                campo.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                campo.dispatchEvent(new Event('keyup', {{ bubbles: true }}));
                                campo.dispatchEvent(new Event('blur', {{ bubbles: true }}));
                            }}
                        }}
                    """, resultado.texto_final)
                    
                    await popup_page.wait_for_timeout(300)
                    valor_digitado = await campo.input_value()
                    logger.info(f"📋 Valor após tentativa 2 (JS): '{valor_digitado}'")
                
                # Se ainda não funcionou, tentar digitação caractere por caractere
                if valor_digitado != resultado.texto_final:
                    logger.info("🔄 Tentativa 3: Digitação caractere por caractere...")
                    
                    await campo.fill('')  # Limpar
                    await campo.focus()
                    await popup_page.wait_for_timeout(200)
                    
                    # Digitar cada caractere individualmente
                    for char in resultado.texto_final:
                        await popup_page.keyboard.type(char)
                        await asyncio.sleep(0.05)  # Pausa entre caracteres (em segundos)
                    
                    valor_digitado = await campo.input_value()
                    logger.info(f"📋 Valor após tentativa 3 (char-by-char): '{valor_digitado}'")
                
                # Se ainda não funcionou, tentar método de seleção e substituição
                if valor_digitado != resultado.texto_final:
                    logger.info("🔄 Tentativa 4: Seleção total e substituição...")
                    
                    await campo.focus()
                    await popup_page.keyboard.press('Control+a')  # Selecionar tudo
                    await asyncio.sleep(0.1)  # Pausa em segundos
                    await popup_page.keyboard.type(resultado.texto_final)  # Digitar substituindo
                    
                    valor_digitado = await campo.input_value()
                    logger.info(f"📋 Valor após tentativa 4 (select-all): '{valor_digitado}'")
                
                # Verificação final
                if valor_digitado == resultado.texto_final:
                    logger.info(f"✅ Sucesso! Valor digitado verificado: '{valor_digitado}'")
                else:
                    logger.warning(f"⚠️ Valor não confere. Esperado: '{resultado.texto_final}', Obtido: '{valor_digitado}'")
                
                # Aguardar um pouco para garantir que o ASP processou
                await popup_page.wait_for_timeout(500)
                
            except Exception as e:
                logger.error(f"❌ Erro ao preencher campo: {e}")
                return None
            
            # 🔍 BUSCA AVANÇADA DO BOTÃO DE CONFIRMAÇÃO
            logger.info("🔍 Iniciando busca avançada por botão de confirmação...")
            
            # Lista completa de seletores para botão
            seletores_botao = [
                'input[type="submit"][value="Confirmar"]',  # 🎯 BOTÃO ESPECÍFICO DO COMPRASNET
                'input[name="Submit"]',                     # 🎯 NOME ESPECÍFICO DO COMPRASNET
                'input[id="idSubmit"]',                     # 🎯 ID ESPECÍFICO DO COMPRASNET
                'input[type="submit"]',
                'input[type="button"][value*="OK"]',
                'input[type="button"][value*="Confirmar"]',
                'input[type="button"][value*="Enviar"]',
                'input[type="button"][value*="Submit"]',
                'button[type="submit"]',
                'button:contains("OK")',
                'button:contains("Confirmar")',
                'button:contains("Enviar")',
                'button',
                'input[type="button"]',
                'input[value*="OK"]',
                'input[value*="Confirmar"]',
                'input[value*="Enviar"]'
            ]
            
            btn = None
            seletor_botao_usado = None
            
            # Primeiro, listar todos os botões para debug
            try:
                todos_botoes = await popup_page.query_selector_all('button, input[type="submit"], input[type="button"]')
                logger.info(f"🔘 Total de botões encontrados no popup: {len(todos_botoes)}")
                
                for i, btn_el in enumerate(todos_botoes):
                    try:
                        value = await btn_el.get_attribute('value') or 'N/A'
                        type_attr = await btn_el.get_attribute('type') or 'N/A'
                        text_content = await btn_el.text_content() or 'N/A'
                        logger.info(f"  Botão {i+1}: value='{value}', type='{type_attr}', texto='{text_content}'")
                    except:
                        logger.info(f"  Botão {i+1}: erro ao obter atributos")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao listar botões: {e}")
            
            # Tentar cada seletor de botão
            for seletor in seletores_botao:
                try:
                    btn = await popup_page.query_selector(seletor)
                    if btn:
                        seletor_botao_usado = seletor
                        logger.info(f"✅ Botão encontrado com seletor: '{seletor}'")
                        break
                except:
                    continue
                
            if not btn:
                logger.error("❌ Monitor: NENHUM botão de confirmação encontrado no popup")
                return None
                
            logger.info(f"✅ Sistema completo pronto! Campo: '{seletor_usado}', Botão: '{seletor_botao_usado}'")
            
            # 🎯 CLIQUE ESPECÍFICO PARA SITES ASP
            logger.info("🚀 Clicando no botão de confirmação (modo ASP)...")
            try:
                # Método 1: Clique direto no elemento
                logger.info("🔄 Tentativa 1: Clique direto no elemento...")
                await btn.click()
                logger.info("✅ Clique direto funcionou!")
                await popup_page.wait_for_timeout(1000)  # Aguardar resultado
                return True  # Retorna True indicando sucesso completo
                
            except Exception as click_error:
                logger.error(f"❌ Erro no clique direto: {click_error}")
                
                # Método 2: Clique com JavaScript (mais compatível com ASP)
                try:
                    logger.info("🔄 Tentativa 2: Clique via JavaScript...")
                    await popup_page.evaluate("""
                        () => {
                            const botao = document.querySelector('input[type="submit"][value="Confirmar"]');
                            if (botao) {
                                botao.focus();
                                botao.click();
                            }
                        }
                    """)
                    logger.info("✅ Clique JavaScript funcionou!")
                    await popup_page.wait_for_timeout(1000)
                    return True
                    
                except Exception as js_click_error:
                    logger.error(f"❌ Erro no clique JavaScript: {js_click_error}")
                    
                    # Método 3: Submit do formulário (específico para ASP)
                    try:
                        logger.info("🔄 Tentativa 3: Submit do formulário ASP...")
                        await popup_page.evaluate("""
                            () => {
                                const botao = document.querySelector('input[type="submit"][value="Confirmar"]');
                                if (botao && botao.form) {
                                    botao.form.submit();
                                } else {
                                    // Procurar por formulário e submeter
                                    const form = document.querySelector('form');
                                    if (form) {
                                        form.submit();
                                    }
                                }
                            }
                        """)
                        logger.info("✅ Submit do formulário funcionou!")
                        await popup_page.wait_for_timeout(1000)
                        return True
                        
                    except Exception as submit_error:
                        logger.error(f"❌ Erro no submit do formulário: {submit_error}")
                        
                        # Método 4: Pressionar Enter no campo (fallback)
                        try:
                            logger.info("🔄 Tentativa 4: Pressionar Enter no campo...")
                            await campo.focus()
                            await popup_page.keyboard.press('Enter')
                            logger.info("✅ Enter no campo funcionou!")
                            await popup_page.wait_for_timeout(1000)
                            return True
                            
                        except Exception as enter_error:
                            logger.error(f"❌ Erro ao pressionar Enter: {enter_error}")
                            return False
        except Exception as e:
            logger.error(f"❌ Monitor: erro ao resolver CAPTCHA no popup: {e}")
            return None
    
    def testar_captcha_com_config(self, arquivo_captcha, resposta_correta, config):
        """Testa uma configuração específica em um CAPTCHA conhecido"""
        try:
            # Carregar imagem
            img = cv2.imread(str(arquivo_captcha))
            if img is None:
                return False
            
            # Processar com OpenCV
            img_processado = self.processar_captcha_opencv_simples(img)
            
            # Testar com EasyOCR
            result = self.reader.readtext(img_processado, detail=1, 
                                        paragraph=config['paragraph'], 
                                        width_ths=config['width_ths'])
            
            for (bbox, text, confidence) in result:
                if confidence > config['threshold']:
                    text_clean = re.sub(r'[^A-Za-z0-9]', '', text.strip())
                    if text_clean.lower() == resposta_correta.lower():
                        return True
            
            return False
            
        except Exception as e:
            return False
    
    def testar_captcha_tesseract(self, arquivo_captcha, resposta_correta, tesseract_config):
        """Testa Tesseract com configuração específica em um CAPTCHA conhecido"""
        try:
            # Carregar imagem
            img = cv2.imread(str(arquivo_captcha))
            if img is None:
                return False
            
            # Processar com OpenCV
            img_processado = self.processar_captcha_opencv_simples(img)
            
            # Testar com Tesseract
            texto = pytesseract.image_to_string(img_processado, config=tesseract_config).strip()
            text_clean = re.sub(r'[^A-Za-z0-9]', '', texto)
            
            return text_clean.lower() == resposta_correta.lower()
            
        except Exception as e:
            return False
    
    def processar_captcha_opencv_simples(self, img):
        """Versão simplificada do processamento OpenCV para treinamento"""
        try:
            # Separar canais RGB
            b, g, r = cv2.split(img)
            
            # Processar canal com melhor contraste
            canais = {'red': r, 'green': g, 'blue': b}
            contraste_scores = {}
            for nome, canal in canais.items():
                contraste_scores[nome] = cv2.Laplacian(canal, cv2.CV_64F).var()
            
            melhor_canal = max(contraste_scores, key=contraste_scores.get)
            img_gray = canais[melhor_canal]
            
            # Processamento básico
            img_gray = cv2.equalizeHist(img_gray)
            img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
            _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return img_thresh
            
        except Exception as e:
            return img
        
    def salvar_captcha_para_estudos(self, captcha_buffer, captcha_resolvido=None, metadados_extras=None):
        """Salva CAPTCHA original para estudos futuros com timestamp, resposta e metadados completos"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Nome do arquivo com timestamp
            if captcha_resolvido:
                nome_arquivo = f"captcha_{timestamp}_{captcha_resolvido}.png"
            else:
                nome_arquivo = f"captcha_{timestamp}_sem_resposta.png"
            
            # Salvar arquivo
            caminho_arquivo = self.captcha_estudos_dir / nome_arquivo
            with open(caminho_arquivo, 'wb') as f:
                f.write(captcha_buffer)
            
            logger.info(f"📚 CAPTCHA salvo para estudos: {nome_arquivo}")
            
            # Criar arquivo de metadados completo
            metadata = {
                "timestamp": timestamp,
                "arquivo_imagem": nome_arquivo,
                "captcha_resolvido": captcha_resolvido or "nao_resolvido",
                "tamanho_buffer": len(captcha_buffer),
                "data_hora": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "status_inicial": "capturado"
            }
            
            # Adicionar metadados extras se fornecidos
            if metadados_extras:
                metadata.update(metadados_extras)
            
            metadata_path = self.captcha_estudos_dir / f"metadata_{timestamp}.txt"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            
            logger.info(f"📊 Metadados salvos: metadata_{timestamp}.txt")
            return nome_arquivo, timestamp
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar CAPTCHA para estudos: {e}")
            return None, None

    async def verificar_resultado_captcha(self, timestamp):
        """
        Verifica sinais na página sobre resultado do CAPTCHA para diagnóstico e metadados.
        IMPORTANTE: Este método NÃO determina sucesso definitivo da resolução.
        O sucesso é determinado EXCLUSIVAMENTE pelo download bem-sucedido.
        """
        try:
            logger.info("🔍 Verificando resultado da submissão do CAPTCHA...")
            
            # Aguardar um pouco para o site processar
            await asyncio.sleep(2)
            
            # Capturar conteúdo da página atual
            conteudo_pagina = await self.page.content()
            url_atual = self.page.url
            
            # Verificar indicadores de sucesso/erro
            resultado = {
                "url_atual": url_atual,
                "timestamp_verificacao": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "status_captcha": "desconhecido"
            }
            
            # Indicadores de erro de CAPTCHA
            erros_captcha = [
                "código inválido", "captcha inválido", "código incorreto",
                "digite novamente", "erro no código", "código de verificação inválido",
                "invalid captcha", "wrong code", "incorrect code"
            ]
            
            # Indicadores de sucesso
            sucessos = [
                "download", "arquivo", "document", "baixar",
                "Content-Disposition", "application/", "attachment"
            ]
            
            conteudo_lower = conteudo_pagina.lower()
            
            # Verificar se houve erro de CAPTCHA
            if any(erro in conteudo_lower for erro in erros_captcha):
                resultado["status_captcha"] = "erro_captcha_invalido"
                resultado["descricao"] = "Site indicou que CAPTCHA estava incorreto"
                logger.warning("❌ Site indicou CAPTCHA incorreto")
            
            # Verificar se houve sucesso (download iniciado)
            elif any(sucesso in conteudo_lower for sucesso in sucessos):
                resultado["status_captcha"] = "sucesso_download"
                resultado["descricao"] = "CAPTCHA correto - download iniciado"
                logger.info("✅ CAPTCHA correto - download detectado")
            
            # Verificar se mudou de página (possível sucesso)
            elif "download.asp" not in url_atual.lower():
                resultado["status_captcha"] = "sucesso_redirecionamento"
                resultado["descricao"] = "CAPTCHA correto - redirecionado para nova página"
                logger.info("✅ CAPTCHA correto - redirecionamento detectado")
            
            # Verificar se ainda está na mesma página (possível erro)
            else:
                resultado["status_captcha"] = "possivel_erro"
                resultado["descricao"] = "Permaneceu na mesma página - possível erro"
                logger.warning("⚠️ Possível erro - permaneceu na mesma página")
            
            # Salvar resultado nos metadados
            self.atualizar_metadata_resultado(timestamp, resultado)
            
            return resultado
            
        except Exception as e:
            logger.error(f"❌ Erro ao verificar resultado do CAPTCHA: {e}")
            return {"status_captcha": "erro_verificacao", "erro": str(e)}

    def atualizar_metadata_resultado(self, timestamp, resultado):
        """Atualiza arquivo de metadados com resultado da verificação"""
        try:
            metadata_path = self.captcha_estudos_dir / f"metadata_{timestamp}.txt"
            
            if metadata_path.exists():
                # Adicionar informações de resultado ao arquivo existente
                with open(metadata_path, 'a', encoding='utf-8') as f:
                    f.write(f"# RESULTADO DA VERIFICAÇÃO\n")
                    for key, value in resultado.items():
                        f.write(f"{key}: {value}\n")
                
                logger.info(f"📊 Metadados atualizados com resultado da verificação")
            else:
                logger.warning(f"⚠️ Arquivo de metadados não encontrado: {metadata_path}")
                
        except Exception as e:
            logger.error(f"❌ Erro ao atualizar metadados: {e}")

    def salvar_estatisticas_resolucao(self, captcha_text, sucesso, metodo="easyocr", detalhes=None):
        """Salva estatísticas detalhadas de resolução de CAPTCHA"""
        try:
            timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            
            estatistica = {
                "timestamp": timestamp,
                "captcha_text": captcha_text,
                "sucesso": sucesso,
                "metodo": metodo,
                "detalhes": detalhes or {}
            }
            
            # Salvar em arquivo de estatísticas
            stats_file = Path("estatisticas_resolucao.txt")
            with open(stats_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {metodo}: '{captcha_text}' -> {'SUCESSO' if sucesso else 'FALHA'}\n")
                if detalhes:
                    for key, value in detalhes.items():
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Contabilizar estatísticas
            total_tentativas = 0
            sucessos = 0
            
            if stats_file.exists():
                with open(stats_file, 'r', encoding='utf-8') as f:
                    linhas = f.readlines()
                    for linha in linhas:
                        if "SUCESSO" in linha or "FALHA" in linha:
                            total_tentativas += 1
                            if "SUCESSO" in linha:
                                sucessos += 1
            
            taxa_sucesso = (sucessos / total_tentativas * 100) if total_tentativas > 0 else 0
            logger.info(f"📊 Taxa de sucesso atual: {taxa_sucesso:.1f}% ({sucessos}/{total_tentativas})")
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar estatísticas: {e}")

    def detectar_cor_letras(self, img):
        """
        Detecta a intensidade das letras/números em escala de cinza
        Funciona com letras maiúsculas, minúsculas e números
        """
        logger.info("🎨 Detectando intensidade das letras/números...")
        
        # Converter para escala de cinza
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Analisar histograma para encontrar picos de intensidade
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_smooth = np.convolve(hist.flatten(), np.ones(5)/5, mode='same')
        
        # Encontrar picos significativos (possíveis intensidades de letras)
        picos = []
        for i in range(10, 246):  # Evitar extremos
            if (hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1] and 
                hist_smooth[i] > hist_smooth.max() * 0.15):  # Pelo menos 15% do pico máximo
                picos.append((i, hist_smooth[i]))
        
        # Ordenar picos por intensidade
        picos.sort(key=lambda x: x[1], reverse=True)
        
        if len(picos) >= 2:
            # O maior pico geralmente é o fundo, segundo maior são as letras
            fundo_intensidade = picos[0][0]
            letra_intensidade = picos[1][0]
            
            # Se a diferença é pequena, pegar o terceiro pico
            if len(picos) >= 3 and abs(fundo_intensidade - letra_intensidade) < 30:
                letra_intensidade = picos[2][0]
        else:
            # Fallback: calcular intensidade média das regiões não-uniformes
            # Usar gradiente para encontrar bordas (onde estão as letras)
            grad = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
            grad_norm = np.abs(grad)
            mask_letras = grad_norm > np.percentile(grad_norm, 75)  # Top 25% de gradientes
            
            if np.sum(mask_letras) > 0:
                letra_intensidade = np.mean(gray[mask_letras])
            else:
                letra_intensidade = 128  # Valor médio padrão
        
        logger.info(f"   Intensidade detectada: {letra_intensidade:.0f}/255")
        
        # Classificar baseado na intensidade detectada
        if letra_intensidade < 85:
            tipo = "escuro"      # Letras/números escuros (0-85)
            mask = cv2.inRange(gray, 0, 120)  # Captura escuros com margem
        elif letra_intensidade > 170:
            tipo = "claro"       # Letras/números claros (170-255)  
            mask = cv2.inRange(gray, 135, 255)  # Captura claros com margem
        else:
            tipo = "medio"       # Letras/números médios (85-170)
            intensidade_min = max(0, int(letra_intensidade - 40))
            intensidade_max = min(255, int(letra_intensidade + 40))
            mask = cv2.inRange(gray, intensidade_min, intensidade_max)
        
        logger.info(f"   Tipo detectado: {tipo}")
        return tipo, mask

    def limpar_borras_rgb(self, img_processada, img_original, metodo_usado):
        """
        Limpa borras e ruídos usando análise avançada de canais RGB
        Mantém apenas as letras que são desenhadas em borras consistentes
        """
        try:
            logger.info("🎨 Analisando canais RGB para detectar borras de letras...")
            
            # Separar canais RGB da imagem original
            b, g, r = cv2.split(img_original)
            
            # 1. DETECÇÃO DE BORRAS POR VARIÂNCIA
            logger.info("📊 Calculando variância de intensidade por canal...")
            
            # Calcular variância local para cada canal (detecta bordas/borras)
            kernel = np.ones((3,3), np.float32) / 9
            
            # Variância local para cada canal
            r_mean = cv2.filter2D(r.astype(np.float32), -1, kernel)
            g_mean = cv2.filter2D(g.astype(np.float32), -1, kernel)
            b_mean = cv2.filter2D(b.astype(np.float32), -1, kernel)
            
            r_var = cv2.filter2D((r.astype(np.float32) - r_mean)**2, -1, kernel)
            g_var = cv2.filter2D((g.astype(np.float32) - g_mean)**2, -1, kernel)
            b_var = cv2.filter2D((b.astype(np.float32) - b_mean)**2, -1, kernel)
            
            # 2. DETECÇÃO DE CONTORNOS DE BORRAS
            logger.info("🔍 Detectando contornos de borras em cada canal...")
            
            # Detectar bordas em cada canal
            r_edges = cv2.Canny(r, 50, 150)
            g_edges = cv2.Canny(g, 50, 150)
            b_edges = cv2.Canny(b, 50, 150)
            
            # 3. ANÁLISE DE CONSISTÊNCIA DE BORRAS
            logger.info("🎯 Analisando consistência de borras entre canais...")
            
            # Combinar informações de variância e bordas
            r_borra = cv2.bitwise_and((r_var > np.percentile(r_var, 70)).astype(np.uint8) * 255, r_edges)
            g_borra = cv2.bitwise_and((g_var > np.percentile(g_var, 70)).astype(np.uint8) * 255, g_edges)
            b_borra = cv2.bitwise_and((b_var > np.percentile(b_var, 70)).astype(np.uint8) * 255, b_edges)
            
            # 4. MÁSCARA DE BORRAS CONSISTENTES
            logger.info("🎭 Criando máscara de borras consistentes...")
            
            # Borras que aparecem em pelo menos 2 canais (mais confiáveis)
            mask_borra_dupla = cv2.bitwise_or(
                cv2.bitwise_and(r_borra, g_borra),
                cv2.bitwise_or(
                    cv2.bitwise_and(r_borra, b_borra),
                    cv2.bitwise_and(g_borra, b_borra)
                )
            )
            
            # 5. DILATAÇÃO PARA CONECTAR LETRAS FRAGMENTADAS
            logger.info("🔗 Conectando fragmentos de letras...")
            
            # Operações morfológicas para conectar partes das letras
            kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_borra_conectada = cv2.morphologyEx(mask_borra_dupla, cv2.MORPH_CLOSE, kernel_connect)
            mask_borra_conectada = cv2.morphologyEx(mask_borra_conectada, cv2.MORPH_DILATE, kernel_connect)
            
            # 6. FILTRAGEM POR TAMANHO DE COMPONENTES
            logger.info("📏 Filtrando componentes por tamanho...")
            
            # Encontrar componentes conectados
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_borra_conectada, connectivity=8)
            
            # Filtrar componentes muito pequenos ou muito grandes
            h, w = img_processada.shape[:2]
            min_area = (h * w) * 0.001  # Mínimo 0.1% da imagem
            max_area = (h * w) * 0.15   # Máximo 15% da imagem
            
            mask_filtrada = np.zeros_like(mask_borra_conectada)
            
            for i in range(1, num_labels):  # Pula o fundo (label 0)
                area = stats[i, cv2.CC_STAT_AREA]
                if min_area <= area <= max_area:
                    # Verificar aspect ratio (letras não são muito largas/altas)
                    x = stats[i, cv2.CC_STAT_LEFT]
                    y = stats[i, cv2.CC_STAT_TOP]
                    w_comp = stats[i, cv2.CC_STAT_WIDTH]
                    h_comp = stats[i, cv2.CC_STAT_HEIGHT]
                    aspect_ratio = max(w_comp, h_comp) / max(min(w_comp, h_comp), 1)
                    
                    if aspect_ratio <= 4:  # Aspect ratio razoável para letras
                        mask_filtrada[labels == i] = 255
            
            # 7. CRIAÇÃO DE FUNDO BRANCO LIMPO COM LETRAS DESTACADAS
            logger.info("🔄 Criando fundo branco limpo com letras destacadas...")
            
            # Criar fundo branco completamente limpo
            h, w = mask_filtrada.shape
            resultado_final = np.full((h, w), 255, dtype=np.uint8)  # Fundo branco
            
            # Detectar tipo de letra na imagem original para decidir a cor
            if len(img_processada.shape) == 3:
                img_gray = cv2.cvtColor(img_processada, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img_processada.copy()
            
            # Analisar intensidade das letras detectadas
            pixels_letras = img_gray[mask_filtrada > 0]
            if len(pixels_letras) > 0:
                intensidade_media = np.mean(pixels_letras)
                logger.info(f"📊 Intensidade média das letras: {intensidade_media:.1f}")
                
                # Determinar cor ideal das letras (preto para contraste máximo)
                if intensidade_media < 127:
                    # Letras escuras originais -> manter pretas
                    cor_letra = 0
                    logger.info("   🖤 Aplicando letras PRETAS em fundo branco")
                else:
                    # Letras claras originais -> converter para pretas para melhor contraste
                    cor_letra = 0
                    logger.info("   🖤 Convertendo para letras PRETAS em fundo branco")
                
                # Aplicar letras na cor escolhida apenas onde há borras
                resultado_final[mask_filtrada > 0] = cor_letra
            
            # 8. PÓS-PROCESSAMENTO PARA MÁXIMA LEGIBILIDADE
            logger.info("✨ Aplicando pós-processamento para máxima legibilidade...")
            
            # Refinamento morfológico para letras mais nítidas
            kernel_refine = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            resultado_final = cv2.morphologyEx(resultado_final, cv2.MORPH_CLOSE, kernel_refine)
            
            # Suavização muito leve para remover ruídos pixelados
            resultado_final = cv2.medianBlur(resultado_final, 3)
            
            # Garantir contraste máximo (binarização final)
            _, resultado_final = cv2.threshold(resultado_final, 127, 255, cv2.THRESH_BINARY)
            
            logger.info("🎯 Fundo branco limpo criado com letras em preto para máximo contraste")
            
            # Salvar imagens debug
            timestamp = datetime.now().strftime("%H%M%S")
            
            # Salvar análise de borras por canal
            debug_r_path = self.debug_rgb_dir / f"borras_red_{timestamp}.png"
            debug_g_path = self.debug_rgb_dir / f"borras_green_{timestamp}.png"
            debug_b_path = self.debug_rgb_dir / f"borras_blue_{timestamp}.png"
            debug_mask_path = self.debug_rgb_dir / f"mask_borras_{timestamp}.png"
            debug_final_path = self.debug_rgb_dir / f"limpo_borras_{timestamp}.png"
            
            cv2.imwrite(str(debug_r_path), r_borra)
            cv2.imwrite(str(debug_g_path), g_borra)
            cv2.imwrite(str(debug_b_path), b_borra)
            cv2.imwrite(str(debug_mask_path), mask_filtrada)
            cv2.imwrite(str(debug_final_path), resultado_final)
            
            # Gerar lâminas com diferentes níveis de ruído para análise
            self.gerar_laminas_ruido(resultado_final, timestamp)
            
            # FOCO EM CARACTERES - Converter para grayscale e isolar caracteres
            logger.info("� FOCANDO EM CARACTERES - Isolamento preciso...")
            resultado_caracteres = self.focar_caracteres_grayscale(img_original, timestamp)
            
            # NOVO: PRÉ-PROCESSAMENTO ADAPTATIVO AVANÇADO
            logger.info("🧠 Aplicando pré-processamento adaptativo avançado...")
            resultado_adaptativo = self.preprocessamento_adaptativo_avancado(img_original)
            
            # Combinar resultado da limpeza RGB + foco em caracteres + adaptativo
            logger.info("🔗 Combinando todas as abordagens: RGB + Caracteres + Adaptativo...")
            
            # Converter resultado final para grayscale se necessário
            if len(resultado_final.shape) == 3:
                resultado_final_gray = cv2.cvtColor(resultado_final, cv2.COLOR_BGR2GRAY)
            else:
                resultado_final_gray = resultado_final
                
            # Analisar densidade de pixels de cada abordagem
            pixels_rgb = cv2.countNonZero(resultado_final_gray)
            pixels_char = cv2.countNonZero(resultado_caracteres)
            pixels_adaptativo = cv2.countNonZero(resultado_adaptativo)
            
            logger.info(f"📊 Pixels detectados - RGB: {pixels_rgb}, Caracteres: {pixels_char}, Adaptativo: {pixels_adaptativo}")
            
            # Selecionar melhor resultado baseado na densidade e qualidade
            candidatos = [
                ("rgb_limpeza", resultado_final_gray, pixels_rgb),
                ("foco_caracteres", resultado_caracteres, pixels_char),
                ("adaptativo", resultado_adaptativo, pixels_adaptativo)
            ]
            
            # Avaliar qualidade de cada candidato
            melhor_candidato = None
            melhor_score = 0
            
            for nome, imagem, pixels in candidatos:
                # Score baseado em densidade ideal (5-20%) e componentes válidos
                h_img, w_img = imagem.shape
                densidade = pixels / (h_img * w_img)
                
                # Analisar componentes conectados
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagem, connectivity=8)
                componentes_validos = 0
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    width = stats[i, cv2.CC_STAT_WIDTH]
                    height = stats[i, cv2.CC_STAT_HEIGHT]
                    aspect_ratio = width / height if height > 0 else 0
                    
                    if 20 <= area <= (h_img * w_img) // 10 and 0.2 <= aspect_ratio <= 5.0:
                        componentes_validos += 1
                
                # Calcular score
                densidade_score = 1.0 if 0.05 <= densidade <= 0.20 else max(0.1, 1.0 - abs(densidade - 0.125) * 8)
                componentes_score = 1.0 if 3 <= componentes_validos <= 7 else max(0.1, 1.0 - abs(componentes_validos - 5) * 0.2)
                score_total = densidade_score * 0.6 + componentes_score * 0.4
                
                logger.info(f"   📈 {nome}: densidade={densidade:.3f}, comp={componentes_validos}, score={score_total:.3f}")
                
                if score_total > melhor_score:
                    melhor_score = score_total
                    melhor_candidato = (nome, imagem)
            
            if melhor_candidato:
                nome_melhor, resultado_final_definitivo = melhor_candidato
                logger.info(f"✅ MELHOR ABORDAGEM: {nome_melhor} (score: {melhor_score:.3f})")
            else:
                logger.info("⚠️ Usando foco em caracteres como fallback")
                resultado_final_definitivo = resultado_caracteres
            
            # Salvar resultado final definitivo
            resultado_definitivo_path = self.debug_rgb_dir / f"RESULTADO_DEFINITIVO_{timestamp}.png"
            cv2.imwrite(str(resultado_definitivo_path), resultado_final_definitivo)
            
            logger.info(f"🖼️ Análise de borras salva: {num_labels-1} componentes encontrados")
            logger.info("✅ Limpeza de borras RGB + foco em caracteres concluída")
            
            return resultado_final_definitivo
            
        except Exception as e:
            logger.error(f"❌ Erro na limpeza de borras RGB: {e}")
            return img_processada

    def gerar_laminas_ruido(self, imagem_limpa, timestamp):
        """
        Gera 5 lâminas com diferentes tipos de ruído REAL para análise futura
        L1: Salt&Pepper + Gaussiano (Alto), L2: Gaussiano + Blur (Médio), L3: JPEG + Distorção (Baixo), L4: Blur mínimo, L5: Limpa
        """
        try:
            logger.info("🔬 Gerando lâminas com diferentes tipos de ruído REAL para análise...")
            
            # Converter para escala de cinza se necessário
            if len(imagem_limpa.shape) == 3:
                img_gray = cv2.cvtColor(imagem_limpa, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = imagem_limpa.copy()
            
            # Definir configurações de ruído REAIS
            laminas_config = [
                ("L1_salt_pepper_alto", "salt_pepper_gaussiano", "L1", "Salt&Pepper + Gaussiano Alto"),
                ("L2_gaussiano_blur", "gaussiano_blur", "L2", "Gaussiano + Blur Médio"), 
                ("L3_jpeg_distorcao", "jpeg_distorcao", "L3", "JPEG + Distorção Baixo"),
                ("L4_blur_minimo", "blur_minimo", "L4", "Blur Mínimo"),
                ("L5_imagem_limpa", "limpa", "L5", "Imagem Limpa")
            ]
            
            laminas_base_dir = Path("laminas_ruido")
            
            for pasta, tipo_ruido, codigo, descricao in laminas_config:
                pasta_lamina = laminas_base_dir / pasta
                
                # Criar diretório se não existe
                pasta_lamina.mkdir(parents=True, exist_ok=True)
                
                # Criar cópia da imagem
                lamina = img_gray.copy()
                detalhes_ruido = []
                
                if tipo_ruido == "salt_pepper_gaussiano":
                    logger.info(f"   🔬 {codigo}: Aplicando {descricao}...")
                    
                    # 1. Salt & Pepper Noise (pixels aleatórios pretos e brancos)
                    lamina = self._aplicar_salt_pepper_noise(lamina, probabilidade=0.02)
                    detalhes_ruido.append("Salt&Pepper: 2% probabilidade")
                    
                    # 2. Ruído Gaussiano aditivo
                    lamina = self._aplicar_ruido_gaussiano(lamina, sigma=15)
                    detalhes_ruido.append("Gaussiano: σ=15")
                    
                elif tipo_ruido == "gaussiano_blur":
                    logger.info(f"   🔬 {codigo}: Aplicando {descricao}...")
                    
                    # 1. Ruído Gaussiano moderado
                    lamina = self._aplicar_ruido_gaussiano(lamina, sigma=8)
                    detalhes_ruido.append("Gaussiano: σ=8")
                    
                    # 2. Motion Blur (simula movimento da câmera)
                    lamina = self._aplicar_motion_blur(lamina, tamanho=3, angulo=45)
                    detalhes_ruido.append("Motion Blur: 3px, 45°")
                    
                elif tipo_ruido == "jpeg_distorcao":
                    logger.info(f"   🔬 {codigo}: Aplicando {descricao}...")
                    
                    # 1. Compressão JPEG (simula artefatos de compressão)
                    lamina = self._aplicar_compressao_jpeg(lamina, qualidade=70)
                    detalhes_ruido.append("JPEG: qualidade 70%")
                    
                    # 2. Distorção geométrica leve
                    lamina = self._aplicar_distorcao_geometrica(lamina, intensidade=0.02)
                    detalhes_ruido.append("Distorção geométrica: 2%")
                    
                elif tipo_ruido == "blur_minimo":
                    logger.info(f"   🔬 {codigo}: Aplicando {descricao}...")
                    
                    # Gaussian Blur muito leve (simula foco ligeiramente fora)
                    lamina = cv2.GaussianBlur(lamina, (3, 3), 0.5)
                    detalhes_ruido.append("Gaussian Blur: kernel 3x3, σ=0.5")
                    
                else:  # limpa
                    logger.info(f"   🔬 {codigo}: {descricao}...")
                    detalhes_ruido.append("Nenhum ruído aplicado")
                
                # Salvar lâmina
                nome_arquivo = f"lamina_{codigo}_{timestamp}.png"
                caminho_lamina = pasta_lamina / nome_arquivo
                cv2.imwrite(str(caminho_lamina), lamina)
                
                # Criar arquivo de metadados detalhado
                metadata_arquivo = pasta_lamina / f"metadata_{codigo}_{timestamp}.txt"
                with open(metadata_arquivo, 'w', encoding='utf-8') as f:
                    f.write(f"LÂMINA {codigo} - ANÁLISE DE RUÍDO REAL\n")
                    f.write(f"========================================\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Descrição: {descricao}\n")
                    f.write(f"Tipo de ruído: {tipo_ruido}\n")
                    f.write(f"Formato: Escala de cinza (GRAYSCALE)\n")
                    f.write(f"Tamanho: {lamina.shape[1]}x{lamina.shape[0]}\n")
                    f.write(f"Tipo: uint8\n")
                    f.write(f"Range de valores: 0-255\n")
                    f.write(f"Detalhes de ruído aplicado:\n")
                    for detalhe in detalhes_ruido:
                        f.write(f"  - {detalhe}\n")
                    f.write(f"Propósito: Teste de robustez OCR contra ruídos reais\n")
                    f.write(f"Baseado em: Problemas comuns em CAPTCHAs web\n")
                
                logger.info(f"   ✅ {codigo}: {descricao} → {caminho_lamina}")
            
            logger.info("🔬 5 lâminas geradas para análise futura em: laminas_ruido/")
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar lâminas de ruído: {e}")

    def _aplicar_salt_pepper_noise(self, imagem, probabilidade=0.02):
        """Aplica ruído Salt & Pepper (pixels aleatórios pretos e brancos)"""
        resultado = imagem.copy()
        h, w = imagem.shape
        
        # Salt noise (pixels brancos)
        salt = np.random.random((h, w)) < (probabilidade / 2)
        resultado[salt] = 255
        
        # Pepper noise (pixels pretos)  
        pepper = np.random.random((h, w)) < (probabilidade / 2)
        resultado[pepper] = 0
        
        return resultado
    
    def _aplicar_ruido_gaussiano(self, imagem, sigma=10):
        """Aplica ruído gaussiano aditivo"""
        ruido = np.random.normal(0, sigma, imagem.shape)
        resultado = imagem.astype(np.float32) + ruido
        resultado = np.clip(resultado, 0, 255)
        return resultado.astype(np.uint8)
    
    def _aplicar_motion_blur(self, imagem, tamanho=5, angulo=0):
        """Aplica motion blur (simula movimento da câmera)"""
        # Criar kernel de motion blur
        kernel = np.zeros((tamanho, tamanho))
        angulo_rad = np.radians(angulo)
        
        for i in range(tamanho):
            x = int(i * np.cos(angulo_rad))
            y = int(i * np.sin(angulo_rad))
            if 0 <= x < tamanho and 0 <= y < tamanho:
                kernel[y, x] = 1
        
        kernel = kernel / np.sum(kernel)
        resultado = cv2.filter2D(imagem, -1, kernel)
        return resultado
    
    def _aplicar_compressao_jpeg(self, imagem, qualidade=70):
        """Simula artefatos de compressão JPEG"""
        # Codificar e decodificar como JPEG para introduzir artefatos
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), qualidade]
        result, encimg = cv2.imencode('.jpg', imagem, encode_param)
        decimg = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
        return decimg
    
    def _aplicar_distorcao_geometrica(self, imagem, intensidade=0.05):
        """Aplica distorção geométrica leve (barrel/pincushion)"""
        h, w = imagem.shape
        
        # Criar mapas de distorção
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        
        center_x, center_y = w // 2, h // 2
        
        for y in range(h):
            for x in range(w):
                # Coordenadas normalizadas (-1 a 1)
                nx = (x - center_x) / center_x
                ny = (y - center_y) / center_y
                
                # Distância do centro
                r = np.sqrt(nx * nx + ny * ny)
                
                # Aplicar distorção barrel
                r_distorted = r * (1 + intensidade * r * r)
                
                # Converter de volta para coordenadas da imagem
                if r > 0:
                    map_x[y, x] = center_x + (nx * r_distorted / r) * center_x
                    map_y[y, x] = center_y + (ny * r_distorted / r) * center_y
                else:
                    map_x[y, x] = x
                    map_y[y, x] = y
        
        # Aplicar remapeamento
        resultado = cv2.remap(imagem, map_x, map_y, cv2.INTER_LINEAR)
        return resultado

    def limpar_fundo_separar_letras(self, imagem, timestamp=""):
        """
        Limpa completamente o fundo mantendo escala de cinza das letras e separa letras individuais
        """
        try:
            logger.info("🧹 LIMPEZA AVANÇADA: Removendo fundo e separando letras...")
            
            # Converter para escala de cinza se necessário
            if len(imagem.shape) == 3:
                img_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = imagem.copy()
            
            # 1. ANÁLISE DE HISTOGRAMA PARA DETECTAR FUNDO
            logger.info("📊 Analisando histograma para detectar pixels de fundo...")
            hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
            
            # Encontrar picos no histograma (cores mais comuns = fundo)
            # Suavizar histograma para melhor detecção de picos
            hist_1d = hist.flatten()
            hist_smooth = np.convolve(hist_1d, np.ones(3)/3, mode='same')  # Média móvel simples
            picos = []
            for i in range(1, 255):
                if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
                    picos.append((i, hist_smooth[i]))
            
            # Ordenar picos por intensidade (maior = mais comum = provavelmente fundo)
            picos.sort(key=lambda x: x[1], reverse=True)
            cor_fundo_principal = picos[0][0] if picos else 240  # Default: quase branco
            
            logger.info(f"🎯 Cor de fundo detectada: {cor_fundo_principal}")
            
            # 2. DETECÇÃO INTELIGENTE DE LETRAS (MÚLTIPLAS ESTRATÉGIAS)
            logger.info("🔍 Detectando letras com múltiplas estratégias...")
            
            # Estratégia 1: Threshold adaptativo
            thresh_adaptativo = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Estratégia 2: Threshold OTSU
            _, thresh_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Estratégia 3: Análise de gradiente (bordas das letras)
            grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
            _, thresh_grad = cv2.threshold(magnitude, 30, 255, cv2.THRESH_BINARY)
            
            # Combinar estratégias (interseção das detecções)
            mask_letras = cv2.bitwise_and(thresh_adaptativo, thresh_otsu)
            mask_letras = cv2.bitwise_or(mask_letras, thresh_grad)
            
            # 3. LIMPEZA MORFOLÓGICA PARA REMOVER RUÍDOS
            logger.info("🔧 Aplicando limpeza morfológica...")
            
            # Remover ruídos pequenos
            kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            mask_letras = cv2.morphologyEx(mask_letras, cv2.MORPH_OPEN, kernel_noise)
            
            # Conectar partes fragmentadas das letras
            kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_letras = cv2.morphologyEx(mask_letras, cv2.MORPH_CLOSE, kernel_connect)
            
            # 4. CRIAR FUNDO LIMPO MANTENDO ESCALA DE CINZA DAS LETRAS
            logger.info("🎨 Criando fundo limpo mantendo escala de cinza original das letras...")
            
            # Criar imagem com fundo branco puro
            imagem_limpa = np.full_like(img_gray, 255, dtype=np.uint8)
            
            # Aplicar letras originais apenas onde detectadas, mantendo escala de cinza
            imagem_limpa[mask_letras > 0] = img_gray[mask_letras > 0]
            
            # 5. DETECÇÃO E SEPARAÇÃO DE LETRAS INDIVIDUAIS
            logger.info("✂️ Separando letras individuais...")
            
            # Encontrar contornos das letras
            contornos, _ = cv2.findContours(mask_letras, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar contornos por tamanho (remover ruídos muito pequenos)
            h, w = img_gray.shape
            area_minima = (h * w) * 0.001  # 0.1% da área total
            area_maxima = (h * w) * 0.3    # 30% da área total
            
            contornos_validos = []
            for contorno in contornos:
                area = cv2.contourArea(contorno)
                if area_minima <= area <= area_maxima:
                    # Verificar se não é muito alongado (provavelmente ruído)
                    x, y, w_cont, h_cont = cv2.boundingRect(contorno)
                    aspect_ratio = w_cont / h_cont if h_cont > 0 else 0
                    if 0.2 <= aspect_ratio <= 5.0:  # Proporções razoáveis para letras
                        contornos_validos.append(contorno)
            
            # Ordenar contornos da esquerda para direita (ordem de leitura)
            contornos_validos.sort(key=lambda c: cv2.boundingRect(c)[0])
            
            logger.info(f"📝 {len(contornos_validos)} letras detectadas e separadas")
            
            # 6. EXTRAIR LETRAS INDIVIDUAIS
            letras_separadas = []
            debug_separacao_dir = self.debug_rgb_dir / "letras_separadas"
            debug_separacao_dir.mkdir(exist_ok=True)
            
            for i, contorno in enumerate(contornos_validos):
                # Obter região da letra
                x, y, w_letra, h_letra = cv2.boundingRect(contorno)
                
                # Adicionar padding para melhor OCR
                padding = max(2, min(w_letra, h_letra) // 10)
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(imagem_limpa.shape[1], x + w_letra + padding)
                y_end = min(imagem_limpa.shape[0], y + h_letra + padding)
                
                # Extrair letra individual
                letra_individual = imagem_limpa[y_start:y_end, x_start:x_end]
                
                # Redimensionar para tamanho padrão (melhor para OCR)
                altura_padrao = 40
                aspecto = w_letra / h_letra if h_letra > 0 else 1
                largura_padrao = int(altura_padrao * aspecto)
                
                if largura_padrao > 0 and altura_padrao > 0:
                    letra_redimensionada = cv2.resize(letra_individual, (largura_padrao, altura_padrao), interpolation=cv2.INTER_CUBIC)
                    letras_separadas.append(letra_redimensionada)
                    
                    # Salvar debug da letra individual
                    if timestamp:
                        letra_path = debug_separacao_dir / f"letra_{i+1}_{timestamp}.png"
                        cv2.imwrite(str(letra_path), letra_redimensionada)
            
            # 7. SALVAR RESULTADOS DEBUG
            if timestamp:
                # Salvar imagem com fundo limpo
                fundo_limpo_path = self.debug_rgb_dir / f"fundo_limpo_{timestamp}.png"
                cv2.imwrite(str(fundo_limpo_path), imagem_limpa)
                
                # Salvar máscara de detecção
                mask_path = self.debug_rgb_dir / f"mask_letras_{timestamp}.png"
                cv2.imwrite(str(mask_path), mask_letras)
                
                # Salvar imagem com contornos marcados
                img_contornos = cv2.cvtColor(imagem_limpa, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(img_contornos, contornos_validos, -1, (0, 255, 0), 2)
                contornos_path = self.debug_rgb_dir / f"contornos_letras_{timestamp}.png"
                cv2.imwrite(str(contornos_path), img_contornos)
            
            logger.info("✅ Fundo limpo e letras separadas com sucesso!")
            logger.info(f"📊 Total de letras extraídas: {len(letras_separadas)}")
            
            return imagem_limpa, letras_separadas
            
        except Exception as e:
            logger.error(f"❌ Erro na limpeza de fundo e separação: {e}")
            # Retornar imagem original e lista vazia em caso de erro
            img_fallback = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) if len(imagem.shape) == 3 else imagem
            return img_fallback, []

    def processar_letras_individuais(self, letras_separadas, tipo_letra="branco"):
        """
        Processa letras separadas individualmente para melhor OCR
        """
        try:
            if not letras_separadas:
                return ""
            
            logger.info(f"🔤 Processando {len(letras_separadas)} letras individuais...")
            
            # Configurações para letra individual
            config_tesseract = "--oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            
            resultado_letras = []
            
            for i, letra in enumerate(letras_separadas):
                try:
                    # Redimensionar letra para tamanho ideal para OCR
                    altura_ocr = 60
                    largura_ocr = int(letra.shape[1] * (altura_ocr / letra.shape[0]))
                    if largura_ocr > 0:
                        letra_redim = cv2.resize(letra, (largura_ocr, altura_ocr), interpolation=cv2.INTER_CUBIC)
                    else:
                        letra_redim = letra
                    
                    # Aplicar threshold para melhor contraste
                    if tipo_letra == 'preto':
                        _, letra_thresh = cv2.threshold(letra_redim, 127, 255, cv2.THRESH_BINARY_INV)
                    else:
                        _, letra_thresh = cv2.threshold(letra_redim, 127, 255, cv2.THRESH_BINARY)
                    
                    # Tentar OCR na letra individual
                    texto_letra = None
                    
                    # Tesseract para letra individual
                    try:
                        texto_tess = pytesseract.image_to_string(letra_thresh, config=config_tesseract).strip()
                        if texto_tess and len(texto_tess) == 1 and texto_tess.isalnum():
                            texto_letra = texto_tess
                            logger.info(f"   📝 Letra {i+1}: '{texto_letra}' (Tesseract)")
                    except:
                        pass
                    
                    # EasyOCR para letra individual se Tesseract falhou
                    if not texto_letra:
                        try:
                            # Converter para formato aceito pelo EasyOCR
                            letra_bgr = cv2.cvtColor(letra_thresh, cv2.COLOR_GRAY2BGR)
                            resultado_easy = self.easyocr_reader.readtext(letra_bgr, paragraph=False, width_ths=0.9)
                            
                            for (bbox, text, conf) in resultado_easy:
                                text_clean = text.strip()
                                if text_clean and len(text_clean) == 1 and text_clean.isalnum() and conf > 0.3:
                                    texto_letra = text_clean
                                    logger.info(f"   📝 Letra {i+1}: '{texto_letra}' (EasyOCR, conf: {conf:.2f})")
                                    break
                        except:
                            pass
                    
                    # Se ainda não conseguiu, usar análise de forma como fallback
                    if not texto_letra:
                        # Análise básica de área e proporção (fallback)
                        area = cv2.countNonZero(letra_thresh)
                        h, w = letra_thresh.shape
                        if area > (h * w * 0.1):  # Se tem conteúdo suficiente
                            texto_letra = "?"  # Placeholder para letra não reconhecida
                            logger.info(f"   ❓ Letra {i+1}: não reconhecida (área: {area})")
                    
                    if texto_letra:
                        resultado_letras.append(texto_letra)
                    
                except Exception as e:
                    logger.warning(f"   ⚠️ Erro ao processar letra {i+1}: {e}")
                    continue
            
            texto_final = "".join(resultado_letras)
            logger.info(f"🎯 Texto combinado das letras individuais: '{texto_final}'")
            
            return texto_final
            
        except Exception as e:
            logger.error(f"❌ Erro no processamento de letras individuais: {e}")
            return ""

    def focar_caracteres_grayscale(self, img_original, timestamp):
        """
        FOCO EM CARACTERES - Converte para grayscale e isola apenas os caracteres
        Abordagem limpa e precisa sem ataques massivos
        """
        try:
            logger.info("📝 INICIANDO FOCO EM CARACTERES EM GRAYSCALE")
            logger.info("🔍 Convertendo para grayscale e detectando caracteres...")
            
            # Converter para escala de cinza
            if len(img_original.shape) == 3:
                img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img_original.copy()
            
            h, w = img_gray.shape
            
            # 1. ANÁLISE DE INTENSIDADE DOS CARACTERES
            logger.info("📊 Analisando intensidade dos caracteres...")
            
            # Calcular histograma para detectar picos de intensidade
            hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
            
            # Encontrar os dois picos principais (fundo e caracteres)
            peaks = []
            for i in range(1, 255):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.max(hist) * 0.1:
                    peaks.append((i, hist[i]))
            
            # Ordenar picos por frequência
            peaks.sort(key=lambda x: x[1], reverse=True)
            
            # MELHORIA: Tratamento para casos extremos
            valores_unicos = np.unique(img_gray)
            
            # Determinar intensidade do fundo e dos caracteres
            if len(peaks) >= 2:
                fundo_intensity = max(peaks[0][0], peaks[1][0])  # Mais claro
                char_intensity = min(peaks[0][0], peaks[1][0])   # Mais escuro
            elif len(valores_unicos) >= 3:
                # CASO ESPECIAL: Imagem com poucos valores únicos (como 3, 4, 255)
                valores_ordenados = sorted(valores_unicos)
                if valores_ordenados[-1] > 200:  # Último valor é muito claro (fundo)
                    fundo_intensity = valores_ordenados[-1]
                    char_intensity = valores_ordenados[0]  # Primeiro valor (mais escuro)
                else:
                    fundo_intensity = valores_ordenados[-1]
                    char_intensity = valores_ordenados[0]
            else:
                # Fallback: assumir fundo claro e caracteres escuros baseado na média
                media = np.mean(img_gray)
                if media > 127:  # Imagem predominantemente clara
                    fundo_intensity = 240
                    char_intensity = 80
                else:  # Imagem predominantemente escura
                    fundo_intensity = 80
                    char_intensity = 240
            
            logger.info(f"📊 Valores únicos: {valores_unicos[:10]}...")  # Mostrar só os primeiros 10
            logger.info(f"📊 Picos detectados: {[p[0] for p in peaks[:5]]}")  # Mostrar só os primeiros 5
            logger.info(f"📊 Fundo: {fundo_intensity}, Caracteres: {char_intensity}")
            
            # 2. THRESHOLD INTELIGENTE BASEADO NA INTENSIDADE
            logger.info("🎯 Aplicando threshold inteligente...")
            
            # Calcular threshold otimizado entre fundo e caracteres
            threshold_value = (fundo_intensity + char_intensity) // 2
            
            # PROTEÇÃO: Ajustar threshold se muito extremo
            if threshold_value < 10:
                threshold_value = 50
            elif threshold_value > 245:
                threshold_value = 200
            
            # Aplicar threshold
            if char_intensity < fundo_intensity:
                # Caracteres são mais escuros que o fundo
                _, img_thresh = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
            else:
                # Caracteres são mais claros que o fundo
                _, img_thresh = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)
            
            # VERIFICAÇÃO: Se resultado muito extremo, ajustar com OTSU
            pixels_brancos = np.sum(img_thresh == 255)
            total_pixels = img_thresh.shape[0] * img_thresh.shape[1]
            percentual_branco = pixels_brancos / total_pixels
            
            if percentual_branco > 0.95 or percentual_branco < 0.01:
                logger.info(f"⚠️ Resultado extremo ({percentual_branco:.3f}), usando OTSU...")
                threshold_auto = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[0]
                _, img_thresh = cv2.threshold(img_gray, threshold_auto, 255, cv2.THRESH_BINARY_INV)
                threshold_value = threshold_auto
            
            logger.info(f"🎯 Threshold aplicado: {threshold_value}")
            
            # 2.5. SUAVIZAÇÃO PRÉVIA PARA REDUZIR RUÍDO
            logger.info("🌊 Aplicando suavização Gaussiana para reduzir ruído...")
            
            # Suavização Gaussiana antes do threshold (inspirado no código fornecido)
            img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
            
            # Recalcular threshold com imagem suavizada para melhor resultado
            if char_intensity < fundo_intensity:
                _, img_thresh = cv2.threshold(img_blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
            else:
                _, img_thresh = cv2.threshold(img_blurred, threshold_value, 255, cv2.THRESH_BINARY)
            
            # ALTERNATIVA: Usar threshold adaptativo se o resultado ainda for extremo
            pixels_brancos_teste = np.sum(img_thresh == 255)
            percentual_teste = pixels_brancos_teste / total_pixels
            
            if percentual_teste > 0.95 or percentual_teste < 0.01:
                logger.info("🎯 Usando threshold adaptativo Gaussiano...")
                img_thresh = cv2.adaptiveThreshold(
                    img_blurred, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    11, 2
                )
            
            # 3. LIMPEZA MORFOLÓGICA AVANÇADA
            logger.info("🧹 Aplicando limpeza morfológica avançada...")
            
            # Usar kernel circular para melhor limpeza (inspirado no código)
            kernel_noise = np.ones((3, 3), np.uint8)
            img_clean = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_noise)
            
            # Conectar fragmentos próximos dos caracteres
            kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            img_clean = cv2.morphologyEx(img_clean, cv2.MORPH_CLOSE, kernel_connect)
            
            # 4. ANÁLISE DE COMPONENTES CONECTADOS
            logger.info("🔍 Analisando componentes conectados...")
            
            # Encontrar componentes conectados
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_clean, connectivity=8)
            
            # Filtrar componentes por tamanho (remover muito pequenos e muito grandes)
            min_area = max(20, (h * w) // 1000)  # Pelo menos 20 pixels ou 0.1% da imagem
            max_area = (h * w) // 5  # No máximo 20% da imagem
            
            img_filtered = np.zeros_like(img_clean)
            
            valid_components = 0
            for i in range(1, num_labels):  # Pular o fundo (label 0)
                area = stats[i, cv2.CC_STAT_AREA]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Filtros para caracteres válidos
                aspect_ratio = width / height if height > 0 else 0
                is_valid_size = min_area <= area <= max_area
                is_valid_aspect = 0.2 <= aspect_ratio <= 5.0  # Proporções razoáveis para caracteres
                is_valid_dimensions = width >= 5 and height >= 8  # Tamanho mínimo para ser um caractere
                
                if is_valid_size and is_valid_aspect and is_valid_dimensions:
                    img_filtered[labels == i] = 255
                    valid_components += 1
                    logger.info(f"   ✅ Componente {valid_components}: {area}px, {width}x{height}, ratio={aspect_ratio:.2f}")
            
            logger.info(f"🔍 {valid_components} componentes válidos encontrados")
            
            # 5. PÓS-PROCESSAMENTO FINAL
            logger.info("✨ Aplicando pós-processamento final...")
            
            # Garantir fundo branco e caracteres pretos
            img_final = img_filtered.copy()
            
            # Aplicar leve suavização para melhorar OCR
            img_final = cv2.medianBlur(img_final, 3)
            
            # NOVO: OCR MELHORADO INSPIRADO NO CÓDIGO FORNECIDO
            logger.info("🔤 Aplicando OCR melhorado...")
            try:
                ocr_result = self.ocr_melhorado_caracteres(img_final)
                if ocr_result and len(ocr_result.strip()) > 0:
                    logger.info(f"✅ OCR detectou: '{ocr_result}'")
                else:
                    logger.info("⚠️ OCR não detectou texto válido")
            except Exception as e:
                logger.warning(f"⚠️ Erro no OCR melhorado: {e}")
            
            # Salvar resultado final
            char_focus_path = self.debug_rgb_dir / f"FOCO_CARACTERES_{timestamp}.png"
            cv2.imwrite(str(char_focus_path), img_final)
            
            logger.info("📝 FOCO EM CARACTERES CONCLUÍDO COM SUCESSO!")
            return img_final
            
        except Exception as e:
            logger.error(f"❌ Erro no foco em caracteres: {e}")
            # Fallback: usar threshold simples
            if len(img_original.shape) == 3:
                img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img_original.copy()
            _, fallback = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return fallback

    def ocr_melhorado_caracteres(self, img_limpa):
        """
        OCR ULTRA FINO - Múltiplas estratégias de pré-processamento e OCR otimizadas
        Aplica análise avançada de imagem e configurações específicas por tipo de caractere
        """
        try:
            logger.info("🔤 INICIANDO OCR ULTRA FINO COM MÚLTIPLAS ESTRATÉGIAS...")
            
            # Se a imagem já está em grayscale, use diretamente
            if len(img_limpa.shape) == 3:
                gray = cv2.cvtColor(img_limpa, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_limpa.copy()
            
            h, w = gray.shape
            timestamp = datetime.now().strftime("%H%M%S")
            
            # 1. ANÁLISE DETALHADA DA IMAGEM
            logger.info("📊 Análise detalhada da imagem...")
            
            # Calcular estatísticas básicas
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            min_intensity = np.min(gray)
            max_intensity = np.max(gray)
            
            # Calcular contraste local usando Laplaciano
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Análise de histograma
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_peaks = []
            for i in range(1, 255):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.max(hist) * 0.1:
                    hist_peaks.append((i, hist[i]))
            
            logger.info(f"📊 Intensidade média: {mean_intensity:.1f}, desvio: {std_intensity:.1f}")
            logger.info(f"📊 Range: {min_intensity}-{max_intensity}, contraste: {laplacian_var:.2f}")
            logger.info(f"📊 Picos no histograma: {len(hist_peaks)}")
            
            # 2. ESTRATÉGIAS DE PRÉ-PROCESSAMENTO MÚLTIPLAS
            logger.info("⚡ Aplicando múltiplas estratégias de pré-processamento...")
            
            estrategias = []
            
            # ESTRATÉGIA 1: Suavização Gaussiana + Threshold Adaptativo Gaussiano
            logger.info("   🌊 Estratégia 1: Gaussiana + Adaptativo Gaussiano")
            blur1 = cv2.GaussianBlur(gray, (3, 3), 0)
            thresh1 = cv2.adaptiveThreshold(blur1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            estrategias.append(("gaussiana_adaptativo", thresh1))
            
            # ESTRATÉGIA 2: Suavização Bilateral + Threshold Adaptativo Mean
            logger.info("   💫 Estratégia 2: Bilateral + Adaptativo Mean")
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            thresh2 = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3)
            estrategias.append(("bilateral_mean", thresh2))
            
            # ESTRATÉGIA 3: CLAHE + OTSU
            logger.info("   ✨ Estratégia 3: CLAHE + OTSU")
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(gray)
            _, thresh3 = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            estrategias.append(("clahe_otsu", thresh3))
            
            # ESTRATÉGIA 4: Morfologia Avançada + Threshold Manual
            logger.info("   🔧 Estratégia 4: Morfologia + Threshold Manual")
            # Usar threshold baseado na análise de intensidade
            if len(hist_peaks) >= 2:
                intensities = [p[0] for p in sorted(hist_peaks, key=lambda x: x[1], reverse=True)]
                fundo = max(intensities[0], intensities[1])
                char = min(intensities[0], intensities[1])
                manual_thresh = (fundo + char) // 2
            else:
                manual_thresh = int(mean_intensity)
            
            _, thresh4_base = cv2.threshold(gray, manual_thresh, 255, cv2.THRESH_BINARY_INV)
            
            # Aplicar morfologia avançada
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh4 = cv2.morphologyEx(thresh4_base, cv2.MORPH_OPEN, kernel_open)
            thresh4 = cv2.morphologyEx(thresh4, cv2.MORPH_CLOSE, kernel_close)
            estrategias.append(("morfologia_manual", thresh4))
            
            # ESTRATÉGIA 5: Filtro de Mediana + Threshold Duplo
            logger.info("   🎯 Estratégia 5: Mediana + Threshold Duplo")
            median_img = cv2.medianBlur(gray, 5)
            # Threshold duplo para capturar diferentes intensidades
            _, thresh5a = cv2.threshold(median_img, manual_thresh - 20, 255, cv2.THRESH_BINARY_INV)
            _, thresh5b = cv2.threshold(median_img, manual_thresh + 20, 255, cv2.THRESH_BINARY_INV)
            thresh5 = cv2.bitwise_or(thresh5a, thresh5b)
            estrategias.append(("mediana_duplo", thresh5))
            
            # 3. AVALIAÇÃO E SELEÇÃO DA MELHOR ESTRATÉGIA
            logger.info("🏆 Avaliando qualidade de cada estratégia...")
            
            scores = []
            for nome, img_estrategia in estrategias:
                # Métricas de qualidade
                
                # 1. Densidade de pixels de texto (ideal: 5-25%)
                pixels_texto = cv2.countNonZero(img_estrategia)
                densidade = pixels_texto / (h * w)
                
                # 2. Número de componentes conectados (ideal: 4-8 para CAPTCHAs típicos)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_estrategia, connectivity=8)
                componentes_validos = 0
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    width = stats[i, cv2.CC_STAT_WIDTH]
                    height = stats[i, cv2.CC_STAT_HEIGHT]
                    aspect_ratio = width / height if height > 0 else 0
                    
                    # Critérios para caracteres válidos
                    if 20 <= area <= (h * w) // 10 and 0.2 <= aspect_ratio <= 5.0:
                        componentes_validos += 1
                
                # 3. Qualidade das bordas (nitidez)
                edges = cv2.Canny(img_estrategia, 50, 150)
                edge_density = cv2.countNonZero(edges) / (h * w)
                
                # Calcular score composto
                densidade_score = 1.0 if 0.05 <= densidade <= 0.25 else max(0.1, 1.0 - abs(densidade - 0.15) * 5)
                componentes_score = 1.0 if 3 <= componentes_validos <= 8 else max(0.1, 1.0 - abs(componentes_validos - 5) * 0.2)
                edge_score = min(1.0, edge_density * 10)  # Normalizar
                
                score_total = (densidade_score * 0.4 + componentes_score * 0.4 + edge_score * 0.2)
                
                scores.append((nome, score_total, densidade, componentes_validos, edge_density))
                logger.info(f"   📈 {nome}: score={score_total:.3f} (densidade={densidade:.3f}, comp={componentes_validos}, edges={edge_density:.3f})")
                
                # Salvar estratégia para debug
                debug_strategy_path = self.debug_rgb_dir / f"estrategia_{nome}_{timestamp}.png"
                cv2.imwrite(str(debug_strategy_path), img_estrategia)
            
            # Selecionar melhor estratégia
            melhor_estrategia = max(scores, key=lambda x: x[1])
            nome_melhor, score_melhor = melhor_estrategia[0], melhor_estrategia[1]
            logger.info(f"🏆 MELHOR ESTRATÉGIA: {nome_melhor} (score: {score_melhor:.3f})")
            
            # Obter imagem da melhor estratégia
            img_otimizada = next(img for nome, img in estrategias if nome == nome_melhor)
            
            # 4. PÓS-PROCESSAMENTO FINO NA MELHOR ESTRATÉGIA
            logger.info("✨ Aplicando pós-processamento fino...")
            
            # Limpeza morfológica específica
            kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            img_final = cv2.morphologyEx(img_otimizada, cv2.MORPH_OPEN, kernel_clean)
            
            # Conectar fragmentos de caracteres
            kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            img_final = cv2.morphologyEx(img_final, cv2.MORPH_CLOSE, kernel_connect)
            
            # Redimensionamento inteligente para melhor OCR
            target_height = 64  # Altura otimizada para Tesseract
            if h < target_height:
                scale_factor = target_height / h
                new_width = int(w * scale_factor)
                img_final = cv2.resize(img_final, (new_width, target_height), interpolation=cv2.INTER_CUBIC)
                logger.info(f"🔍 Redimensionado para {new_width}x{target_height} (escala: {scale_factor:.2f}x)")
            
            # 5. SALVAR RESULTADO FINAL
            debug_final_path = self.debug_rgb_dir / f"OCR_ULTRA_FINO_{nome_melhor}_{timestamp}.png"
            cv2.imwrite(str(debug_final_path), img_final)
            
            # 6. OCR COM MÚLTIPLAS CONFIGURAÇÕES OTIMIZADAS
            logger.info("🔍 Executando OCR com configurações múltiplas...")
            
            # Configurações específicas otimizadas
            configs_tesseract = [
                # Configuração básica otimizada
                r'--oem 3 --psm 8 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                # Configuração para texto de linha única
                r'--oem 3 --psm 7 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                # Configuração para caractere único (tentativa palavra por palavra)
                r'--oem 3 --psm 10 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                # Configuração com menos restrições
                r'--oem 3 --psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            ]
            
            resultados_ocr = []
            
            try:
                import pytesseract
                
                for i, config in enumerate(configs_tesseract):
                    try:
                        text = pytesseract.image_to_string(img_final, config=config).strip()
                        text_clean = ''.join(c for c in text if c.isalnum())
                        
                        if text_clean and len(text_clean) >= 3:  # Mínimo 3 caracteres
                            # Calcular confiança baseada no comprimento e caracteres válidos
                            confianca = min(1.0, len(text_clean) / 6.0)  # 6 chars = 100% confiança
                            resultados_ocr.append((text_clean, confianca, f"config_{i+1}"))
                            logger.info(f"📝 Config {i+1}: '{text_clean}' (conf: {confianca:.2f})")
                    
                    except Exception as e:
                        logger.debug(f"Erro na config {i+1}: {e}")
                        continue
                
                # Selecionar melhor resultado
                if resultados_ocr:
                    # Ordenar por confiança e comprimento
                    resultados_ocr.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
                    melhor_resultado = resultados_ocr[0]
                    text_final = melhor_resultado[0]
                    
                    logger.info(f"🏆 MELHOR OCR: '{text_final}' ({melhor_resultado[2]}, conf: {melhor_resultado[1]:.2f})")
                    
                    # NOVA FUNCIONALIDADE: Análise e correção pós-OCR
                    logger.info("🔍 Aplicando análise e correção pós-OCR...")
                    text_corrigido = self.analisar_e_corrigir_ocr(text_final, img_limpa)
                    
                    if text_corrigido != text_final:
                        logger.info(f"✅ Texto corrigido: '{text_final}' → '{text_corrigido}'")
                        return text_corrigido
                    else:
                        logger.info("✅ Texto validado sem necessidade de correção")
                        return text_final
                else:
                    logger.warning("⚠️ Nenhum resultado OCR válido encontrado")
                    return ""
                    
            except ImportError:
                logger.warning("⚠️ pytesseract não disponível")
                return ""
            except Exception as e:
                logger.error(f"❌ Erro no OCR: {e}")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Erro no OCR ultra fino: {e}")
            return ""
                
    def preprocessamento_adaptativo_avancado(self, img_original):
        """
        Sistema de pré-processamento adaptativo ultra avançado
        Analisa a imagem e aplica a estratégia de limpeza mais adequada
        """
        try:
            logger.info("🧠 INICIANDO PRÉ-PROCESSAMENTO ADAPTATIVO AVANÇADO")
            
            timestamp = datetime.now().strftime("%H%M%S")
            
            # Converter para escala de cinza se necessário
            if len(img_original.shape) == 3:
                img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img_original.copy()
            
            h, w = img_gray.shape
            
            # 1. ANÁLISE ESTATÍSTICA PROFUNDA DA IMAGEM
            logger.info("📊 Análise estatística profunda...")
            
            # Estatísticas básicas
            mean_val = np.mean(img_gray)
            std_val = np.std(img_gray)
            min_val = np.min(img_gray)
            max_val = np.max(img_gray)
            
            # Análise de distribuição
            percentis = np.percentile(img_gray, [10, 25, 50, 75, 90])
            range_dinamico = max_val - min_val
            
            # Análise de complexidade
            laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
            
            # Análise de entropia (complexidade da informação)
            hist, _ = np.histogram(img_gray, bins=256, range=(0, 255))
            hist_norm = hist / hist.sum()
            entropia = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
            
            logger.info(f"📊 Média: {mean_val:.1f}, Desvio: {std_val:.1f}")
            logger.info(f"📊 Range: {min_val}-{max_val} (dinâmico: {range_dinamico})")
            logger.info(f"📊 Percentis: 10%={percentis[0]:.0f}, 50%={percentis[2]:.0f}, 90%={percentis[4]:.0f}")
            logger.info(f"📊 Complexidade (Laplacian): {laplacian_var:.2f}")
            logger.info(f"📊 Entropia: {entropia:.2f}")
            
            # 2. CLASSIFICAÇÃO AUTOMÁTICA DO TIPO DE IMAGEM
            logger.info("🏷️ Classificando tipo de imagem...")
            
            tipo_imagem = "indefinido"
            estrategia_recomendada = "padrao"
            
            # Classificação baseada em múltiplos critérios
            if range_dinamico < 100:
                tipo_imagem = "baixo_contraste"
                estrategia_recomendada = "clahe_agressivo"
            elif std_val < 30:
                tipo_imagem = "uniforme"
                estrategia_recomendada = "threshold_adaptativo"
            elif laplacian_var > 1000:
                tipo_imagem = "alta_complexidade"
                estrategia_recomendada = "morfologia_avancada"
            elif mean_val < 80:
                tipo_imagem = "predominante_escuro"
                estrategia_recomendada = "inversao_otimizada"
            elif mean_val > 180:
                tipo_imagem = "predominante_claro"
                estrategia_recomendada = "threshold_otsu"
            elif entropia > 6:
                tipo_imagem = "alta_entropia"
                estrategia_recomendada = "filtragem_avancada"
            else:
                tipo_imagem = "balanceado"
                estrategia_recomendada = "pipeline_completo"
            
            logger.info(f"🏷️ Tipo identificado: {tipo_imagem}")
            logger.info(f"🎯 Estratégia recomendada: {estrategia_recomendada}")
            
            # 3. APLICAR ESTRATÉGIA ESPECÍFICA
            logger.info(f"⚡ Aplicando estratégia: {estrategia_recomendada}")
            
            if estrategia_recomendada == "clahe_agressivo":
                resultado = self._estrategia_clahe_agressivo(img_gray)
            elif estrategia_recomendada == "threshold_adaptativo":
                resultado = self._estrategia_threshold_adaptativo(img_gray)
            elif estrategia_recomendada == "morfologia_avancada":
                resultado = self._estrategia_morfologia_avancada(img_gray)
            elif estrategia_recomendada == "inversao_otimizada":
                resultado = self._estrategia_inversao_otimizada(img_gray)
            elif estrategia_recomendada == "threshold_otsu":
                resultado = self._estrategia_threshold_otsu(img_gray)
            elif estrategia_recomendada == "filtragem_avancada":
                resultado = self._estrategia_filtragem_avancada(img_gray)
            else:  # pipeline_completo
                resultado = self._estrategia_pipeline_completo(img_gray)
            
            # 4. VALIDAÇÃO E AJUSTE FINO
            logger.info("✨ Validação e ajuste fino...")
            
            # Verificar qualidade do resultado
            pixels_texto = cv2.countNonZero(resultado)
            densidade = pixels_texto / (h * w)
            
            logger.info(f"📊 Densidade de pixels de texto: {densidade:.3f}")
            
            # Se densidade muito alta ou baixa, aplicar correção
            if densidade > 0.8:
                logger.info("⚠️ Densidade muito alta, aplicando erosão...")
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                resultado = cv2.erode(resultado, kernel, iterations=1)
            elif densidade < 0.02:
                logger.info("⚠️ Densidade muito baixa, aplicando dilatação...")
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                resultado = cv2.dilate(resultado, kernel, iterations=1)
            
            # 5. SALVAR RESULTADO E METADADOS
            resultado_path = self.debug_rgb_dir / f"ADAPTATIVO_{estrategia_recomendada}_{timestamp}.png"
            cv2.imwrite(str(resultado_path), resultado)
            
            # Salvar metadados da análise
            metadata_path = self.debug_rgb_dir / f"adaptativo_metadata_{timestamp}.txt"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(f"ANÁLISE DE PRÉ-PROCESSAMENTO ADAPTATIVO\n")
                f.write(f"=======================================\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Tipo identificado: {tipo_imagem}\n")
                f.write(f"Estratégia aplicada: {estrategia_recomendada}\n")
                f.write(f"Dimensões: {w}x{h}\n")
                f.write(f"Estatísticas:\n")
                f.write(f"  - Média: {mean_val:.1f}\n")
                f.write(f"  - Desvio padrão: {std_val:.1f}\n")
                f.write(f"  - Range dinâmico: {range_dinamico}\n")
                f.write(f"  - Laplacian variance: {laplacian_var:.2f}\n")
                f.write(f"  - Entropia: {entropia:.2f}\n")
                f.write(f"Resultado:\n")
                f.write(f"  - Densidade final: {densidade:.3f}\n")
            
            logger.info("🧠 PRÉ-PROCESSAMENTO ADAPTATIVO CONCLUÍDO COM SUCESSO!")
            return resultado
            
        except Exception as e:
            logger.error(f"❌ Erro no pré-processamento adaptativo: {e}")
            # Fallback para threshold simples
            if len(img_original.shape) == 3:
                img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img_original.copy()
            _, fallback = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            return fallback

    def _estrategia_clahe_agressivo(self, img_gray):
        """Estratégia para imagens de baixo contraste"""
        logger.info("   🔆 Aplicando CLAHE agressivo...")
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(img_gray)
        _, result = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return result

    def _estrategia_threshold_adaptativo(self, img_gray):
        """Estratégia para imagens uniformes"""
        logger.info("   🎯 Aplicando threshold adaptativo múltiplo...")
        # Combinar diferentes tipos de threshold adaptativo
        thresh1 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        thresh2 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3)
        result = cv2.bitwise_or(thresh1, thresh2)
        return result

    def _estrategia_morfologia_avancada(self, img_gray):
        """Estratégia para imagens complexas"""
        logger.info("   🔧 Aplicando morfologia avançada...")
        # Threshold inicial
        _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morfologia avançada com múltiplos kernels
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel2)
        return closed

    def _estrategia_inversao_otimizada(self, img_gray):
        """Estratégia para imagens escuras"""
        logger.info("   🔄 Aplicando inversão otimizada...")
        # Inverter e aplicar threshold
        inverted = cv2.bitwise_not(img_gray)
        enhanced = cv2.equalizeHist(inverted)
        _, result = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return result

    def _estrategia_threshold_otsu(self, img_gray):
        """Estratégia para imagens claras"""
        logger.info("   📊 Aplicando threshold OTSU otimizado...")
        # Suavização prévia + OTSU
        blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
        _, result = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return result

    def _estrategia_filtragem_avancada(self, img_gray):
        """Estratégia para imagens com alta entropia (ruidosas)"""
        logger.info("   🧹 Aplicando filtragem avançada...")
        # Múltiplos filtros
        median_filtered = cv2.medianBlur(img_gray, 5)
        bilateral = cv2.bilateralFilter(median_filtered, 9, 75, 75)
        _, result = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return result

    def _estrategia_pipeline_completo(self, img_gray):
        """Estratégia pipeline completo para imagens balanceadas"""
        logger.info("   🚀 Aplicando pipeline completo...")
        # Pipeline de múltiplas etapas
        
        # 1. Suavização
        smoothed = cv2.bilateralFilter(img_gray, 5, 50, 50)
        
        # 2. Realce de contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(smoothed)
        
        # 3. Threshold adaptativo
        thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # 4. Limpeza morfológica
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        result = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel2)
        
        return result

    def analisar_e_corrigir_ocr(self, texto_ocr_bruto, img_original):
        """
        Análise e correção pós-OCR usando padrões de CAPTCHA e validação visual
        """
        try:
            logger.info(f"🔍 Analisando e corrigindo resultado OCR: '{texto_ocr_bruto}'")
            
            if not texto_ocr_bruto or len(texto_ocr_bruto.strip()) == 0:
                return texto_ocr_bruto
            
            texto_limpo = ''.join(c for c in texto_ocr_bruto if c.isalnum())
            
            # 1. CORREÇÕES BASEADAS EM PADRÕES COMUNS
            logger.info("🔧 Aplicando correções baseadas em padrões...")
            
            # Correções comuns OCR -> caractere correto
            correcoes_comuns = {
                # Números confundidos com letras
                '0': ['O', 'D', 'Q'],
                '1': ['I', 'l', '|'],
                '2': ['Z'],
                '5': ['S'],
                '6': ['G'],
                '8': ['B'],
                # Letras confundidas com números
                'O': ['0'],
                'I': ['1', 'l'],
                'S': ['5'],
                'G': ['6'],
                'B': ['8'],
                'Z': ['2'],
                # Letras minúsculas/maiúsculas confundidas
                'l': ['I', '1'],
                'o': ['O', '0'],
                'c': ['C'],
                'v': ['V'],
                'w': ['W'],
                'x': ['X'],
                'z': ['Z']
            }
            
            # 2. ANÁLISE CONTEXTUAL DO COMPRIMENTO
            logger.info("📏 Análise contextual do comprimento...")
            
            comprimento_esperado = 5  # CAPTCHAs típicos têm 4-6 caracteres
            
            if len(texto_limpo) > comprimento_esperado:
                logger.info(f"⚠️ Texto muito longo ({len(texto_limpo)} chars), pode ter ruído")
                # Tentar identificar a sequência mais provável
                # Buscar sequência contígua de caracteres alfanuméricos
                melhor_sequencia = ""
                for i in range(len(texto_limpo) - comprimento_esperado + 1):
                    subsequencia = texto_limpo[i:i + comprimento_esperado]
                    if len(subsequencia) == comprimento_esperado:
                        melhor_sequencia = subsequencia
                        break
                
                if melhor_sequencia:
                    logger.info(f"✂️ Extraída sequência provável: '{melhor_sequencia}'")
                    texto_limpo = melhor_sequencia
            
            elif len(texto_limpo) < 3:
                logger.warning(f"⚠️ Texto muito curto ({len(texto_limpo)} chars), pode estar incompleto")
            
            # 3. VALIDAÇÃO VISUAL CONTRA A IMAGEM
            logger.info("👁️ Validação visual contra a imagem...")
            
            # Converter imagem para análise
            if len(img_original.shape) == 3:
                img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img_original.copy()
            
            # Analisar componentes conectados para verificar número de caracteres
            _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            # Contar componentes que parecem caracteres
            h, w = img_gray.shape
            caracteres_visuais = 0
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                aspect_ratio = width / height if height > 0 else 0
                
                # Critérios para ser um caractere
                min_area = (h * w) // 500  # Pelo menos 0.2% da imagem
                max_area = (h * w) // 10   # No máximo 10% da imagem
                
                if (min_area <= area <= max_area and 
                    0.2 <= aspect_ratio <= 4.0 and 
                    width >= 4 and height >= 6):
                    caracteres_visuais += 1
            
            logger.info(f"👁️ Caracteres detectados visualmente: {caracteres_visuais}")
            logger.info(f"📝 Caracteres no OCR: {len(texto_limpo)}")
            
            # Validar consistência
            if abs(caracteres_visuais - len(texto_limpo)) > 2:
                logger.warning(f"⚠️ Inconsistência: {caracteres_visuais} visuais vs {len(texto_limpo)} OCR")
            
            # 4. APLICAR CORREÇÕES INTELIGENTES
            logger.info("🧠 Aplicando correções inteligentes...")
            
            texto_corrigido = ""
            
            for char in texto_limpo:
                # Para cada caractere, verificar se é provável baseado no contexto
                char_corrigido = char
                
                # Se o caractere está nas correções comuns, avaliar contexto
                if char in correcoes_comuns:
                    # Por enquanto manter o caractere original
                    # Em versões futuras, pode implementar análise de contexto mais sofisticada
                    pass
                
                texto_corrigido += char_corrigido
            
            # 5. VALIDAÇÕES FINAIS
            logger.info("✅ Validações finais...")
            
            # Remover caracteres especiais que podem ter vazado
            texto_final = ''.join(c for c in texto_corrigido if c.isalnum())
            
            # Verificar se resultado é válido
            if len(texto_final) >= 3 and len(texto_final) <= 8:
                logger.info(f"✅ Resultado final válido: '{texto_final}'")
                return texto_final
            else:
                logger.warning(f"⚠️ Resultado suspeito: '{texto_final}' (comprimento: {len(texto_final)})")
                return texto_final  # Retornar mesmo assim, pode ser útil
            
        except Exception as e:
            logger.error(f"❌ Erro na análise e correção: {e}")
            return texto_ocr_bruto

    def ataque_massivo_pixels(self, img_original, timestamp):
        try:
            logger.info("⚔️ INICIANDO ATAQUE MASSIVO AOS PIXELS ⚔️")
            logger.info("🎯 Combinando todas as armas disponíveis...")
            
            # Converter para escala de cinza se necessário
            if len(img_original.shape) == 3:
                img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img_original.copy()
            
            h, w = img_gray.shape
            resultados_ataques = {}
            
            # 1. ATAQUE POR CANAIS RGB INDIVIDUAIS
            logger.info("🔴 ATAQUE 1: Análise RGB individual por canal...")
            if len(img_original.shape) == 3:
                b, g, r = cv2.split(img_original)
                
                # Ataque ao canal vermelho
                r_limpo = self.processar_canal_individual(r, "VERMELHO", timestamp)
                resultados_ataques['vermelho'] = r_limpo
                
                # Ataque ao canal verde
                g_limpo = self.processar_canal_individual(g, "VERDE", timestamp)
                resultados_ataques['verde'] = g_limpo
                
                # Ataque ao canal azul
                b_limpo = self.processar_canal_individual(b, "AZUL", timestamp)
                resultados_ataques['azul'] = b_limpo
            
            # 2. ATAQUE POR VARIÂNCIA EXTREMA
            logger.info("📊 ATAQUE 2: Variância extrema pixel a pixel...")
            ataque_variancia = self.ataque_variancia_extrema(img_gray, timestamp)
            resultados_ataques['variancia'] = ataque_variancia
            
            # 3. ATAQUE POR MORFOLOGIA AGRESSIVA
            logger.info("🔧 ATAQUE 3: Morfologia agressiva...")
            ataque_morfologia = self.ataque_morfologico_agressivo(img_gray, timestamp)
            resultados_ataques['morfologia'] = ataque_morfologia
            
            # 4. ATAQUE POR CLUSTERING DE PIXELS
            logger.info("🎯 ATAQUE 4: Clustering agressivo de pixels...")
            ataque_clustering = self.ataque_clustering_pixels(img_gray, timestamp)
            resultados_ataques['clustering'] = ataque_clustering
            
            # 5. ATAQUE POR GRADIENTE EXTREMO
            logger.info("⚡ ATAQUE 5: Gradiente extremo...")
            ataque_gradiente = self.ataque_gradiente_extremo(img_gray, timestamp)
            resultados_ataques['gradiente'] = ataque_gradiente
            
            # 6. ATAQUE POR THRESHOLD ADAPTATIVO MÚLTIPLO
            logger.info("🎭 ATAQUE 6: Threshold adaptativo múltiplo...")
            ataque_threshold = self.ataque_threshold_multiplo(img_gray, timestamp)
            resultados_ataques['threshold'] = ataque_threshold
            
            # 7. COMBINAÇÃO FINAL - FUSÃO DE TODOS OS ATAQUES
            logger.info("💥 ATAQUE FINAL: Fusão de todos os métodos...")
            resultado_final = self.fusao_todos_ataques(resultados_ataques, img_gray, timestamp)
            
            # Aplicar lâminas de ruído ao resultado final
            self.gerar_laminas_ruido(resultado_final, f"ataque_{timestamp}")
            
            logger.info("⚔️ ATAQUE MASSIVO CONCLUÍDO COM SUCESSO! ⚔️")
            return resultado_final
            
        except Exception as e:
            logger.error(f"❌ Erro no ataque massivo aos pixels: {e}")
            return img_gray

    def processar_canal_individual(self, canal, nome_canal, timestamp):
        """Processa um canal RGB individual com técnicas agressivas"""
        try:
            logger.info(f"   🎯 Atacando canal {nome_canal}...")
            
            # Equalização de histograma agressiva
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4))
            canal_eq = clahe.apply(canal)
            
            # Threshold múltiplo
            _, thresh1 = cv2.threshold(canal_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, thresh2 = cv2.threshold(canal_eq, 127, 255, cv2.THRESH_BINARY)
            thresh_combined = cv2.bitwise_or(thresh1, thresh2)
            
            # Operações morfológicas agressivas
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            
            # Abertura e fechamento
            opening = cv2.morphologyEx(thresh_combined, cv2.MORPH_OPEN, kernel_small)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_large)
            
            # Salvar resultado do canal
            canal_path = self.debug_rgb_dir / f"ataque_canal_{nome_canal.lower()}_{timestamp}.png"
            cv2.imwrite(str(canal_path), closing)
            
            logger.info(f"   ✅ Canal {nome_canal} atacado e salvo")
            return closing
            
        except Exception as e:
            logger.error(f"❌ Erro no ataque ao canal {nome_canal}: {e}")
            return canal

    def ataque_variancia_extrema(self, img, timestamp):
        """Ataque usando variância extrema para detectar bordas sutis"""
        try:
            # Variância local extrema
            kernel_var = np.ones((5,5), np.float32) / 25
            img_float = img.astype(np.float32)
            
            # Múltiplas escalas de variância
            mean_3x3 = cv2.filter2D(img_float, -1, np.ones((3,3), np.float32) / 9)
            var_3x3 = cv2.filter2D((img_float - mean_3x3)**2, -1, np.ones((3,3), np.float32) / 9)
            
            mean_7x7 = cv2.filter2D(img_float, -1, np.ones((7,7), np.float32) / 49)
            var_7x7 = cv2.filter2D((img_float - mean_7x7)**2, -1, np.ones((7,7), np.float32) / 49)
            
            # Combinar variâncias
            var_combined = var_3x3 + var_7x7
            var_normalized = cv2.normalize(var_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Threshold extremo
            _, result = cv2.threshold(var_normalized, np.percentile(var_normalized, 85), 255, cv2.THRESH_BINARY)
            
            # Salvar resultado
            var_path = self.debug_rgb_dir / f"ataque_variancia_{timestamp}.png"
            cv2.imwrite(str(var_path), result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro no ataque de variância: {e}")
            return img

    def ataque_morfologico_agressivo(self, img, timestamp):
        """Ataque morfológico extremamente agressivo"""
        try:
            # Múltiplos kernels morfológicos
            kernels = [
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)),
                cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)),
                cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            ]
            
            # Threshold inicial
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            results = []
            for i, kernel in enumerate(kernels):
                # Sequência morfológica agressiva
                opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
                gradient = cv2.morphologyEx(closed, cv2.MORPH_GRADIENT, kernel)
                results.append(gradient)
            
            # Combinar todos os resultados
            result_combined = results[0]
            for result in results[1:]:
                result_combined = cv2.bitwise_or(result_combined, result)
            
            # Salvar resultado
            morph_path = self.debug_rgb_dir / f"ataque_morfologia_{timestamp}.png"
            cv2.imwrite(str(morph_path), result_combined)
            
            return result_combined
            
        except Exception as e:
            logger.error(f"❌ Erro no ataque morfológico: {e}")
            return img

    def ataque_clustering_pixels(self, img, timestamp):
        """Ataque usando clustering K-means agressivo"""
        try:
            # Preparar dados para clustering
            pixel_values = img.reshape((-1, 1)).astype(np.float32)
            
            # K-means com múltiplos K
            resultados_k = []
            for k in [2, 3, 4, 5]:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
                _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                
                # Reconstruir imagem
                centers = np.uint8(centers)
                segmented_data = centers[labels.flatten()]
                segmented_image = segmented_data.reshape((img.shape))
                
                # Threshold no resultado
                _, thresh_seg = cv2.threshold(segmented_image, 127, 255, cv2.THRESH_BINARY)
                resultados_k.append(thresh_seg)
            
            # Combinar resultados de diferentes K
            result_final = resultados_k[0]
            for resultado in resultados_k[1:]:
                result_final = cv2.bitwise_or(result_final, resultado)
            
            # Salvar resultado
            cluster_path = self.debug_rgb_dir / f"ataque_clustering_{timestamp}.png"
            cv2.imwrite(str(cluster_path), result_final)
            
            return result_final
            
        except Exception as e:
            logger.error(f"❌ Erro no ataque de clustering: {e}")
            return img

    def ataque_gradiente_extremo(self, img, timestamp):
        """Ataque usando gradientes em múltiplas direções"""
        try:
            # Gradientes Sobel em múltiplas escalas
            sobel_x1 = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y1 = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            sobel_x2 = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
            sobel_y2 = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
            
            # Magnitude dos gradientes
            magnitude1 = np.sqrt(sobel_x1**2 + sobel_y1**2)
            magnitude2 = np.sqrt(sobel_x2**2 + sobel_y2**2)
            
            # Combinar magnitudes
            magnitude_combined = magnitude1 + magnitude2
            magnitude_norm = cv2.normalize(magnitude_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Threshold extremo
            _, result = cv2.threshold(magnitude_norm, np.percentile(magnitude_norm, 75), 255, cv2.THRESH_BINARY)
            
            # Salvar resultado
            grad_path = self.debug_rgb_dir / f"ataque_gradiente_{timestamp}.png"
            cv2.imwrite(str(grad_path), result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro no ataque de gradiente: {e}")
            return img

    def ataque_threshold_multiplo(self, img, timestamp):
        """Ataque usando múltiplos tipos de threshold"""
        try:
            resultados_thresh = []
            
            # Threshold binário simples
            _, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            resultados_thresh.append(thresh1)
            
            # Threshold Otsu
            _, thresh2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            resultados_thresh.append(thresh2)
            
            # Threshold Triangle
            _, thresh3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
            resultados_thresh.append(thresh3)
            
            # Threshold adaptativo Gaussiano
            thresh4 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            resultados_thresh.append(thresh4)
            
            # Threshold adaptativo Mean
            thresh5 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            resultados_thresh.append(thresh5)
            
            # Combinar todos os thresholds
            result_final = resultados_thresh[0]
            for resultado in resultados_thresh[1:]:
                result_final = cv2.bitwise_or(result_final, resultado)
            
            # Salvar resultado
            thresh_path = self.debug_rgb_dir / f"ataque_threshold_{timestamp}.png"
            cv2.imwrite(str(thresh_path), result_final)
            
            return result_final
            
        except Exception as e:
            logger.error(f"❌ Erro no ataque de threshold: {e}")
            return img

    def fusao_todos_ataques(self, resultados_ataques, img_original, timestamp):
        """Funde todos os resultados dos ataques em uma imagem final"""
        try:
            logger.info("🔥 Fusionando todos os ataques...")
            
            # Começar com fundo branco
            h, w = img_original.shape
            fusao_final = np.full((h, w), 255, dtype=np.uint8)
            
            # Combinar todos os resultados usando OR
            for nome_ataque, resultado in resultados_ataques.items():
                if resultado is not None:
                    # Garantir que seja binário
                    _, resultado_bin = cv2.threshold(resultado, 127, 255, cv2.THRESH_BINARY)
                    # Aplicar letras pretas (pixels escuros onde há detecção)
                    fusao_final[resultado_bin == 255] = 0
                    logger.info(f"   ✅ Integrado ataque: {nome_ataque}")
            
            # Pós-processamento final
            # Remover ruídos muito pequenos
            kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            fusao_final = cv2.morphologyEx(fusao_final, cv2.MORPH_OPEN, kernel_clean)
            
            # Conectar fragmentos próximos
            kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fusao_final = cv2.morphologyEx(fusao_final, cv2.MORPH_CLOSE, kernel_connect)
            
            # Salvar resultado final
            fusao_path = self.debug_rgb_dir / f"FUSAO_FINAL_ATAQUES_{timestamp}.png"
            cv2.imwrite(str(fusao_path), fusao_final)
            
            logger.info("💥 FUSÃO COMPLETA - Todos os ataques combinados!")
            return fusao_final
            
        except Exception as e:
            logger.error(f"❌ Erro na fusão dos ataques: {e}")
            return img_original

    def processar_captcha_opencv(self, captcha_buffer, retornar_letras_separadas=False):
        """
        Processa CAPTCHA usando método otimizado de preservação de preto + sombras
        Baseado em análise extensiva que identificou threshold ≤85 como ótimo
        """
        try:
            logger.info("🎯 Processando CAPTCHA com método otimizado (Threshold ≤85)...")
            
            # Converter buffer para imagem numpy
            nparr = np.frombuffer(captcha_buffer, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("❌ Erro ao decodificar imagem")
                return captcha_buffer
            
            # Aplicar método otimizado de limpeza
            processed = self.limpar_captcha_otimizado(img)
            
            logger.info(f"✨ CAPTCHA processado com método otimizado (79.1% de confiança comprovada)")
            
            # Converter de volta para buffer
            _, buffer = cv2.imencode('.png', processed)
            logger.info("✅ CAPTCHA processado com método otimizado")
            
            if retornar_letras_separadas:
                # Para compatibilidade, retornar lista vazia de letras separadas
                return buffer.tobytes(), []
            else:
                return buffer.tobytes()
            
        except Exception as e:
            logger.error(f"❌ Erro no processamento: {e}")
            return captcha_buffer

    def limpar_captcha_otimizado(self, img):
        """
        Método otimizado de limpeza de CAPTCHA baseado em análise extensiva
        
        Estratégia: Preservar pixels pretos (0) + sombras (≤85) com análise automática
        Comprovadamente efetivo: 79.1% de confiança vs 36.5% original (+117% melhoria)
        
        SALVA AUTOMATICAMENTE imagens limpas em pasta 'captchas_limpos'
        """
        try:
            logger.info("🎯 Aplicando método de limpeza otimizado...")
            
            # Criar pasta para CAPTCHAs limpos se não existir
            captchas_limpos_dir = Path("captchas_limpos")
            captchas_limpos_dir.mkdir(exist_ok=True)
            
            # Converter para escala de cinza
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Análise automática da distribuição de pixels
            unique_values, counts = np.unique(img_gray, return_counts=True)
            total_pixels = img_gray.size
            
            # Detectar se é CAPTCHA com texto preto puro ou com sombras
            pixels_preto_puro = np.sum(img_gray == 0)
            percentual_preto = (pixels_preto_puro / total_pixels) * 100
            
            logger.info(f"📊 Análise do CAPTCHA:")
            logger.info(f"   • Pixels preto puro (0): {pixels_preto_puro:,} ({percentual_preto:.2f}%)")
            logger.info(f"   • Valores únicos: {len(unique_values)}")
            
            # Determinar threshold automático baseado na análise
            if percentual_preto >= 10.0:
                # CAPTCHA com muito texto preto puro - usar threshold conservador
                threshold_otimo = 10
                logger.info("🖤 CAPTCHA com texto preto dominante - usando threshold ≤10")
                
            elif percentual_preto >= 1.0:
                # CAPTCHA com texto preto moderado - usar threshold padrão otimizado 
                threshold_otimo = 85
                logger.info("⚫ CAPTCHA padrão - usando threshold ≤85 (comprovadamente ótimo)")
                
            else:
                # CAPTCHA com pouco preto - texto está nas sombras
                threshold_otimo = 85
                logger.info("🌑 CAPTCHA com texto em sombras - usando threshold ≤85")
            
            # Aplicar método de preservação de preto + sombras
            img_limpa = np.full_like(img_gray, 255, dtype=np.uint8)  # Fundo branco
            img_limpa[img_gray <= threshold_otimo] = 0  # Preservar texto + sombras
            
            # Contar resultado
            pixels_preservados = np.sum(img_limpa == 0)
            percentual_preservado = (pixels_preservados / total_pixels) * 100
            
            logger.info(f"✨ Resultado da limpeza:")
            logger.info(f"   • Threshold aplicado: ≤{threshold_otimo}")
            logger.info(f"   • Pixels preservados: {pixels_preservados:,} ({percentual_preservado:.2f}%)")
            logger.info(f"   • Pixels removidos: {total_pixels - pixels_preservados:,} ({100 - percentual_preservado:.2f}%)")
            
            # Aplicar limpeza morfológica leve para melhorar qualidade
            logger.info("🔧 Aplicando refinamentos morfológicos...")
            
            # Remover ruído pequeno
            kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            img_limpa = cv2.morphologyEx(img_limpa, cv2.MORPH_OPEN, kernel_noise)
            
            # Conectar partes quebradas das letras (muito sutil)
            kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            img_limpa = cv2.morphologyEx(img_limpa, cv2.MORPH_CLOSE, kernel_connect)
            
            # Gerar timestamp único para salvar imagens
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Incluir milissegundos
            
            # NOVO: Salvar imagem ORIGINAL para comparação
            img_original_path = captchas_limpos_dir / f"captcha_original_{timestamp}.png"
            cv2.imwrite(str(img_original_path), img_gray)
            logger.info(f"📸 Imagem ORIGINAL salva: {img_original_path}")
            
            # NOVO: Salvar imagem LIMPA (principal)
            img_limpa_path = captchas_limpos_dir / f"captcha_limpo_t{threshold_otimo}_{timestamp}.png"
            cv2.imwrite(str(img_limpa_path), img_limpa)
            logger.info(f"✨ Imagem LIMPA salva: {img_limpa_path}")
            
            # NOVO: Salvar metadados da limpeza
            metadata_path = captchas_limpos_dir / f"metadata_{timestamp}.txt"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(f"METADADOS DA LIMPEZA DE CAPTCHA\n")
                f.write(f"===============================\n")
                f.write(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Threshold aplicado: ≤{threshold_otimo}\n")
                f.write(f"Tipo detectado: {'texto preto dominante' if threshold_otimo == 10 else 'texto em sombras'}\n")
                f.write(f"Pixels originais: {total_pixels:,}\n")
                f.write(f"Pixels preservados: {pixels_preservados:,} ({percentual_preservado:.2f}%)\n")
                f.write(f"Pixels removidos: {total_pixels - pixels_preservados:,} ({100 - percentual_preservado:.2f}%)\n")
                f.write(f"Valores únicos originais: {len(unique_values)}\n")
                f.write(f"Pixels preto puro (0): {pixels_preto_puro:,} ({percentual_preto:.2f}%)\n")
                f.write(f"Método: Preservação de preto + sombras (comprovadamente 79.1% efetivo)\n")
                f.write(f"Arquivo original: {img_original_path.name}\n")
                f.write(f"Arquivo limpo: {img_limpa_path.name}\n")
            
            logger.info(f"📄 Metadados salvos: {metadata_path}")
            
            # Salvar imagem debug (mantendo para compatibilidade)
            debug_path = self.debug_rgb_dir / f"captcha_otimizado_t{threshold_otimo}_{timestamp}.png"
            cv2.imwrite(str(debug_path), img_limpa)
            logger.info(f"🖼️ Imagem debug salva: {debug_path}")
            
            return img_limpa
            
        except Exception as e:
            logger.error(f"❌ Erro na limpeza otimizada: {e}")
            # Fallback: retornar imagem em escala de cinza simples
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def resolver_captcha_easyocr(self, captcha_buffer):
        """Resolve CAPTCHA usando OpenCV + EasyOCR com detecção inteligente de cor"""
        try:
            logger.info("🔍 Resolvendo CAPTCHA com detecção inteligente de letras...")
            
            # PRIMEIRO: Verificar se existe treinamento manual para este CAPTCHA
            try:
                # Salvar CAPTCHA temporariamente para identificação
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_filename = f"temp_captcha_{timestamp}.png"
                
                # Converter buffer para imagem e salvar temporariamente
                nparr = np.frombuffer(captcha_buffer, np.uint8)
                img_temp = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                cv2.imwrite(temp_filename, img_temp)
                
                # Verificar se este CAPTCHA foi treinado manualmente
                # (comparando com CAPTCHAs existentes por similaridade de hash)
                resposta_manual = self.verificar_captcha_treinado(img_temp)
                
                # Remover arquivo temporário
                try:
                    os.remove(temp_filename)
                except:
                    pass
                
                if resposta_manual:
                    logger.info(f"🎓 CAPTCHA encontrado no treinamento manual: '{resposta_manual}'")
                    self.taxa_sucesso_atual = min(100.0, self.taxa_sucesso_atual + 0.5)  # Pequeno boost na taxa
                    logger.info(f"📊 Taxa de sucesso atual: {self.taxa_sucesso_atual:.1f}% ({self.total_resolvidos}/{self.total_tentativas})")
                    return resposta_manual
                    
            except Exception as e:
                logger.debug(f"⚠️ Erro ao verificar treinamento manual: {e}")
            
            # Primeiro detectar se precisamos de análise especial
            nparr_original = np.frombuffer(captcha_buffer, np.uint8)
            img_original = cv2.imdecode(nparr_original, cv2.IMREAD_COLOR)
            
            logger.info("🎯 Aplicando método de limpeza otimizado (comprovadamente efetivo)")
            
            # Processar imagem com método otimizado
            captcha_processado = self.processar_captcha_opencv(captcha_buffer)
            
            # Converter buffer para imagem numpy
            nparr = np.frombuffer(captcha_processado, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Aplicar processamentos finais para OCR
            img_final = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Gerar timestamp único para salvar imagens
            timestamp_final = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            # NOVO: Criar pasta para imagens finais processadas
            captchas_processados_dir = Path("captchas_processados")
            captchas_processados_dir.mkdir(exist_ok=True)
            
            # NOVO: Salvar imagem final processada para OCR
            final_path = captchas_processados_dir / f"captcha_final_ocr_{timestamp_final}.png"
            cv2.imwrite(str(final_path), img_final)
            logger.info(f"🎯 Imagem FINAL para OCR salva: {final_path}")
            
            # Salvar imagem processada para debug (mantendo compatibilidade)
            debug_path = self.debug_rgb_dir / f"captcha_otimizado_final_{datetime.now().strftime('%H%M%S')}.png"
            cv2.imwrite(str(debug_path), img_final)
            logger.info(f"🖼️ Imagem otimizada salva: {debug_path}")
            
            # Configurações otimizadas para o método comprovado
            configuracoes_otimizadas = [
                {'paragraph': False, 'width_ths': 0.6, 'threshold': 0.15},  # Config principal
                {'paragraph': False, 'width_ths': 0.7, 'threshold': 0.2},   # Config alternativa 1
                {'paragraph': False, 'width_ths': 0.5, 'threshold': 0.18},  # Config alternativa 2
                {'paragraph': False, 'width_ths': 0.8, 'threshold': 0.25},  # Config backup
            ]
            
            # Múltiplas tentativas de reconhecimento
            todos_textos = []
            
            # TENTATIVA 0: CNN OCR Customizado (maior prioridade)
            if hasattr(self, 'cnn_ocr') and self.cnn_ocr:
                logger.info("🧠 Tentativa com CNN OCR customizado...")
                try:
                    # Usar imagem processada para CNN
                    texto_cnn, confianca_cnn = self.cnn_ocr.predizer_com_cnn(img_final)
                    if texto_cnn and confianca_cnn > 0.3:  # Threshold baixo para CNN
                        todos_textos.append((texto_cnn, confianca_cnn + 0.2, "cnn_customizado"))
                        logger.info(f"🧠 CNN OCR: '{texto_cnn}' (conf: {confianca_cnn:.2f})")
                        
                        # Se CNN tem alta confiança, dar prioridade máxima
                        if confianca_cnn > 0.7:
                            logger.info("🎯 CNN com alta confiança - priorizando resultado")
                            
                except Exception as e:
                    logger.warning(f"⚠️ Erro no CNN OCR: {e}")
            
            # Usar configuração otimizada se disponível (do treinamento)
            if hasattr(self, 'melhor_config') and isinstance(self.melhor_config, dict):
                logger.info(f"🎯 Aplicando configuração otimizada do treinamento: {self.melhor_config}")
                result_otimizada = self.reader.readtext(img_final, detail=1, 
                                                      paragraph=self.melhor_config['paragraph'], 
                                                      width_ths=self.melhor_config['width_ths'])
                for (bbox, text, confidence) in result_otimizada:
                    text_clean = re.sub(r'[^A-Za-z0-9]', '', text.strip())
                    if len(text_clean) >= 4 and confidence > self.melhor_config['threshold']:
                        todos_textos.append((text_clean, confidence + 0.1, "otimizada_treino"))
            
            # Tentativas com configurações otimizadas
            for i, config in enumerate(configuracoes_otimizadas):
                logger.info(f"🔍 Tentativa {i+1} com configuração otimizada...")
                result = self.reader.readtext(img_final, detail=1, **config)
                for (bbox, text, confidence) in result:
                    text_clean = re.sub(r'[^A-Za-z0-9]', '', text.strip())
                    if len(text_clean) >= 4 and confidence > config['threshold']:
                        todos_textos.append((text_clean, confidence, f"otimizada_{i+1}"))
            
            # Tentativa com imagem invertida
            img_inverted = cv2.bitwise_not(img_final)
            logger.info("🔍 Tentativa com imagem invertida...")
            result_inv = self.reader.readtext(img_inverted, detail=1, **configuracoes_otimizadas[0])
            for (bbox, text, confidence) in result_inv:
                text_clean = re.sub(r'[^A-Za-z0-9]', '', text.strip())
                if len(text_clean) >= 4 and confidence > configuracoes_otimizadas[0]['threshold']:
                    todos_textos.append((text_clean, confidence, "invertida_otimizada"))
            
            # Tentativa com Tesseract OCR otimizado
            try:
                logger.info("🔍 Tentando Tesseract com configuração otimizada...")
                
                tesseract_config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
                
                # Tesseract na imagem normal
                tesseract_text1 = pytesseract.image_to_string(img_final, config=tesseract_config).strip()
                text_clean1 = re.sub(r'[^A-Za-z0-9]', '', tesseract_text1)
                if len(text_clean1) >= 4:
                    todos_textos.append((text_clean1, 0.85, "tesseract_otimizado"))
                    logger.info(f"🔍 Tesseract otimizado: '{text_clean1}'")
                
                # Tesseract na imagem invertida
                tesseract_text2 = pytesseract.image_to_string(img_inverted, config=tesseract_config).strip()
                text_clean2 = re.sub(r'[^A-Za-z0-9]', '', tesseract_text2)
                if len(text_clean2) >= 4:
                    todos_textos.append((text_clean2, 0.85, "tesseract_inv_otimizado"))
                    logger.info(f"🔍 Tesseract invertido otimizado: '{text_clean2}'")
                    
            except Exception as e:
                logger.warning(f"⚠️ Erro no Tesseract: {e}")
            
            # Tentativa backup com configuração padrão
            logger.info("🔍 Tentativa backup com configuração padrão...")
            result_backup = self.reader.readtext(img_final, detail=1, paragraph=False, width_ths=0.7)
            for (bbox, text, confidence) in result_backup:
                text_clean = re.sub(r'[^A-Za-z0-9]', '', text.strip())
                if len(text_clean) >= 4 and confidence > 0.2:
                    todos_textos.append((text_clean, confidence * 0.9, "backup"))  # Penaliza backup
            
            # Remover duplicatas e ordenar por confiança
            textos_unicos = {}
            for text, conf, metodo in todos_textos:
                if text not in textos_unicos or textos_unicos[text][1] < conf:
                    textos_unicos[text] = (text, conf, metodo)
            
            textos_encontrados = [(text, conf, metodo) for text, conf, metodo in textos_unicos.values()]
            textos_encontrados.sort(key=lambda x: x[1], reverse=True)
            
            # Log dos resultados com método usado
            for text_clean, confidence, metodo in textos_encontrados:
                logger.info(f"📝 '{text_clean}' (confiança: {confidence:.2f}) via {metodo} [CASE SENSITIVE]")
            
            if textos_encontrados:
                # Pegar o texto com maior confiança
                melhor_resultado = textos_encontrados[0]
                captcha_text = melhor_resultado[0]
                confianca = melhor_resultado[1]
                metodo = melhor_resultado[2]
                
                logger.info(f"✅ CAPTCHA resolvido: '{captcha_text}' (confiança: {confianca:.2f}) via {metodo}")
                logger.info(f"🎯 Método de limpeza: Otimizado (Threshold ≤85)")
                
                # NOVO: Salvar resultado final com metadados completos
                resultado_path = captchas_processados_dir / f"resultado_{timestamp_final}.txt"
                with open(resultado_path, 'w', encoding='utf-8') as f:
                    f.write(f"RESULTADO DA RESOLUÇÃO DE CAPTCHA\n")
                    f.write(f"=================================\n")
                    f.write(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
                    f.write(f"Timestamp: {timestamp_final}\n")
                    f.write(f"CAPTCHA RESOLVIDO: '{captcha_text}'\n")
                    f.write(f"Confiança: {confianca:.2f}\n")
                    f.write(f"Método vencedor: {metodo}\n")
                    f.write(f"Método de limpeza: Otimizado (Threshold ≤85)\n")
                    f.write(f"Total candidatos: {len(textos_encontrados)}\n")
                    f.write(f"Configurações testadas: {len(configuracoes_otimizadas)}\n")
                    f.write(f"Imagem final: {final_path.name}\n")
                    f.write(f"\nTodos os candidatos encontrados:\n")
                    for i, (texto, conf, met) in enumerate(textos_encontrados, 1):
                        f.write(f"  {i}. '{texto}' (conf: {conf:.2f}) via {met}\n")
                
                logger.info(f"📄 Resultado completo salvo: {resultado_path}")
                
                # Salvar estatísticas de sucesso com informações detalhadas
                detalhes = {
                    'metodo_limpeza': 'otimizado_threshold_85',
                    'metodo_vencedor': metodo,
                    'total_candidatos': len(textos_encontrados),
                    'configuracoes_testadas': len(configuracoes_otimizadas),
                    'timestamp': timestamp_final
                }
                self.salvar_estatisticas_resolucao(captcha_text, True, metodo, detalhes)
                return captcha_text
            else:
                logger.warning("❌ Nenhum texto válido encontrado no CAPTCHA")
                
                # NOVO: Salvar caso de falha para análise futura
                falha_path = captchas_processados_dir / f"falha_{timestamp_final}.txt"
                with open(falha_path, 'w', encoding='utf-8') as f:
                    f.write(f"FALHA NA RESOLUÇÃO DE CAPTCHA\n")
                    f.write(f"=============================\n")
                    f.write(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
                    f.write(f"Timestamp: {timestamp_final}\n")
                    f.write(f"RESULTADO: FALHA - Nenhum texto válido encontrado\n")
                    f.write(f"Método de limpeza: Otimizado (Threshold ≤85)\n")
                    f.write(f"Total tentativas: {len(todos_textos)}\n")
                    f.write(f"Configurações testadas: {len(configuracoes_otimizadas)}\n")
                    f.write(f"Imagem final: {final_path.name}\n")
                    f.write(f"\nTodos os textos tentados:\n")
                    for i, (texto, conf, met) in enumerate(todos_textos, 1):
                        f.write(f"  {i}. '{texto}' (conf: {conf:.2f}) via {met}\n")
                
                logger.info(f"📄 Análise de falha salva: {falha_path}")
                
                # Salvar estatísticas de falha
                detalhes = {
                    'metodo_limpeza': 'otimizado_threshold_85',
                    'total_tentativas': len(todos_textos),
                    'configuracoes_testadas': len(configuracoes_otimizadas),
                    'timestamp': timestamp_final
                }
                self.salvar_estatisticas_resolucao(None, False, "nenhum", detalhes)
                return None
                
        except Exception as e:
            logger.error(f"❌ Erro resolver CAPTCHA com EasyOCR: {e}")
            self.salvar_estatisticas_resolucao(None, False, "erro", {'erro': str(e)})
            return None
    
    # ========================================
    # 🎯 MÉTODOS OCR PARA HEURÍSTICA DUAL
    # ========================================
    
    async def _easyocr_heuristico(self, img_processada: np.ndarray):
        """Método EasyOCR otimizado para heurística"""
        try:
            # Converter para RGB se necessário
            if len(img_processada.shape) == 3:
                img_rgb = cv2.cvtColor(img_processada, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = cv2.cvtColor(img_processada, cv2.COLOR_GRAY2RGB)
            
            # OCR com EasyOCR
            resultados = self.reader.readtext(
                img_rgb,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                width_ths=0.4,
                height_ths=0.4,
                paragraph=False
            )
            
            if resultados:
                texto = resultados[0][1].strip().upper()
                confianca = float(resultados[0][2])
                
                # Limpar texto
                texto_limpo = ''.join(c for c in texto if c.isalnum())
                
                return texto_limpo, confianca
            
            return "", 0.0
            
        except Exception as e:
            logger.debug(f"Erro EasyOCR heurístico: {e}")
            return "", 0.0
    
    async def _tesseract_heuristico(self, img_processada: np.ndarray):
        """Método Tesseract configurado para heurística"""
        try:
            # Configuração otimizada
            config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            
            texto = pytesseract.image_to_string(img_processada, config=config).strip().upper()
            texto_limpo = ''.join(c for c in texto if c.isalnum())
            
            # Estimar confiança (Tesseract não dá confiança direta)
            if len(texto_limpo) >= 4:
                confianca = 0.85  # Boa confiança se texto tem tamanho esperado
            elif len(texto_limpo) >= 3:
                confianca = 0.70
            else:
                confianca = 0.30
            
            return texto_limpo, confianca
            
        except Exception as e:
            logger.debug(f"Erro Tesseract heurístico: {e}")
            return "", 0.0
    
    async def _easyocr_multiplo_heuristico(self, img_processada: np.ndarray):
        """Método EasyOCR com múltiplas configurações para heurística"""
        try:
            resultados_configs = []
            
            # Configuração 1: Padrão
            config1 = {
                'allowlist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                'width_ths': 0.4,
                'height_ths': 0.4
            }
            
            # Configuração 2: Mais restritiva
            config2 = {
                'allowlist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                'width_ths': 0.3,
                'height_ths': 0.3
            }
            
            # Converter imagem
            if len(img_processada.shape) == 3:
                img_rgb = cv2.cvtColor(img_processada, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = cv2.cvtColor(img_processada, cv2.COLOR_GRAY2RGB)
            
            # Testar configurações
            for i, config in enumerate([config1, config2], 1):
                try:
                    resultados = self.reader.readtext(img_rgb, paragraph=False, **config)
                    if resultados:
                        texto = resultados[0][1].strip().upper()
                        confianca = float(resultados[0][2])
                        texto_limpo = ''.join(c for c in texto if c.isalnum())
                        
                        if len(texto_limpo) >= 3:
                            resultados_configs.append((texto_limpo, confianca))
                except:
                    continue
            
            # Retornar melhor resultado
            if resultados_configs:
                return max(resultados_configs, key=lambda x: x[1])
            
            return "", 0.0
            
        except Exception as e:
            logger.debug(f"Erro EasyOCR múltiplo heurístico: {e}")
            return "", 0.0
    
    async def _tesseract_psm_heuristico(self, img_processada: np.ndarray):
        """Método Tesseract com variações de PSM para heurística"""
        try:
            psm_configs = [
                '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            ]
            
            resultados = []
            
            for config in psm_configs:
                try:
                    texto = pytesseract.image_to_string(img_processada, config=config).strip().upper()
                    texto_limpo = ''.join(c for c in texto if c.isalnum())
                    
                    if len(texto_limpo) >= 3:
                        # Estimar confiança baseada no PSM
                        if '--psm 7' in config:
                            confianca_base = 0.85
                        elif '--psm 8' in config:
                            confianca_base = 0.75
                        else:
                            confianca_base = 0.65
                        
                        # Ajustar pela qualidade do texto
                        if len(texto_limpo) >= 4:
                            confianca = confianca_base
                        else:
                            confianca = confianca_base * 0.8
                        
                        resultados.append((texto_limpo, confianca))
                except:
                    continue
            
            # Retornar melhor resultado
            if resultados:
                return max(resultados, key=lambda x: x[1])
            
            return "", 0.0
            
        except Exception as e:
            logger.debug(f"Erro Tesseract PSM heurístico: {e}")
            return "", 0.0

    async def digitar_humano(self, elemento, texto):
        """Digitação humana MUITO LENTA com delays realistas e case sensitivity"""
        try:
            logger.info(f"⌨️ Digitação humana LENTA CASE SENSITIVE: '{texto}'")
            
            # Clicar no campo e aguardar foco (mais tempo)
            await elemento.click()
            await asyncio.sleep(random.uniform(0.5, 0.8))
            
            # Limpar campo com seleção total + delete (mais devagar)
            await elemento.press('Control+a')
            await asyncio.sleep(random.uniform(0.3, 0.5))
            await elemento.press('Delete')
            await asyncio.sleep(random.uniform(0.4, 0.6))
            
            # Digitar caractere por caractere com delays MUITO LENTOS
            for i, char in enumerate(texto):
                # Delays muito mais longos (500ms a 1.2s entre caracteres)
                delay = random.uniform(0.5, 1.2)
                
                # Hesitações ainda mais frequentes e longas
                if random.random() < 0.25:  # 25% chance de hesitação longa
                    hesitacao = random.uniform(0.8, 2.0)
                    delay += hesitacao
                    logger.info(f"   🤔 Hesitação longa ({hesitacao:.1f}s) no caractere '{char}'")
                
                # Simular pequenas correções ocasionais
                if random.random() < 0.10:  # 10% chance de "erro" e correção
                    await elemento.type('x')  # digitar caractere errado
                    await asyncio.sleep(random.uniform(0.2, 0.4))
                    await elemento.press('Backspace')  # corrigir
                    await asyncio.sleep(random.uniform(0.3, 0.6))
                    logger.info(f"   ✏️ Simulou correção antes de '{char}'")
                
                await elemento.type(char)
                logger.info(f"   ✍️ Digitado: '{char}' (delay: {delay:.1f}s)")
                await asyncio.sleep(delay)
            
            # Pausa final muito longa antes de confirmar
            pausa_final = random.uniform(1.0, 2.5)
            logger.info(f"   ⏱️ Pausa final: {pausa_final:.1f}s")
            await asyncio.sleep(pausa_final)
            logger.info(f"✅ Digitação LENTA concluída: '{texto}' (preservando maiúsculas/minúsculas)")
                
        except Exception as e:
            logger.error(f"❌ Erro na digitação humana: {e}")

    async def verificar_downloads_recentes(self):
        """Verifica se há downloads recentes"""
        try:
            logger.info("📁 Verificando downloads recentes...")
            
            # Listar arquivos recentes (últimas 2 horas)
            agora = datetime.now().timestamp()
            arquivos_recentes = []
            
            for arquivo in self.downloads_dir.glob("*"):
                if arquivo.is_file():
                    tempo_mod = arquivo.stat().st_mtime
                    if (agora - tempo_mod) < 7200:  # 2 horas
                        arquivos_recentes.append(arquivo)
            
            if arquivos_recentes:
                logger.info(f"✅ {len(arquivos_recentes)} download(s) recente(s) encontrado(s)")
                for arquivo in arquivos_recentes:
                    logger.info(f"   📄 {arquivo.name}")
                return True
            else:
                logger.warning("⚠️ Nenhum download recente encontrado")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erro ao verificar downloads: {e}")
            return False

    async def inicializar_browser(self):
        """Inicializa o browser com configurações otimizadas"""
        try:
            logger.info("🚀 Iniciando Playwright com monitor de download...")
            
            playwright = await async_playwright().start()
            
            # Configurações do browser otimizadas para downloads
            self.browser = await playwright.chromium.launch(
                headless=False,  # SEMPRE visível - downloads não funcionam em modo anônimo
                proxy={
                    "server": "http://filtroweb.tjes.jus.br:9090"
                },
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor',
                    '--disable-download-protection',  # Permitir downloads
                    '--disable-popup-blocking',       # Permitir popups
                    '--allow-running-insecure-content',
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                ],
                downloads_path=str(self.downloads_dir)  # Pasta específica para downloads
            )
            
            # Criar contexto com downloads habilitados
            context = await self.browser.new_context(
                accept_downloads=True,
                java_script_enabled=True,
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
            )
            
            # Mascarar automação
            await context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
            """)
            
            self.page = await context.new_page()
            
            # Headers para parecer mais humano
            await self.page.set_extra_http_headers({
                'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
            })
            
            logger.info("✅ Browser inicializado com monitor de download")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar browser: {e}")
            return False

    async def fazer_download(self):
        """Executa o processo de download do ComprasNet seguindo sequência: OK → Itens e Download → Download"""
        try:
            logger.info("🌐 Navegando para ComprasNet - Página inicial da licitação...")
            # Primeiro navegar para a página da licitação específica (sem parâmetros de download)
            await self.page.goto("https://comprasnet.gov.br/ConsultaLicitacoes/ConsLicitacao.asp?coduasg=925968&numprp=900372025&modprp=5", 
                                 wait_until="domcontentloaded", timeout=30000)
            
            await asyncio.sleep(3)
            logger.info("✅ Página da licitação carregada")
            
            # ETAPA 1: Procurar e clicar no botão OK primeiro
            logger.info("🔍 Procurando botão OK...")
            botao_ok = await self.page.query_selector('input[type="button"][id="ok"][name="ok"][value="OK"]')
            if not botao_ok:
                # Fallback: buscar por qualquer botão OK
                botao_ok = await self.page.query_selector('input[value="OK"]')
            
            if botao_ok:
                logger.info("✅ Botão OK encontrado - clicando...")
                await botao_ok.click()
                await asyncio.sleep(2)
                logger.info("✅ Botão OK clicado")
            else:
                logger.warning("⚠️ Botão OK não encontrado - continuando...")
            
            # ETAPA 2: Procurar e clicar no botão "Itens e Download"
            logger.info("🔍 Procurando botão 'Itens e Download'...")
            botao_itens = await self.page.query_selector('input[type="button"][name="itens"][value="Itens e Download"]')
            if not botao_itens:
                # Fallback: buscar por texto similar
                botao_itens = await self.page.query_selector('input[value*="Itens"][value*="Download"]')
            
            if botao_itens:
                logger.info("✅ Botão 'Itens e Download' encontrado - clicando...")
                await botao_itens.click()
                await asyncio.sleep(3)
                logger.info("✅ Navegando para página de itens...")
                
                # Aguardar carregamento da nova página
                await self.page.wait_for_load_state("domcontentloaded", timeout=15000)
                
                # ETAPA 3: Procurar e clicar no botão "Download" na nova página
                logger.info("🔍 Procurando botão 'Download' na página de itens...")
                botao_download = await self.page.query_selector('input[type="button"][name="Download"][value="Download"]')
                if not botao_download:
                    # Fallback: buscar por onclick com ValidaCodigo
                    botao_download = await self.page.query_selector('input[onclick*="ValidaCodigo"]')
                
                if botao_download:
                    logger.info("✅ Botão 'Download' encontrado - clicando...")
                    await botao_download.click()
                    await asyncio.sleep(3)
                    logger.info("✅ Navegando para página de download...")
                    
                    # Aguardar carregamento da página de download (pop-up)
                    await self.page.wait_for_load_state("domcontentloaded", timeout=15000)
                    
                    # Verificar se estamos na página de download correta
                    url_atual = self.page.url
                    logger.info(f"📍 URL atual: {url_atual}")
                    
                    if "Download.asp" in url_atual:
                        logger.info("✅ Página de download carregada com sucesso")
                    else:
                        logger.warning(f"⚠️ URL inesperada: {url_atual}")
                else:
                    logger.error("❌ Botão 'Download' não encontrado na página de itens")
                    return False
            else:
                logger.error("❌ Botão 'Itens e Download' não encontrado")
                return False
            
            await asyncio.sleep(2)
            logger.info("✅ Sequência de navegação concluída - agora na página de download")
            
            # Processar CAPTCHA diretamente (sem formulário)
            captcha_img = await self.page.query_selector('img[src*="captcha.aspx"]')
            if captcha_img:
                logger.info("📷 CAPTCHA encontrado")
                
                # Capturar informações iniciais do CAPTCHA
                captcha_src = await captcha_img.get_attribute('src')
                captcha_buffer = await captcha_img.screenshot()
                
                # Metadados extras para o CAPTCHA
                metadados_extras = {
                    "url_origem": self.page.url,
                    "captcha_src": captcha_src,
                    "metodo_captura": "playwright_screenshot",
                    "tamanho_imagem": f"{len(captcha_buffer)} bytes"
                }
                
                # SALVAR CAPTCHA ORIGINAL PARA ESTUDOS COM METADADOS
                logger.info("📚 Salvando CAPTCHA original para estudos futuros...")
                nome_arquivo, timestamp = self.salvar_captcha_para_estudos(captcha_buffer, None, metadados_extras)
                
                # 🎯 RESOLUÇÃO INTELIGENTE COM HEURÍSTICA DUAL
                captcha_text = None
                confianca = 0.0
                metodo_usado = "tradicional"
                
                # PRIMEIRA PRIORIDADE: HEURÍSTICA DUAL
                if self.usar_heuristica and self.integrador_heuristico:
                    logger.info("🎯 APLICANDO HEURÍSTICA DUAL PARA 98%+ DE CONFIANÇA (PROTEGIDA)...")
                    
                    try:
                        # Converter buffer para numpy array
                        nparr = np.frombuffer(captcha_buffer, np.uint8)
                        img_captcha = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if img_captcha is not None:
                            # Preparar métodos OCR para a heurística
                            metodos_ocr_heuristica = {
                                'easyocr_otimizado': self._easyocr_heuristico,
                                'tesseract_configurado': self._tesseract_heuristico,
                                'easyocr_multiplo': self._easyocr_multiplo_heuristico,
                                'tesseract_psm': self._tesseract_psm_heuristico,
                            }
                            
                            # EXECUTAR HEURÍSTICA DUAL
                            resultado_heuristico = await self.integrador_heuristico.resolver_captcha_inteligente(
                                img_captcha, 
                                metodos_ocr_heuristica,
                                salvar_resultados=True
                            )
                            
                            # Verificar se atingiu a meta
                            if resultado_heuristico.meta_98_atingida:
                                captcha_text = resultado_heuristico.texto_final
                                confianca = resultado_heuristico.confianca_final
                                metodo_usado = f"heuristica_dual_{resultado_heuristico.metodo_vencedor}"
                                
                                logger.info(f"🎉 HEURÍSTICA DUAL SUCESSO: '{captcha_text}' ({confianca:.1%})")
                                logger.info(f"    Estratégia: {resultado_heuristico.estrategia_usada}")
                                logger.info(f"    Método: {resultado_heuristico.metodo_vencedor}")
                                logger.info(f"    Tempo: {resultado_heuristico.tempo_total_processamento:.2f}s")
                            
                            elif resultado_heuristico.confianca_final >= 0.35:
                                # Aceitar se confiança razoável - ajustado para ser mais realista
                                captcha_text = resultado_heuristico.texto_final
                                confianca = resultado_heuristico.confianca_final
                                metodo_usado = f"heuristica_parcial_{resultado_heuristico.metodo_vencedor}"
                                
                                logger.info(f"✅ HEURÍSTICA PARCIAL: '{captcha_text}' ({confianca:.1%})")
                            
                            else:
                                logger.warning(f"⚠️ Heurística com baixa confiança: {resultado_heuristico.confianca_final:.1%}")
                        
                    except Exception as e:
                        logger.error(f"❌ Erro na heurística dual: {e}")
                        logger.info("🔄 Continuando com método tradicional...")
                
                # SEGUNDA PRIORIDADE: SISTEMA ML/DL EXISTENTE
                if not captcha_text and hasattr(self, 'ml_analyzer') and self.ml_analyzer:
                    logger.info(f"🧠 Analisando CAPTCHA com {self.ml_tipo.upper()}...")
                    
                    try:
                        if self.ml_tipo == "deep_learning":
                            # Usar Deep Learning
                            analise_dl = self.ml_analyzer.analisar_captcha_dl(captcha_buffer)
                            
                            if analise_dl:
                                dificuldade = analise_dl['dificuldade']
                                metodo_recomendado = analise_dl['metodo_recomendado']
                                confianca_dl = analise_dl.get('confianca_dl', 0.7)
                                
                                logger.info(f"🎯 DL Analysis: {dificuldade} → {metodo_recomendado} (conf: {confianca_dl:.2f})")
                                
                                # Se DL tem OCR próprio e confiança alta
                                if analise_dl.get('texto_ocr') and confianca_dl > 0.8:
                                    captcha_text = analise_dl['texto_ocr']
                                    confianca = confianca_dl
                                    metodo_usado = "deep_learning_ocr"
                                    logger.info(f"� DL OCR direto: '{captcha_text}' (conf: {confianca:.2f})")
                                
                                # Senão, usar método recomendado
                                elif metodo_recomendado == "tesseract":
                                    logger.info("🧠 DL recomenda Tesseract")
                                    captcha_text = self._resolver_tesseract_simples(captcha_buffer)
                                    metodo_usado = "tesseract_dl_guided"
                                    confianca = 0.8
                                elif metodo_recomendado == "easyocr":
                                    logger.info("🧠 DL recomenda EasyOCR")
                                    captcha_text = self.resolver_captcha_easyocr(captcha_buffer)
                                    metodo_usado = "easyocr_dl_guided"
                                    confianca = 0.7
                            
                        else:
                            # Usar ML simples como fallback
                            analise_ml = self.ml_analyzer.analisar_captcha_ml(captcha_buffer)
                            
                            if analise_ml:
                                metodo_recomendado = analise_ml['metodo_recomendado']
                                logger.info(f"🤖 ML recomenda: {metodo_recomendado}")
                                
                                if metodo_recomendado == "tesseract":
                                    captcha_text = self._resolver_tesseract_simples(captcha_buffer)
                                    metodo_usado = "tesseract_ml_guided"
                                else:
                                    captcha_text = self.resolver_captcha_easyocr(captcha_buffer)
                                    metodo_usado = "easyocr_ml_guided"
                                    
                                confianca = 0.7
                    
                    except Exception as e:
                        logger.error(f"❌ Erro no ML/DL: {e}")
                
                # Fallback tradicional se ML falhou
                if not captcha_text:
                    logger.info("🔄 Usando método tradicional como fallback...")
                    captcha_text = self.resolver_captcha_easyocr(captcha_buffer)
                    metodo_usado = "easyocr_fallback"
                    confianca = 0.6
                
                if captcha_text:
                    # ATUALIZAR METADADOS COM RESPOSTA RESOLVIDA
                    metadados_extras.update({
                        "captcha_resolvido": captcha_text,
                        "metodo_resolucao": metodo_usado,
                        "confianca_ia": round(confianca, 3),
                        "ml_tipo_usado": getattr(self, 'ml_tipo', 'none'),
                        "status_inicial": "resolvido_automaticamente"
                    })
                    
                    logger.info(f"✅ CAPTCHA resolvido via {metodo_usado}: '{captcha_text}' (conf: {confianca:.2f})")
                    
                    # Preencher campo CAPTCHA
                    campo_captcha = await self.page.query_selector('input[name="idLetra"]')
                    if campo_captcha:
                        await self.digitar_humano(campo_captcha, captcha_text)
                        logger.info("✅ CAPTCHA preenchido automaticamente")
                        
                        # Tentar submeter formulário
                        logger.info("🚀 Submetendo formulário...")
                        # Buscar diferentes tipos de botões de submit
                        botao_submit = await self.page.query_selector('input[name="Submit"]') or \
                                     await self.page.query_selector('input[type="submit"]') or \
                                     await self.page.query_selector('button[type="submit"]') or \
                                     await self.page.query_selector('input[value*="Download"]') or \
                                     await self.page.query_selector('input[value*="Baixar"]')
                        
                        if botao_submit:
                            # ========== CRITÉRIO DEFINITIVO DE SUCESSO ==========
                            # SUCESSO = Download bem-sucedido (prova irrefutável)
                            # FALHA = Qualquer coisa diferente de download bem-sucedido
                            # ====================================================
                            try:
                                async with self.page.expect_download(timeout=30000) as download_info:
                                    await botao_submit.click()
                                    
                                download = await download_info.value
                                logger.info(f"✅ Download capturado: {download.suggested_filename}")
                                
                                # Salvar download
                                download_path = self.downloads_dir / download.suggested_filename
                                await download.save_as(download_path)
                                logger.info(f"📁 Download salvo: {download_path}")
                                
                                # SUCESSO: CAPTCHA resolvido corretamente - download bem-sucedido é a prova definitiva
                                resultado = {
                                    "status_captcha": "sucesso_download_confirmado",
                                    "download_realizado": True,
                                    "nome_arquivo_download": download.suggested_filename,
                                    "descricao": "CAPTCHA resolvido com sucesso - download realizado",
                                    "url_atual": self.page.url,
                                    "timestamp_verificacao": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                                }
                                self.atualizar_metadata_resultado(timestamp, resultado)
                                
                                # Salvar estatísticas de SUCESSO (download é prova definitiva)
                                detalhes_stats = {
                                    "download_realizado": True,
                                    "arquivo_download": download.suggested_filename,
                                    "status_verificacao": "sucesso_download_confirmado",
                                    "prova_sucesso": "download_bem_sucedido"
                                }
                                self.salvar_estatisticas_resolucao(captcha_text, True, "easyocr", detalhes_stats)
                                
                                # Salvar amostra para treinamento CNN (SUCESSO)
                                if hasattr(self, 'cnn_ocr') and self.cnn_ocr and captcha_text:
                                    try:
                                        # Obter imagem original do CAPTCHA
                                        img_element = await self.page.query_selector('img[alt="Captcha"]')
                                        if img_element:
                                            img_buffer = await img_element.screenshot()
                                            nparr = np.frombuffer(img_buffer, np.uint8)
                                            img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                            if img_original is not None:
                                                self.cnn_ocr.salvar_amostra_treinamento(img_original, captcha_text, True)
                                                logger.info("💾 Amostra de SUCESSO salva para treinamento CNN")
                                    except Exception as e:
                                        logger.warning(f"⚠️ Erro ao salvar amostra CNN (sucesso): {e}")
                                
                                return True
                                
                            except Exception as e:
                                logger.warning(f"⚠️ Timeout ou erro no download: {e}")
                                # FALHA: Sem download não podemos confirmar se CAPTCHA estava correto
                                # Verificar se apareceu mensagem de erro de CAPTCHA na página
                                resultado = await self.verificar_resultado_captcha(timestamp)
                                resultado["download_realizado"] = False
                                resultado["erro_download"] = str(e)
                                self.atualizar_metadata_resultado(timestamp, resultado)
                                
                                # Só contabilizar como SUCESSO se houve download bem-sucedido
                                # Sem download = FALHA na resolução (independente do motivo)
                                detalhes_stats = {
                                    "download_realizado": False,
                                    "erro_download": str(e),
                                    "status_verificacao": resultado.get("status_captcha", "falha_sem_download"),
                                    "motivo_falha": "sem_download_nao_confirma_captcha"
                                }
                                # Sempre FALSE se não houve download - download é critério definitivo
                                self.salvar_estatisticas_resolucao(captcha_text, False, "easyocr", detalhes_stats)
                                
                                # Salvar amostra para treinamento CNN (FALHA)
                                if hasattr(self, 'cnn_ocr') and self.cnn_ocr and captcha_text:
                                    try:
                                        # Obter imagem original do CAPTCHA
                                        img_element = await self.page.query_selector('img[alt="Captcha"]')
                                        if img_element:
                                            img_buffer = await img_element.screenshot()
                                            nparr = np.frombuffer(img_buffer, np.uint8)
                                            img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                            if img_original is not None:
                                                self.cnn_ocr.salvar_amostra_treinamento(img_original, captcha_text, False)
                                                logger.info("💾 Amostra de FALHA salva para treinamento CNN")
                                    except Exception as e:
                                        logger.warning(f"⚠️ Erro ao salvar amostra CNN (falha): {e}")
                                
                                return False
                        else:
                            logger.error("❌ Botão Submit não encontrado")
                            # FALHA: Sem botão submit não conseguimos testar o CAPTCHA
                            resultado = {"status_captcha": "erro_botao_submit", "descricao": "Botão de submit não encontrado"}
                            self.atualizar_metadata_resultado(timestamp, resultado)
                            
                            # Erro técnico - não conseguimos nem tentar submeter o CAPTCHA
                            detalhes_stats = {
                                "erro": "botao_submit_nao_encontrado",
                                "motivo_falha": "erro_tecnico_interface"
                            }
                            self.salvar_estatisticas_resolucao(captcha_text, False, "easyocr", detalhes_stats)
                    else:
                        logger.error("❌ Campo CAPTCHA não encontrado")
                        # FALHA: Sem campo CAPTCHA não conseguimos nem inserir o texto
                        resultado = {"status_captcha": "erro_campo_captcha", "descricao": "Campo de CAPTCHA não encontrado"}
                        self.atualizar_metadata_resultado(timestamp, resultado)
                        
                        # Erro técnico - interface não disponível
                        detalhes_stats = {
                            "erro": "campo_captcha_nao_encontrado",
                            "motivo_falha": "erro_tecnico_interface"
                        }
                        self.salvar_estatisticas_resolucao(captcha_text, False, "easyocr", detalhes_stats)
                else:
                    logger.warning("❌ Não foi possível resolver o CAPTCHA automaticamente")
                    # FALHA: Não conseguimos resolver o CAPTCHA (algoritmo falhou)
                    resultado = {"status_captcha": "erro_resolucao", "descricao": "Não foi possível resolver automaticamente"}
                    self.atualizar_metadata_resultado(timestamp, resultado)
                    
                    # Falha na resolução algorítmica - não conseguimos decifrar
                    detalhes_stats = {
                        "erro": "nao_foi_possivel_resolver",
                        "motivo_falha": "algoritmo_nao_conseguiu_decifrar"
                    }
                    self.salvar_estatisticas_resolucao("", False, "easyocr", detalhes_stats)
                
                # Atualizar página para tentar próximo CAPTCHA
                logger.info("🔄 Atualizando página para próximo CAPTCHA...")
                await self.page.reload()
                return False  # Retornar para tentar novamente
            
        except Exception as e:
            logger.error(f"❌ Erro no download: {e}")
            return False

    async def run(self):
        """Executa o monitor principal"""
        try:
            logger.info("🤖 ComprasNet Download Monitor")
            logger.info("========================================")
            
            # Verificar downloads recentes
            await self.verificar_downloads_recentes()
            
            # Inicializar browser
            if not await self.inicializar_browser():
                return
            
            # Loop principal
            tentativas = 0
            max_tentativas = 15
            
            while tentativas < max_tentativas:
                logger.info(f"🔄 Tentativa {tentativas + 1}/{max_tentativas}")
                
                sucesso = await self.fazer_download()
                if sucesso:
                    logger.info("🎉 Download concluído com sucesso!")
                    break
                
                tentativas += 1
                await asyncio.sleep(5)  # Aguardar antes da próxima tentativa
            
            if tentativas >= max_tentativas:
                logger.warning("⚠️ Máximo de tentativas atingido")
            
        except KeyboardInterrupt:
            logger.info("🛑 Interrompido pelo usuário")
        except Exception as e:
            logger.error(f"❌ Erro no sistema: {e}")
        finally:
            # Cleanup
            if self.browser:
                await self.browser.close()
            logger.info("👋 Sistema finalizado")
    
    def aplicar_treinamento_manual_aos_modelos(self):
        """Aplica o treinamento manual aos modelos de ML"""
        try:
            logger.info("🎓 Aplicando treinamento manual aos modelos...")
            
            # Carregar todos os treinamentos
            treinamento_json = self.carregar_treinamento_manual()
            treinamento_db = self.carregar_treinamento_banco()
            
            # Combinar treinamentos (DB tem prioridade)
            todos_treinamentos = {**treinamento_json, **treinamento_db}
            
            if not todos_treinamentos:
                logger.info("📚 Nenhum treinamento manual disponível")
                return
            
            logger.info(f"📚 {len(todos_treinamentos)} CAPTCHAs treinados encontrados")
            
            # Aqui você pode implementar lógica para aplicar aos modelos
            # Por enquanto, apenas carregamos para uso durante execução
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao aplicar treinamento: {e}")
            return False

    def carregar_treinamento_manual(self, arquivo_treinamento="captchas_treinados.json"):
        """Carrega treinamento manual dos CAPTCHAs"""
        try:
            if not os.path.exists(arquivo_treinamento):
                logger.info("📚 Nenhum arquivo de treinamento manual encontrado")
                return {}
            
            with open(arquivo_treinamento, 'r', encoding='utf-8') as f:
                treinamentos = json.load(f)
            
            logger.info(f"📚 Carregados {len(treinamentos)} CAPTCHAs do treinamento manual")
            return treinamentos
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar treinamento manual: {e}")
            return {}
    
    def carregar_treinamento_banco(self, db_path="captcha_treinamento.db"):
        """Carrega treinamento do banco de dados SQLite"""
        try:
            if not os.path.exists(db_path):
                logger.info("🗄️ Nenhum banco de treinamento encontrado")
                return {}
            
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT nome_arquivo, resposta_correta, confianca, dificuldade 
                FROM captcha_treinamento 
                WHERE confianca >= 0.7
                ORDER BY data_treinamento DESC
            ''')
            
            treinamentos = {}
            for row in cursor.fetchall():
                nome_arquivo, resposta, confianca, dificuldade = row
                treinamentos[nome_arquivo] = {
                    'resposta': resposta,
                    'confianca': confianca,
                    'dificuldade': dificuldade
                }
            
            conn.close()
            logger.info(f"🗄️ Carregados {len(treinamentos)} CAPTCHAs do banco de dados")
            return treinamentos
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar do banco: {e}")
            return {}
    
    def verificar_captcha_treinado(self, img_current):
        """Verifica se o CAPTCHA atual foi treinado manualmente comparando com os existentes"""
        try:
            # Carregar treinamentos disponíveis
            treinamento_json = self.carregar_treinamento_manual()
            treinamento_db = self.carregar_treinamento_banco()
            todos_treinamentos = {**treinamento_json, **treinamento_db}
            
            if not todos_treinamentos:
                return None
            
            # Calcular hash da imagem atual
            hash_atual = self.calcular_hash_imagem(img_current)
            
            # Comparar com CAPTCHAs treinados
            captcha_dir = Path("captcha_estudos")
            for hash_treinado, dados in todos_treinamentos.items():
                try:
                    arquivo_nome = dados.get('arquivo', '')
                    arquivo_path = captcha_dir / arquivo_nome
                    if arquivo_path.exists():
                        img_treinado = cv2.imread(str(arquivo_path))
                        if img_treinado is not None:
                            hash_treinado_calc = self.calcular_hash_imagem(img_treinado)
                            
                            # Comparar hashes (similaridade)
                            similaridade = self.comparar_hashes(hash_atual, hash_treinado_calc)
                            
                            # Se muito similar (>85%), usar resposta treinada
                            if similaridade > 0.85:
                                if isinstance(dados, dict):
                                    resposta = dados.get('resposta', dados.get('texto', ''))
                                else:
                                    resposta = str(dados)
                                
                                logger.info(f"🎯 CAPTCHA similar encontrado: {arquivo_nome} (similaridade: {similaridade:.1%})")
                                return resposta
                                
                except Exception as e:
                    logger.debug(f"⚠️ Erro ao comparar com {dados.get('arquivo', 'N/A')}: {e}")
            
            return None
            
        except Exception as e:
            logger.debug(f"❌ Erro na verificação de treinamento: {e}")
            return None
    
    def calcular_hash_imagem(self, img):
        """Calcula hash perceptual da imagem para comparação"""
        try:
            # Redimensionar para tamanho padrão
            img_resized = cv2.resize(img, (64, 32))
            
            # Converter para grayscale
            if len(img_resized.shape) == 3:
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img_resized
            
            # Calcular média
            media = np.mean(img_gray)
            
            # Criar hash binário
            hash_binario = (img_gray > media).astype(np.uint8)
            
            return hash_binario.flatten()
            
        except Exception as e:
            logger.debug(f"❌ Erro ao calcular hash: {e}")
            return np.array([])
    
    def comparar_hashes(self, hash1, hash2):
        """Compara dois hashes e retorna similaridade (0-1)"""
        try:
            if len(hash1) != len(hash2) or len(hash1) == 0:
                return 0.0
            
            # Calcular diferença bit a bit
            diferencas = np.sum(hash1 != hash2)
            total_bits = len(hash1)
            
            # Retornar similaridade (1 - % de diferenças)
            similaridade = 1.0 - (diferencas / total_bits)
            return similaridade
            
        except Exception as e:
            logger.debug(f"❌ Erro ao comparar hashes: {e}")
            return 0.0

class CNNCaptchaOCR:
    """
    🧠 Sistema de OCR Customizado usando CNN (Keras + TensorFlow)
    =============================================================
    Treina modelos neurais específicos para CAPTCHAs do ComprasNet
    """
    
    def __init__(self, model_dir="models_cnn_ocr"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.char_model = None
        self.word_model = None
        self.label_encoder = None
        
        # Configurações do modelo
        self.img_height = 50
        self.img_width = 200
        self.channels = 1  # Grayscale
        
        # Dataset paths
        self.dataset_dir = Path("dataset_cnn_captcha")
        self.dataset_dir.mkdir(exist_ok=True)
        
        logger.info("🧠 CNN OCR Customizado inicializado")
        
        if CNN_AVAILABLE:
            self.carregar_ou_criar_modelos()
        else:
            logger.warning("⚠️ TensorFlow não disponível - CNN OCR desabilitado")
    
    def preprocessar_imagem_para_cnn(self, img):
        """Preprocessa imagem para entrada na CNN"""
        try:
            # Redimensionar para tamanho fixo
            img_resized = cv2.resize(img, (self.img_width, self.img_height))
            
            # Converter para grayscale se necessário
            if len(img_resized.shape) == 3:
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img_resized
            
            # Normalizar pixels (0-1)
            img_normalized = img_gray.astype(np.float32) / 255.0
            
            # Adicionar dimensão do canal
            img_final = np.expand_dims(img_normalized, axis=-1)
            
            # Adicionar dimensão do batch
            img_batch = np.expand_dims(img_final, axis=0)
            
            return img_batch
            
        except Exception as e:
            logger.error(f"❌ Erro no preprocessamento CNN: {e}")
            return None
    
    def criar_modelo_caractere(self):
        """Cria modelo CNN para reconhecer caracteres individuais"""
        if not CNN_AVAILABLE:
            return None
            
        try:
            model = models.Sequential([
                # Camada de entrada
                layers.Input(shape=(self.img_height, self.img_width, self.channels)),
                
                # Primeira camada convolucional
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Segunda camada convolucional
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Terceira camada convolucional
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Camadas densas
                layers.Flatten(),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),
                
                # Saída para classificação de caracteres (A-Z, a-z, 0-9)
                layers.Dense(62, activation='softmax')
            ])
            
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("🧠 Modelo CNN para caracteres criado")
            return model
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar modelo CNN: {e}")
            return None
    
    def criar_modelo_palavra(self):
        """Cria modelo CNN para reconhecer palavras completas"""
        if not CNN_AVAILABLE:
            return None
            
        try:
            model = models.Sequential([
                # Camada de entrada
                layers.Input(shape=(self.img_height, self.img_width, self.channels)),
                
                # Camadas convolucionais
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.BatchNormalization(),
                layers.Dropout(0.25),
                
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.BatchNormalization(),
                layers.Dropout(0.25),
                
                layers.Conv2D(256, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.BatchNormalization(),
                layers.Dropout(0.25),
                
                # Camadas densas
                layers.Flatten(),
                layers.Dense(1024, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                
                # Saída para sequência de caracteres (será ajustada dinamicamente)
                layers.Dense(1000, activation='softmax')  # Placeholder - será ajustado
            ])
            
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("🧠 Modelo CNN para palavras criado")
            return model
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar modelo CNN palavras: {e}")
            return None
    
    def carregar_ou_criar_modelos(self):
        """Carrega modelos existentes ou cria novos"""
        if not CNN_AVAILABLE:
            return
            
        try:
            # Tentar carregar modelo de caracteres
            char_model_path = self.model_dir / "char_model.h5"
            if char_model_path.exists():
                self.char_model = keras.models.load_model(str(char_model_path))
                logger.info("✅ Modelo CNN de caracteres carregado")
            else:
                self.char_model = self.criar_modelo_caractere()
                logger.info("🆕 Novo modelo CNN de caracteres criado")
            
            # Tentar carregar modelo de palavras
            word_model_path = self.model_dir / "word_model.h5"
            if word_model_path.exists():
                self.word_model = keras.models.load_model(str(word_model_path))
                logger.info("✅ Modelo CNN de palavras carregado")
            else:
                self.word_model = self.criar_modelo_palavra()
                logger.info("🆕 Novo modelo CNN de palavras criado")
            
            # Carregar label encoder
            encoder_path = self.model_dir / "label_encoder.pkl"
            if encoder_path.exists():
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                logger.info("✅ Label encoder carregado")
            else:
                self.criar_label_encoder()
                
        except Exception as e:
            logger.error(f"❌ Erro ao carregar/criar modelos CNN: {e}")
    
    def criar_label_encoder(self):
        """Cria encoder para caracteres"""
        try:
            # Caracteres possíveis em CAPTCHAs
            chars = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(chars)
            
            # Salvar encoder
            encoder_path = self.model_dir / "label_encoder.pkl"
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            logger.info("✅ Label encoder criado e salvo")
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar label encoder: {e}")
    
    def salvar_amostra_treinamento(self, img_original, texto_correto, sucesso):
        """Salva amostra para treinamento futuro"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Salvar imagem original
            img_path = self.dataset_dir / f"captcha_{timestamp}_{sucesso}.png"
            cv2.imwrite(str(img_path), img_original)
            
            # Salvar metadados
            metadata = {
                "timestamp": timestamp,
                "texto_correto": texto_correto,
                "sucesso": sucesso,
                "img_path": str(img_path)
            }
            
            # Converter tipos antes de salvar
            metadata = convert_numpy_types(metadata)
            
            metadata_path = self.dataset_dir / f"metadata_{timestamp}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Amostra salva para treinamento: {texto_correto} ({'✅' if sucesso else '❌'})")
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar amostra: {e}")
    
    def predizer_com_cnn(self, img):
        """Usa CNN para predizer texto do CAPTCHA"""
        if not CNN_AVAILABLE or not self.char_model:
            return None, 0.0
            
        try:
            # Preprocessar imagem
            img_processed = self.preprocessar_imagem_para_cnn(img)
            if img_processed is None:
                return None, 0.0
            
            # Predição com modelo de caracteres
            if self.char_model:
                prediction = self.char_model.predict(img_processed, verbose=0)
                confidence = np.max(prediction)
                
                if self.label_encoder:
                    predicted_idx = np.argmax(prediction)
                    if predicted_idx < len(self.label_encoder.classes_):
                        predicted_char = self.label_encoder.classes_[predicted_idx]
                        
                        logger.info(f"🧠 CNN predição: '{predicted_char}' (conf: {confidence:.2f})")
                        return predicted_char, confidence
            
            return None, 0.0
            
        except Exception as e:
            logger.error(f"❌ Erro na predição CNN: {e}")
            return None, 0.0
    
    def treinar_com_dataset_existente(self, epochs=50, batch_size=32):
        """Treina modelo com dataset existente"""
        if not CNN_AVAILABLE:
            logger.warning("⚠️ TensorFlow não disponível para treinamento")
            return False
            
        try:
            logger.info("🎓 Iniciando treinamento CNN com dataset existente...")
            
            # Carregar dados de treinamento
            X_train, y_train = self.carregar_dataset_treinamento()
            
            if len(X_train) < 10:
                logger.warning("⚠️ Dataset muito pequeno para treinamento CNN")
                return False
            
            logger.info(f"📊 Dataset: {len(X_train)} amostras carregadas")
            
            # Dividir em treino e validação
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            # Callbacks para treinamento
            callbacks_list = [
                callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
                callbacks.ModelCheckpoint(
                    str(self.model_dir / "best_model.h5"),
                    save_best_only=True
                )
            ]
            
            # Treinar modelo
            if self.char_model:
                history = self.char_model.fit(
                    X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks_list,
                    verbose=1
                )
                
                # Salvar modelo treinado
                model_path = self.model_dir / "char_model.h5"
                self.char_model.save(str(model_path))
                
                logger.info("✅ Treinamento CNN concluído com sucesso!")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erro no treinamento CNN: {e}")
            return False
    
    def carregar_dataset_treinamento(self):
        """Carrega dataset de imagens e labels para treinamento"""
        try:
            X_data = []
            y_data = []
            
            # Percorrer arquivos de metadata
            for metadata_file in self.dataset_dir.glob("metadata_*.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Carregar apenas amostras com sucesso
                    if metadata.get('sucesso', False) and metadata.get('texto_correto'):
                        img_path = Path(metadata['img_path'])
                        if img_path.exists():
                            # Carregar e preprocessar imagem
                            img = cv2.imread(str(img_path))
                            if img is not None:
                                img_processed = self.preprocessar_imagem_para_cnn(img)
                                if img_processed is not None:
                                    X_data.append(img_processed[0])  # Remove batch dimension
                                    
                                    # Para caracteres individuais, usar primeiro caractere
                                    texto = metadata['texto_correto']
                                    if texto and self.label_encoder:
                                        primeiro_char = texto[0].lower()
                                        if primeiro_char in self.label_encoder.classes_:
                                            label_encoded = self.label_encoder.transform([primeiro_char])[0]
                                            # One-hot encoding
                                            label_onehot = keras.utils.to_categorical(
                                                label_encoded, 
                                                num_classes=len(self.label_encoder.classes_)
                                            )
                                            y_data.append(label_onehot)
                
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao processar {metadata_file}: {e}")
                    continue
            
            if X_data and y_data:
                X_array = np.array(X_data)
                y_array = np.array(y_data)
                logger.info(f"📊 Dataset carregado: {X_array.shape[0]} amostras")
                return X_array, y_array
            else:
                logger.warning("⚠️ Nenhuma amostra válida encontrada no dataset")
                return np.array([]), np.array([])
                
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dataset: {e}")
            return np.array([]), np.array([])

async def main():
    """🎯 FUNÇÃO PRINCIPAL COM HEURÍSTICA DUAL"""
    
    logger.info("🎯 INICIANDO COMPRASNET COM HEURÍSTICA DUAL")
    logger.info("=" * 60)
    logger.info("🚀 Sistema avançado para 98%+ de confiança em CAPTCHAs")
    logger.info("=" * 60)
    
    monitor = ComprasNetDownloadMonitor()
    
    # Verificar se heurística está disponível
    if monitor.usar_heuristica:
        logger.info("✅ Sistema Heurístico Dual ATIVO!")
        logger.info("🎯 Meta: 98%+ de confiança na resolução de CAPTCHAs")
    else:
        logger.warning("⚠️ Heurística dual não disponível, usando sistema tradicional")
    
    # Aplicar treinamento manual se disponível
    logger.info("🎓 Verificando treinamento manual...")
    monitor.aplicar_treinamento_manual_aos_modelos()
    
    # Executar sistema
    await monitor.run()

if __name__ == "__main__":
    asyncio.run(main())