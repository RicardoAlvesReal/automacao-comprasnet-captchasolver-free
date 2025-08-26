#!/usr/bin/env python3
"""
Sistema Avançado de CAPTCHA - Versão 2.0
Foca em reconhecimento de 6+ caracteres com alta precisão
"""

import cv2
import numpy as np
import pytesseract
import easyocr
import os
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime
import re

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CaptchaResolverV2:
    """Sistema avançado de resolução de CAPTCHA com validação de 6 caracteres"""
    
    def __init__(self):
        self.easyocr_reader = None
        
        # Estrutura de debug organizada
        self.debug_dir = "debug_sistema"
        self.captchas_resolvidos_dir = os.path.join(self.debug_dir, "captchas_resolvidos")
        self.captchas_falhas_dir = os.path.join(self.debug_dir, "captchas_falhas")
        self.opencv_processamento_dir = os.path.join(self.debug_dir, "opencv_processamento")
        
        # Pastas de treinamento ML
        self.captchas_limpos_dir = "captchas_limpos"
        self.captchas_processados_dir = "captchas_processados"
        
        # Criar todas as pastas necessárias
        os.makedirs(self.debug_dir, exist_ok=True)
        os.makedirs(self.captchas_resolvidos_dir, exist_ok=True)
        os.makedirs(self.captchas_falhas_dir, exist_ok=True)
        os.makedirs(self.opencv_processamento_dir, exist_ok=True)
        os.makedirs(self.captchas_limpos_dir, exist_ok=True)
        os.makedirs(self.captchas_processados_dir, exist_ok=True)
        
    def inicializar_easyocr(self):
        """Inicializa EasyOCR se ainda não estiver carregado"""
        if self.easyocr_reader is None:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
                logger.info("✅ EasyOCR inicializado")
            except Exception as e:
                logger.error(f"❌ Erro ao inicializar EasyOCR: {e}")
    
    def preprocessar_imagem_avancado(self, img: np.ndarray) -> List[np.ndarray]:
        """Preprocessa imagem com múltiplas estratégias para maximizar reconhecimento"""
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Lista de imagens processadas para teste
        imagens_processadas = []
        
        # 1. Imagem original em escala de cinza
        imagens_processadas.append(("original", gray))
        
        # 2. Binarização adaptativa
        binary_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
        imagens_processadas.append(("binary_adaptativa", binary_adapt))
        
        # 3. Threshold simples com múltiplos valores
        for thresh_val in [128, 100, 150, 80, 180]:
            _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            imagens_processadas.append((f"threshold_{thresh_val}", binary))
        
        # 4. Melhoria de contraste CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        imagens_processadas.append(("clahe", enhanced))
        
        # 5. Filtro de mediana para ruído
        median = cv2.medianBlur(gray, 3)
        imagens_processadas.append(("median", median))
        
        # 6. Operações morfológicas
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        imagens_processadas.append(("morfologia", morph))
        
        # 7. Inversão para texto branco em fundo preto
        inverted = cv2.bitwise_not(gray)
        imagens_processadas.append(("invertida", inverted))
        
        # 8. Nova técnica OpenCV + Tesseract (Técnica do usuário)
        opencv_enhanced = self.aplicar_tecnica_opencv_avancada(gray)
        imagens_processadas.append(("opencv_avancada", opencv_enhanced))
        
        return imagens_processadas
    
    def aplicar_tecnica_opencv_avancada(self, img: np.ndarray) -> np.ndarray:
        """
        Técnica avançada OpenCV + Tesseract sugerida pelo usuário
        - Resize 2x com interpolação cúbica
        - GaussianBlur para remover ruído
        - Threshold adaptativo
        - Inversão dupla para texto preto em fundo branco
        """
        try:
            # 1. Aumenta a imagem em 2x com interpolação cúbica (melhor para ampliar)
            imagem_maior = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # 2. Converter para escala de cinza se necessário
            if len(imagem_maior.shape) == 3:
                imagem_cinza = cv2.cvtColor(imagem_maior, cv2.COLOR_BGR2GRAY)
            else:
                imagem_cinza = imagem_maior.copy()
            
            # 3. Aplicar GaussianBlur para remover ruído mantendo estrutura
            imagem_borrada = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)
            
            # 4. Threshold adaptativo - Tesseract prefere texto preto em fundo branco
            imagem_binaria = cv2.adaptiveThreshold(
                imagem_borrada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # 5. Inverte novamente para ter texto preto em fundo branco
            imagem_final = cv2.bitwise_not(imagem_binaria)
            
            return imagem_final
            
        except Exception as e:
            logger.error(f"❌ Erro na técnica OpenCV avançada: {e}")
            # Retorna imagem original em caso de erro
            return img
    
    def aplicar_tecnica_opencv_variantes(self, img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Múltiplas variantes da técnica OpenCV avançada com diferentes parâmetros
        Para maximizar chances de sucesso em CAPTCHAs variados
        """
        variantes = []
        
        try:
            # Variante 1: Parâmetros originais do usuário
            variantes.append(("opencv_user_original", self.aplicar_tecnica_opencv_avancada(img)))
            
            # Variante 2: Resize 3x para CAPTCHAs muito pequenos
            img_3x = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            if len(img_3x.shape) == 3:
                gray_3x = cv2.cvtColor(img_3x, cv2.COLOR_BGR2GRAY)
            else:
                gray_3x = img_3x.copy()
            blur_3x = cv2.GaussianBlur(gray_3x, (5, 5), 0)
            thresh_3x = cv2.adaptiveThreshold(blur_3x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            final_3x = cv2.bitwise_not(thresh_3x)
            variantes.append(("opencv_resize_3x", final_3x))
            
            # Variante 3: GaussianBlur mais suave (3x3)
            img_2x = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            if len(img_2x.shape) == 3:
                gray_2x = cv2.cvtColor(img_2x, cv2.COLOR_BGR2GRAY)
            else:
                gray_2x = img_2x.copy()
            blur_suave = cv2.GaussianBlur(gray_2x, (3, 3), 0)
            thresh_suave = cv2.adaptiveThreshold(blur_suave, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            final_suave = cv2.bitwise_not(thresh_suave)
            variantes.append(("opencv_blur_suave", final_suave))
            
            # Variante 4: Threshold adaptativo com parâmetros diferentes
            img_2x_alt = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            if len(img_2x_alt.shape) == 3:
                gray_2x_alt = cv2.cvtColor(img_2x_alt, cv2.COLOR_BGR2GRAY)
            else:
                gray_2x_alt = img_2x_alt.copy()
            blur_alt = cv2.GaussianBlur(gray_2x_alt, (5, 5), 0)
            thresh_alt = cv2.adaptiveThreshold(blur_alt, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
            final_alt = cv2.bitwise_not(thresh_alt)
            variantes.append(("opencv_thresh_alt", final_alt))
            
        except Exception as e:
            logger.error(f"❌ Erro nas variantes OpenCV: {e}")
            # Pelo menos retornar a técnica original
            if not variantes:
                variantes.append(("opencv_fallback", self.aplicar_tecnica_opencv_avancada(img)))
        
        return variantes
    
    def criar_imagem_limpa(self, img: np.ndarray) -> np.ndarray:
        """Cria imagem limpa onde as letras ficam em valor 0 (preto) e o resto é descartado"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Aplicar threshold para destacar letras em preto (valor 0)
        # Inverter primeiro para que texto fique branco
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Operações morfológicas para limpar ruído
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Inverter de volta para que letras fiquem em valor 0 (preto)
        resultado = cv2.bitwise_not(cleaned)
        
        return resultado
    
    def criar_imagem_com_contraste(self, img: np.ndarray) -> np.ndarray:
        """Cria imagem com contraste melhorado para algoritmo"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Aplicar CLAHE para melhorar contraste
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        contrasted = clahe.apply(gray)
        
        # Aplicar filtro bilateral para suavizar mas manter bordas
        smooth = cv2.bilateralFilter(contrasted, 9, 75, 75)
        
        # Normalizar para garantir contraste máximo
        normalized = cv2.normalize(smooth, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized
    
    def salvar_opencv_debug(self, img_original: np.ndarray, img_processada: np.ndarray, texto_resultado: str, timestamp: str = None) -> bool:
        """Salva debug específico da técnica OpenCV"""
        try:
            if timestamp is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            # Salvar imagem original
            original_path = os.path.join(self.opencv_processamento_dir, f"opencv_original_{timestamp}.png")
            cv2.imwrite(original_path, img_original)
            
            # Salvar imagem processada pela técnica OpenCV
            processada_path = os.path.join(self.opencv_processamento_dir, f"opencv_processada_{timestamp}_{texto_resultado}.png")
            cv2.imwrite(processada_path, img_processada)
            
            logger.info(f"🔧 OpenCV Debug salvo: {processada_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro salvando debug OpenCV: {e}")
            return False
    
    def salvar_captcha_como_sucesso(self, img: np.ndarray, texto_resolvido: str, timestamp: str = None) -> bool:
        """Salva CAPTCHA como sucesso após confirmar que resultou em download"""
        try:
            if timestamp is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            # Salvar imagem original limpa (letras em valor 0)
            imagem_limpa = self.criar_imagem_limpa(img)
            limpo_path = os.path.join(self.captchas_limpos_dir, f"captcha_limpo_{timestamp}_{texto_resolvido}.png")
            cv2.imwrite(limpo_path, imagem_limpa)
            
            # Salvar imagem processada com contraste melhorado
            imagem_processada = self.criar_imagem_com_contraste(img)
            processado_path = os.path.join(self.captchas_processados_dir, f"captcha_processado_{timestamp}_{texto_resolvido}.png")
            cv2.imwrite(processado_path, imagem_processada)
            
            logger.info(f"🎯 SUCESSO ML: CAPTCHA '{texto_resolvido}' salvo para treinamento")
            logger.info(f"✅ Limpo: {limpo_path}")
            logger.info(f"✅ Processado: {processado_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar CAPTCHA como sucesso: {e}")
            return False
    
    def reconhecer_com_tesseract(self, img: np.ndarray, config: str = "") -> Tuple[str, float]:
        """Reconhece texto usando Tesseract com configuração específica otimizada para CAPTCHA"""
        try:
            # Configuração otimizada para CAPTCHAs: PSM 7 = linha única (IDEAL)
            if not config:
                config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            
            text = pytesseract.image_to_string(img, config=config).strip()
            
            # Tentar obter dados de confiança
            try:
                data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                confidence = np.mean(confidences) / 100.0 if confidences else 0.0
            except:
                confidence = 0.5 if text else 0.0
                
            return text, confidence
        except Exception as e:
            logger.debug(f"Erro Tesseract: {e}")
            return "", 0.0
    
    def reconhecer_com_easyocr(self, img: np.ndarray) -> Tuple[str, float]:
        """Reconhece texto usando EasyOCR"""
        try:
            self.inicializar_easyocr()
            if self.easyocr_reader is None:
                return "", 0.0
            
            results = self.easyocr_reader.readtext(img, detail=1, 
                                                 allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
            
            if results:
                # Pegar o resultado com maior confiança
                best_result = max(results, key=lambda x: x[2])
                text = best_result[1].strip()
                confidence = best_result[2]
                return text, confidence
            else:
                return "", 0.0
                
        except Exception as e:
            logger.debug(f"Erro EasyOCR: {e}")
            return "", 0.0
    
    def validar_resultado(self, texto: str) -> bool:
        """Valida se o resultado atende aos critérios de CAPTCHA"""
        if not texto:
            return False
        
        # Remover espaços e caracteres especiais
        texto_limpo = re.sub(r'[^a-zA-Z0-9]', '', texto)
        
        # Verificar comprimento mínimo de 3 caracteres (pode ser menos que 6 em alguns casos)
        if len(texto_limpo) < 3:
            return False
        
        # Verificar se contém apenas caracteres alfanuméricos
        if not texto_limpo.isalnum():
            return False
        
        return True
    
    def resolver_captcha_completo(self, img: np.ndarray, salvar_debug: bool = True, salvar_sucesso: bool = False) -> Dict[str, Any]:
        """Resolve CAPTCHA usando todas as estratégias disponíveis
        
        Args:
            img: Imagem do CAPTCHA
            salvar_debug: Se deve salvar para debug (sempre, independente do resultado)
            salvar_sucesso: Se deve salvar como sucesso (só após confirmar download)
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        logger.info("🎯 INICIANDO RESOLUÇÃO AVANÇADA V2.0")
        logger.info(f"📏 Imagem: {img.shape}")
        
        # ESTRATÉGIA PRIORITÁRIA: Técnica OpenCV Avançada primeiro
        logger.info("🚀 Testando TÉCNICA OPENCV AVANÇADA (prioritária)")
        img_opencv_avancada = self.aplicar_tecnica_opencv_avancada(img)
        
        # Teste prioritário com AMBOS Tesseract E EasyOCR
        texto_tess, conf_tess = self.reconhecer_com_tesseract(
            img_opencv_avancada, 
            "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        )
        
        texto_easy, conf_easy = self.reconhecer_com_easyocr(img_opencv_avancada)
        
        # Escolher melhor resultado da técnica OpenCV
        if self.validar_resultado(texto_tess) and len(re.sub(r'[^a-zA-Z0-9]', '', texto_tess)) >= 6:
            texto_opencv = texto_tess
            conf_opencv = conf_tess
            metodo_opencv = "opencv_tesseract_prioritaria"
        elif self.validar_resultado(texto_easy) and len(re.sub(r'[^a-zA-Z0-9]', '', texto_easy)) >= 6:
            texto_opencv = texto_easy
            conf_opencv = conf_easy
            metodo_opencv = "opencv_easyocr_prioritaria"
        else:
            # Usar o melhor dos dois, mesmo que não tenha 6+ chars
            if conf_easy > conf_tess:
                texto_opencv = texto_easy
                conf_opencv = conf_easy
                metodo_opencv = "opencv_easyocr_fallback"
            else:
                texto_opencv = texto_tess
                conf_opencv = conf_tess
                metodo_opencv = "opencv_tesseract_fallback"
        
        if self.validar_resultado(texto_opencv) and len(re.sub(r'[^a-zA-Z0-9]', '', texto_opencv)) >= 6:
            texto_limpo = re.sub(r'[^a-zA-Z0-9]', '', texto_opencv)
            resultado_final = {
                'texto': texto_limpo,
                'confianca': conf_opencv,
                'metodo': metodo_opencv,
                'estrategia': 'opencv_avancada',
                'sucesso': True,
                'chars': len(texto_limpo)
            }
            logger.info(f"🎯 SUCESSO PRIORITÁRIO! '{texto_limpo}' (conf: {conf_opencv:.2f}) - {len(texto_limpo)} chars via {metodo_opencv}")
            
            # Salvar debug específico da técnica OpenCV
            self.salvar_opencv_debug(img, img_opencv_avancada, texto_limpo, timestamp)
            
            # Salvar debug se solicitado
            if salvar_debug:
                debug_path = os.path.join(self.captchas_resolvidos_dir, f"opencv_sucesso_{timestamp}_{texto_limpo}.png")
                cv2.imwrite(debug_path, img)
                logger.info(f"✅ Sucesso OpenCV salvo: {debug_path}")
            
            # Salvar como sucesso se solicitado (após confirmação de download)
            if salvar_sucesso:
                self.salvar_captcha_como_sucesso(img, texto_limpo, timestamp)
            
            return resultado_final
        
        # Se técnica prioritária não funcionou, continuar com estratégias convencionais
        logger.info("⚠️ Técnica prioritária não atingiu 6+ chars, testando outras estratégias...")
        
        # ESTRATÉGIA SECUNDÁRIA: Variantes da técnica OpenCV
        logger.info("🔄 Testando VARIANTES OPENCV AVANÇADAS")
        variantes_opencv = self.aplicar_tecnica_opencv_variantes(img)
        
        for nome_variante, img_variante in variantes_opencv:
            # Testar cada variante com AMBOS Tesseract e EasyOCR
            texto_tess, conf_tess = self.reconhecer_com_tesseract(
                img_variante,
                "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            )
            
            texto_easy, conf_easy = self.reconhecer_com_easyocr(img_variante)
            
            # Testar primeiro com Tesseract
            if self.validar_resultado(texto_tess) and len(re.sub(r'[^a-zA-Z0-9]', '', texto_tess)) >= 6:
                texto_limpo = re.sub(r'[^a-zA-Z0-9]', '', texto_tess)
                resultado_final = {
                    'texto': texto_limpo,
                    'confianca': conf_tess,
                    'metodo': f'{nome_variante}_tesseract_secundaria',
                    'estrategia': nome_variante,
                    'sucesso': True,
                    'chars': len(texto_limpo)
                }
                logger.info(f"🎯 SUCESSO VARIANTE! '{texto_limpo}' via {nome_variante}+Tesseract (conf: {conf_tess:.2f}) - {len(texto_limpo)} chars")
                
                if salvar_debug:
                    self.salvar_debug_completo(img, resultado_final, timestamp)
                if salvar_sucesso:
                    self.salvar_captcha_como_sucesso(img, texto_limpo, timestamp)
                
                return resultado_final
            
            # Testar com EasyOCR se Tesseract não funcionou
            if self.validar_resultado(texto_easy) and len(re.sub(r'[^a-zA-Z0-9]', '', texto_easy)) >= 6:
                texto_limpo = re.sub(r'[^a-zA-Z0-9]', '', texto_easy)
                resultado_final = {
                    'texto': texto_limpo,
                    'confianca': conf_easy,
                    'metodo': f'{nome_variante}_easyocr_secundaria',
                    'estrategia': nome_variante,
                    'sucesso': True,
                    'chars': len(texto_limpo)
                }
                logger.info(f"🎯 SUCESSO VARIANTE! '{texto_limpo}' via {nome_variante}+EasyOCR (conf: {conf_easy:.2f}) - {len(texto_limpo)} chars")
                
                if salvar_debug:
                    self.salvar_debug_completo(img, resultado_final, timestamp)
                if salvar_sucesso:
                    self.salvar_captcha_como_sucesso(img, texto_limpo, timestamp)
                
                return resultado_final
        
        # Se variantes OpenCV não funcionaram, usar estratégias convencionais como fallback
        logger.info("⚠️ Variantes OpenCV não atingiram 6+ chars, usando estratégias convencionais...")
        
        # Preprocessar imagem com múltiplas estratégias
        imagens_processadas = self.preprocessar_imagem_avancado(img)
        
        melhores_resultados = []
        
        # Testar cada estratégia de preprocessamento
        for nome_estrategia, img_processada in imagens_processadas:
            
            # Tesseract com diferentes configurações (PSM otimizados para CAPTCHA)
            tesseract_configs = [
                "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",  # IDEAL: linha única
                "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",  # GEMINI: bloco uniforme
                "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",  # PALAVRA: mais restritivo
                "--psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"  # CRUA: sem segmentação
            ]
            
            for i, config in enumerate(tesseract_configs):
                texto, confianca = self.reconhecer_com_tesseract(img_processada, config)
                if self.validar_resultado(texto):
                    resultado = {
                        'texto': re.sub(r'[^a-zA-Z0-9]', '', texto),
                        'confianca': confianca,
                        'metodo': f'tesseract_{nome_estrategia}_psm{[7,6,8,13][i]}',
                        'estrategia': nome_estrategia
                    }
                    melhores_resultados.append(resultado)
                    logger.info(f"📝 {resultado['metodo']}: '{resultado['texto']}' (conf: {confianca:.2f})")
            
            # EasyOCR
            texto, confianca = self.reconhecer_com_easyocr(img_processada)
            if self.validar_resultado(texto):
                resultado = {
                    'texto': re.sub(r'[^a-zA-Z0-9]', '', texto),
                    'confianca': confianca,
                    'metodo': f'easyocr_{nome_estrategia}',
                    'estrategia': nome_estrategia
                }
                melhores_resultados.append(resultado)
                logger.info(f"📝 {resultado['metodo']}: '{resultado['texto']}' (conf: {confianca:.2f})")
        
        # Selecionar melhor resultado
        if melhores_resultados:
            # Priorizar resultados com 6+ caracteres e alta confiança
            resultados_6_plus = [r for r in melhores_resultados if len(r['texto']) >= 6]
            
            if resultados_6_plus:
                melhor = max(resultados_6_plus, key=lambda x: x['confianca'])
                logger.info(f"🏆 MELHOR RESULTADO (6+ chars): '{melhor['texto']}' - {melhor['metodo']} (conf: {melhor['confianca']:.2f})")
            else:
                # Se não há resultados com 6+ caracteres, pegar o de maior confiança
                melhor = max(melhores_resultados, key=lambda x: x['confianca'])
                logger.info(f"⚠️ MELHOR DISPONÍVEL (<6 chars): '{melhor['texto']}' - {melhor['metodo']} (conf: {melhor['confianca']:.2f})")
            
            # Salvar debug sempre se solicitado (independente do resultado)
            if salvar_debug:
                # Salvar em pasta de sucessos ou falhas conforme resultado
                if len(melhor['texto']) >= 6:
                    debug_path = os.path.join(self.captchas_resolvidos_dir, f"captcha_resolvido_{timestamp}_{melhor['texto']}.png")
                    cv2.imwrite(debug_path, img)
                    logger.info(f"✅ Sucesso salvo no debug: {debug_path}")
                else:
                    debug_path = os.path.join(self.captchas_falhas_dir, f"captcha_insuficiente_{timestamp}_{melhor['texto']}.png")
                    cv2.imwrite(debug_path, img)
                    logger.info(f"⚠️ Resultado insuficiente salvo no debug: {debug_path}")
            
            # Salvar como sucesso APENAS se confirmado que resultou em download
            if salvar_sucesso:
                self.salvar_captcha_como_sucesso(img, melhor['texto'], timestamp)
            
            return {
                'sucesso': True,
                'texto': melhor['texto'],
                'confianca': melhor['confianca'],
                'metodo': melhor['metodo'],
                'estrategia': melhor['estrategia'],
                'total_candidatos': len(melhores_resultados),
                'tem_6_chars': len(melhor['texto']) >= 6,
                'chars': len(melhor['texto']),
                'timestamp': timestamp
            }
        else:
            logger.warning("❌ Nenhum resultado válido encontrado")
            
            if salvar_debug:
                # Salvar falhas na pasta específica de falhas
                debug_path = os.path.join(self.captchas_falhas_dir, f"captcha_falha_{timestamp}.png")
                cv2.imwrite(debug_path, img)
                logger.info(f"❌ Falha salva no debug: {debug_path}")
                
            # Falhas NUNCA são salvas como sucesso para treinamento ML
            
            return {
                'sucesso': False,
                'texto': '',
                'confianca': 0.0,
                'metodo': 'nenhum',
                'estrategia': 'nenhuma',
                'total_candidatos': 0,
                'tem_6_chars': False,
                'chars': 0,
                'timestamp': timestamp
            }

def teste_captcha_v2():
    """Testa o sistema V2.0 com CAPTCHAs reais"""
    
    resolver = CaptchaResolverV2()
    
    # Testar com algumas imagens recentes
    captchas_teste = [
        "C:/Users/LURBRANDAO/Documents/workspace/auto-comprasnet-tic-open/captcha_estudos/captcha_20250825_162423_Lt4Dqb.png",
        "C:/Users/LURBRANDAO/Documents/workspace/auto-comprasnet-tic-open/captcha_estudos/captcha_20250825_163606_T7Wlkz.png",
        "C:/Users/LURBRANDAO/Documents/workspace/auto-comprasnet-tic-open/captcha_estudos/captcha_20250825_162559_grrits.png",
        "C:/Users/LURBRANDAO/Documents/workspace/auto-comprasnet-tic-open/captcha_estudos/captcha_20250825_163753_7cj226.png",
        "C:/Users/LURBRANDAO/Documents/workspace/auto-comprasnet-tic-open/heuristica_resultados/original_20250825_163840_187.png",
        "C:/Users/LURBRANDAO/Documents/workspace/auto-comprasnet-tic-open/heuristica_resultados/original_20250825_163812_759.png"
    ]
    
    sucessos = 0
    sucessos_6_chars = 0
    
    for i, caminho_captcha in enumerate(captchas_teste, 1):
        if not os.path.exists(caminho_captcha):
            print(f"⚠️ Arquivo não encontrado: {caminho_captcha}")
            continue
            
        print(f"\n{'='*60}")
        print(f"🧪 TESTE {i}: {os.path.basename(caminho_captcha)}")
        print(f"{'='*60}")
        
        img = cv2.imread(caminho_captcha)
        if img is None:
            print(f"❌ Erro ao carregar imagem: {caminho_captcha}")
            continue
        
        resultado = resolver.resolver_captcha_completo(img, salvar_debug=True)
        
        print(f"\n📊 RESULTADO FINAL:")
        print(f"   ✅ Sucesso: {resultado['sucesso']}")
        print(f"   📝 Texto: '{resultado['texto']}'")
        print(f"   🎯 Confiança: {resultado['confianca']:.2f}")
        print(f"   🔧 Método: {resultado['metodo']}")
        print(f"   📏 Tem 6+ chars: {resultado['tem_6_chars']}")
        print(f"   🔍 Total candidatos: {resultado['total_candidatos']}")
        
        if resultado['sucesso']:
            sucessos += 1
            if resultado['tem_6_chars']:
                sucessos_6_chars += 1
    
    print(f"\n{'='*60}")
    print(f"📈 ESTATÍSTICAS FINAIS:")
    print(f"   ✅ Sucessos: {sucessos}/{len(captchas_teste)} ({100*sucessos/len(captchas_teste):.1f}%)")
    print(f"   📏 Com 6+ caracteres: {sucessos_6_chars}/{sucessos} ({100*sucessos_6_chars/sucessos if sucessos > 0 else 0:.1f}%)")
    print(f"{'='*60}")

if __name__ == "__main__":
    teste_captcha_v2()
