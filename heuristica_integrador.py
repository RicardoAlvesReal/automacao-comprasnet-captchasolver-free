#!/usr/bin/env python3
"""
🔗 INTEGRADOR HEURÍSTICO - ORQUESTRADOR DUAL
============================================
Sistema que integra as duas heurísticas para resolver CAPTCHAs com 98%+ de confiança
Coordena Analisador + Reconhecedor para máxima eficácia
"""

import cv2
import numpy as np
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple
import json

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

from heuristica_analisador import AnalisadorInteligente, AnaliseCompleta, EstrategiaProcessamento
from heuristica_reconhecedor import ReconhecedorAdaptativo, DecisaoFinal, MetodoOCR

logger = logging.getLogger(__name__)

@dataclass
class ResultadoHeuristicaDual:
    """Resultado completo do sistema de heurística dual"""
    # Resultados das heurísticas
    analise_inteligente: AnaliseCompleta
    decisao_ocr: DecisaoFinal
    
    # Resultado final
    texto_final: str
    confianca_final: float
    meta_98_atingida: bool
    
    # Performance
    tempo_total_processamento: float
    tempo_analise: float
    tempo_reconhecimento: float
    eficiencia_score: float
    
    # Metadados
    timestamp: str
    estrategia_usada: str
    metodo_vencedor: str
    melhorias_aplicadas: List[str]

class ProcessadorEstrategia:
    """Processador que implementa as diferentes estratégias de processamento"""
    

    def _converter_para_json(self, obj):
        """Converte tipos numpy para JSON serializável"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._converter_para_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._converter_para_json(item) for item in obj]
        else:
            return obj
    
    def __init__(self):
        logger.info("⚙️ Processador de Estratégias inicializado")
    
    def aplicar_estrategia(self, img: np.ndarray, analise: AnaliseCompleta) -> np.ndarray:
        """Aplica a estratégia recomendada pela análise"""
        estrategia = analise.estrategia_principal
        parametros = analise.parametros_otimizados
        
        logger.info(f"⚙️ Aplicando estratégia: {estrategia.value}")
        
        if estrategia == EstrategiaProcessamento.THRESHOLD_SIMPLES:
            return self._threshold_simples(img, analise.threshold_recomendado, parametros)
        
        elif estrategia == EstrategiaProcessamento.THRESHOLD_ADAPTATIVO:
            return self._threshold_adaptativo(img, parametros)
        
        elif estrategia == EstrategiaProcessamento.MORFOLOGIA_AVANCADA:
            return self._morfologia_avancada(img, analise.threshold_recomendado, parametros)
        
        elif estrategia == EstrategiaProcessamento.DENOISE_AGRESSIVO:
            return self._denoise_agressivo(img, parametros)
        
        elif estrategia == EstrategiaProcessamento.SUPER_RESOLUTION:
            return self._super_resolution(img, parametros)
        
        elif estrategia == EstrategiaProcessamento.SEGMENTACAO_WATERSHED:
            return self._segmentacao_watershed(img, parametros)
        
        elif estrategia == EstrategiaProcessamento.BINARIZACAO_TEXTO_PRETO:
            return self._binarizacao_texto_preto(img, analise.threshold_recomendado, parametros)
        
        elif estrategia == EstrategiaProcessamento.PIPELINE_COMPLETO:
            # PROTEÇÃO ANTI-RECURSÃO: Usar estratégia segura
            logger.warning('⚠️ Pipeline completo bloqueado - usando threshold adaptativo')
            return self._threshold_adaptativo(img, parametros)
            # Fallback para método padrão otimizado
            return self._threshold_simples(img, analise.threshold_recomendado, {})
    
    def _threshold_simples(self, img: np.ndarray, threshold: int, parametros: Dict) -> np.ndarray:
        """
        Estratégia de threshold simples otimizada
        REGRA: Caracteres verdadeiros estão em valor 0, tudo acima é ruído
        """
        # Converter para escala de cinza
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Aplicar blur se necessário para suavizar
        if parametros.get('blur_kernel', (1, 1)) != (1, 1):
            kernel = parametros['blur_kernel']
            gray = cv2.GaussianBlur(gray, kernel, 0)
        
        # THRESHOLD OTIMIZADO: Preservar apenas valores ≤ threshold
        # Caracteres verdadeiros estão em 0, ruído está em valores > 0
        img_limpa = np.full_like(gray, 255, dtype=np.uint8)  # Fundo branco
        img_limpa[gray <= threshold] = 0  # Preservar apenas caracteres (valores baixos)
        
        # Morfologia básica para conectar caracteres fragmentados
        if parametros.get('apply_closing', False):
            kernel_size = parametros.get('morph_kernel_size', 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            img_limpa = cv2.morphologyEx(img_limpa, cv2.MORPH_CLOSE, kernel)
        
        return img_limpa
    
    def _threshold_adaptativo(self, img: np.ndarray, parametros: Dict) -> np.ndarray:
        """
        Estratégia de threshold adaptativo
        REGRA: Preserva valores baixos (caracteres) e remove valores altos (ruído)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Parâmetros otimizados
        block_size = parametros.get('block_size', 11)
        C = parametros.get('C_constant', 2)
        method = parametros.get('method', 'gaussian')
        
        # Aplicar threshold adaptativo
        if method == 'gaussian':
            thresh_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        else:
            thresh_type = cv2.ADAPTIVE_THRESH_MEAN_C
        
        # Usar THRESH_BINARY para preservar valores baixos como preto
        img_thresh = cv2.adaptiveThreshold(
            gray, 255, thresh_type, cv2.THRESH_BINARY, block_size, C
        )
        
        # Inverter se necessário para garantir que texto fique preto
        if np.mean(img_thresh) > 127:  # Maioria branca
            img_thresh = cv2.bitwise_not(img_thresh)
        
        # Limpeza morfológica leve para conectar caracteres
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        img_limpa = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
        
        return img_limpa
    
    def _morfologia_avancada(self, img: np.ndarray, threshold: int, parametros: Dict) -> np.ndarray:
        """
        Estratégia de morfologia avançada
        REGRA: Aplicar threshold baseado em valor 0 para caracteres + morfologia
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Threshold inicial seguindo regra: caracteres em valor 0
        img_binary = np.full_like(gray, 255, dtype=np.uint8)  # Fundo branco
        img_binary[gray <= threshold] = 0  # Preservar apenas valores baixos (caracteres)
        
        # Parâmetros otimizados
        opening_kernel = parametros.get('opening_kernel', (2, 2))
        closing_kernel = parametros.get('closing_kernel', (3, 3))
        iterations = parametros.get('iterations', 1)
        
        # Operações morfológicas para limpeza e conectividade
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, opening_kernel)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, closing_kernel)
        
        # Opening para remover ruído
        img_opened = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel_open, iterations=iterations)
        
        # Closing para conectar fragmentos
        img_closed = cv2.morphologyEx(img_opened, cv2.MORPH_CLOSE, kernel_close, iterations=iterations)
        
        return img_closed
    
    def _binarizacao_texto_preto(self, img: np.ndarray, threshold: int, parametros: Dict) -> np.ndarray:
        """
        Estratégia específica para texto preto (valor 0) com OCR case-sensitive
        REGRA: Texto verdadeiro está em valor 0, fundo/ruído deve ser 255
        """
        try:
            if len(img.shape) == 3:
                gray = img.copy()
            else:
                gray = img.copy()
            
            # Validar se a imagem não está vazia
            if gray is None or gray.size == 0:
                logger.error("❌ Imagem vazia ou inválida na binarização de texto preto")
                return np.ones((64, 256), dtype=np.uint8) * 255  # Imagem em branco padrão
                
            # Usar o detector de caracteres valor 0 para processamento completo
            from detector_caracteres_zero import DetectorCaracteresZero
            detector = DetectorCaracteresZero()
            
            # Preparar parâmetros para o detector
            parametros_detector = {
                'melhorar_contraste_automatico': parametros.get('melhorar_contraste', True),
                'timestamp': parametros.get('timestamp', None)
            }
            
            # Processar com detecção de valor 0 e OCR
            resultado_completo = detector.processar_captcha_completo(gray, **parametros_detector)
            
            # Usar a imagem processada pelo detector
            if 'imagem_tradicional' in resultado_completo:
                img_processada = resultado_completo['imagem_tradicional']
            else:
                # Re-detectar pixels valor 0 para obter imagem binária
                img_processada, _ = detector.detectar_pixels_zero(gray)
            
            # Armazenar resultado OCR nos parâmetros para uso posterior
            if hasattr(self, 'ultimo_resultado_ocr'):
                self.ultimo_resultado_ocr = resultado_completo
            
            # Log do resultado OCR
            if resultado_completo['sucesso']:
                logger.info(f"🎯 OCR detectado: '{resultado_completo['texto_detectado']}' (confiança: {resultado_completo['confianca']:.2f})")
            else:
                logger.warning(f"⚠️ OCR falhou: '{resultado_completo['texto_detectado']}' (confiança: {resultado_completo['confianca']:.2f})")
            
            # Validar resultado antes de retornar
            if img_processada is None or img_processada.size == 0:
                logger.error("❌ Processamento resultou em imagem vazia")
                return np.ones_like(gray) * 255  # Fundo branco padrão
                
            return img_processada
            
        except Exception as e:
            logger.error(f"❌ Erro na binarização de texto preto: {e}")
            # Fallback: binarização simples
            gray_fallback = gray if 'gray' in locals() else img
            if len(gray_fallback.shape) == 3:
                gray_fallback = cv2.cvtColor(gray_fallback, cv2.COLOR_BGR2GRAY)
            img_fallback = np.where(gray_fallback <= threshold, 0, 255).astype(np.uint8)
            return img_fallback
    
    def _denoise_agressivo(self, img: np.ndarray, parametros: Dict) -> np.ndarray:
        """Estratégia de denoise agressivo"""
        if len(img.shape) == 3:
            img_color = img.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Parâmetros otimizados
        d = parametros.get('bilateral_d', 9)
        sigma_color = parametros.get('bilateral_sigma_color', 75)
        sigma_space = parametros.get('bilateral_sigma_space', 75)
        median_kernel = parametros.get('median_kernel', 5)
        
        # Filtro bilateral (preserva bordas)
        img_bilateral = cv2.bilateralFilter(img_color, d, sigma_color, sigma_space)
        gray_filtered = cv2.cvtColor(img_bilateral, cv2.COLOR_BGR2GRAY)
        
        # Filtro de mediana (remove ruído sal e pimenta)
        gray_median = cv2.medianBlur(gray_filtered, median_kernel)
        
        # Threshold OTSU após denoise
        _, img_thresh = cv2.threshold(gray_median, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return img_thresh
    
    def _super_resolution(self, img: np.ndarray, parametros: Dict) -> np.ndarray:
        """Estratégia de super resolution (para imagens pequenas)"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        h, w = gray.shape
        
        # Redimensionar para pelo menos 64 pixels de altura
        if h < 64:
            scale_factor = 64 / h
            new_w = int(w * scale_factor)
            gray_upscaled = cv2.resize(gray, (new_w, 64), interpolation=cv2.INTER_CUBIC)
        else:
            gray_upscaled = gray
        
        # Aplicar unsharp masking para realçar bordas
        gaussian = cv2.GaussianBlur(gray_upscaled, (0, 0), 2.0)
        unsharp = cv2.addWeighted(gray_upscaled, 1.5, gaussian, -0.5, 0)
        
        # Threshold
        _, img_thresh = cv2.threshold(unsharp, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return img_thresh
    
    def _segmentacao_watershed(self, img: np.ndarray, parametros: Dict) -> np.ndarray:
        """Estratégia de segmentação watershed (para caracteres colados)"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Threshold inicial
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Operações morfológicas para preparar watershed
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Distância transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        
        # Threshold na distância para encontrar foreground
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        return sure_fg
    
    def _pipeline_completo(self, img: np.ndarray, analise: AnaliseCompleta) -> np.ndarray:
        """Pipeline completo que combina múltiplas estratégias"""
        # Aplicar estratégia principal
        resultado_principal = self.aplicar_estrategia(img, analise)
        
        # Se há estratégias alternativas, testar e combinar
        if analise.estrategias_alternativas:
            # Para pipeline completo, usar apenas a estratégia principal por ora
            # Futuras versões podem implementar ensemble de estratégias
            pass
        
        return resultado_principal

class IntegradorHeuristico:
    """
    🔗 ORQUESTRADOR PRINCIPAL DAS HEURÍSTICAS DUAIS
    
    Coordena Analisador + Reconhecedor para atingir 98%+ de confiança
    """
    
    def __init__(self, meta_confianca: float = 0.98):
        self.meta_confianca = meta_confianca
        
        # Inicializar componentes
        self.analisador = AnalisadorInteligente()
        self.reconhecedor = ReconhecedorAdaptativo(meta_confianca)
        self.processador = ProcessadorEstrategia()
        
        # Estatísticas
        self.historico_resultados = []
        self.estatisticas = {
            'total_processados': 0,
            'meta_atingida': 0,
            'tempo_medio': 0.0,
            'estrategias_eficazes': {},
            'metodos_eficazes': {}
        }
        
        # Criar pastas para resultados
        self.resultados_dir = Path("heuristica_resultados")
        self.resultados_dir.mkdir(exist_ok=True)
        
        logger.info(f"🔗 Integrador Heurístico inicializado (meta: {meta_confianca:.1%})")
    
    async def resolver_captcha_inteligente(self, img_original: np.ndarray, 
                                         metodos_ocr: Dict[str, Callable],
                                         salvar_resultados: bool = True) -> ResultadoHeuristicaDual:
        """
        🎯 MÉTODO PRINCIPAL: Resolve CAPTCHA com heurística dual para 98%+ confiança
        """
        inicio_total = datetime.now()
        timestamp = inicio_total.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        logger.info("🚀 INICIANDO RESOLUÇÃO COM HEURÍSTICA DUAL")
        logger.info(f"🎯 Meta de confiança: {self.meta_confianca:.1%}")
        
        # FASE 1: ANÁLISE INTELIGENTE
        logger.info("🧠 FASE 1: Análise Inteligente...")
        inicio_analise = datetime.now()
        
        analise = self.analisador.analisar_captcha(img_original)
        
        tempo_analise = (datetime.now() - inicio_analise).total_seconds()
        logger.info(f"✅ Análise concluída em {tempo_analise:.3f}s")
        logger.info(f"   📊 Tipo: {analise.tipo_captcha.value}")
        logger.info(f"   📊 Complexidade: {analise.score_complexidade:.1f}/100")
        logger.info(f"   📊 Estratégia: {analise.estrategia_principal.value}")
        logger.info(f"   📊 Threshold: ≤{analise.threshold_recomendado}")
        
        # FASE 2: PROCESSAMENTO ESTRATÉGICO
        logger.info("⚙️ FASE 2: Processamento Estratégico...")
        
        img_processada = self.processador.aplicar_estrategia(img_original, analise)
        
        # Salvar imagem processada se solicitado
        if salvar_resultados and img_processada is not None and img_processada.size > 0:
            img_processada_path = self.resultados_dir / f"processada_{timestamp}.png"
            try:
                cv2.imwrite(str(img_processada_path), img_processada)
                logger.info(f"💾 Imagem processada salva: {img_processada_path}")
            except Exception as e:
                logger.error(f"❌ Erro ao salvar imagem processada: {e}")
        elif salvar_resultados:
            logger.warning("⚠️ Imagem processada vazia - não foi salva")
        
        # FASE 3: RECONHECIMENTO ADAPTATIVO
        logger.info("🔤 FASE 3: Reconhecimento Adaptativo...")
        inicio_reconhecimento = datetime.now()
        
        decisao_ocr = await self.reconhecedor.reconhecer_com_meta(
            img_processada, metodos_ocr, {
                'analise_original': analise,
                'timestamp': timestamp
            }
        )
        
        tempo_reconhecimento = (datetime.now() - inicio_reconhecimento).total_seconds()
        
        # FASE 4: COMPILAÇÃO DO RESULTADO FINAL
        tempo_total = (datetime.now() - inicio_total).total_seconds()
        
        resultado = ResultadoHeuristicaDual(
            analise_inteligente=analise,
            decisao_ocr=decisao_ocr,
            texto_final=decisao_ocr.texto_final,
            confianca_final=decisao_ocr.confianca_final,
            meta_98_atingida=decisao_ocr.meta_atingida,
            tempo_total_processamento=tempo_total,
            tempo_analise=tempo_analise,
            tempo_reconhecimento=tempo_reconhecimento,
            eficiencia_score=self._calcular_eficiencia(decisao_ocr, tempo_total),
            timestamp=timestamp,
            estrategia_usada=analise.estrategia_principal.value,
            metodo_vencedor=decisao_ocr.metodo_vencedor.value,
            melhorias_aplicadas=self._identificar_melhorias_aplicadas(analise, decisao_ocr)
        )
        
        # Atualizar estatísticas
        self._atualizar_estatisticas(resultado)
        
        # Salvar resultado completo se solicitado
        if salvar_resultados:
            await self._salvar_resultado_completo(resultado, img_original, img_processada)
        
        # Log do resultado final
        self._log_resultado_final(resultado)
        
        return resultado
    
    def _calcular_eficiencia(self, decisao: DecisaoFinal, tempo_total: float) -> float:
        """Calcula score de eficiência do processo"""
        
        # Componentes da eficiência
        confianca_score = decisao.confianca_final  # 0-1
        velocidade_score = max(0, 1.0 - (tempo_total / 15.0))  # Penaliza se > 15s
        iteracoes_score = max(0, 1.0 - (decisao.iteracoes / 10.0))  # Penaliza se > 10 iterações
        
        # Score ponderado
        eficiencia = (
            confianca_score * 0.6 +      # 60% - confiança é mais importante
            velocidade_score * 0.25 +    # 25% - velocidade
            iteracoes_score * 0.15       # 15% - eficiência de iterações
        )
        
        return min(eficiencia, 1.0)
    
    def _identificar_melhorias_aplicadas(self, analise: AnaliseCompleta, 
                                       decisao: DecisaoFinal) -> List[str]:
        """Identifica quais melhorias foram aplicadas"""
        melhorias = []
        
        # Melhorias da análise
        if analise.score_complexidade > 70:
            melhorias.append("analise_complexidade_alta")
        
        if analise.threshold_recomendado != 85:
            melhorias.append(f"threshold_otimizado_{analise.threshold_recomendado}")
        
        if analise.estrategia_principal != EstrategiaProcessamento.THRESHOLD_SIMPLES:
            melhorias.append(f"estrategia_avancada_{analise.estrategia_principal.value}")
        
        # Melhorias do reconhecimento
        if decisao.meta_atingida:
            melhorias.append("meta_98_atingida")
        
        if len(decisao.candidatos_testados) > 1:
            melhorias.append("ensemble_multiplos_metodos")
        
        if decisao.iteracoes <= 3:
            melhorias.append("reconhecimento_eficiente")
        
        return melhorias
    
    def _atualizar_estatisticas(self, resultado: ResultadoHeuristicaDual):
        """Atualiza estatísticas globais"""
        self.estatisticas['total_processados'] += 1
        
        if resultado.meta_98_atingida:
            self.estatisticas['meta_atingida'] += 1
        
        # Atualizar tempo médio
        total = self.estatisticas['total_processados']
        tempo_anterior = self.estatisticas['tempo_medio'] * (total - 1)
        self.estatisticas['tempo_medio'] = (tempo_anterior + resultado.tempo_total_processamento) / total
        
        # Contabilizar estratégias eficazes
        estrategia = resultado.estrategia_usada
        if estrategia not in self.estatisticas['estrategias_eficazes']:
            self.estatisticas['estrategias_eficazes'][estrategia] = {'total': 0, 'sucessos': 0}
        
        self.estatisticas['estrategias_eficazes'][estrategia]['total'] += 1
        if resultado.meta_98_atingida:
            self.estatisticas['estrategias_eficazes'][estrategia]['sucessos'] += 1
        
        # Contabilizar métodos eficazes
        metodo = resultado.metodo_vencedor
        if metodo not in self.estatisticas['metodos_eficazes']:
            self.estatisticas['metodos_eficazes'][metodo] = {'total': 0, 'sucessos': 0}
        
        self.estatisticas['metodos_eficazes'][metodo]['total'] += 1
        if resultado.meta_98_atingida:
            self.estatisticas['metodos_eficazes'][metodo]['sucessos'] += 1
        
        # Adicionar ao histórico (manter apenas últimos 50)
        self.historico_resultados.append(resultado)
        if len(self.historico_resultados) > 50:
            self.historico_resultados.pop(0)
    
    async def _salvar_resultado_completo(self, resultado: ResultadoHeuristicaDual, 
                                       img_original: np.ndarray, img_processada: np.ndarray):
        """Salva resultado completo com todos os detalhes"""
        timestamp = resultado.timestamp
        
        # Salvar imagem original
        original_path = self.resultados_dir / f"original_{timestamp}.png"
        cv2.imwrite(str(original_path), img_original)
        
        # Salvar relatório JSON
        relatorio_path = self.resultados_dir / f"relatorio_{timestamp}.json"
        relatorio_data = {
            'timestamp': timestamp,
            'texto_final': resultado.texto_final,
            'confianca_final': resultado.confianca_final,
            'meta_atingida': resultado.meta_98_atingida,
            'tempo_total': resultado.tempo_total_processamento,
            'eficiencia_score': resultado.eficiencia_score,
            'analise': {
                'tipo_captcha': resultado.analise_inteligente.tipo_captcha.value,
                'complexidade': resultado.analise_inteligente.score_complexidade,
                'estrategia': resultado.analise_inteligente.estrategia_principal.value,
                'threshold': resultado.analise_inteligente.threshold_recomendado,
                'confianca_analise': resultado.analise_inteligente.confianca_analise
            },
            'reconhecimento': {
                'metodo_vencedor': resultado.decisao_ocr.metodo_vencedor.value,
                'status_parada': resultado.decisao_ocr.status_parada.value,
                'iteracoes': resultado.decisao_ocr.iteracoes,
                'candidatos_testados': len(resultado.decisao_ocr.candidatos_testados)
            },
            'melhorias_aplicadas': resultado.melhorias_aplicadas
        }
        
        with open(relatorio_path, 'w', encoding='utf-8') as f:
            # Converter tipos numpy antes de salvar
            relatorio_convertido = convert_numpy_types(relatorio_data)
            json.dump(relatorio_convertido, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Relatório completo salvo: {relatorio_path}")
    
    def _log_resultado_final(self, resultado: ResultadoHeuristicaDual):
        """Log detalhado do resultado final"""
        status_emoji = "🎉" if resultado.meta_98_atingida else "⚠️"
        
        logger.info("=" * 60)
        logger.info(f"{status_emoji} RESULTADO FINAL DA HEURÍSTICA DUAL")
        logger.info("=" * 60)
        logger.info(f"📝 Texto: '{resultado.texto_final}'")
        logger.info(f"🎯 Confiança: {resultado.confianca_final:.1%}")
        logger.info(f"✅ Meta 98% atingida: {'SIM' if resultado.meta_98_atingida else 'NÃO'}")
        logger.info(f"⏱️ Tempo total: {resultado.tempo_total_processamento:.2f}s")
        logger.info(f"📊 Eficiência: {resultado.eficiencia_score:.1%}")
        logger.info(f"🧠 Estratégia: {resultado.estrategia_usada}")
        logger.info(f"🔤 Método: {resultado.metodo_vencedor}")
        logger.info(f"🔧 Melhorias: {', '.join(resultado.melhorias_aplicadas)}")
        logger.info("=" * 60)
        
        # Estatísticas globais
        taxa_sucesso = self.estatisticas['meta_atingida'] / max(1, self.estatisticas['total_processados'])
        logger.info(f"📈 Taxa de sucesso global: {taxa_sucesso:.1%} ({self.estatisticas['meta_atingida']}/{self.estatisticas['total_processados']})")
    
    def obter_relatorio_estatisticas(self) -> str:
        """Gera relatório detalhado das estatísticas"""
        stats = self.estatisticas
        
        if stats['total_processados'] == 0:
            return "📊 Nenhum CAPTCHA processado ainda."
        
        taxa_sucesso = stats['meta_atingida'] / stats['total_processados']
        
        relatorio = f"""
📊 ESTATÍSTICAS DA HEURÍSTICA DUAL
=================================
Total processados: {stats['total_processados']}
Meta 98% atingida: {stats['meta_atingida']} ({taxa_sucesso:.1%})
Tempo médio: {stats['tempo_medio']:.2f}s

🎯 ESTRATÉGIAS MAIS EFICAZES:
"""
        
        for estrategia, dados in stats['estrategias_eficazes'].items():
            taxa = dados['sucessos'] / max(1, dados['total'])
            relatorio += f"   {estrategia}: {taxa:.1%} ({dados['sucessos']}/{dados['total']})\n"
        
        relatorio += "\n🔤 MÉTODOS MAIS EFICAZES:\n"
        for metodo, dados in stats['metodos_eficazes'].items():
            taxa = dados['sucessos'] / max(1, dados['total'])
            relatorio += f"   {metodo}: {taxa:.1%} ({dados['sucessos']}/{dados['total']})\n"
        
        return relatorio
    
    def salvar_conhecimento(self):
        """Salva todo o conhecimento adquirido"""
        self.analisador.salvar_conhecimento()
        
        # Salvar estatísticas próprias
        try:
            stats_path = self.resultados_dir / "estatisticas_integrador.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                # Converter tipos numpy antes de salvar
                estatisticas_convertidas = convert_numpy_types(self.estatisticas)
                json.dump(estatisticas_convertidas, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Estatísticas do integrador salvas: {stats_path}")
        except Exception as e:
            logger.error(f"❌ Erro ao salvar estatísticas: {e}")

if __name__ == "__main__":
    # Teste do integrador
    logging.basicConfig(level=logging.INFO, format='🔗 [%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    
    print("🔗 Testando Integrador Heurístico...")
    
    async def teste_integrador():
        integrador = IntegradorHeuristico(meta_confianca=0.98)
        
        # Simular métodos OCR
        async def mock_easyocr(img):
            await asyncio.sleep(0.1)
            return "TEST123", 0.95
        
        metodos_mock = {
            'easyocr_otimizado': mock_easyocr
        }
        
        # Criar imagem de teste
        img_teste = np.random.randint(0, 255, (40, 120, 3), dtype=np.uint8)
        
        # Executar teste
        resultado = await integrador.resolver_captcha_inteligente(img_teste, metodos_mock)
        
        # Mostrar estatísticas
        print(integrador.obter_relatorio_estatisticas())
    
    asyncio.run(teste_integrador())
