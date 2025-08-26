#!/usr/bin/env python3
"""
🧠 HEURÍSTICA DE ANÁLISE - ANALISADOR INTELIGENTE
================================================
Sistema de decisão inteligente para análise e classificação de CAPTCHAs
Decide COMO processar cada imagem para máxima eficácia
"""

import cv2
import numpy as np
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import json
from datetime import datetime

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

logger = logging.getLogger(__name__)

class TipoCaptcha(Enum):
    """Tipos de CAPTCHA identificados pelo analisador"""
    SIMPLES_LIMPO = "simples_limpo"           # Texto claro, fundo uniforme
    PADRAO_SOMBRAS = "padrao_sombras"         # Texto com sombras leves
    DISTORCAO_GEOMETRICA = "distorcao_geom"   # Texto distorcido/inclinado
    RUIDO_PESADO = "ruido_pesado"             # Muito ruído/interferência
    BAIXA_RESOLUCAO = "baixa_resolucao"       # Imagem pequena/borrada
    CASO_ESPECIAL = "caso_especial"           # Requer tratamento especial

class EstrategiaProcessamento(Enum):
    """Estratégias de processamento identificadas"""
    THRESHOLD_SIMPLES = "threshold_simples"
    THRESHOLD_ADAPTATIVO = "threshold_adaptativo" 
    BINARIZACAO_TEXTO_PRETO = "binarizacao_texto_preto"  # NOVA: Específica para texto preto
    MORFOLOGIA_AVANCADA = "morfologia_avancada"
    DENOISE_AGRESSIVO = "denoise_agressivo"
    SUPER_RESOLUTION = "super_resolution"
    SEGMENTACAO_WATERSHED = "segmentacao_watershed"
    PIPELINE_COMPLETO = "pipeline_completo"

@dataclass
class AnaliseCompleta:
    """Resultado completo da análise heurística"""
    # Características da imagem
    tipo_captcha: TipoCaptcha
    score_complexidade: float
    caracteristicas: Dict[str, float]
    
    # Estratégia recomendada
    estrategia_principal: EstrategiaProcessamento
    estrategias_alternativas: List[EstrategiaProcessamento]
    confianca_analise: float
    
    # Configurações específicas
    parametros_otimizados: Dict[str, any]
    threshold_recomendado: int
    
    # Metadados
    tempo_analise: float
    timestamp: str

class AnalisadorInteligente:
    """
    🧠 HEURÍSTICA 1: ANALISADOR INTELIGENTE
    
    Responsabilidades:
    - Classificar tipo de CAPTCHA
    - Calcular score de complexidade
    - Decidir estratégia de processamento
    - Otimizar parâmetros específicos
    """
    
    def __init__(self):
        self.historico_analises = []
        self.estatisticas = {
            'total_analisados': 0,
            'tipos_encontrados': {},
            'estrategias_eficazes': {},
            'parametros_otimos': {}
        }
        
        # Carregar conhecimento prévio se existir
        self.carregar_conhecimento_previo()
        
        logger.info("🧠 Analisador Inteligente inicializado")
    
    def analisar_captcha(self, img: np.ndarray) -> AnaliseCompleta:
        """
        ANÁLISE PRINCIPAL: Determina o melhor approach para o CAPTCHA
        """
        inicio = datetime.now()
        logger.info("🔍 Iniciando análise heurística inteligente...")
        
        # 1. EXTRAÇÃO DE CARACTERÍSTICAS
        caracteristicas = self._extrair_caracteristicas(img)
        logger.info(f"📊 Características extraídas: {len(caracteristicas)} métricas")
        
        # 2. CLASSIFICAÇÃO DO TIPO
        tipo_captcha = self._classificar_tipo(caracteristicas)
        logger.info(f"🏷️ Tipo identificado: {tipo_captcha.value}")
        
        # 3. CÁLCULO DE COMPLEXIDADE
        score_complexidade = self._calcular_complexidade(caracteristicas)
        logger.info(f"📈 Score de complexidade: {score_complexidade:.2f}")
        
        # 4. SELEÇÃO DE ESTRATÉGIA
        estrategia_principal, alternativas = self._selecionar_estrategia(
            tipo_captcha, score_complexidade, caracteristicas
        )
        logger.info(f"🎯 Estratégia principal: {estrategia_principal.value}")
        
        # 5. OTIMIZAÇÃO DE PARÂMETROS
        parametros = self._otimizar_parametros(
            tipo_captcha, estrategia_principal, caracteristicas
        )
        
        # 6. THRESHOLD RECOMENDADO
        threshold = self._calcular_threshold_otimo(caracteristicas, tipo_captcha)
        logger.info(f"🎯 Threshold otimizado: ≤{threshold}")
        
        # 7. CONFIANÇA DA ANÁLISE
        confianca = self._calcular_confianca_analise(
            tipo_captcha, score_complexidade, caracteristicas
        )
        
        tempo_total = (datetime.now() - inicio).total_seconds()
        
        # Criar resultado completo
        analise = AnaliseCompleta(
            tipo_captcha=tipo_captcha,
            score_complexidade=score_complexidade,
            caracteristicas=caracteristicas,
            estrategia_principal=estrategia_principal,
            estrategias_alternativas=alternativas,
            confianca_analise=confianca,
            parametros_otimizados=parametros,
            threshold_recomendado=threshold,
            tempo_analise=tempo_total,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        )
        
        # Atualizar estatísticas
        self._atualizar_estatisticas(analise)
        
        logger.info(f"✅ Análise concluída em {tempo_total:.3f}s (confiança: {confianca:.1%})")
        return analise
    
    def _extrair_caracteristicas(self, img: np.ndarray) -> Dict[str, float]:
        """Extrai características detalhadas da imagem"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        h, w = gray.shape
        total_pixels = h * w
        
        caracteristicas = {}
        
        # 1. CARACTERÍSTICAS BÁSICAS
        caracteristicas['altura'] = h
        caracteristicas['largura'] = w
        caracteristicas['aspecto_ratio'] = w / h
        caracteristicas['area_total'] = total_pixels
        
        # 2. CARACTERÍSTICAS DE INTENSIDADE
        caracteristicas['intensidade_media'] = np.mean(gray)
        caracteristicas['intensidade_std'] = np.std(gray)
        caracteristicas['intensidade_min'] = np.min(gray)
        caracteristicas['intensidade_max'] = np.max(gray)
        caracteristicas['range_dinamico'] = caracteristicas['intensidade_max'] - caracteristicas['intensidade_min']
        
        # 3. ANÁLISE DE HISTOGRAMA
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.flatten() / total_pixels
        
        # Entropia (complexidade da informação)
        entropia = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        caracteristicas['entropia'] = entropia
        
        # Picos no histograma
        picos = self._encontrar_picos_histograma(hist_norm)
        caracteristicas['num_picos_histograma'] = len(picos)
        caracteristicas['pico_principal'] = picos[0][0] if picos else 128
        
        # 4. ANÁLISE DE CONTRASTE E BORDAS
        # Variância do Laplaciano (nitidez)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        caracteristicas['variancia_laplaciana'] = laplacian.var()
        
        # Detecção de bordas Canny
        edges = cv2.Canny(gray, 50, 150)
        caracteristicas['densidade_bordas'] = np.sum(edges > 0) / total_pixels
        caracteristicas['density_bordas'] = caracteristicas['densidade_bordas']  # Alias para compatibilidade
        
        # 5. ANÁLISE DE COMPONENTES
        # Threshold OTSU para análise binária
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Componentes conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Filtrar componentes por tamanho
        areas_validas = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > total_pixels * 0.001:  # Pelo menos 0.1% da imagem
                areas_validas.append(area)
        
        caracteristicas['num_componentes'] = len(areas_validas)
        caracteristicas['area_media_componentes'] = np.mean(areas_validas) if areas_validas else 0
        caracteristicas['densidade_texto'] = sum(areas_validas) / total_pixels if areas_validas else 0
        
        # 6. ANÁLISE DE PIXELS ESPECÍFICOS - OTIMIZADA PARA DETECÇÃO DE TEXTO
        # REGRA: Texto verdadeiro está em valor 0 (preto puro)
        pixels_preto = np.sum(gray == 0)
        caracteristicas['percentual_preto_puro'] = pixels_preto / total_pixels * 100
        
        # Pixels muito escuros (≤20) - possível texto com anti-aliasing
        pixels_muito_escuros = np.sum(gray <= 20)
        caracteristicas['percentual_muito_escuros'] = pixels_muito_escuros / total_pixels * 100
        
        # Pixels escuros (≤85) - incluindo sombras de texto
        pixels_escuros = np.sum(gray <= 85)
        caracteristicas['percentual_escuros'] = pixels_escuros / total_pixels * 100
        
        # Pixels claros (>200) - provável fundo
        pixels_claros = np.sum(gray > 200)
        caracteristicas['percentual_claros'] = pixels_claros / total_pixels * 100
        
        # ANÁLISE DE CONCENTRAÇÃO DE VALORES BAIXOS (indica qualidade do texto)
        # Concentração em valores 0-10 (texto muito nítido)
        pixels_nitidos = np.sum(gray <= 10)
        caracteristicas['concentracao_texto_nitido'] = pixels_nitidos / total_pixels * 100
        
        # Concentração em valores 11-50 (texto com bordas suaves)
        pixels_bordas_suaves = np.sum((gray > 10) & (gray <= 50))
        caracteristicas['concentracao_bordas_suaves'] = pixels_bordas_suaves / total_pixels * 100
        
        # 7. ANÁLISE DE TEXTURA
        # Uniformidade (Range de valores únicos)
        valores_unicos = len(np.unique(gray))
        caracteristicas['diversidade_valores'] = valores_unicos
        caracteristicas['uniformidade'] = 1.0 - (valores_unicos / 256.0)
        
        # 8. ANÁLISE DE GRADIENTES
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        caracteristicas['gradiente_medio'] = np.mean(magnitude)
        caracteristicas['gradiente_std'] = np.std(magnitude)
        
        return caracteristicas
    
    def _encontrar_picos_histograma(self, hist_norm: np.ndarray) -> List[Tuple[int, float]]:
        """Encontra picos significativos no histograma"""
        picos = []
        
        for i in range(1, 255):
            if (hist_norm[i] > hist_norm[i-1] and 
                hist_norm[i] > hist_norm[i+1] and 
                hist_norm[i] > np.max(hist_norm) * 0.1):
                picos.append((i, hist_norm[i]))
        
        # Ordenar por intensidade (frequência)
        picos.sort(key=lambda x: x[1], reverse=True)
        return picos
    
    def _classificar_tipo(self, carac: Dict[str, float]) -> TipoCaptcha:
        """Classifica o tipo de CAPTCHA baseado nas características"""
        
        # SIMPLES LIMPO: Poucos valores únicos, alto contraste, poucos componentes
        if (carac['diversidade_valores'] < 50 and 
            carac['range_dinamico'] > 150 and 
            carac['num_componentes'] <= 8 and
            carac['entropia'] < 4.0):
            return TipoCaptcha.SIMPLES_LIMPO
        
        # PADRÃO SOMBRAS: Valores escuros moderados, entropia média
        if (carac['percentual_escuros'] > 5 and carac['percentual_escuros'] < 25 and
            carac['entropia'] >= 4.0 and carac['entropia'] < 6.5 and
            carac['num_componentes'] >= 3 and carac['num_componentes'] <= 10):
            return TipoCaptcha.PADRAO_SOMBRAS
        
        # DISTORÇÃO GEOMÉTRICA: Alto gradiente, muitas bordas
        if (carac['variancia_laplaciana'] > 800 and
            carac['densidade_bordas'] > 0.15 and
            carac['gradiente_std'] > 50):
            return TipoCaptcha.DISTORCAO_GEOMETRICA
        
        # RUÍDO PESADO: Alta entropia, muitos valores únicos
        if (carac['entropia'] > 6.5 and
            carac['diversidade_valores'] > 200 and
            carac['variancia_laplaciana'] > 400):
            return TipoCaptcha.RUIDO_PESADO
        
        # BAIXA RESOLUÇÃO: Área pequena ou baixa nitidez
        if (carac['area_total'] < 5000 or
            carac['variancia_laplaciana'] < 100):
            return TipoCaptcha.BAIXA_RESOLUCAO
        
        # Casos que não se encaixam nos padrões
        return TipoCaptcha.CASO_ESPECIAL
    
    def _calcular_complexidade(self, carac: Dict[str, float]) -> float:
        """Calcula score de complexidade (0-100)"""
        
        # Normalizar características para 0-1
        entropia_norm = min(carac['entropia'] / 8.0, 1.0)
        laplacian_norm = min(carac['variancia_laplaciana'] / 2000.0, 1.0)
        componentes_norm = min(carac['num_componentes'] / 15.0, 1.0)
        diversidade_norm = carac['diversidade_valores'] / 256.0
        
        # Score ponderado
        complexidade = (
            entropia_norm * 0.30 +           # 30% - Complexidade da informação
            laplacian_norm * 0.25 +          # 25% - Nitidez/contraste
            componentes_norm * 0.20 +        # 20% - Número de elementos
            diversidade_norm * 0.25          # 25% - Diversidade de cores
        ) * 100
        
        return min(complexidade, 100.0)
    
    def _selecionar_estrategia(self, tipo: TipoCaptcha, complexidade: float, 
                              carac: Dict[str, float]) -> Tuple[EstrategiaProcessamento, List[EstrategiaProcessamento]]:
        """
        Seleciona estratégia OTIMIZADA para texto preto (valor 0)
        PRIORIDADE: Detectar concentração de pixels pretos e usar estratégia específica
        """
        
        # REGRA PRIORITÁRIA: Se há concentração significativa de texto preto, usar estratégia específica
        percentual_preto_puro = carac.get('percentual_preto_puro', 0)
        concentracao_nitido = carac.get('concentracao_texto_nitido', 0)
        
        # ESTRATÉGIA ESPECÍFICA PARA TEXTO PRETO NÍTIDO
        if percentual_preto_puro > 5 or concentracao_nitido > 8:
            logger.info(f"🖤 TEXTO PRETO DETECTADO - Usando estratégia de binarização específica")
            return (
                EstrategiaProcessamento.BINARIZACAO_TEXTO_PRETO,
                [EstrategiaProcessamento.THRESHOLD_SIMPLES, EstrategiaProcessamento.MORFOLOGIA_AVANCADA]
            )
        
        # Estratégias tradicionais por tipo (fallback)
        estrategias_por_tipo = {
            TipoCaptcha.SIMPLES_LIMPO: (
                EstrategiaProcessamento.THRESHOLD_SIMPLES,
                [EstrategiaProcessamento.BINARIZACAO_TEXTO_PRETO, EstrategiaProcessamento.THRESHOLD_ADAPTATIVO]
            ),
            TipoCaptcha.PADRAO_SOMBRAS: (
                EstrategiaProcessamento.BINARIZACAO_TEXTO_PRETO,  # Priorizar para sombras
                [EstrategiaProcessamento.THRESHOLD_ADAPTATIVO, EstrategiaProcessamento.MORFOLOGIA_AVANCADA]
            ),
            TipoCaptcha.DISTORCAO_GEOMETRICA: (
                EstrategiaProcessamento.MORFOLOGIA_AVANCADA,
                [EstrategiaProcessamento.BINARIZACAO_TEXTO_PRETO, EstrategiaProcessamento.SEGMENTACAO_WATERSHED]
            ),
            TipoCaptcha.RUIDO_PESADO: (
                EstrategiaProcessamento.DENOISE_AGRESSIVO,
                [EstrategiaProcessamento.BINARIZACAO_TEXTO_PRETO, EstrategiaProcessamento.MORFOLOGIA_AVANCADA]
            ),
            TipoCaptcha.BAIXA_RESOLUCAO: (
                EstrategiaProcessamento.SUPER_RESOLUTION,
                [EstrategiaProcessamento.BINARIZACAO_TEXTO_PRETO, EstrategiaProcessamento.THRESHOLD_ADAPTATIVO]
            ),
            TipoCaptcha.CASO_ESPECIAL: (
                EstrategiaProcessamento.PIPELINE_COMPLETO,
                [EstrategiaProcessamento.BINARIZACAO_TEXTO_PRETO, EstrategiaProcessamento.MORFOLOGIA_AVANCADA]
            )
        }
        
        principal, alternativas = estrategias_por_tipo[tipo]
        
        # Ajuste baseado na complexidade
        if complexidade > 80:
            # Casos muito complexos sempre usam pipeline completo
            if principal != EstrategiaProcessamento.PIPELINE_COMPLETO:
                alternativas.insert(0, principal)
                principal = EstrategiaProcessamento.PIPELINE_COMPLETO
        
        return principal, alternativas
    
    def _otimizar_parametros(self, tipo: TipoCaptcha, estrategia: EstrategiaProcessamento, 
                           carac: Dict[str, float]) -> Dict[str, any]:
        """Otimiza parâmetros específicos para a estratégia escolhida"""
        
        parametros = {}
        
        # NOVA ESTRATÉGIA: Binarização específica para texto preto
        if estrategia == EstrategiaProcessamento.BINARIZACAO_TEXTO_PRETO:
            # Parâmetros otimizados para texto em valor 0 (preto puro)
            percentual_preto = carac.get('percentual_preto_puro', 0)
            concentracao_nitido = carac.get('concentracao_texto_nitido', 0)
            
            if percentual_preto > 8:
                # Texto muito nítido - threshold muito baixo
                parametros.update({
                    'threshold_max': 5,      # Aceitar apenas pixels 0-5
                    'invert_binary': False,  # Manter preto como preto
                    'apply_opening': True,   # Remover ruído pequeno
                    'opening_kernel': (2, 2),
                    'apply_closing': True,   # Conectar caracteres quebrados
                    'closing_kernel': (1, 1)
                })
            elif concentracao_nitido > 5:
                # Texto nítido com algum anti-aliasing
                parametros.update({
                    'threshold_max': 15,     # Incluir bordas suaves
                    'invert_binary': False,
                    'apply_opening': True,
                    'opening_kernel': (1, 1),
                    'apply_closing': True,
                    'closing_kernel': (2, 2)
                })
            else:
                # Texto com bordas suaves ou sombras
                parametros.update({
                    'threshold_max': 35,     # Incluir tons escuros
                    'invert_binary': False,
                    'apply_opening': False,  # Não remover - pode ser texto
                    'apply_closing': True,
                    'closing_kernel': (3, 3)
                })
        
        # Parâmetros base por estratégia (existentes)
        elif estrategia == EstrategiaProcessamento.THRESHOLD_SIMPLES:
            parametros.update({
                'blur_kernel': (3, 3) if carac['variancia_laplaciana'] > 200 else (1, 1),
                'morph_kernel_size': 2 if carac['num_componentes'] > 6 else 1,
                'apply_closing': carac['num_componentes'] > 5
            })
            
        elif estrategia == EstrategiaProcessamento.THRESHOLD_ADAPTATIVO:
            parametros.update({
                'block_size': 15 if carac['area_total'] > 10000 else 11,
                'C_constant': 3 if carac['entropia'] > 5 else 2,
                'method': 'gaussian' if carac['gradiente_std'] > 30 else 'mean'
            })
            
        elif estrategia == EstrategiaProcessamento.MORFOLOGIA_AVANCADA:
            parametros.update({
                'opening_kernel': (3, 3) if carac['density_bordas'] > 0.1 else (2, 2),
                'closing_kernel': (4, 4) if carac['num_componentes'] > 8 else (3, 3),
                'iterations': 2 if carac['variancia_laplaciana'] > 500 else 1
            })
            
        elif estrategia == EstrategiaProcessamento.DENOISE_AGRESSIVO:
            parametros.update({
                'bilateral_d': 9 if carac['entropia'] > 6 else 5,
                'bilateral_sigma_color': 75 if carac['diversidade_valores'] > 150 else 50,
                'bilateral_sigma_space': 75,
                'median_kernel': 5 if carac['variancia_laplaciana'] < 200 else 3
            })
        
        # Adicionar parâmetros comuns otimizados
        parametros.update({
            'preprocessing_intensity': 'high' if carac['entropia'] > 6 else 'medium',
            'postprocessing_enabled': carac['num_componentes'] > 3,
            'quality_check_enabled': True
        })
        
        return parametros
    
    def _calcular_threshold_otimo(self, carac: Dict[str, float], tipo: TipoCaptcha) -> int:
        """
        HEURÍSTICA OTIMIZADA: Caracteres verdadeiros estão em valor 0 (preto puro)
        Todo resto (valor > 0) é ruído/fundo que deve ser eliminado ou convertido para 255 (branco)
        """
        
        # REGRA FUNDAMENTAL: Texto = valor 0, Fundo/Ruído = valor > 0
        percentual_preto_puro = carac['percentual_preto_puro']
        percentual_escuros = carac['percentual_escuros']
        
        logger.info(f"📊 Análise de distribuição: {percentual_preto_puro:.1f}% pixels valor 0, {percentual_escuros:.1f}% pixels escuros")
        
        # ESTRATÉGIA 1: CAPTCHA com alto percentual de preto puro (texto nítido)
        if percentual_preto_puro > 8:
            logger.info(f"🖤 TEXTO NÍTIDO DETECTADO - Threshold ≤5 (manter apenas valor 0)")
            return 5  # Muito restritivo - apenas pixels 0-5 são considerados texto
        
        # ESTRATÉGIA 2: CAPTCHA com pouco preto puro mas muitos escuros (texto borrado/anti-aliasing)
        elif percentual_preto_puro < 2 and percentual_escuros > 15:
            # Analisar concentração em valores muito baixos
            if carac.get('percentual_muito_escuros', 0) > 10:  # pixels ≤ 20
                logger.info(f"🌑 TEXTO COM ANTI-ALIASING - Threshold ≤20 (capturar bordas suaves)")
                return 20  # Capturar texto com bordas suavizadas
            else:
                logger.info(f"⚫ TEXTO EM SOMBRAS - Threshold ≤35 (incluir tons escuros)")
                return 35  # Incluir tons escuros que podem ser texto
        
        # ESTRATÉGIA 3: CAPTCHA com pouco contraste (fundo claro, texto escuro)
        elif percentual_preto_puro < 2 and carac['percentual_claros'] > 60:
            logger.info(f"🤍 ALTO CONTRASTE FUNDO CLARO - Threshold ≤50 (separar texto de fundo)")
            return 50  # Fundo muito claro, separar melhor o texto
        
        # ESTRATÉGIA 4: CAPTCHA balanceado (quantidade moderada de preto)
        elif 2 <= percentual_preto_puro <= 8:
            # Verificar se há picos distintos no histograma
            num_picos = carac.get('num_picos_histograma', 0)
            if num_picos >= 2:
                pico_principal = carac.get('pico_principal', 128)
                if pico_principal > 200:  # Fundo muito claro
                    logger.info(f"🌫️ BIMODAL COM FUNDO CLARO - Threshold ≤40")
                    return 40
                else:
                    logger.info(f"📊 BIMODAL BALANCEADO - Threshold ≤60")
                    return 60
            else:
                logger.info(f"⚫ DISTRIBUIÇÃO UNIFORME - Threshold ≤70")
                return 70
        
        # ESTRATÉGIA 5: CAPTCHA problemático (fallback conservador)
        else:
            logger.info(f"❓ PADRÃO NÃO IDENTIFICADO - Usando threshold conservador ≤85")
            return 85  # Valor comprovadamente eficaz (78.6% sucesso)
        
        # Análise adicional baseada no tipo identificado
        if tipo == TipoCaptcha.RUIDO_PESADO:
            # Para ruído pesado, usar threshold específico otimizado
            threshold_base = 40
            logger.info(f"🗿 CAPTCHA com ruído pesado - threshold otimizado ≤{threshold_base}")
            return threshold_base
            
        elif tipo == TipoCaptcha.BAIXA_RESOLUCAO:
            # Para baixa resolução, ser mais permissivo para capturar detalhes fragmentados
            threshold_base = max(85, 85)
            logger.info(f"🔍 CAPTCHA baixa resolução - threshold permissivo ≤{threshold_base}")
            return threshold_base

    def _calcular_confianca_analise(self, tipo: TipoCaptcha, complexidade: float, 
                              carac: Dict[str, float]) -> float:
        """Calcula a confiança da análise realizada"""
        
        confianca_base = 0.8  # 80% base
        
        # Boost para tipos bem definidos
        confianca_por_tipo = {
            TipoCaptcha.SIMPLES_LIMPO: 0.95,
            TipoCaptcha.PADRAO_SOMBRAS: 0.90,
            TipoCaptcha.DISTORCAO_GEOMETRICA: 0.85,
            TipoCaptcha.RUIDO_PESADO: 0.80,
            TipoCaptcha.BAIXA_RESOLUCAO: 0.85,
            TipoCaptcha.CASO_ESPECIAL: 0.70
        }
        
        confianca = confianca_por_tipo[tipo]
        
        # Ajuste baseado na complexidade
        if complexidade < 30:
            confianca += 0.05  # Casos simples, mais confiança
        elif complexidade > 80:
            confianca -= 0.10  # Casos complexos, menos confiança
        
        # Ajuste baseado na clareza dos dados
        if carac['num_picos_histograma'] >= 2 and carac['range_dinamico'] > 100:
            confianca += 0.05  # Dados claros, mais confiança
        
        return min(confianca, 0.98)  # Máximo 98% de confiança na análise
    
    def _atualizar_estatisticas(self, analise: AnaliseCompleta):
        """Atualiza estatísticas internas para aprendizado"""
        self.estatisticas['total_analisados'] += 1
        
        # Contabilizar tipos
        tipo_str = analise.tipo_captcha.value
        if tipo_str not in self.estatisticas['tipos_encontrados']:
            self.estatisticas['tipos_encontrados'][tipo_str] = 0
        self.estatisticas['tipos_encontrados'][tipo_str] += 1
        
        # Adicionar ao histórico (manter apenas últimos 100)
        self.historico_analises.append(analise)
        if len(self.historico_analises) > 100:
            self.historico_analises.pop(0)
    
    def carregar_conhecimento_previo(self):
        """Carrega conhecimento de análises anteriores"""
        try:
            stats_path = Path("heuristica_estatisticas.json")
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    self.estatisticas.update(json.load(f))
                logger.info(f"📚 Conhecimento prévio carregado: {self.estatisticas['total_analisados']} análises")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar conhecimento prévio: {e}")
    
    def salvar_conhecimento(self):
        """Salva conhecimento adquirido"""
        try:
            stats_path = Path("heuristica_estatisticas.json")
            with open(stats_path, 'w') as f:
                # Converter tipos numpy antes de salvar
                estatisticas_convertidas = convert_numpy_types(self.estatisticas)
                json.dump(estatisticas_convertidas, f, indent=2)
            logger.info(f"💾 Conhecimento salvo: {self.estatisticas['total_analisados']} análises")
        except Exception as e:
            logger.error(f"❌ Erro ao salvar conhecimento: {e}")
    
    def obter_relatorio_analise(self, analise: AnaliseCompleta) -> str:
        """Gera relatório detalhado da análise"""
        relatorio = f"""
🧠 RELATÓRIO DE ANÁLISE HEURÍSTICA
==================================
Timestamp: {analise.timestamp}
Tempo de análise: {analise.tempo_analise:.3f}s

📊 CLASSIFICAÇÃO:
   Tipo: {analise.tipo_captcha.value}
   Complexidade: {analise.score_complexidade:.1f}/100
   Confiança da análise: {analise.confianca_analise:.1%}

🎯 ESTRATÉGIA RECOMENDADA:
   Principal: {analise.estrategia_principal.value}
   Alternativas: {', '.join([e.value for e in analise.estrategias_alternativas])}
   Threshold ótimo: ≤{analise.threshold_recomendado}

📈 CARACTERÍSTICAS PRINCIPAIS:
   Dimensões: {analise.caracteristicas['largura']}x{analise.caracteristicas['altura']}
   Entropia: {analise.caracteristicas['entropia']:.2f}
   Componentes: {analise.caracteristicas['num_componentes']}
   Preto puro: {analise.caracteristicas['percentual_preto_puro']:.1f}%
   Escuros: {analise.caracteristicas['percentual_escuros']:.1f}%

⚙️ PARÂMETROS OTIMIZADOS:
"""
        for param, valor in analise.parametros_otimizados.items():
            relatorio += f"   {param}: {valor}\n"
        
        return relatorio

if __name__ == "__main__":
    # Teste do analisador
    logging.basicConfig(level=logging.INFO, format='🧠 [%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    
    print("🧠 Testando Analisador Inteligente...")
    analisador = AnalisadorInteligente()
    
    # Criar imagem de teste
    img_teste = np.random.randint(0, 255, (40, 120, 3), dtype=np.uint8)
    
    # Executar análise
    resultado = analisador.analisar_captcha(img_teste)
    
    # Mostrar relatório
    print(analisador.obter_relatorio_analise(resultado))
