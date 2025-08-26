#!/usr/bin/env python3
"""
🔤 HEURÍSTICA DE RECONHECIMENTO - RECONHECEDOR ADAPTATIVO
========================================================
Sistema de decisão inteligente para OCR com ensemble dinâmico
Decide QUAL método usar e QUANDO parar para atingir 98%+ de confiança
"""

import cv2
import numpy as np
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Callable
from pathlib import Path
import json
import time
from datetime import datetime
import re
import asyncio

logger = logging.getLogger(__name__)

class MetodoOCR(Enum):
    """Métodos de OCR disponíveis no ensemble"""
    DETECTOR_VALOR_ZERO = "detector_valor_zero"  # NOVO: Detecção específica valor 0
    TREINAMENTO_MANUAL = "treinamento_manual"
    EASYOCR_OTIMIZADO = "easyocr_otimizado"  
    EASYOCR_ALTERNATIVO = "easyocr_alternativo"
    TESSERACT_CUSTOM = "tesseract_custom"
    TESSERACT_ALTERNATIVO = "tesseract_alternativo"
    CNN_PROPRIO = "cnn_proprio"
    PADDLEOCR = "paddleocr"
    ENSEMBLE_VOTING = "ensemble_voting"

class StatusDecisao(Enum):
    """Status da decisão de parada"""
    CONTINUAR = "continuar"
    CONFIANCA_ALTA = "confianca_alta"
    CONSENSO_MULTIPLO = "consenso_multiplo"
    VALIDACAO_CRUZADA = "validacao_cruzada"
    LIMITE_TENTATIVAS = "limite_tentativas"
    TIMEOUT = "timeout"
    META_ATINGIDA = "meta_atingida"

@dataclass
class ResultadoOCR:
    """Resultado individual de um método de OCR"""
    metodo: MetodoOCR
    texto: str
    confianca_bruta: float
    confianca_ajustada: float
    tempo_execucao: float
    metadados: Dict[str, any]
    timestamp: str

@dataclass
class DecisaoFinal:
    """Decisão final do reconhecedor adaptativo"""
    texto_final: str
    confianca_final: float
    metodo_vencedor: MetodoOCR
    status_parada: StatusDecisao
    candidatos_testados: List[ResultadoOCR]
    validacoes_realizadas: Dict[str, float]
    tempo_total: float
    iteracoes: int
    meta_atingida: bool

class ReconhecedorAdaptativo:
    """
    🔤 HEURÍSTICA 2: RECONHECEDOR ADAPTATIVO
    
    Responsabilidades:
    - Ensemble dinâmico de métodos OCR
    - Validação cruzada inteligente
    - Decisão de parada com meta de 98%
    - Aprendizado contínuo de eficácia
    """
    
    def __init__(self, meta_confianca: float = 0.98):
        self.meta_confianca = meta_confianca
        self.historico_resultados = []
        
        # Configuração de métodos com pesos iniciais baseados em performance histórica
        self.configuracao_metodos = {
            MetodoOCR.TREINAMENTO_MANUAL: {
                'peso': 0.98,           # Prioridade máxima
                'confianca_minima': 0.85,
                'timeout': 2.0,
                'habilitado': True
            },
            MetodoOCR.EASYOCR_OTIMIZADO: {
                'peso': 0.85,
                'confianca_minima': 0.70,
                'timeout': 5.0,
                'habilitado': True
            },
            MetodoOCR.EASYOCR_ALTERNATIVO: {
                'peso': 0.80,
                'confianca_minima': 0.65,
                'timeout': 5.0,
                'habilitado': True
            },
            MetodoOCR.TESSERACT_CUSTOM: {
                'peso': 0.78,
                'confianca_minima': 0.60,
                'timeout': 3.0,
                'habilitado': True
            },
            MetodoOCR.TESSERACT_ALTERNATIVO: {
                'peso': 0.75,
                'confianca_minima': 0.55,
                'timeout': 3.0,
                'habilitado': True
            },
            MetodoOCR.CNN_PROPRIO: {
                'peso': 0.72,
                'confianca_minima': 0.50,
                'timeout': 4.0,
                'habilitado': True
            },
            MetodoOCR.PADDLEOCR: {
                'peso': 0.70,
                'confianca_minima': 0.45,
                'timeout': 6.0,
                'habilitado': False  # Será habilitado quando implementado
            }
        }
        
        # Limites de execução
        self.max_tentativas = 10
        self.timeout_total = 15.0  # 15 segundos máximo
        
        # Carregar conhecimento prévio
        self.carregar_estatisticas_metodos()
        
        logger.info(f"🔤 Reconhecedor Adaptativo inicializado (meta: {meta_confianca:.1%})")
    
    async def reconhecer_com_meta(self, img_processada: np.ndarray, 
                                 metodos_ocr: Dict[str, Callable],
                                 contexto_imagem: Dict = None) -> DecisaoFinal:
        """
        RECONHECIMENTO PRINCIPAL: Executa ensemble até atingir meta de confiança
        """
        inicio = time.time()
        logger.info(f"🎯 Iniciando reconhecimento com meta de {self.meta_confianca:.1%}...")
        
        candidatos_testados = []
        validacoes = {}
        iteracao = 0
        
        # Ordenar métodos por prioridade (peso * sucesso histórico)
        metodos_ordenados = self._ordenar_metodos_por_prioridade()
        
        # LOOP PRINCIPAL: Testar métodos até atingir meta
        for metodo in metodos_ordenados:
            iteracao += 1
            tempo_atual = time.time() - inicio
            
            # Verificar timeout geral
            if tempo_atual >= self.timeout_total:
                logger.warning(f"⏰ Timeout geral atingido ({self.timeout_total}s)")
                break
            
            # Verificar limite de tentativas
            if iteracao > self.max_tentativas:
                logger.warning(f"🔄 Limite de tentativas atingido ({self.max_tentativas})")
                break
            
            config = self.configuracao_metodos[metodo]
            if not config['habilitado']:
                continue
            
            logger.info(f"🔍 Tentativa {iteracao}: {metodo.value} (peso: {config['peso']:.2f})")
            
            # Executar método específico
            resultado = await self._executar_metodo_ocr(
                metodo, img_processada, metodos_ocr, config
            )
            
            if resultado and resultado.texto:
                candidatos_testados.append(resultado)
                
                logger.info(f"📝 '{resultado.texto}' (conf: {resultado.confianca_ajustada:.2f}) via {metodo.value}")
                
                # Verificar se atingiu a meta
                if resultado.confianca_ajustada >= self.meta_confianca:
                    logger.info(f"🎉 META ATINGIDA! {resultado.confianca_ajustada:.1%} >= {self.meta_confianca:.1%}")
                    
                    # Validação final para confirmar
                    validacao_final = self._validar_resultado_final(resultado, img_processada, candidatos_testados)
                    validacoes['validacao_final'] = validacao_final
                    
                    if validacao_final >= 0.95:  # 95% de validação
                        decisao = DecisaoFinal(
                            texto_final=resultado.texto,
                            confianca_final=resultado.confianca_ajustada,
                            metodo_vencedor=metodo,
                            status_parada=StatusDecisao.META_ATINGIDA,
                            candidatos_testados=candidatos_testados,
                            validacoes_realizadas=validacoes,
                            tempo_total=time.time() - inicio,
                            iteracoes=iteracao,
                            meta_atingida=True
                        )
                        
                        # Atualizar estatísticas de sucesso
                        self._atualizar_estatisticas_sucesso(metodo, resultado.confianca_ajustada)
                        
                        logger.info(f"✅ Reconhecimento concluído: '{resultado.texto}' em {iteracao} tentativas")
                        return decisao
        
        # Meta não atingida - usar melhor candidato disponível
        logger.warning(f"⚠️ Meta {self.meta_confianca:.1%} não atingida em {iteracao} tentativas")
        
        if candidatos_testados:
            # Análise de ensemble para decidir melhor resultado
            decisao_ensemble = self._decidir_por_ensemble(candidatos_testados, validacoes, inicio, iteracao)
            return decisao_ensemble
        else:
            # Nenhum resultado válido
            return DecisaoFinal(
                texto_final="",
                confianca_final=0.0,
                metodo_vencedor=MetodoOCR.ENSEMBLE_VOTING,
                status_parada=StatusDecisao.LIMITE_TENTATIVAS,
                candidatos_testados=[],
                validacoes_realizadas={},
                tempo_total=time.time() - inicio,
                iteracoes=iteracao,
                meta_atingida=False
            )
    
    def _ordenar_metodos_por_prioridade(self) -> List[MetodoOCR]:
        """Ordena métodos por prioridade (peso + histórico de sucesso)"""
        metodos_com_score = []
        
        for metodo, config in self.configuracao_metodos.items():
            if not config['habilitado']:
                continue
            
            # Score = peso base + boost por sucesso histórico
            sucesso_historico = self._obter_taxa_sucesso_metodo(metodo)
            score_final = config['peso'] + (sucesso_historico * 0.1)  # Boost de até 10%
            
            metodos_com_score.append((metodo, score_final))
        
        # Ordenar por score (maior primeiro)
        metodos_com_score.sort(key=lambda x: x[1], reverse=True)
        
        return [metodo for metodo, score in metodos_com_score]
    
    async def _executar_metodo_ocr(self, metodo: MetodoOCR, img: np.ndarray, 
                                  metodos_ocr: Dict[str, Callable], 
                                  config: Dict) -> Optional[ResultadoOCR]:
        """Executa um método específico de OCR com timeout"""
        inicio_metodo = time.time()
        
        try:
            # Executar com timeout
            resultado_texto = None
            confianca_bruta = 0.0
            metadados = {}
            
            if metodo == MetodoOCR.DETECTOR_VALOR_ZERO:
                # NOVO: Detector específico para caracteres valor 0
                from detector_caracteres_zero import DetectorCaracteresZero
                detector = DetectorCaracteresZero()
                
                resultado_completo = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    detector.processar_captcha_completo,
                    img, 
                    True  # melhorar_contraste_automatico
                )
                
                if resultado_completo['sucesso']:
                    resultado_texto = resultado_completo['texto_detectado']
                    confianca_bruta = resultado_completo['confianca']
                    metadados = {
                        'metodo_ocr_usado': resultado_completo['metodo_usado'],
                        'contraste_melhorado': resultado_completo['contraste_melhorado'],
                        'pixels_valor_zero': resultado_completo['estatisticas_pixels']['percentual_zero'],
                        'tesseract_texto': resultado_completo['detalhes_ocr']['tesseract_case_sensitive'],
                        'tesseract_confianca': resultado_completo['detalhes_ocr']['tesseract_confianca'],
                        'easyocr_texto': resultado_completo['detalhes_ocr']['easyocr_case_sensitive'],
                        'easyocr_confianca': resultado_completo['detalhes_ocr']['easyocr_confianca']
                    }
                    logger.info(f"🎯 Detector valor 0: '{resultado_texto}' (confiança: {confianca_bruta:.2f}, método: {resultado_completo['metodo_usado']})")
                
            elif metodo == MetodoOCR.TREINAMENTO_MANUAL:
                if 'treinamento_manual' in metodos_ocr:
                    resultado_texto = await asyncio.wait_for(
                        metodos_ocr['treinamento_manual'](img),
                        timeout=config['timeout']
                    )
                    confianca_bruta = 0.95  # Alta confiança para treinamento manual
                    
            elif metodo == MetodoOCR.EASYOCR_OTIMIZADO:
                if 'easyocr_otimizado' in metodos_ocr:
                    resultado = await asyncio.wait_for(
                        metodos_ocr['easyocr_otimizado'](img),
                        timeout=config['timeout']
                    )
                    if isinstance(resultado, tuple):
                        resultado_texto, confianca_bruta = resultado
                    else:
                        resultado_texto = resultado
                        confianca_bruta = 0.7
                        
            elif metodo == MetodoOCR.TESSERACT_CUSTOM:
                if 'tesseract_custom' in metodos_ocr:
                    resultado_texto = await asyncio.wait_for(
                        metodos_ocr['tesseract_custom'](img),
                        timeout=config['timeout']
                    )
                    confianca_bruta = 0.65
            
            # Adicionar outros métodos conforme necessário...
            
            if not resultado_texto:
                return None
            
            # Limpar e validar texto
            texto_limpo = self._limpar_texto_ocr(resultado_texto)
            if not self._validar_texto_basico(texto_limpo):
                return None
            
            # Calcular confiança ajustada
            confianca_ajustada = self._calcular_confianca_ajustada(
                texto_limpo, confianca_bruta, metodo, img
            )
            
            tempo_execucao = time.time() - inicio_metodo
            
            return ResultadoOCR(
                metodo=metodo,
                texto=texto_limpo,
                confianca_bruta=confianca_bruta,
                confianca_ajustada=confianca_ajustada,
                tempo_execucao=tempo_execucao,
                metadados=metadados,
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            )
            
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Timeout no método {metodo.value} ({config['timeout']}s)")
            return None
        except Exception as e:
            logger.error(f"❌ Erro no método {metodo.value}: {e}")
            return None
    
    def _limpar_texto_ocr(self, texto: str) -> str:
        """Limpa e normaliza texto resultado do OCR"""
        if not texto:
            return ""
        
        # Remover caracteres não alfanuméricos
        texto_limpo = re.sub(r'[^A-Za-z0-9]', '', texto.strip())
        
        # Correções comuns de OCR
        correcoes = {
            '0': ['O', 'D', 'Q'],
            '1': ['I', 'l', '|'],
            '2': ['Z'],
            '5': ['S'],
            '6': ['G'], 
            '8': ['B'],
            'O': ['0'],
            'I': ['1'],
            'S': ['5'],
            'G': ['6'],
            'B': ['8'],
            'Z': ['2']
        }
        
        # Por enquanto, apenas retornar texto limpo
        # Correções automáticas podem ser implementadas posteriormente
        
        return texto_limpo
    
    def _validar_texto_basico(self, texto: str) -> bool:
        """Validação básica do texto OCR"""
        if not texto:
            return False
        
        # Comprimento típico de CAPTCHA
        if len(texto) < 3 or len(texto) > 8:
            return False
        
        # Deve ser alfanumérico
        if not texto.isalnum():
            return False
        
        return True
    
    def _calcular_confianca_ajustada(self, texto: str, confianca_bruta: float, 
                                   metodo: MetodoOCR, img: np.ndarray) -> float:
        """Calcula confiança ajustada baseada em múltiplos fatores"""
        
        # Começar com confiança bruta
        confianca = confianca_bruta
        
        # Ajuste por peso do método
        peso_metodo = self.configuracao_metodos[metodo]['peso']
        confianca *= peso_metodo
        
        # Boost para treinamento manual
        if metodo == MetodoOCR.TREINAMENTO_MANUAL:
            confianca = min(confianca * 1.15, 0.99)  # 15% boost, máximo 99%
        
        # Ajuste por comprimento do texto (CAPTCHAs típicos: 4-6 chars)
        if 4 <= len(texto) <= 6:
            confianca *= 1.05  # 5% boost
        elif len(texto) == 3 or len(texto) == 7:
            confianca *= 0.95  # 5% penalidade
        else:
            confianca *= 0.85  # 15% penalidade
        
        # Ajuste por padrões conhecidos
        if self._texto_segue_padroes_conhecidos(texto):
            confianca *= 1.08  # 8% boost
        
        # Ajuste por histórico de sucesso do método
        taxa_sucesso = self._obter_taxa_sucesso_metodo(metodo)
        if taxa_sucesso > 0.8:
            confianca *= 1.03  # 3% boost para métodos com bom histórico
        elif taxa_sucesso < 0.5:
            confianca *= 0.95  # 5% penalidade para métodos com histórico ruim
        
        return min(confianca, 0.999)  # Máximo 99.9%
    
    def _texto_segue_padroes_conhecidos(self, texto: str) -> bool:
        """Verifica se o texto segue padrões conhecidos de CAPTCHAs"""
        
        # Padrões típicos observados
        padroes_comuns = [
            r'^[A-Z]{2}[0-9]{2}[A-Z]{2}$',    # Ex: AB12CD
            r'^[A-Z][0-9][A-Z][0-9][A-Z][0-9]$',  # Ex: A1B2C3
            r'^[A-Za-z]{4,6}$',               # Ex: AbCdEf
            r'^[0-9]{4,6}$',                  # Ex: 123456
            r'^[A-Za-z0-9]{5}$',              # Ex: Ab1C2
        ]
        
        for padrao in padroes_comuns:
            if re.match(padrao, texto):
                return True
        
        return False
    
    def _validar_resultado_final(self, resultado: ResultadoOCR, img: np.ndarray, 
                               outros_candidatos: List[ResultadoOCR]) -> float:
        """Validação cruzada final do resultado"""
        
        score_validacao = 0.8  # Base
        
        # 1. Consistência com outros candidatos
        textos_similares = 0
        for candidato in outros_candidatos:
            if candidato.texto == resultado.texto:
                textos_similares += 1
            elif self._calcular_similaridade_texto(resultado.texto, candidato.texto) > 0.8:
                textos_similares += 0.5
        
        if textos_similares >= 2:
            score_validacao += 0.15  # Consenso forte
        elif textos_similares >= 1:
            score_validacao += 0.08  # Consenso moderado
        
        # 2. Validação contra características da imagem
        if self._validar_contra_imagem(resultado.texto, img):
            score_validacao += 0.05
        
        return min(score_validacao, 1.0)
    
    def _calcular_similaridade_texto(self, texto1: str, texto2: str) -> float:
        """Calcula similaridade entre dois textos (Levenshtein normalizado)"""
        if not texto1 or not texto2:
            return 0.0
        
        if texto1 == texto2:
            return 1.0
        
        # Implementação simples de distância de Levenshtein normalizada
        len_max = max(len(texto1), len(texto2))
        if len_max == 0:
            return 1.0
        
        # Distância simples por caracteres diferentes
        diferencas = sum(c1 != c2 for c1, c2 in zip(texto1, texto2))
        diferencas += abs(len(texto1) - len(texto2))
        
        similaridade = 1.0 - (diferencas / len_max)
        return max(0.0, similaridade)
    
    def _validar_contra_imagem(self, texto: str, img: np.ndarray) -> bool:
        """Validação básica do texto contra características da imagem"""
        
        # Análise simples: número de caracteres vs componentes na imagem
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Threshold para análise binária
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Contar componentes conectados
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Filtrar componentes por tamanho
        h, w = gray.shape
        componentes_validos = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > (h * w) * 0.001:  # Pelo menos 0.1% da imagem
                componentes_validos += 1
        
        # Verificar se número de caracteres é compatível com componentes
        diferenca = abs(len(texto) - componentes_validos)
        return diferenca <= 2  # Tolerância de 2 componentes
    
    def _decidir_por_ensemble(self, candidatos: List[ResultadoOCR], validacoes: Dict, 
                            inicio: float, iteracoes: int) -> DecisaoFinal:
        """Decide resultado final por ensemble quando meta não é atingida"""
        
        if not candidatos:
            return DecisaoFinal(
                texto_final="",
                confianca_final=0.0,
                metodo_vencedor=MetodoOCR.ENSEMBLE_VOTING,
                status_parada=StatusDecisao.LIMITE_TENTATIVAS,
                candidatos_testados=candidatos,
                validacoes_realizadas=validacoes,
                tempo_total=time.time() - inicio,
                iteracoes=iteracoes,
                meta_atingida=False
            )
        
        # Ordenar candidatos por confiança ajustada
        candidatos_ordenados = sorted(candidatos, key=lambda x: x.confianca_ajustada, reverse=True)
        melhor_candidato = candidatos_ordenados[0]
        
        # Verificar consenso
        consenso_score = self._calcular_consenso(candidatos)
        validacoes['consenso'] = consenso_score
        
        # Decidir status de parada
        if consenso_score >= 0.8:
            status = StatusDecisao.CONSENSO_MULTIPLO
        elif melhor_candidato.confianca_ajustada >= 0.90:
            status = StatusDecisao.CONFIANCA_ALTA
        else:
            status = StatusDecisao.LIMITE_TENTATIVAS
        
        return DecisaoFinal(
            texto_final=melhor_candidato.texto,
            confianca_final=melhor_candidato.confianca_ajustada,
            metodo_vencedor=melhor_candidato.metodo,
            status_parada=status,
            candidatos_testados=candidatos,
            validacoes_realizadas=validacoes,
            tempo_total=time.time() - inicio,
            iteracoes=iteracoes,
            meta_atingida=False
        )
    
    def _calcular_consenso(self, candidatos: List[ResultadoOCR]) -> float:
        """Calcula score de consenso entre candidatos"""
        if len(candidatos) < 2:
            return 0.0
        
        # Contar ocorrências de cada texto
        contagem_textos = {}
        for candidato in candidatos:
            texto = candidato.texto
            if texto not in contagem_textos:
                contagem_textos[texto] = []
            contagem_textos[texto].append(candidato.confianca_ajustada)
        
        # Encontrar texto com maior consenso ponderado
        melhor_consenso = 0.0
        for texto, confiancas in contagem_textos.items():
            # Score = número de ocorrências + média de confiança
            score_consenso = (len(confiancas) / len(candidatos)) + (np.mean(confiancas) * 0.5)
            melhor_consenso = max(melhor_consenso, score_consenso)
        
        return min(melhor_consenso, 1.0)
    
    def _obter_taxa_sucesso_metodo(self, metodo: MetodoOCR) -> float:
        """Obtém taxa de sucesso histórica do método"""
        # Por enquanto, retornar valores baseados na configuração
        # Posteriormente será implementado histórico real
        pesos_base = {
            MetodoOCR.TREINAMENTO_MANUAL: 0.95,
            MetodoOCR.EASYOCR_OTIMIZADO: 0.80,
            MetodoOCR.EASYOCR_ALTERNATIVO: 0.75,
            MetodoOCR.TESSERACT_CUSTOM: 0.70,
            MetodoOCR.TESSERACT_ALTERNATIVO: 0.65,
            MetodoOCR.CNN_PROPRIO: 0.60,
            MetodoOCR.PADDLEOCR: 0.55
        }
        
        return pesos_base.get(metodo, 0.50)
    
    def _atualizar_estatisticas_sucesso(self, metodo: MetodoOCR, confianca: float):
        """Atualiza estatísticas de sucesso do método"""
        # Implementar sistema de aprendizado contínuo
        logger.info(f"📊 Atualizando estatísticas: {metodo.value} com confiança {confianca:.2f}")
    
    def carregar_estatisticas_metodos(self):
        """Carrega estatísticas históricas dos métodos"""
        try:
            stats_path = Path("heuristica_ocr_stats.json")
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                    # Atualizar configurações baseado no histórico
                logger.info("📚 Estatísticas de métodos carregadas")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar estatísticas: {e}")
    
    def gerar_relatorio_decisao(self, decisao: DecisaoFinal) -> str:
        """Gera relatório detalhado da decisão final"""
        status_emoji = "🎉" if decisao.meta_atingida else "⚠️"
        
        relatorio = f"""
{status_emoji} RELATÓRIO DE RECONHECIMENTO ADAPTATIVO
=============================================
Resultado: '{decisao.texto_final}'
Confiança: {decisao.confianca_final:.1%}
Método vencedor: {decisao.metodo_vencedor.value}
Status: {decisao.status_parada.value}
Meta atingida: {'SIM' if decisao.meta_atingida else 'NÃO'}

⏱️ PERFORMANCE:
   Tempo total: {decisao.tempo_total:.2f}s
   Iterações: {decisao.iteracoes}
   Candidatos testados: {len(decisao.candidatos_testados)}

📊 CANDIDATOS TESTADOS:
"""
        for i, candidato in enumerate(decisao.candidatos_testados, 1):
            relatorio += f"   {i}. '{candidato.texto}' ({candidato.confianca_ajustada:.2f}) via {candidato.metodo.value}\n"
        
        relatorio += f"\n✅ VALIDAÇÕES:\n"
        for validacao, score in decisao.validacoes_realizadas.items():
            relatorio += f"   {validacao}: {score:.2f}\n"
        
        return relatorio

if __name__ == "__main__":
    # Teste do reconhecedor
    logging.basicConfig(level=logging.INFO, format='🔤 [%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    
    print("🔤 Testando Reconhecedor Adaptativo...")
    reconhecedor = ReconhecedorAdaptativo(meta_confianca=0.98)
    
    # Simular métodos OCR
    async def mock_easyocr(img):
        await asyncio.sleep(0.1)
        return "ABC123", 0.85
    
    metodos_mock = {
        'easyocr_otimizado': mock_easyocr
    }
    
    # Criar imagem de teste
    img_teste = np.random.randint(0, 255, (40, 120), dtype=np.uint8)
    
    # Executar teste
    async def teste():
        resultado = await reconhecedor.reconhecer_com_meta(img_teste, metodos_mock)
        print(reconhecedor.gerar_relatorio_decisao(resultado))
    
    asyncio.run(teste())
