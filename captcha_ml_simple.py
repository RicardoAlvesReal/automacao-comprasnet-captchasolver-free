#!/usr/bin/env python3
"""
Sistema ML Simples para Análise de CAPTCHA
Implementação básica sem dependências pesadas
"""

import cv2
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import json
import re

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleCaptchaMLAnalyzer:
    """Analisador ML simples para CAPTCHA sem dependências pesadas"""
    
    def __init__(self):
        self.patterns_database = {}
        self.confidence_threshold = 0.5
        self.learning_data = []
        
        # Criar diretórios necessários
        os.makedirs("models", exist_ok=True)
        os.makedirs("captchas_limpos", exist_ok=True)
        
        # Tentar carregar dados de aprendizado existentes
        self.load_learning_data()
        
    def load_learning_data(self):
        """Carrega dados de aprendizado salvos"""
        try:
            if os.path.exists("models/simple_ml_data.json"):
                with open("models/simple_ml_data.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.patterns_database = data.get("patterns", {})
                    self.learning_data = data.get("learning_data", [])
                logger.info(f"✅ Dados ML carregados: {len(self.patterns_database)} padrões")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar dados ML: {e}")
            
    def save_learning_data(self):
        """Salva dados de aprendizado"""
        try:
            data = {
                "patterns": self.patterns_database,
                "learning_data": self.learning_data,
                "timestamp": datetime.now().isoformat()
            }
            with open("models/simple_ml_data.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("✅ Dados ML salvos")
        except Exception as e:
            logger.error(f"❌ Erro ao salvar dados ML: {e}")
    
    def extract_features(self, img: np.ndarray) -> Dict[str, float]:
        """Extrai características básicas da imagem"""
        try:
            # Converter para escala de cinza se necessário
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Características básicas
            height, width = gray.shape
            
            # Densidade de pixels escuros
            dark_pixels = np.sum(gray < 128)
            density = dark_pixels / (height * width)
            
            # Variância (complexidade)
            variance = np.var(gray)
            
            # Entropia aproximada
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist / np.sum(hist)  # Normalizar
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            # Detecção de bordas (complexidade de contornos)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            
            # Aspecto da imagem
            aspect_ratio = width / height
            
            return {
                "density": density,
                "variance": variance / 1000.0,  # Normalizar
                "entropy": entropy / 8.0,  # Normalizar
                "edge_density": edge_density,
                "aspect_ratio": aspect_ratio,
                "width": width / 200.0,  # Normalizar assumindo ~200px
                "height": height / 100.0   # Normalizar assumindo ~100px
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao extrair características: {e}")
            return {}
    
    def predict_difficulty(self, img: np.ndarray) -> Dict[str, Any]:
        """Prediz dificuldade do CAPTCHA"""
        try:
            features = self.extract_features(img)
            if not features:
                return {"difficulty": "medium", "confidence": 0.5}
            
            # Análise heurística simples baseada em características
            difficulty_score = 0.0
            
            # Densidade alta = mais difícil
            if features.get("density", 0) > 0.3:
                difficulty_score += 0.3
            
            # Variância alta = mais complexo
            if features.get("variance", 0) > 0.5:
                difficulty_score += 0.2
            
            # Entropia alta = mais aleatório
            if features.get("entropy", 0) > 0.7:
                difficulty_score += 0.3
            
            # Densidade de bordas alta = mais detalhado
            if features.get("edge_density", 0) > 0.2:
                difficulty_score += 0.2
            
            # Classificar dificuldade
            if difficulty_score < 0.3:
                difficulty = "easy"
            elif difficulty_score < 0.7:
                difficulty = "medium"
            else:
                difficulty = "hard"
            
            confidence = min(0.9, 0.5 + difficulty_score)
            
            return {
                "difficulty": difficulty,
                "confidence": confidence,
                "features": features,
                "score": difficulty_score
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na predição: {e}")
            return {"difficulty": "medium", "confidence": 0.5}
    
    def suggest_strategy(self, img: np.ndarray) -> Dict[str, Any]:
        """Sugere estratégia de processamento baseada na análise"""
        try:
            analysis = self.predict_difficulty(img)
            difficulty = analysis.get("difficulty", "medium")
            
            strategies = {
                "easy": {
                    "primary": "easyocr_original",
                    "fallback": ["tesseract_psm7", "easyocr_clahe"],
                    "preprocessing": "minimal",
                    "confidence_threshold": 0.6
                },
                "medium": {
                    "primary": "easyocr_clahe",
                    "fallback": ["easyocr_morfologia", "tesseract_psm6", "easyocr_median"],
                    "preprocessing": "standard",
                    "confidence_threshold": 0.7
                },
                "hard": {
                    "primary": "easyocr_morfologia",
                    "fallback": ["easyocr_invertida", "opencv_advanced", "tesseract_psm13"],
                    "preprocessing": "aggressive",
                    "confidence_threshold": 0.8
                }
            }
            
            strategy = strategies.get(difficulty, strategies["medium"])
            strategy["analysis"] = analysis
            
            return strategy
            
        except Exception as e:
            logger.error(f"❌ Erro ao sugerir estratégia: {e}")
            return {
                "primary": "easyocr_clahe",
                "fallback": ["easyocr_original", "tesseract_psm7"],
                "preprocessing": "standard",
                "confidence_threshold": 0.7
            }
    
    def learn_from_success(self, img: np.ndarray, text_result: str, method: str, confidence: float):
        """Aprende com resultado bem-sucedido"""
        try:
            features = self.extract_features(img)
            if not features:
                return
            
            learning_entry = {
                "text": text_result,
                "method": method,
                "confidence": confidence,
                "features": features,
                "timestamp": datetime.now().isoformat(),
                "chars": len(text_result)
            }
            
            self.learning_data.append(learning_entry)
            
            # Manter apenas os últimos 1000 registros
            if len(self.learning_data) > 1000:
                self.learning_data = self.learning_data[-1000:]
            
            # Atualizar padrões
            pattern_key = f"{method}_{len(text_result)}chars"
            if pattern_key not in self.patterns_database:
                self.patterns_database[pattern_key] = {
                    "count": 0,
                    "avg_confidence": 0.0,
                    "avg_features": {}
                }
            
            pattern = self.patterns_database[pattern_key]
            pattern["count"] += 1
            
            # Atualizar média de confiança
            pattern["avg_confidence"] = (
                (pattern["avg_confidence"] * (pattern["count"] - 1) + confidence) / pattern["count"]
            )
            
            logger.info(f"📚 ML aprendeu: {method} -> '{text_result}' (conf: {confidence:.2f})")
            
            # Salvar periodicamente
            if pattern["count"] % 10 == 0:
                self.save_learning_data()
                
        except Exception as e:
            logger.error(f"❌ Erro no aprendizado: {e}")
    
    def get_method_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas dos métodos"""
        try:
            stats = {}
            
            for pattern_key, data in self.patterns_database.items():
                method = pattern_key.split("_")[0] if "_" in pattern_key else pattern_key
                
                if method not in stats:
                    stats[method] = {
                        "total_count": 0,
                        "avg_confidence": 0.0,
                        "char_distributions": {}
                    }
                
                stats[method]["total_count"] += data["count"]
                stats[method]["avg_confidence"] = (
                    (stats[method]["avg_confidence"] + data["avg_confidence"]) / 2
                )
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter estatísticas: {e}")
            return {}
    
    def analyze_image(self, img: np.ndarray) -> Dict[str, Any]:
        """Análise completa da imagem"""
        try:
            difficulty_analysis = self.predict_difficulty(img)
            strategy = self.suggest_strategy(img)
            
            return {
                "difficulty": difficulty_analysis,
                "recommended_strategy": strategy,
                "ml_confidence": difficulty_analysis.get("confidence", 0.5),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na análise completa: {e}")
            return {
                "difficulty": {"difficulty": "medium", "confidence": 0.5},
                "recommended_strategy": {
                    "primary": "easyocr_clahe",
                    "fallback": ["easyocr_original"],
                    "confidence_threshold": 0.7
                },
                "ml_confidence": 0.5
            }

# Função de conveniência para uso externo
def create_simple_ml_analyzer() -> SimpleCaptchaMLAnalyzer:
    """Cria uma instância do analisador ML simples"""
    return SimpleCaptchaMLAnalyzer()

if __name__ == "__main__":
    # Teste básico
    analyzer = create_simple_ml_analyzer()
    
    # Criar uma imagem de teste
    test_img = np.random.randint(0, 255, (90, 200, 3), dtype=np.uint8)
    
    analysis = analyzer.analyze_image(test_img)
    print("🧪 Teste do ML Simples:")
    print(f"   Dificuldade: {analysis['difficulty']['difficulty']}")
    print(f"   Confiança: {analysis['ml_confidence']:.2f}")
    print(f"   Estratégia recomendada: {analysis['recommended_strategy']['primary']}")
