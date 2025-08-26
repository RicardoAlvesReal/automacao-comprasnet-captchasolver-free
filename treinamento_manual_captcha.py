#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎓 SISTEMA DE TREINAMENTO MANUAL DE CAPTCHAS

Este sistema permite ensinar o código a resolver CAPTCHAs manualmente,
usando as imagens que você já possui. O sistema irá:

1. Mostrar cada CAPTCHA não resolvido
2. Permitir que você digite a resposta correta
3. Salvar o treinamento no banco de dados
4. Atualizar os modelos automaticamente
5. Melhorar a precisão do sistema progressivamente

COMO USAR:
- Execute o script
- Para cada CAPTCHA mostrado, digite a resposta correta
- Digite 'pular' para pular um CAPTCHA difícil
- Digite 'sair' para finalizar o treinamento
- Digite 'auto' para tentar resolver automaticamente primeiro
"""

import os
import sys
import cv2
import json
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from comprasnet_download_monitor_open import ComprasNetDownloadMonitor

class TreinadorManualCaptcha:
    def __init__(self):
        self.monitor = ComprasNetDownloadMonitor()
        self.db_path = "captcha_treinamento.db"
        self.captcha_dir = Path("captcha_estudos")
        self.contador_treinados = 0
        self.contador_pulados = 0
        
        # Criar banco de dados se não existir
        self.criar_banco_dados()
        
        print("🎓 SISTEMA DE TREINAMENTO MANUAL DE CAPTCHAS")
        print("=" * 60)
        print("📚 Sistema inicializado com sucesso!")
    
    def criar_banco_dados(self):
        """Cria o banco de dados para armazenar o treinamento"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS captcha_treinamento (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome_arquivo TEXT UNIQUE NOT NULL,
                resposta_correta TEXT NOT NULL,
                data_treinamento TEXT NOT NULL,
                confianca REAL DEFAULT 1.0,
                observacoes TEXT,
                largura INTEGER,
                altura INTEGER,
                dificuldade TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historico_treinamento (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sessao_id TEXT NOT NULL,
                total_treinados INTEGER,
                total_pulados INTEGER,
                data_sessao TEXT NOT NULL,
                observacoes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print("🗄️ Banco de dados de treinamento preparado")
    
    def listar_captchas_nao_resolvidos(self):
        """Lista todos os CAPTCHAs que ainda não foram treinados"""
        if not self.captcha_dir.exists():
            print(f"❌ Diretório {self.captcha_dir} não encontrado!")
            return []
        
        # Buscar todos os arquivos de imagem
        extensoes = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        todos_captchas = []
        
        for ext in extensoes:
            todos_captchas.extend(self.captcha_dir.glob(f"*{ext}"))
            todos_captchas.extend(self.captcha_dir.glob(f"*{ext.upper()}"))
        
        # Verificar quais já foram treinados
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT nome_arquivo FROM captcha_treinamento")
        treinados = {row[0] for row in cursor.fetchall()}
        conn.close()
        
        # Filtrar apenas os não treinados
        nao_resolvidos = [img for img in todos_captchas if img.name not in treinados]
        
        print(f"📊 Total de CAPTCHAs encontrados: {len(todos_captchas)}")
        print(f"✅ Já treinados: {len(treinados)}")
        print(f"⏳ Pendentes de treinamento: {len(nao_resolvidos)}")
        
        return sorted(nao_resolvidos)
    
    def mostrar_captcha(self, caminho_imagem):
        """Mostra o CAPTCHA na tela para o usuário visualizar"""
        try:
            # Carregar e mostrar a imagem
            img = cv2.imread(str(caminho_imagem))
            if img is None:
                print(f"❌ Erro ao carregar: {caminho_imagem}")
                return False
            
            # Redimensionar se muito grande
            h, w = img.shape[:2]
            if w > 800 or h > 600:
                scale = min(800/w, 600/h)
                new_w, new_h = int(w*scale), int(h*scale)
                img = cv2.resize(img, (new_w, new_h))
            
            # Mostrar a imagem
            cv2.imshow('CAPTCHA para Treinamento', img)
            cv2.moveWindow('CAPTCHA para Treinamento', 100, 100)
            
            print(f"🖼️ Imagem mostrada: {caminho_imagem.name}")
            print(f"📐 Dimensões: {w}x{h}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao mostrar imagem: {e}")
            return False
    
    def tentar_resolucao_automatica(self, caminho_imagem):
        """Tenta resolver o CAPTCHA automaticamente primeiro"""
        try:
            print("🤖 Tentando resolução automática...")
            
            # Carregar imagem
            img = cv2.imread(str(caminho_imagem))
            _, buffer = cv2.imencode('.png', img)
            captcha_buffer = buffer.tobytes()
            
            # Tentar diferentes métodos
            resultados = {}
            
            # Método 1: OpenCV
            try:
                resultado_opencv = self.monitor.processar_captcha_opencv(captcha_buffer)
                if resultado_opencv and len(str(resultado_opencv)) > 2:
                    resultados['OpenCV'] = str(resultado_opencv)
            except:
                pass
            
            # Método 2: EasyOCR
            try:
                resultado_easyocr = self.monitor.resolver_captcha_easyocr(captcha_buffer)
                if resultado_easyocr and len(resultado_easyocr) > 2:
                    resultados['EasyOCR'] = resultado_easyocr
            except:
                pass
            
            # Método 3: Pré-processamento + OCR melhorado
            try:
                img_preprocessada = self.monitor.preprocessamento_adaptativo_avancado(img)
                resultado_melhorado = self.monitor.ocr_melhorado_caracteres(img_preprocessada)
                if resultado_melhorado:
                    resultado_final = self.monitor.analisar_e_corrigir_ocr(resultado_melhorado, img_preprocessada)
                    if resultado_final and len(resultado_final) > 2:
                        resultados['Melhorado'] = resultado_final
            except:
                pass
            
            if resultados:
                print("🎯 Sugestões automáticas:")
                for i, (metodo, resultado) in enumerate(resultados.items(), 1):
                    print(f"   {i}. {metodo}: '{resultado}'")
                return resultados
            else:
                print("⚠️ Nenhuma sugestão automática disponível")
                return {}
                
        except Exception as e:
            print(f"❌ Erro na resolução automática: {e}")
            return {}
    
    def classificar_dificuldade(self, img_path):
        """Classifica a dificuldade do CAPTCHA baseado em características visuais"""
        try:
            img = cv2.imread(str(img_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Métricas de dificuldade
            h, w = gray.shape
            contraste = np.ptp(gray)
            complexidade = cv2.Laplacian(gray, cv2.CV_64F).var()
            edges = cv2.Canny(gray, 50, 150)
            densidade_bordas = np.sum(edges > 0) / edges.size
            
            score_dificuldade = 0
            
            # Baixo contraste = mais difícil
            if contraste < 100:
                score_dificuldade += 2
            elif contraste < 150:
                score_dificuldade += 1
            
            # Muitas bordas = mais complexo
            if densidade_bordas > 0.15:
                score_dificuldade += 2
            elif densidade_bordas > 0.10:
                score_dificuldade += 1
            
            # Alta variação = mais ruído
            if complexidade > 1000:
                score_dificuldade += 2
            elif complexidade > 500:
                score_dificuldade += 1
            
            # Tamanho pequeno = mais difícil
            if h < 30 or w < 80:
                score_dificuldade += 1
            
            if score_dificuldade >= 5:
                return "muito_difícil"
            elif score_dificuldade >= 3:
                return "difícil"
            elif score_dificuldade >= 1:
                return "médio"
            else:
                return "fácil"
                
        except:
            return "desconhecido"
    
    def salvar_treinamento(self, nome_arquivo, resposta_correta, observacoes="", confianca=1.0):
        """Salva o treinamento no banco de dados"""
        try:
            # Obter informações da imagem
            img_path = self.captcha_dir / nome_arquivo
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            dificuldade = self.classificar_dificuldade(img_path)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO captcha_treinamento 
                (nome_arquivo, resposta_correta, data_treinamento, confianca, 
                 observacoes, largura, altura, dificuldade)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                nome_arquivo,
                resposta_correta,
                datetime.now().isoformat(),
                confianca,
                observacoes,
                w, h, dificuldade
            ))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Treinamento salvo: '{resposta_correta}' para {nome_arquivo}")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao salvar treinamento: {e}")
            return False
    
    def iniciar_treinamento_interativo(self):
        """Inicia o processo de treinamento interativo"""
        captchas_pendentes = self.listar_captchas_nao_resolvidos()
        
        if not captchas_pendentes:
            print("🎉 Todos os CAPTCHAs já foram treinados!")
            return
        
        print(f"\n🎓 Iniciando treinamento interativo...")
        print(f"📝 {len(captchas_pendentes)} CAPTCHAs para treinar")
        print("\n💡 COMANDOS:")
        print("   - Digite a resposta correta do CAPTCHA")
        print("   - 'pular' = pular este CAPTCHA")
        print("   - 'auto' = tentar resolução automática primeiro")
        print("   - 'sair' = finalizar treinamento")
        print("   - 'info' = mostrar informações do CAPTCHA")
        
        sessao_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, captcha_path in enumerate(captchas_pendentes, 1):
            print(f"\n" + "="*60)
            print(f"📸 CAPTCHA {i}/{len(captchas_pendentes)}: {captcha_path.name}")
            
            # Mostrar a imagem
            if not self.mostrar_captcha(captcha_path):
                continue
            
            while True:
                try:
                    resposta = input(f"\n🎯 Digite a resposta para '{captcha_path.name}': ").strip()
                    
                    if resposta.lower() == 'sair':
                        print("👋 Finalizando treinamento...")
                        cv2.destroyAllWindows()
                        self.salvar_historico_sessao(sessao_id)
                        return
                    
                    elif resposta.lower() == 'pular':
                        print("⏭️ Pulando este CAPTCHA...")
                        self.contador_pulados += 1
                        break
                    
                    elif resposta.lower() == 'auto':
                        sugestoes = self.tentar_resolucao_automatica(captcha_path)
                        if sugestoes:
                            print("💡 Escolha uma das sugestões digitando o número, ou digite a resposta correta:")
                            continue
                        else:
                            print("⚠️ Nenhuma sugestão automática. Digite a resposta manualmente:")
                            continue
                    
                    elif resposta.lower() == 'info':
                        img = cv2.imread(str(captcha_path))
                        h, w = img.shape[:2]
                        dificuldade = self.classificar_dificuldade(captcha_path)
                        print(f"📊 Informações do CAPTCHA:")
                        print(f"   📐 Dimensões: {w}x{h}")
                        print(f"   🎚️ Dificuldade: {dificuldade}")
                        print(f"   📁 Arquivo: {captcha_path}")
                        continue
                    
                    elif resposta.isdigit() and len(resposta) == 1:
                        # Usuário escolheu uma sugestão automática
                        try:
                            sugestoes = self.tentar_resolucao_automatica(captcha_path)
                            if sugestoes:
                                opcoes = list(sugestoes.values())
                                idx = int(resposta) - 1
                                if 0 <= idx < len(opcoes):
                                    resposta_escolhida = opcoes[idx]
                                    confirma = input(f"✅ Confirma '{resposta_escolhida}'? (s/n): ").strip().lower()
                                    if confirma in ['s', 'sim', 'y', 'yes']:
                                        resposta = resposta_escolhida
                                    else:
                                        continue
                                else:
                                    print("❌ Opção inválida!")
                                    continue
                            else:
                                print("❌ Nenhuma sugestão disponível!")
                                continue
                        except:
                            print("❌ Erro ao processar sugestão!")
                            continue
                    
                    # Validar resposta
                    if not resposta or len(resposta) < 2:
                        print("⚠️ Resposta muito curta! Digite pelo menos 2 caracteres.")
                        continue
                    
                    if len(resposta) > 10:
                        print("⚠️ Resposta muito longa! Máximo 10 caracteres.")
                        continue
                    
                    # Solicitar confirmação
                    print(f"✅ Resposta: '{resposta}'")
                    confirma = input("📝 Confirma esta resposta? (s/n): ").strip().lower()
                    
                    if confirma in ['s', 'sim', 'y', 'yes']:
                        # Salvar treinamento
                        if self.salvar_treinamento(captcha_path.name, resposta):
                            self.contador_treinados += 1
                            print(f"🎉 Treinamento salvo! Total treinados: {self.contador_treinados}")
                            break
                        else:
                            print("❌ Erro ao salvar. Tente novamente.")
                    else:
                        print("🔄 Digite a resposta novamente...")
                        
                except KeyboardInterrupt:
                    print("\n\n⚠️ Treinamento interrompido pelo usuário!")
                    cv2.destroyAllWindows()
                    self.salvar_historico_sessao(sessao_id)
                    return
                except Exception as e:
                    print(f"❌ Erro: {e}")
            
            # Fechar janela do CAPTCHA atual
            cv2.destroyAllWindows()
        
        # Finalizar sessão
        print(f"\n🎉 Treinamento concluído!")
        self.salvar_historico_sessao(sessao_id)
        self.mostrar_estatisticas_finais()
    
    def salvar_historico_sessao(self, sessao_id):
        """Salva o histórico da sessão de treinamento"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO historico_treinamento 
                (sessao_id, total_treinados, total_pulados, data_sessao, observacoes)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                sessao_id,
                self.contador_treinados,
                self.contador_pulados,
                datetime.now().isoformat(),
                f"Treinamento manual - {self.contador_treinados} treinados, {self.contador_pulados} pulados"
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Erro ao salvar histórico: {e}")
    
    def mostrar_estatisticas_finais(self):
        """Mostra estatísticas finais do treinamento"""
        print(f"\n📊 ESTATÍSTICAS DO TREINAMENTO:")
        print(f"✅ CAPTCHAs treinados: {self.contador_treinados}")
        print(f"⏭️ CAPTCHAs pulados: {self.contador_pulados}")
        
        # Buscar estatísticas do banco
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM captcha_treinamento")
            total_treinados = cursor.fetchone()[0]
            
            cursor.execute("SELECT dificuldade, COUNT(*) FROM captcha_treinamento GROUP BY dificuldade")
            por_dificuldade = cursor.fetchall()
            
            conn.close()
            
            print(f"🗄️ Total no banco: {total_treinados} CAPTCHAs treinados")
            
            if por_dificuldade:
                print(f"📈 Por dificuldade:")
                for dif, count in por_dificuldade:
                    print(f"   {dif}: {count}")
                    
        except Exception as e:
            print(f"⚠️ Erro ao buscar estatísticas: {e}")
    
    def exportar_treinamento(self, arquivo_saida="treinamento_captchas.json"):
        """Exporta todo o treinamento para um arquivo JSON"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT nome_arquivo, resposta_correta, data_treinamento, 
                       confianca, observacoes, largura, altura, dificuldade
                FROM captcha_treinamento 
                ORDER BY data_treinamento
            ''')
            
            treinamentos = []
            for row in cursor.fetchall():
                treinamentos.append({
                    'nome_arquivo': row[0],
                    'resposta_correta': row[1],
                    'data_treinamento': row[2],
                    'confianca': row[3],
                    'observacoes': row[4],
                    'largura': row[5],
                    'altura': row[6],
                    'dificuldade': row[7]
                })
            
            conn.close()
            
            with open(arquivo_saida, 'w', encoding='utf-8') as f:
                json.dump(treinamentos, f, indent=2, ensure_ascii=False)
            
            print(f"📤 Treinamento exportado para: {arquivo_saida}")
            print(f"📊 Total de registros: {len(treinamentos)}")
            
        except Exception as e:
            print(f"❌ Erro ao exportar: {e}")

def main():
    """Função principal do treinamento manual"""
    print("🎓 INICIANDO SISTEMA DE TREINAMENTO MANUAL")
    print("=" * 60)
    
    try:
        treinador = TreinadorManualCaptcha()
        
        while True:
            print(f"\n📋 MENU PRINCIPAL:")
            print("1. 🎓 Iniciar treinamento interativo")
            print("2. 📊 Mostrar estatísticas")
            print("3. 📤 Exportar treinamento")
            print("4. 📋 Listar CAPTCHAs pendentes")
            print("5. 🚪 Sair")
            
            escolha = input("\n👉 Escolha uma opção (1-5): ").strip()
            
            if escolha == '1':
                treinador.iniciar_treinamento_interativo()
            
            elif escolha == '2':
                treinador.mostrar_estatisticas_finais()
            
            elif escolha == '3':
                nome_arquivo = input("📁 Nome do arquivo (Enter para padrão): ").strip()
                if not nome_arquivo:
                    nome_arquivo = "treinamento_captchas.json"
                treinador.exportar_treinamento(nome_arquivo)
            
            elif escolha == '4':
                pendentes = treinador.listar_captchas_nao_resolvidos()
                if pendentes:
                    print(f"\n📋 CAPTCHAs pendentes ({len(pendentes)}):")
                    for i, captcha in enumerate(pendentes[:10], 1):
                        print(f"   {i}. {captcha.name}")
                    if len(pendentes) > 10:
                        print(f"   ... e mais {len(pendentes) - 10}")
                else:
                    print("✅ Todos os CAPTCHAs foram treinados!")
            
            elif escolha == '5':
                print("👋 Obrigado por usar o sistema de treinamento!")
                break
            
            else:
                print("❌ Opção inválida! Escolha entre 1-5.")
    
    except KeyboardInterrupt:
        print("\n\n👋 Sistema encerrado pelo usuário!")
    except Exception as e:
        print(f"❌ Erro no sistema: {e}")

if __name__ == "__main__":
    main()
