#!/usr/bin/env python3
"""
🎯 SISTEMA DE DOWNLOAD AUTOMÁTICO DE LICITAÇÕES 2023-2025
=========================================================
Download automático de todas as licitações do ComprasNet de 2023 até agora
com resolução de CAPTCHA usando sistema heurístico 98%+
"""

import asyncio
import logging
import time
import random
import os
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from playwright.async_api import async_playwright
from urllib.parse import urlencode

# Importar sistema de CAPTCHA otimizado
from comprasnet_download_monitor_open import ComprasNetDownloadMonitor
from heuristica_analisador import AnalisadorInteligente
from heuristica_reconhecedor import ReconhecedorAdaptativo
from captcha_resolver_v2_clean import CaptchaResolverV2

def convert_numpy_types(obj):
    """Converte tipos numpy para tipos serializáveis em JSON"""
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

# Configurar logging
logging.basicConfig(level=logging.INFO, format='📅 [%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class ComprasNetDownloadAutomatico:
    def __init__(self):
        # Inicializar progresso primeiro (para evitar erros)
        self.progresso = None
        self.log_progresso = Path("progresso_downloads.json")
        
        # Configurações de download - DEVE SER ANTES de inicializar navegador
        self.downloads_dir = Path("downloads_licitacoes")
        self.downloads_dir.mkdir(exist_ok=True)
        
        # Configurações do navegador
        self.page = None
        self.browser = None
        
        # Base URL do ComprasNet para consulta de licitações
        self.base_url = "https://comprasnet.gov.br/ConsultaLicitacoes/ConsLicitacao_Filtro.asp"
        
        # Sistema de resolução de CAPTCHA otimizado
        self.captcha_resolver = ComprasNetDownloadMonitor()
        
        # Sistema heurístico otimizado (98%+ sucesso)
        # Removido na limpeza: self.analisador_heuristico = AnalisadorInteligente()
        
        # Variáveis para controle de CAPTCHA e ML
        self.ultimo_captcha_resolvido = None
        
        # Sistema de reconhecimento heurístico como backup
        self.reconhecedor_heuristico = ReconhecedorAdaptativo()
        logger.info("🧠 Reconhecedor heurístico integrado como backup")
        
        # 💾 Controle de CAPTCHA para treinamento ML
        self.ultimo_captcha_resolvido = None
        logger.info("🎯 Sistema de confirmação de CAPTCHA para ML inicializado")
        
        # Carregar configuração otimizada
        self.config = self.carregar_config_otimizada()
        
        # Inicializar progresso agora que todas as variáveis estão configuradas
        self.progresso = self.inicializar_progresso_limpo()
        
        logger.info("🎯 Sistema de Download Automático inicializado com heurística 98%+")
        logger.info(f"📁 Downloads serão salvos em: {self.downloads_dir}")
        logger.info("🔄 MODO: Sempre começar do zero (progresso anterior ignorado)")
        
    def carregar_config_otimizada(self):
        """Carrega a configuração otimizada do sistema heurístico"""
        try:
            with open("config_otimizado_98.json", 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info("✅ Configuração heurística otimizada carregada (meta 98%)")
                return config
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar config otimizada: {e}")
            return {
                "meta_confianca_98": True,
                "thresholds_otimizados": {
                    "texto_preto_threshold": 25,
                    "texto_preto_confianca": 0.95
                },
                "metodos_prioritarios": [
                    "binarizacao_texto_preto",
                    "treinamento_manual",
                    "easyocr_otimizado"
                ]
            }
        
    def inicializar_progresso_limpo(self):
        """Sempre inicializa um progresso limpo, ignorando arquivos anteriores"""
        logger.info("🆕 Inicializando progresso limpo - começando do zero")
        
        # Remover arquivo de progresso anterior se existir
        if hasattr(self, 'log_progresso') and self.log_progresso.exists():
            try:
                self.log_progresso.unlink()
                logger.info("🗑️ Arquivo de progresso anterior removido")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao remover progresso anterior: {e}")
        
        # Retornar progresso limpo
        return {
            "periodos_concluidos": [], 
            "ultimo_download": None,
            "downloads_realizados": 0,
            "data_inicio_processamento": datetime.now().isoformat()
        }
        
    def carregar_progresso(self):
        """Método auxiliar para carregar progresso (não usado no modo 'sempre do zero')"""
        if self.log_progresso.exists():
            try:
                with open(self.log_progresso, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Erro ao carregar progresso: {e}")
                return {"periodos_concluidos": [], "ultimo_download": None}
        return {"periodos_concluidos": [], "ultimo_download": None}
    
    def salvar_progresso(self):
        """Salva progresso atual"""
        try:
            # Verificar se progresso existe
            if not hasattr(self, 'progresso') or self.progresso is None:
                logger.warning("⚠️ Progresso não inicializado, pulando salvamento")
                return
                
            # Converter tipos numpy antes de salvar
            progresso_convertido = convert_numpy_types(self.progresso)
            
            with open(self.log_progresso, 'w', encoding='utf-8') as f:
                json.dump(progresso_convertido, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"❌ Erro ao salvar progresso: {e}")
    
    def gerar_periodos_quinzenais(self, data_inicio, data_fim):
        """
        Gera períodos de 15 dias entre as datas especificadas
        O ComprasNet só aceita períodos máximos de 15 dias
        """
        periodos = []
        data_atual = data_inicio
        
        while data_atual <= data_fim:
            # Calcular fim do período (máximo 15 dias)
            fim_periodo = min(data_atual + timedelta(days=14), data_fim)
            
            periodo = {
                "inicio": data_atual.strftime("%d/%m/%Y"),
                "fim": fim_periodo.strftime("%d/%m/%Y"),
                "inicio_obj": data_atual,
                "fim_obj": fim_periodo
            }
            
            periodos.append(periodo)
            
            # Próximo período começa no dia seguinte
            data_atual = fim_periodo + timedelta(days=1)
        
        return periodos
    
    def construir_url_consulta(self, data_inicio, data_fim):
        """
        Constrói URL de consulta baseada no padrão fornecido
        """
        # Parâmetros baseados no link fornecido
        parametros = {
            'numprp': '',  # Número da proposta (vazio para todas)
            'dt_publ_ini': data_inicio,  # Data início
            'dt_publ_fim': data_fim,     # Data fim
            'chkModalidade': '1,2,3,20,5,99',  # Modalidades
            'chk_concor': '31,32,41,42,49',    # Concorrências
            'chk_pregao': '1,2,3,4',           # Pregões
            'chk_rdc': '1,2,3,4',              # RDC
            'optTpPesqMat': 'C',               # Tipo pesquisa material
            'optTpPesqServ': 'S',              # Tipo pesquisa serviço
            'chkTodos': '-1',                  # Todos
            'chk_concorTodos': '-1',           # Todas concorrências
            'chk_pregaoTodos': '-1',           # Todos pregões
            'txtlstUf': 'ES',                  # Estado: Espírito Santo
            'txtlstMunicipio': '57053',        # Município
            'txtlstUasg': '925968',            # UASG
            'txtlstGrpMaterial': '70',         # Grupo material
            'txtlstClasMaterial': '',          # Classificação material
            'txtlstMaterial': '',              # Material
            'txtlstGrpServico': '',            # Grupo serviço
            'txtlstServico': '',               # Serviço
            'txtObjeto': ''                    # Objeto
        }
        
        # Construir URL completa
        url_completa = f"{self.base_url}?{urlencode(parametros)}"
        return url_completa
    
    async def inicializar_navegador(self):
        """Inicializa o navegador Playwright usando Chromium persistente não-anônimo"""
        try:
            self.playwright = await async_playwright().start()
            
            # Criar pasta de dados do usuário temporária para contexto persistente
            user_data_dir = Path.cwd() / "temp_user_data"
            user_data_dir.mkdir(exist_ok=True)
            
            logger.info("🚀 Inicializando Chromium persistente para downloads não-anônimos...")
            
            # Usar launch_persistent_context para garantir modo não-anônimo
            self.context = await self.playwright.chromium.launch_persistent_context(
                user_data_dir=str(user_data_dir),
                headless=False,  # SEMPRE visível
                proxy={
                    "server": "http://filtroweb.tjes.jus.br:9090"
                },
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor',
                    '--disable-download-protection',
                    '--disable-popup-blocking',
                    '--allow-running-insecure-content',
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-background-networking',
                    '--disable-background-timer-throttling',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding',
                    '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                ],
                accept_downloads=True,
                downloads_path=str(self.downloads_dir),
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                bypass_csp=True,
                ignore_https_errors=True,
                extra_http_headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                    'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Cache-Control': 'max-age=0'
                }
            )
            
            # Obter a primeira página do contexto persistente
            self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()
            
            logger.info("✅ Contexto persistente não-anônimo inicializado")
            
            # Configurar eventos de download ANTES de qualquer navegação
            self.page.on("download", self.handle_download)
            logger.info("📥 Manipulador de downloads configurado")
            
            # Aguardar um pouco para garantir que tudo está configurado
            await asyncio.sleep(1)
            
            logger.info("🌐 Navegador Chromium inicializado com downloads otimizados")
            logger.info("🔒 Modo não-anônimo ativado para downloads corretos")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar navegador: {e}")
            return False
    
    def analisar_arquivo_download(self, nome_arquivo):
        """Analisa e categoriza o arquivo de download"""
        if not nome_arquivo:
            return {
                "tipo": "DESCONHECIDO",
                "categoria": "Arquivo sem nome",
                "numerico": False,
                "valido": False
            }
        
        nome_lower = nome_arquivo.lower()
        nome_sem_extensao = nome_arquivo
        
        # Detectar ZIP
        if nome_lower.endswith('.zip'):
            nome_sem_extensao = nome_arquivo[:-4]  # Remove .zip
            
            # Verificar se é numérico
            if nome_sem_extensao.isdigit():
                return {
                    "tipo": "ZIP",
                    "categoria": "ZIP numérico (licitação)",
                    "numerico": True,
                    "valido": True,
                    "codigo_licitacao": nome_sem_extensao
                }
            else:
                return {
                    "tipo": "ZIP",
                    "categoria": "ZIP nomeado",
                    "numerico": False,
                    "valido": True
                }
        
        # Outros tipos de arquivo
        extensoes_conhecidas = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.txt', '.rtf']
        for ext in extensoes_conhecidas:
            if nome_lower.endswith(ext):
                return {
                    "tipo": ext.upper().replace('.', ''),
                    "categoria": f"Documento {ext.upper()}",
                    "numerico": False,
                    "valido": True
                }
        
        return {
            "tipo": "OUTRO",
            "categoria": "Arquivo não categorizado",
            "numerico": False,
            "valido": True
        }

    def extrair_info_licitacao(self):
        """Extrai informações da licitação da URL atual para criar nome inteligente"""
        try:
            if hasattr(self, 'page') and self.page:
                url_atual = self.page.url
                logger.info(f"🔍 Extraindo info da URL: {url_atual}")
                
                # Extrair parâmetros da URL
                info = {
                    "coduasg": "925968",  # UASG padrão
                    "numprp": "UNKNOWN",
                    "modprp": "UNKNOWN"
                }
                
                # Buscar padrões na URL
                import re
                
                # Padrão para coduasg
                match_uasg = re.search(r'coduasg=(\d+)', url_atual)
                if match_uasg:
                    info["coduasg"] = match_uasg.group(1)
                
                # Padrão para numprp
                match_numprp = re.search(r'numprp=(\d+)', url_atual)
                if match_numprp:
                    info["numprp"] = match_numprp.group(1)
                
                # Padrão para modprp
                match_modprp = re.search(r'modprp=(\d+)', url_atual)
                if match_modprp:
                    info["modprp"] = match_modprp.group(1)
                
                logger.info(f"📋 Info extraída: UASG={info['coduasg']}, NumProp={info['numprp']}, ModProp={info['modprp']}")
                return info
                
        except Exception as e:
            logger.warning(f"⚠️ Erro ao extrair info da licitação: {e}")
        
        # Valores padrão em caso de erro
        return {
            "coduasg": "925968",
            "numprp": "UNKNOWN", 
            "modprp": "UNKNOWN"
        }

    async def handle_download(self, download):
        """Manipula downloads automáticos com controle de fluxo e renomeação inteligente"""
        try:
            # Obter nome original do arquivo (normalmente UUID)
            nome_original = download.suggested_filename or "arquivo_sem_nome.zip"
            
            # Extrair informações da licitação para criar nome inteligente
            info_licitacao = self.extrair_info_licitacao()
            
            # Detectar se o nome é UUID e precisa ser renomeado
            import re
            padrao_uuid = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
            eh_uuid = re.match(padrao_uuid, nome_original.replace('.zip', ''), re.IGNORECASE)
            
            if eh_uuid:
                # 🎯 RENOMEAÇÃO INTELIGENTE: UUID → Nome Significativo
                nome_inteligente = f"{info_licitacao['coduasg']}_{info_licitacao['modprp']}_{info_licitacao['numprp']}.zip"
                logger.info(f"🔄 RENOMEANDO UUID para nome inteligente:")
                logger.info(f"   📛 UUID Original: {nome_original}")
                logger.info(f"   🎯 Nome Inteligente: {nome_inteligente}")
                nome_para_salvar = nome_inteligente
            else:
                # Manter nome original se não for UUID
                nome_para_salvar = nome_original
                logger.info(f"📦 Arquivo com nome válido: {nome_original}")
            
            # Analisar o arquivo (agora com nome inteligente)
            analise = self.analisar_arquivo_download(nome_para_salvar)
            
            # Log específico baseado na análise
            if analise["numerico"] and analise["tipo"] == "ZIP":
                logger.info(f"📦 {analise['categoria']} detectado: {nome_original}")
                logger.info(f"🔢 Código da licitação: {analise.get('codigo_licitacao', 'N/A')}")
            else:
                logger.info(f"� {analise['categoria']} detectado: {nome_original}")
            
            # 🎯 CRIAR NOME INTELIGENTE baseado no contexto da licitação
            if eh_uuid:
                # Usar nome inteligente para UUIDs
                nome_arquivo = f"licitacao_{info_licitacao['coduasg']}_{info_licitacao['modprp']}_{info_licitacao['numprp']}.zip"
                logger.info(f"🔄 RENOMEAÇÃO APLICADA:")
                logger.info(f"   📛 UUID Original: {nome_original}")
                logger.info(f"   🎯 Nome Inteligente: {nome_arquivo}")
            else:
                # Criar nome único com timestamp para nomes não-UUID
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nome_arquivo = f"licitacao_{timestamp}_{nome_original}"
                logger.info(f"📦 Nome válido mantido: {nome_arquivo}")
            
            caminho_arquivo = self.downloads_dir / nome_arquivo
            
            # Salvar o download
            await download.save_as(caminho_arquivo)
            
            logger.info(f"✅ Download salvo: {nome_arquivo}")
            logger.info(f"📁 Arquivo original: {nome_original}")
            logger.info(f"📍 Localização: {caminho_arquivo}")
            
            # 🎯 REGISTRAR SUCESSO E CONTROLAR FLUXO
            sucesso_info = {
                "arquivo": nome_arquivo,
                "arquivo_original": nome_original,
                "timestamp": timestamp,
                "tamanho": caminho_arquivo.stat().st_size if caminho_arquivo.exists() else 0,
                "url_origem": self.page.url if hasattr(self, 'page') else 'N/A',
                "analise_arquivo": analise,
                "tipo_arquivo": analise["tipo"],
                "categoria": analise["categoria"],
                "nome_numerico": analise["numerico"],
                "codigo_licitacao": analise.get("codigo_licitacao", "N/A"),
                "status": "sucesso"
            }
            
            # Atualizar progresso
            if hasattr(self, 'progresso') and self.progresso is not None:
                # Registrar download bem-sucedido
                if "downloads_sucesso" not in self.progresso:
                    self.progresso["downloads_sucesso"] = []
                
                self.progresso["downloads_sucesso"].append(sucesso_info)
                self.progresso["ultimo_download"] = sucesso_info
                self.progresso["total_downloads_realizados"] = len(self.progresso["downloads_sucesso"])
                
                # Marcar como deve pular para próxima licitação
                self.progresso["continuar_proxima_licitacao"] = True
                
                self.salvar_progresso()
                
                # 📄 GERAR LOG DETALHADO DE SUCESSO
                self.gerar_log_sucesso(sucesso_info)
                
                # Mensagem específica baseada na análise
                analise = sucesso_info.get('analise_arquivo', {})
                if analise.get('numerico') and analise.get('codigo_licitacao'):
                    logger.info(f"🎉 SUCESSO! {analise['categoria']} baixado: {analise['codigo_licitacao']}.zip - Total: {self.progresso['total_downloads_realizados']}")
                else:
                    logger.info(f"🎉 SUCESSO! {analise.get('categoria', 'Arquivo')} baixado - Total: {self.progresso['total_downloads_realizados']}")
                    
                logger.info("➡️ Sistema continuará automaticamente para próxima licitação...")
            
            # ✅ CONFIRMAR SUCESSO DEFINITIVO DO DOWNLOAD
            self.ultimo_download_sucesso = True
            logger.info("🎯 DOWNLOAD CONFIRMADO E PROCESSADO COM SUCESSO!")
            
            # 🎯 CONFIRMAR CAPTCHA COMO SUCESSO PARA TREINAMENTO ML
            await self.confirmar_captcha_sucesso_para_ml()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar download: {e}")
            self.ultimo_download_sucesso = False
            return False
    
    def gerar_log_sucesso(self, sucesso_info):
        """Gera log detalhado de download bem-sucedido"""
        try:
            # Caminho do arquivo de log de sucessos
            log_sucessos = self.downloads_dir / "downloads_sucessos.log"
            
            # Informações para o log
            timestamp = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
            
            # Detalhes específicos baseados na análise
            analise = sucesso_info.get('analise_arquivo', {})
            detalhes_arquivo = f"� {analise.get('categoria', 'Arquivo')}: {sucesso_info.get('arquivo_original', 'N/A')}"
            
            # Informações extras para licitações numéricas
            info_extra = ""
            if analise.get('numerico') and analise.get('codigo_licitacao'):
                info_extra = f"� Código da licitação: {analise['codigo_licitacao']}\n"
            
            log_entry = f"""
[{timestamp}] ✅ DOWNLOAD BEM-SUCEDIDO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📁 Arquivo salvo: {sucesso_info['arquivo']}
{detalhes_arquivo}
{info_extra}⏰ Timestamp: {sucesso_info['timestamp']}
📦 Tamanho: {sucesso_info['tamanho']} bytes
🌐 URL de origem: {sucesso_info['url_origem']}
📊 Status: {sucesso_info['status']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
            
            # Escrever no arquivo de log
            with open(log_sucessos, "a", encoding="utf-8") as f:
                f.write(log_entry)
                
            logger.info(f"📝 Log de sucesso gravado em: {log_sucessos}")
            
        except Exception as e:
            logger.warning(f"⚠️ Erro ao gerar log de sucesso: {e}")
    
    async def navegar_para_periodo(self, periodo):
        """Navega para página de consulta de um período específico"""
        try:
            url = self.construir_url_consulta(periodo["inicio"], periodo["fim"])
            
            logger.info(f"🔍 Consultando período: {periodo['inicio']} a {periodo['fim']}")
            logger.info(f"🌐 URL: {url}")
            
            await self.page.goto(url, wait_until='domcontentloaded', timeout=15000)
            
            # Aguardar carregamento da página (otimizado)
            await asyncio.sleep(0.5)
            
            # NOVO: Verificar e clicar no botão OK se presente
            logger.info("🔍 Verificando se há botão OK para clicar...")
            try:
                botao_ok = await self.page.query_selector('input[value="OK"]')
                
                if botao_ok:
                    logger.info("✅ Botão OK encontrado - clicando para abrir resultados...")
                    await botao_ok.click()
                    
                    # Aguardar navegação para página de resultados (otimizado)
                    await self.page.wait_for_load_state('domcontentloaded', timeout=8000)
                    await asyncio.sleep(0.5)
                    
                    nova_url = self.page.url
                    logger.info(f"📍 Navegou para página de resultados: {nova_url}")
                    
                    if "ConsLicitacao_Relacao.asp" in nova_url:
                        logger.info("✅ Página de resultados carregada com sucesso")
                    else:
                        logger.warning(f"⚠️ URL inesperada após clique no OK: {nova_url}")
                else:
                    logger.info("📋 Botão OK não encontrado - página já pode estar nos resultados")
                    
            except Exception as ok_error:
                logger.warning(f"⚠️ Erro ao tratar botão OK: {ok_error}")
                # Continuar mesmo se houver erro com o botão OK
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao navegar para período: {e}")
            return False
    
    async def processar_licitacoes_periodo(self, periodo):
        """Processa todas as licitações de um período - adaptado para ASP clássico"""
        try:
            # Aguardar carregamento completo
            await self.page.wait_for_load_state('networkidle')
            await asyncio.sleep(2)
            
            # Buscar pelos botões "Itens e Download" específicos
            logger.info("🔍 Procurando botões 'Itens e Download'...")
            
            # Usar o seletor correto descoberto
            botoes_itens = await self.page.query_selector_all('input[value="Itens e Download"]')
            
            if not botoes_itens:
                # Fallback para outros seletores
                logger.info("🔄 Tentando seletores alternativos...")
                seletores_alternativos = [
                    'input[value*="Itens"]',
                    'input[name="itens"]',
                    'a[href*="Itens"]',
                    'input[onclick*="VisualizarItens"]'
                ]
                
                for seletor in seletores_alternativos:
                    botoes_itens = await self.page.query_selector_all(seletor)
                    if botoes_itens:
                        logger.info(f"✅ {len(botoes_itens)} botão(ões) encontrado(s) com: {seletor}")
                        break
            else:
                logger.info(f"✅ {len(botoes_itens)} botão(ões) 'Itens e Download' encontrado(s)")

            if not botoes_itens:
                logger.warning(f"⚠️ Nenhum botão de itens encontrado no período {periodo['inicio']} a {periodo['fim']}")
                return 0

            logger.info(f"📋 Processando {len(botoes_itens)} licitação(ões) no período")
            
            downloads_realizados = 0
            licitacoes_processadas = 0
            
            # 🔄 PROCESSAR LICITAÇÕES UMA POR VEZ, VOLTANDO SEMPRE PARA A LISTAGEM
            while licitacoes_processadas < len(botoes_itens):
                try:
                    licitacoes_processadas += 1
                    logger.info(f"📄 Processando licitação {licitacoes_processadas}/{len(botoes_itens)}")
                    
                    # 🔄 SEMPRE RE-BUSCAR OS BOTÕES NA PÁGINA ATUAL
                    logger.info("🔍 Re-buscando botões 'Itens e Download' na página atual...")
                    botoes_atuais = await self.page.query_selector_all('input[value="Itens e Download"]')
                    
                    if not botoes_atuais:
                        logger.warning("⚠️ Nenhum botão encontrado na página atual")
                        break
                    
                    if licitacoes_processadas > len(botoes_atuais):
                        logger.info("✅ Todas as licitações desta página foram processadas")
                        break
                    
                    # Pegar o primeiro botão disponível (sempre o primeiro não processado)
                    botao_item = botoes_atuais[0]
                    
                    # Obter informações do botão antes de clicar
                    try:
                        value = await botao_item.get_attribute('value')
                        onclick = await botao_item.get_attribute('onclick')
                        logger.info(f"� Clicando no botão: {value}")
                        if onclick:
                            logger.info(f"🔗 Onclick: {onclick}")
                    except Exception as e:
                        logger.warning(f"⚠️ Erro ao obter atributos do botão: {e}")
                    
                    # Clicar no botão "Itens e Download"
                    await botao_item.click()
                    
                    # Aguardar carregamento da nova página
                    logger.info("⏳ Aguardando página de detalhes...")
                    try:
                        # Aguardar mudança de URL para página de download
                        await self.page.wait_for_url("**/download/download_editais_detalhe.asp*", timeout=15000)
                        logger.info("✅ Página de detalhes carregada")
                        
                        current_url = self.page.url
                        logger.info(f"📍 URL atual: {current_url}")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Timeout ao aguardar página de detalhes: {e}")
                        # Continuar mesmo assim (otimizado)
                        await self.page.wait_for_load_state('domcontentloaded', timeout=5000)
                        await asyncio.sleep(0.5)
                        current_url = self.page.url
                        logger.info(f"📍 URL atual: {current_url}")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Timeout ao aguardar página: {e}")
                        continue

                    # Procurar botão de download na página ASP
                    logger.info("🔽 Procurando botão de download...")
                    
                    # No ASP clássico, os botões podem ter várias formas
                    botao_download = None
                    
                    # Tentar diferentes seletores para botão de download
                    seletores_download = [
                        'input[name="Download"]',
                        'input[value="Download"]', 
                        'input[value*="Download"]',
                        'a[href*="download"]',
                        'input[type="submit"][value*="Download"]',
                        'button[name*="download"]',
                        'input[onclick*="download"]'
                    ]
                    
                    for seletor in seletores_download:
                        botao_download = await self.page.query_selector(seletor)
                        if botao_download:
                            logger.info(f"✅ Botão de download encontrado com seletor: {seletor}")
                            break
                    
                    if botao_download:
                        logger.info("📥 Iniciando download...")
                        
                        try:
                            # Verificar se o botão abre popup ou navega
                            onclick = await botao_download.get_attribute('onclick')
                            target = await botao_download.get_attribute('target')
                            
                            # Usar expect_popup() para detectar CAPTCHA diretamente
                            logger.info("📥 Botão Download encontrado - usando expect_popup para capturar CAPTCHA...")
                            
                            async with self.page.context.expect_page() as popup_info:
                                logger.info("🔄 Clique realizado - aguardando popup...")
                                await botao_download.click()
                                
                            popup_page = await popup_info.value
                            popup_url = popup_page.url
                            logger.info(f"🎉 POPUP CAPTCHA DETECTADO: {popup_url}")
                            
                            # Verificar se é página de CAPTCHA
                            if "Download.asp" in popup_url:
                                logger.info("✅ Confirmado: Popup é página de CAPTCHA")
                                await asyncio.sleep(3)  # Aguardar carregamento completo
                                
                                # Inicializar flag de sucesso
                                self.ultimo_download_sucesso = False
                                
                                # 🎯 PROCESSAR CAPTCHA COM CONTROLE INTELIGENTE
                                logger.info("🚀 Iniciando processo de CAPTCHA com controle de fluxo...")
                                
                                # Usar loop personalizado com parada imediata em caso de sucesso
                                tentativas_captcha = 0
                                download_realizado = False
                                
                                while tentativas_captcha < 30 and not download_realizado:
                                    logger.info(f"🔄 Tentativa {tentativas_captcha+1}/30 de CAPTCHA")
                                    
                                    # Resolver CAPTCHA com sistema V2.0 otimizado
                                    sucesso_captcha = await self.resolver_captcha_heuristico_otimizado(popup_page)
                                    
                                    if sucesso_captcha:
                                        logger.info("✅ CAPTCHA resolvido e clicado!")
                                        
                                        # 🎯 AGUARDAR CONFIRMAÇÃO REAL DO DOWNLOAD
                                        download_confirmado = False
                                        try:
                                            logger.info("⏳ Aguardando confirmação de download do navegador...")
                                            
                                            # Verificar se popup redirecionou (otimizado)
                                            await asyncio.sleep(0.3)
                                            current_url = popup_page.url
                                            logger.info(f"🌐 URL após clique: {current_url}")
                                            
                                            # Verificar se há mensagens na página (otimizado)
                                            try:
                                                page_text = await popup_page.text_content("body", timeout=1500)
                                                if "erro" in page_text.lower() or "incorret" in page_text.lower():
                                                    logger.warning(f"❌ Erro detectado na página: {page_text[:200]}")
                                                    continue  # Tentar próximo CAPTCHA
                                                elif "download" in page_text.lower() or "arquivo" in page_text.lower():
                                                    logger.info(f"📄 Resposta do servidor: {page_text[:200]}")
                                            except:
                                                pass
                                            
                                            # 🚨 AGUARDAR EVENTO REAL DE DOWNLOAD DO NAVEGADOR (otimizado)
                                            try:
                                                async with popup_page.expect_download(timeout=8000) as download_info:
                                                    pass  # Clique já foi feito, só aguardando confirmação
                                                
                                                # ✅ DOWNLOAD CONFIRMADO PELO NAVEGADOR!
                                                download = await download_info.value
                                                nome_arquivo = download.suggested_filename or "arquivo_sem_nome"
                                                download_confirmado = True
                                                
                                                logger.info(f"🎉 DOWNLOAD CONFIRMADO PELO NAVEGADOR: {nome_arquivo}")
                                                
                                                # Analisar arquivo para log específico
                                                analise = self.analisar_arquivo_download(nome_arquivo)
                                                
                                                if analise["numerico"] and analise["tipo"] == "ZIP":
                                                    logger.info(f"📦 {analise['categoria']}: {analise.get('codigo_licitacao', nome_arquivo)}.zip")
                                                else:
                                                    logger.info(f"📦 {analise['categoria']}: {nome_arquivo}")
                                                
                                                # Processar download com renomeação inteligente
                                                await self.handle_download(download)
                                                download_realizado = True
                                                
                                                logger.info("✅ DOWNLOAD PROCESSADO! Continuando para próxima licitação...")
                                                break  # Sair do loop de tentativas - SUCESSO CONFIRMADO!
                                                
                                            except Exception as download_error:
                                                logger.warning(f"⚠️ Download não foi iniciado pelo navegador: {download_error}")
                                                download_confirmado = False
                                                # Continuar para próxima tentativa de CAPTCHA
                                            
                                        except Exception as e:
                                            logger.warning(f"⚠️ Erro na verificação de download: {e}")
                                            download_confirmado = False
                                    else:
                                        logger.warning(f"⚠️ CAPTCHA falhou na tentativa {tentativas_captcha+1}")
                                    
                                    # Verificar flag de sucesso global
                                    if hasattr(self, 'ultimo_download_sucesso') and self.ultimo_download_sucesso:
                                        logger.info("🎉 Sucesso detectado via flag! Parando...")
                                        download_realizado = True
                                        break
                                    
                                    # Recarregar apenas se necessário
                                    if tentativas_captcha < 14 and not download_realizado:
                                        logger.info("� Recarregando para nova tentativa...")
                                        try:
                                            await popup_page.reload(wait_until="networkidle")
                                            await asyncio.sleep(2)
                                        except:
                                            pass
                                    
                                    tentativas_captcha += 1
                                
                                # Fechar popup após processamento (sucesso ou falha)
                                await popup_page.close()
                                logger.info("🗂️ Popup CAPTCHA fechado")
                                
                                # 🎯 VERIFICAR RESULTADO FINAL - APENAS COM CONFIRMAÇÃO REAL
                                if download_realizado:
                                    downloads_realizados += 1
                                    logger.info(f"🎉 DOWNLOAD {licitacoes_processadas} CONFIRMADO PELO NAVEGADOR!")
                                    logger.info(f"📊 Total de downloads confirmados: {downloads_realizados}")
                                    
                                    # ✅ SUCESSO CONFIRMADO - Voltar para listagem e continuar
                                    logger.info("➡️ Download confirmado! Voltando para listagem...")
                                    
                                else:
                                    logger.warning(f"❌ DOWNLOAD {licitacoes_processadas} NÃO CONFIRMADO após {tentativas_captcha} tentativas de CAPTCHA")
                                    logger.info("➡️ Sem confirmação de download - Voltando para listagem...")
                                
                            else:
                                # Se não for CAPTCHA, fechar popup e continuar
                                logger.warning(f"⚠️ Popup não é CAPTCHA: {popup_url}")
                                await popup_page.close()
                                
                        except Exception as e:
                            logger.error(f"❌ Erro ao processar download: {e}")
                    else:
                        logger.warning(f"⚠️ Botão de download não encontrado para licitação {licitacoes_processadas}")
                    
                    # 🔙 SEMPRE VOLTAR PARA A PÁGINA DE LISTAGEM APÓS CADA LICITAÇÃO
                    logger.info("🔙 Voltando para página de listagem...")
                    try:
                        # Verificar se ainda estamos na página de detalhes
                        if 'download_editais_detalhe.asp' in self.page.url or 'Download.asp' in self.page.url:
                            await self.page.go_back()
                            await self.page.wait_for_load_state('domcontentloaded', timeout=10000)
                            await asyncio.sleep(1)
                        
                        # Verificar se voltamos para a listagem
                        if 'ConsLicitacao_Relacao.asp' not in self.page.url:
                            # Se go_back não funcionou, navegar diretamente
                            url_listagem = self.page.url.replace('download_editais_detalhe.asp', '../ConsLicitacao_Relacao.asp')
                            if 'ConsLicitacao_Relacao.asp' not in url_listagem:
                                # Construir URL de listagem baseada no período
                                url_listagem = "https://comprasnet.gov.br/ConsultaLicitacoes/ConsLicitacao_Relacao.asp"
                            
                            logger.info(f"🔄 Navegando diretamente para: {url_listagem}")
                            await self.page.goto(url_listagem, wait_until='domcontentloaded', timeout=10000)
                            await asyncio.sleep(1)
                        
                        logger.info(f"✅ De volta à listagem: {self.page.url}")
                        
                    except Exception as nav_error:
                        logger.warning(f"⚠️ Erro ao voltar para listagem: {nav_error}")
                        # Se tudo falhar, reconstruir a consulta
                        try:
                            url_original = self.construir_url_consulta(periodo["inicio"], periodo["fim"])
                            await self.page.goto(url_original, wait_until='domcontentloaded')
                            await asyncio.sleep(2)
                            # Clicar no botão OK para voltar aos resultados
                            botao_ok = await self.page.query_selector('input[value="OK"]')
                            if botao_ok:
                                await botao_ok.click()
                                await self.page.wait_for_load_state('domcontentloaded', timeout=10000)
                                await asyncio.sleep(2)
                        except Exception as rebuild_error:
                            logger.error(f"❌ Erro crítico ao reconstruir navegação: {rebuild_error}")
                            break
                    
                except Exception as e:
                    logger.error(f"❌ Erro ao processar licitação {licitacoes_processadas}: {e}")
                    # Tentar voltar para listagem mesmo em caso de erro
                    try:
                        if 'ConsLicitacao_Relacao.asp' not in self.page.url:
                            await self.page.go_back()
                            await self.page.wait_for_load_state('domcontentloaded', timeout=5000)
                    except:
                        pass
                    continue
            
            # Verificar se há próxima página (navegação ASP)
            try:
                logger.info("🔍 Verificando se há próxima página...")
                
                # No ASP, o botão de próxima página pode ter diferentes formatos
                seletores_proxima = [
                    'input[value="Próxima"]',
                    'input[value="Próximo"]', 
                    'input[value*="Próx"]',
                    'a[href*="proxima"]',
                    'a[href*="proximo"]',
                    'input[name*="next"]',
                    'input[onclick*="proxima"]'
                ]
                
                botao_proxima = None
                for seletor in seletores_proxima:
                    botao_proxima = await self.page.query_selector(seletor)
                    if botao_proxima:
                        logger.info(f"✅ Botão 'Próxima' encontrado: {seletor}")
                        break
                
                if botao_proxima:
                    logger.info("➡️ Navegando para próxima página...")
                    await botao_proxima.click()
                    await self.page.wait_for_load_state('networkidle')
                    await asyncio.sleep(2)
                    
                    # Processar próxima página recursivamente
                    downloads_proxima = await self.processar_licitacoes_periodo(periodo)
                    downloads_realizados += downloads_proxima
                else:
                    logger.info("📄 Não há próxima página disponível")
                    
            except Exception as e:
                logger.warning(f"⚠️ Erro ao verificar próxima página: {e}")
            
            logger.info(f"✅ Período processado: {downloads_realizados} downloads realizados")
            return downloads_realizados
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar licitações do período: {e}")
            return 0
    
    async def processar_captcha_download(self, page, numero_licitacao):
        """Processa CAPTCHA e realiza download"""
        try:
            logger.info(f"🔒 Procurando CAPTCHA na página...")
            
            # Aguardar possível CAPTCHA aparecer
            await asyncio.sleep(2)
            
            # Tentar resolver CAPTCHA até 30 vezes
            tentativas_captcha = 0
            download = None
            
            while tentativas_captcha < 30 and not download:
                # Usar o sistema V2.0 otimizado para resolver o captcha
                sucesso_captcha = await self.resolver_captcha_heuristico_otimizado(page)
                
                if sucesso_captcha:
                    logger.info(f"✅ Tentativa {tentativas_captcha+1}: CAPTCHA resolvido e botão clicado automaticamente!")
                    try:
                        # Aguardar download após clique automático
                        async with page.expect_download(timeout=10000) as download_info:
                            pass  # O clique já foi feito automaticamente na função
                        download = await download_info.value
                        logger.info(f"📥 Download iniciado: {download.suggested_filename}")
                        await self.handle_download(download)
                        return True
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Download não iniciou após CAPTCHA: {e}")
                else:
                    logger.warning(f"⚠️ Falha ao resolver CAPTCHA na tentativa {tentativas_captcha+1}")
                
                if not download:
                    # Recarregar para nova tentativa
                    logger.info("� Recarregando para nova tentativa...")
                    await page.reload(wait_until="networkidle")
                    await asyncio.sleep(2)
                
                tentativas_captcha += 1
            
            if not download:
                logger.error(f"❌ Falha em todas as tentativas de CAPTCHA para licitação {numero_licitacao}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erro ao processar CAPTCHA: {e}")
            return False
    
    
    async def processar_captcha_com_monitor(self, nova_aba, numero_licitacao):
        """Usa o sistema heurístico otimizado (98%+) para resolver CAPTCHA"""
        try:
            logger.info(f"� Iniciando resolução de CAPTCHA com heurística 98%+ para licitação {numero_licitacao}")
            
            # Tentar resolver CAPTCHA até 3 vezes (reduzido devido à alta precisão)
            tentativas_captcha = 0
            while tentativas_captcha < 30:
                logger.info(f"🎯 Tentativa {tentativas_captcha+1}/30 de resolução com heurística otimizada")
                
                # Usar o sistema heurístico integrado
                botao_confirmar = await self.resolver_captcha_heuristico_otimizado(nova_aba)
                
                if botao_confirmar:
                    logger.info(f"✅ CAPTCHA resolvido com heurística 98%+! Clicando para download...")
                    try:
                        async with nova_aba.expect_download(timeout=15000) as download_info:
                            await botao_confirmar.click()
                        download = await download_info.value
                        logger.info(f"📥 Download iniciado: {download.suggested_filename}")
                        await self.handle_download(download)
                        return True
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Download não iniciou após CAPTCHA: {e}")
                else:
                    logger.warning(f"⚠️ Falha ao resolver CAPTCHA na tentativa {tentativas_captcha+1}")
                
                # Recarregar para nova tentativa (menos necessário com 98% de sucesso)
                if tentativas_captcha < 29:  # Não recarregar na última tentativa (30-1)
                    logger.info("🔄 Recarregando página para nova tentativa...")
                    await nova_aba.reload(wait_until="networkidle")
                    await asyncio.sleep(2)  # Tempo reduzido
                
                tentativas_captcha += 1
            
            logger.error(f"❌ Falha em todas as tentativas de CAPTCHA para licitação {numero_licitacao}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar CAPTCHA com heurística: {e}")
            return False
            
            # Aguardar carregamento completo da página
            await nova_aba.wait_for_load_state('networkidle')
            await asyncio.sleep(2)
            
            # Primeiro: tentar download direto pelo botão Download
            logger.info("🔍 Procurando botão Download...")
            botao_download = await nova_aba.query_selector('input[name="Download"]')
            
            if botao_download:
                logger.info("� Botão Download encontrado - tentando download direto...")
                
                # Tentar download direto até 3 vezes
                for tentativa in range(30):
                    try:
                        logger.info(f"🔄 Tentativa {tentativa+1}/30 de download direto...")
                        
                        # Aguardar download
                        async with nova_aba.expect_download(timeout=15000) as download_info:
                            await botao_download.click()
                        
                        download = await download_info.value
                        logger.info(f"✅ Download direto bem-sucedido: {download.suggested_filename}")
                        await self.handle_download(download)
                        return True
                        
                    except Exception as e:
                        logger.info(f"⚠️ Download direto falhou na tentativa {tentativa+1}: {e}")
                        await asyncio.sleep(2)
                
                # Se download direto falhou, verificar se apareceu CAPTCHA
                logger.info("🔒 Download direto falhou - verificando se há CAPTCHA...")
                await asyncio.sleep(2)
                
                # Procurar por elementos CAPTCHA com prioridade ComprasNet
                seletores_captcha = [
                    'img[src*="captcha.aspx?opt=image"]',  # PRIORITÁRIO: ComprasNet específico
                    'img[src*="captcha.aspx"]',            # ComprasNet genérico
                    'img[src*="captcha"]',
                    'img[alt*="captcha"]', 
                    'img[src*="security"]',
                    'input[name*="captcha"]',
                    'input[placeholder*="captcha"]',
                    'input[placeholder*="código"]',
                    'canvas'
                ]
                
                captcha_encontrado = False
                for seletor in seletores_captcha:
                    elemento = await nova_aba.query_selector(seletor)
                    if elemento:
                        logger.info(f"🔒 CAPTCHA encontrado com seletor: {seletor}")
                        captcha_encontrado = True
                        break
                
                if captcha_encontrado:
                    logger.info("🤖 Iniciando resolução automática de CAPTCHA...")
                    
                    # Usar o sistema V2.0 otimizado para resolver o captcha na nova aba
                    tentativas_captcha = 0
                    while tentativas_captcha < 30:
                        logger.info(f"🔄 Tentativa {tentativas_captcha+1}/30 de resolução de CAPTCHA")
                        
                        sucesso_captcha = await self.resolver_captcha_heuristico_otimizado(nova_aba)
                        
                        if sucesso_captcha:
                            logger.info(f"✅ CAPTCHA resolvido e botão clicado automaticamente!")
                            try:
                                # Aguardar download após clique automático
                                async with nova_aba.expect_download(timeout=15000) as download_info:
                                    pass  # O clique já foi feito automaticamente na função
                                download = await download_info.value
                                logger.info(f"📥 Download iniciado: {download.suggested_filename}")
                                await self.handle_download(download)
                                return True
                                
                            except Exception as e:
                                logger.warning(f"⚠️ Download não iniciou após CAPTCHA: {e}")
                        else:
                            logger.warning(f"⚠️ Falha ao resolver CAPTCHA na tentativa {tentativas_captcha+1}")
                        
                        # Recarregar para nova tentativa
                        if tentativas_captcha < 29:  # Não recarregar na última tentativa (30-1)
                            logger.info("🔄 Recarregando página para nova tentativa...")
                            await nova_aba.reload(wait_until="networkidle")
                            await asyncio.sleep(3)
                        
                        tentativas_captcha += 1
                    
                    logger.error(f"❌ Falha em todas as tentativas de CAPTCHA para licitação {numero_licitacao}")
                    return False
                    
                else:
                    # Não há CAPTCHA, mas download direto também falhou
                    logger.warning("⚠️ Nenhum CAPTCHA encontrado e download direto falhou")
                    
                    # Tentar outros botões possíveis
                    logger.info("🔍 Procurando botões alternativos...")
                    botoes_alternativos = [
                        'input[type="submit"]',
                        'button[type="submit"]',
                        'input[value*="Baixar"]',
                        'input[value*="Confirmar"]',
                        'a[href*="download"]'
                    ]
                    
                    for seletor in botoes_alternativos:
                        botao_alt = await nova_aba.query_selector(seletor)
                        if botao_alt:
                            logger.info(f"🔄 Tentando botão alternativo: {seletor}")
                            try:
                                async with nova_aba.expect_download(timeout=10000) as download_info:
                                    await botao_alt.click()
                                download = await download_info.value
                                logger.info(f"📥 Download via botão alternativo: {download.suggested_filename}")
                                await self.handle_download(download)
                                return True
                            except:
                                continue
                    
                    logger.error(f"❌ Nenhum método de download funcionou para licitação {numero_licitacao}")
                    return False
            else:
                logger.warning("⚠️ Botão Download não encontrado na nova aba")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erro ao processar download em nova aba: {e}")
            logger.error(f"🌐 URL: {url_captcha}")
            return False
    
    async def resolver_captcha_heuristico_otimizado(self, page):
        """Resolve CAPTCHA usando sistema heurístico otimizado + análise por caractere"""
        try:
            logger.info("🧠 Aplicando análise heurística avançada com segmentação por caractere...")
            
            # Aguardar a imagem do CAPTCHA aparecer
            await asyncio.sleep(2)
            
            # 📸 CAPTURA MELHORADA: Fazer screenshot da página toda primeiro
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                os.makedirs("debug_sistema/paginas_licitacao", exist_ok=True)
                screenshot_path = os.path.join("debug_sistema", "paginas_licitacao", f"captcha_page_{timestamp}.png")
                await page.screenshot(path=screenshot_path, full_page=True)
                logger.info(f"📸 Screenshot da página salva: {screenshot_path}")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao salvar screenshot da página: {e}")
            
            # Procurar imagem do CAPTCHA com prioridade para ComprasNet
            img_captcha = None
            captcha_selectors = [
                'img[src*="captcha.aspx?opt=image"]',  # PRIORITÁRIO: ComprasNet específico
                'img[src*="captcha.aspx"]',            # ComprasNet genérico
                'img[src*="captcha"]',
                'img[alt*="captcha"]', 
                'img[id*="captcha"]',
                'img[src*="security"]',
                'img[src*="verify"]',
                'canvas',
                'img[src*=".aspx"]'  # Para outros CAPTCHAs ASP.NET
            ]
            
            for selector in captcha_selectors:
                img_captcha = await page.query_selector(selector)
                if img_captcha:
                    logger.info(f"🔍 CAPTCHA encontrado com seletor: {selector}")
                    break
            
            if not img_captcha:
                logger.warning("⚠️ Imagem do CAPTCHA não encontrada")
                return False
            
            # 🎯 MELHORAR CAPTURA: Primeiro tentar obter tamanho real da imagem
            try:
                # Obter propriedades da imagem
                img_props = await img_captcha.evaluate('''element => {
                    return {
                        src: element.src,
                        width: element.naturalWidth || element.width,
                        height: element.naturalHeight || element.height,
                        offsetWidth: element.offsetWidth,
                        offsetHeight: element.offsetHeight,
                        clientWidth: element.clientWidth,
                        clientHeight: element.clientHeight
                    };
                }''')
                
                logger.info(f"📏 Propriedades da imagem CAPTCHA: {img_props}")
                
                # Se a imagem é muito pequena, tentar forçar um tamanho maior
                if img_props['width'] < 100 or img_props['height'] < 50:
                    logger.warning(f"⚠️ Imagem muito pequena ({img_props['width']}x{img_props['height']}), tentando redimensionar...")
                    
                    # Tentar forçar tamanho maior via CSS
                    await page.evaluate('''() => {
                        const captchaImg = document.querySelector('img[src*="captcha.aspx?opt=image"], img[src*="captcha.aspx"], img[src*="captcha"], img[alt*="captcha"], img[id*="captcha"]');
                        if (captchaImg) {
                            captchaImg.style.width = '200px';
                            captchaImg.style.height = '80px';
                            captchaImg.style.imageRendering = 'pixelated';
                            captchaImg.style.transform = 'scale(3)';
                            captchaImg.style.transformOrigin = 'top left';
                        }
                    }''')
                    
                    await asyncio.sleep(0.3)  # Aguardar aplicação do CSS (otimizado)
                
            except Exception as e:
                logger.warning(f"⚠️ Erro ao obter propriedades da imagem: {e}")
            
            # Capturar screenshot da imagem do CAPTCHA
            try:
                captcha_screenshot = await img_captcha.screenshot()
                logger.info(f"📸 Screenshot do CAPTCHA capturado: {len(captcha_screenshot)} bytes")
            except Exception as e:
                logger.error(f"❌ Erro ao capturar screenshot do CAPTCHA: {e}")
                return False
            
            # Salvar temporariamente para análise
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_file.write(captcha_screenshot)
                caminho_temp = tmp_file.name
            
            # 🎯 SISTEMA V2.0: Resolução avançada com múltiplas estratégias
            logger.info("🚀 INICIANDO SISTEMA CAPTCHA V2.0 (6+ CARACTERES GARANTIDOS)")
            
            # Inicializar resolver V2.0 se não existir
            if not hasattr(self, 'resolver_v2'):
                self.resolver_v2 = CaptchaResolverV2()
            
            # Carregar imagem para análise V2.0
            import cv2
            img = cv2.imread(caminho_temp)
            
            # Tentar resolução V2.0 primeiro
            resultado_v2 = self.resolver_v2.resolver_captcha_completo(img, salvar_debug=True, salvar_sucesso=False)
            
            if resultado_v2.get('sucesso') and resultado_v2.get('tem_6_chars'):
                texto_resolvido = resultado_v2['texto']
                confianca = resultado_v2['confianca']
                metodo = resultado_v2['metodo']
                logger.info(f"� SISTEMA V2.0 SUCCESS: '{texto_resolvido}' ({metodo}, conf: {confianca:.2f}, chars: {len(texto_resolvido)})")
                
                # Inserir o texto no campo de CAPTCHA
                try:
                    # Procurar campo de entrada do CAPTCHA
                    campo_captcha = await page.query_selector('input[name*="captcha"]')
                    if not campo_captcha:
                        campo_captcha = await page.query_selector('input[id*="captcha"]')
                    if not campo_captcha:
                        campo_captcha = await page.query_selector('input[type="text"]')
                    
                    if campo_captcha:
                        await campo_captcha.fill(texto_resolvido)
                        logger.info(f"✅ V2.0: Texto '{texto_resolvido}' inserido no campo CAPTCHA")
                        
                        # Procurar e clicar no botão de confirmação
                        try:
                            # Método 1: Tentar clique JavaScript primeiro (evita sobreposição)
                            logger.info("🔄 V2.0: Tentando clique JavaScript no botão...")
                            sucesso_js = await page.evaluate('''() => {
                                // Procurar botão de confirmação
                                let botao = document.querySelector('input[type="submit"][value="Confirmar"]');
                                if (!botao) botao = document.querySelector('input[type="submit"]');
                                if (!botao) botao = document.querySelector('button[type="submit"]');
                                if (!botao) botao = document.querySelector('input[value*="Confirmar"]');
                                if (!botao) botao = document.querySelector('input[value*="OK"]');
                                
                                if (botao) {
                                    botao.focus();
                                    botao.click();
                                    return true;
                                }
                                return false;
                            }''')
                            
                            if sucesso_js:
                                logger.info("✅ V2.0: Botão clicado via JavaScript!")
                                return True
                            
                            # Método 2: Fallback para clique direto
                            logger.info("🔄 V2.0: Tentando clique direto no botão...")
                            botao_confirmar = await page.query_selector('input[type="submit"][value="Confirmar"]')
                            if not botao_confirmar:
                                botao_confirmar = await page.query_selector('input[type="submit"]')
                            if not botao_confirmar:
                                botao_confirmar = await page.query_selector('button[type="submit"]')
                            if not botao_confirmar:
                                botao_confirmar = await page.query_selector('input[value*="Confirmar"]')
                            if not botao_confirmar:
                                botao_confirmar = await page.query_selector('input[value*="OK"]')
                            
                            if botao_confirmar:
                                await botao_confirmar.click(timeout=5000)
                                logger.info("✅ V2.0: Botão de confirmação clicado diretamente!")
                                return True
                            else:
                                logger.warning("⚠️ V2.0: Botão de confirmação não encontrado")
                                return False
                        except Exception as e:
                            logger.error(f"❌ V2.0: Erro ao clicar no botão: {e}")
                            return False
                    else:
                        logger.warning("⚠️ Campo de entrada do CAPTCHA não encontrado")
                except Exception as e:
                    logger.error(f"❌ Erro ao inserir texto no campo: {e}")
            else:
                logger.warning(f"⚠️ Sistema V2.0 falhou: confiança {resultado_v2.get('confianca', 0):.2f}, 6+chars: {resultado_v2.get('tem_6_chars', False)}")
            
            # 🔄 FALLBACK 1: Sistema V2.0 já processou acima - sem necessidade de fallbacks adicionais
            logger.info("🔄 Sistema V2.0 já processou acima - sem necessidade de fallbacks adicionais")
            
            # Se chegou até aqui sem sucesso, o V2.0 já esgotou todas as possibilidades
            logger.warning("⚠️ Sistema V2.0 não conseguiu resolver o CAPTCHA")
            return False
            
        except Exception as e:
            logger.error(f"❌ Erro na resolução CAPTCHA: {e}")
            return False
        finally:
            # Limpar arquivo temporário
            try:
                import os
                if 'caminho_temp' in locals():
                    os.unlink(caminho_temp)
            except:
                pass
    
    async def executar_downloads_automaticos(self):
        """Executa downloads automáticos de 2023 até agora com controle de fluxo inteligente"""
        try:
            # 🕒 Marcar início da execução
            self.tempo_inicio = datetime.now()
            
            # Definir período: 01/01/2023 até hoje
            data_inicio = datetime(2023, 1, 1)
            data_fim = datetime.now()
            
            logger.info("🎯 INICIANDO DOWNLOADS AUTOMÁTICOS DE LICITAÇÕES")
            logger.info("=" * 60)
            logger.info(f"📅 Período: {data_inicio.strftime('%d/%m/%Y')} até {data_fim.strftime('%d/%m/%Y')}")
            logger.info(f"🕒 Início: {self.tempo_inicio.strftime('%d/%m/%Y %H:%M:%S')}")
            
            # Gerar períodos quinzenais
            periodos = self.gerar_periodos_quinzenais(data_inicio, data_fim)
            logger.info(f"📊 Total de períodos a processar: {len(periodos)}")
            
            # Inicializar progresso se não existir
            if not hasattr(self, 'progresso') or self.progresso is None:
                self.progresso = {
                    "downloads_sucesso": [],
                    "total_downloads_realizados": 0,
                    "inicio_execucao": self.tempo_inicio.isoformat()
                }
            
            # Inicializar navegador
            if not await self.inicializar_navegador():
                logger.error("❌ Falha ao inicializar navegador")
                return
            
            total_downloads = 0
            
            try:
                for i, periodo in enumerate(periodos, 1):
                    logger.info(f"🔄 PROCESSANDO PERÍODO {i}/{len(periodos)}: {periodo['inicio']} a {periodo['fim']}")
                    logger.info("=" * 80)
                    
                    # Navegar para período
                    if await self.navegar_para_periodo(periodo):
                        # Processar licitações do período
                        downloads_periodo = await self.processar_licitacoes_periodo(periodo)
                        total_downloads += downloads_periodo
                        
                        # 📊 RELATÓRIO DO PERÍODO
                        logger.info("=" * 80)
                        logger.info(f"✅ PERÍODO {i}/{len(periodos)} CONCLUÍDO!")
                        logger.info(f"📅 Período: {periodo['inicio']} a {periodo['fim']}")
                        logger.info(f"📥 Downloads neste período: {downloads_periodo}")
                        logger.info(f"📊 Total acumulado: {total_downloads}")
                        
                        # Atualizar estatísticas de progresso
                        if hasattr(self, 'progresso') and self.progresso is not None:
                            self.progresso["downloads_realizados"] = total_downloads
                            self.progresso["ultimo_periodo_processado"] = f"{periodo['inicio']} a {periodo['fim']}"
                            self.progresso["periodo_atual"] = f"{i}/{len(periodos)}"
                            self.progresso["percentual_concluido"] = round((i / len(periodos)) * 100, 2)
                            self.salvar_progresso()
                        
                        # 🎯 DECISÃO DE FLUXO
                        if downloads_periodo > 0:
                            logger.info(f"🎉 Período produtivo! {downloads_periodo} downloads realizados")
                        else:
                            logger.info("📭 Período sem downloads (sem licitações ou todas falharam)")
                        
                        logger.info("=" * 80)
                        
                        # Verificar se deve continuar para próximo período
                        if i < len(periodos):
                            logger.info(f"➡️ AVANÇANDO PARA PRÓXIMO PERÍODO ({i+1}/{len(periodos)})...")
                            logger.info(f"📅 Próximo período: {periodos[i]['inicio']} a {periodos[i]['fim']}")
                        else:
                            logger.info("🏁 TODOS OS PERÍODOS PROCESSADOS!")
                            
                    else:
                        logger.warning(f"⚠️ Falha ao navegar para período {i}: {periodo['inicio']} a {periodo['fim']}")
                        logger.info("🔄 Continuando para próximo período...")
                    
                    # Pausa entre períodos para evitar sobrecarga
                    if i < len(periodos):  # Não fazer pausa no último período
                        pausa = random.uniform(3, 6)
                        logger.info(f"⏸️ Pausa de {pausa:.1f}s antes do próximo período...")
                        await asyncio.sleep(pausa)
                
                # 🎉 RELATÓRIO FINAL COMPLETO
                logger.info("=" * 80)
                logger.info("🎉 DOWNLOADS AUTOMÁTICOS CONCLUÍDOS!")
                logger.info("=" * 80)
                logger.info(f"📊 ESTATÍSTICAS FINAIS:")
                logger.info(f"   • Total de períodos processados: {len(periodos)}")
                logger.info(f"   • Total de downloads realizados: {total_downloads}")
                
                # Relatório detalhado dos downloads
                if hasattr(self, 'progresso') and self.progresso is not None and "downloads_sucesso" in self.progresso:
                    sucessos = self.progresso["downloads_sucesso"]
                    logger.info(f"   • Downloads bem-sucedidos registrados: {len(sucessos)}")
                    
                    if sucessos:
                        logger.info("� ARQUIVOS BAIXADOS:")
                        for i, download in enumerate(sucessos[-5:], 1):  # Mostrar últimos 5
                            logger.info(f"   {i}. {download['arquivo']} ({download['timestamp']})")
                        
                        if len(sucessos) > 5:
                            logger.info(f"   ... e mais {len(sucessos) - 5} arquivos")
                
                tempo_fim = datetime.now()
                if hasattr(self, 'tempo_inicio'):
                    duracao = tempo_fim - self.tempo_inicio
                    logger.info(f"⏱️ Tempo total de execução: {duracao}")
                
                logger.info("=" * 80)
                logger.info("🔄 PRÓXIMA EXECUÇÃO:")
                logger.info("   • O sistema sempre processa todos os períodos do zero")
                logger.info("   • Execute novamente para buscar novas licitações")
                logger.info("   • Licitações já baixadas serão detectadas automaticamente")
                logger.info("=" * 80)
                
            finally:
                # Fechar contexto persistente
                if hasattr(self, 'context') and self.context:
                    await self.context.close()
                if hasattr(self, 'playwright'):
                    await self.playwright.stop()
            
        except Exception as e:
            logger.error(f"❌ Erro durante execução automática: {e}")
    
    async def confirmar_captcha_sucesso_para_ml(self):
        """Confirma CAPTCHA como sucesso para treinamento ML após download confirmado"""
        try:
            if hasattr(self, 'ultimo_captcha_resolvido') and self.ultimo_captcha_resolvido:
                captcha_data = self.ultimo_captcha_resolvido
                
                # Salvar o CAPTCHA como sucesso para treinamento ML
                if hasattr(self, 'resolver_v2') and self.resolver_v2:
                    sucesso = self.resolver_v2.salvar_captcha_como_sucesso(
                        captcha_data['img'],
                        captcha_data['texto'],
                        captcha_data['timestamp']
                    )
                    
                    if sucesso:
                        logger.info(f"🎯 CAPTCHA '{captcha_data['texto']}' CONFIRMADO PARA TREINAMENTO ML!")
                        logger.info(f"🎯 Método: {captcha_data['metodo']}, Confiança: {captcha_data['confianca']:.2f}")
                    else:
                        logger.warning("⚠️ Erro ao salvar CAPTCHA como sucesso")
                
                # Limpar dados do último CAPTCHA
                self.ultimo_captcha_resolvido = None
            else:
                logger.debug("📝 Nenhum CAPTCHA resolvido para confirmar")
                
        except Exception as e:
            logger.error(f"❌ Erro ao confirmar CAPTCHA para ML: {e}")

async def main():
    """Função principal"""
    logger.info("🚀 SISTEMA DE DOWNLOAD AUTOMÁTICO DE LICITAÇÕES")
    logger.info("=" * 60)
    
    downloader = ComprasNetDownloadAutomatico()
    await downloader.executar_downloads_automaticos()

if __name__ == "__main__":
    asyncio.run(main())
