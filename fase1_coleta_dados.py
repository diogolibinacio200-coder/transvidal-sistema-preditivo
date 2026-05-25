"""
=============================================================================
FASE 1 — ENGENHARIA DE DADOS
Sistema Preditivo de Gargalos Portuários | Vidal Transportes
=============================================================================

Módulos:
  1. ScraperLineup     → Coleta o line-up de navios (Paranaguá + Santos)
  2. ColetaClima       → Coleta previsão do tempo via OpenWeatherMap
  3. ColetaTelemetria  → Puxa posição dos caminhões via SQL (frota Vidal)
  4. Orquestrador      → Executa tudo e salva no banco PostgreSQL
  5. Agendador         → Roda automaticamente toda madrugada (APScheduler)

Dependências:
  pip install requests beautifulsoup4 selenium pandas psycopg2-binary
              sqlalchemy apscheduler python-dotenv webdriver-manager

Configuração:
  Crie um arquivo .env na mesma pasta com:
    OPENWEATHER_API_KEY=sua_chave_aqui
    DB_URL=postgresql://usuario:senha@host:5432/vidal_portos
    DB_FROTA_URL=postgresql://usuario:senha@host_frota:5432/telemetria
=============================================================================
"""

import os
import time
import logging
from datetime import datetime, timedelta

import requests
import pandas as pd
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, text
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

# Selenium (usado como fallback se o site renderizar com JS)
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURAÇÕES GLOBAIS
# ---------------------------------------------------------------------------

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "SUA_CHAVE_AQUI")
DB_URL = os.getenv("DB_URL", "postgresql://user:pass@localhost:5432/vidal_portos")
DB_FROTA_URL = os.getenv("DB_FROTA_URL", "postgresql://user:pass@localhost:5432/telemetria")

# Coordenadas dos portos para consulta de clima
# =============================================================================
# MAPA COMPLETO DE PORTOS E TERMINAIS — BRASIL
# Todos os pontos onde a Transvidal opera ou monitora
# =============================================================================

PORTOS = {
    # ── SUL ──────────────────────────────────────────────────────────────────
    "paranagua": {
        "nome": "Paranaguá", "uf": "PR", "regiao": "Sul",
        "lat": -25.5169, "lon": -48.5133,
        "tipo": "porto",
        "principais_cargas": ["soja", "milho", "açúcar", "fertilizantes"],
        "url_lineup": "https://www.appaweb.appa.pr.gov.br/appaweb/pesquisa.aspx?WCI=relLineUpNavios",
    },
    "rio_grande": {
        "nome": "Rio Grande", "uf": "RS", "regiao": "Sul",
        "lat": -32.0350, "lon": -52.0986,
        "tipo": "porto",
        "principais_cargas": ["soja", "milho", "óleo vegetal", "arroz"],
        "url_lineup": "https://www.portoriogrande.com.br/site/operacao_lineup_navios.php",
    },
    "itajai": {
        "nome": "Itajaí", "uf": "SC", "regiao": "Sul",
        "lat": -26.9078, "lon": -48.6700,
        "tipo": "porto",
        "principais_cargas": ["contêineres", "frigorificados", "papel"],
        "url_lineup": "https://www.portoitajai.com.br",
    },
    "sao_francisco_sul": {
        "nome": "São Francisco do Sul", "uf": "SC", "regiao": "Sul",
        "lat": -26.2430, "lon": -48.6350,
        "tipo": "porto",
        "principais_cargas": ["grãos", "fertilizantes", "combustíveis"],
        "url_lineup": "https://www.apsfs.sc.gov.br",
    },
    "imbituba": {
        "nome": "Imbituba", "uf": "SC", "regiao": "Sul",
        "lat": -28.2330, "lon": -48.6560,
        "tipo": "porto",
        "principais_cargas": ["carvão", "potássio", "contêineres"],
        "url_lineup": "https://www.portodeimbituba.com.br",
    },
    # ── SUDESTE ──────────────────────────────────────────────────────────────
    "santos": {
        "nome": "Santos", "uf": "SP", "regiao": "Sudeste",
        "lat": -23.9608, "lon": -46.3336,
        "tipo": "porto",
        "principais_cargas": ["soja", "milho", "açúcar", "contêineres", "combustíveis"],
        "url_lineup": "https://www.portodesantos.com.br/informacoes-operacionais/navios-no-porto/",
    },
    "rio_de_janeiro": {
        "nome": "Rio de Janeiro", "uf": "RJ", "regiao": "Sudeste",
        "lat": -22.8942, "lon": -43.1729,
        "tipo": "porto",
        "principais_cargas": ["contêineres", "combustíveis", "cargas gerais"],
        "url_lineup": "https://www.portosrio.gov.br",
    },
    "vitoria": {
        "nome": "Vitória / VALE", "uf": "ES", "regiao": "Sudeste",
        "lat": -20.3222, "lon": -40.3381,
        "tipo": "porto",
        "principais_cargas": ["minério de ferro", "pellets", "celulose", "grãos"],
        "url_lineup": "https://www.codesa.gov.br",
    },
    "angra_dos_reis": {
        "nome": "Angra dos Reis", "uf": "RJ", "regiao": "Sudeste",
        "lat": -23.0067, "lon": -44.3183,
        "tipo": "terminal",
        "principais_cargas": ["combustíveis", "granéis líquidos"],
        "url_lineup": "",
    },
    # ── NORDESTE ─────────────────────────────────────────────────────────────
    "suape": {
        "nome": "Suape", "uf": "PE", "regiao": "Nordeste",
        "lat": -8.4030, "lon": -34.9700,
        "tipo": "porto",
        "principais_cargas": ["contêineres", "combustíveis", "grãos"],
        "url_lineup": "https://www.suape.pe.gov.br",
    },
    "fortaleza": {
        "nome": "Fortaleza", "uf": "CE", "regiao": "Nordeste",
        "lat": -3.7160, "lon": -38.4830,
        "tipo": "porto",
        "principais_cargas": ["granéis sólidos", "combustíveis", "contêineres"],
        "url_lineup": "https://www.portodoceara.com.br",
    },
    "salvador": {
        "nome": "Salvador", "uf": "BA", "regiao": "Nordeste",
        "lat": -12.9714, "lon": -38.5124,
        "tipo": "porto",
        "principais_cargas": ["veículos", "contêineres", "granéis"],
        "url_lineup": "https://www.codeba.com.br",
    },
    "itaqui": {
        "nome": "Itaqui (São Luís)", "uf": "MA", "regiao": "Nordeste",
        "lat": -2.5847, "lon": -44.3628,
        "tipo": "porto",
        "principais_cargas": ["soja", "milho", "minério", "combustíveis"],
        "url_lineup": "https://www.portodoitaqui.ma.gov.br",
    },
    # ── NORTE (ARCO NORTE) ───────────────────────────────────────────────────
    "vila_do_conde": {
        "nome": "Vila do Conde (Barcarena)", "uf": "PA", "regiao": "Norte",
        "lat": -1.5167, "lon": -48.6333,
        "tipo": "porto",
        "principais_cargas": ["soja", "milho", "alumínio", "caulim"],
        "url_lineup": "https://www.portodovilladoconde.com.br",
    },
    "santarem": {
        "nome": "Santarém (Cargill)", "uf": "PA", "regiao": "Norte",
        "lat": -2.4430, "lon": -54.7080,
        "tipo": "terminal",
        "principais_cargas": ["soja", "milho"],
        "url_lineup": "",
    },
    "belem": {
        "nome": "Belém", "uf": "PA", "regiao": "Norte",
        "lat": -1.4558, "lon": -48.4902,
        "tipo": "porto",
        "principais_cargas": ["grãos", "madeira", "contêineres"],
        "url_lineup": "https://www.cdp.com.br",
    },
    "porto_velho": {
        "nome": "Porto Velho", "uf": "RO", "regiao": "Norte",
        "lat": -8.7612, "lon": -63.9004,
        "tipo": "terminal_hidroviario",
        "principais_cargas": ["soja", "milho", "combustíveis"],
        "url_lineup": "",
    },
    # ── TERMINAIS INTERIORES (SILOS / TRADINGS) ──────────────────────────────
    "rondonopolis": {
        "nome": "Rondonópolis (MT)", "uf": "MT", "regiao": "Centro-Oeste",
        "lat": -16.4726, "lon": -54.6358,
        "tipo": "terminal_interior",
        "principais_cargas": ["soja", "milho", "algodão"],
        "url_lineup": "",
    },
    "sorriso": {
        "nome": "Sorriso (MT)", "uf": "MT", "regiao": "Centro-Oeste",
        "lat": -12.5438, "lon": -55.7212,
        "tipo": "terminal_interior",
        "principais_cargas": ["soja", "milho"],
        "url_lineup": "",
    },
    "ponta_grossa": {
        "nome": "Ponta Grossa (PR)", "uf": "PR", "regiao": "Sul",
        "lat": -25.0945, "lon": -50.1633,
        "tipo": "terminal_interior",
        "principais_cargas": ["soja", "milho", "fertilizantes"],
        "url_lineup": "",
    },
    "maringa": {
        "nome": "Maringá (PR)", "uf": "PR", "regiao": "Sul",
        "lat": -23.4253, "lon": -51.9381,
        "tipo": "terminal_interior",
        "principais_cargas": ["soja", "milho", "açúcar"],
        "url_lineup": "",
    },
    "cascavel": {
        "nome": "Cascavel (PR)", "uf": "PR", "regiao": "Sul",
        "lat": -24.9578, "lon": -53.4595,
        "tipo": "terminal_interior",
        "principais_cargas": ["soja", "milho"],
        "url_lineup": "",
    },
    "ribeirao_preto": {
        "nome": "Ribeirão Preto (SP)", "uf": "SP", "regiao": "Sudeste",
        "lat": -21.1767, "lon": -47.8208,
        "tipo": "terminal_interior",
        "principais_cargas": ["açúcar", "etanol", "soja"],
        "url_lineup": "",
    },
    "uberlandia": {
        "nome": "Uberlândia (MG)", "uf": "MG", "regiao": "Sudeste",
        "lat": -18.9186, "lon": -48.2772,
        "tipo": "terminal_interior",
        "principais_cargas": ["soja", "milho", "granéis"],
        "url_lineup": "",
    },
    "rio_verde": {
        "nome": "Rio Verde (GO)", "uf": "GO", "regiao": "Centro-Oeste",
        "lat": -17.7983, "lon": -50.9269,
        "tipo": "terminal_interior",
        "principais_cargas": ["soja", "milho", "carne"],
        "url_lineup": "",
    },
}

# Rodovias federais monitoradas (corredores logísticos da Vidal)
BRS_MONITORADAS = {
    "br_163": {"nome": "BR-163", "trecho": "Cuiabá → Santarém", "extensao_km": 1780, "carga_principal": "soja"},
    "br_116": {"nome": "BR-116", "trecho": "Rio de Janeiro → Porto Alegre", "extensao_km": 4500, "carga_principal": "geral"},
    "br_101": {"nome": "BR-101", "trecho": "Touros/RN → São José do Norte/RS", "extensao_km": 4630, "carga_principal": "geral"},
    "br_364": {"nome": "BR-364", "trecho": "Limeira/SP → Porto Velho/RO", "extensao_km": 3100, "carga_principal": "soja/milho"},
    "br_153": {"nome": "BR-153 (Belém-Brasília)", "trecho": "Belém/PA → Anápolis/GO", "extensao_km": 2120, "carga_principal": "soja"},
    "br_060": {"nome": "BR-060", "trecho": "Brasília → Corumbá", "extensao_km": 1500, "carga_principal": "geral"},
    "br_376": {"nome": "BR-376", "trecho": "Apucarana → Garuva", "extensao_km": 400, "carga_principal": "soja/milho"},
    "br_277": {"nome": "BR-277", "trecho": "Paranaguá → Foz do Iguaçu", "extensao_km": 730, "carga_principal": "soja"},
}


# ---------------------------------------------------------------------------
# MÓDULO 1 — SCRAPER DE LINE-UP DE NAVIOS
# ---------------------------------------------------------------------------

class ScraperLineup:
    """
    Coleta a tabela de navios aguardando (line-up) nos portos.

    Estratégia:
      - Tenta primeiro com requests + BeautifulSoup (mais rápido).
      - Se o site usar JavaScript para renderizar a tabela, usa Selenium.
    """

    def _get_driver(self) -> webdriver.Chrome:
        """Retorna um driver Chrome headless (sem abrir janela)."""
        opts = Options()
        opts.add_argument("--headless")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=opts)

    # ------------------------------------------------------------------
    # Paranaguá — site usa ASP.NET, renderiza no servidor (sem JS)
    # ------------------------------------------------------------------
    def coletar_paranagua(self) -> pd.DataFrame:
        """
        Acessa o sistema APPA (Administração dos Portos de Paranaguá e Antonina)
        e extrai o line-up de navios aguardando atracação.

        Retorna DataFrame com colunas:
          porto, navio, bandeira, tipo_carga, tev (tonelagem),
          eta (chegada estimada), status, coletado_em
        """
        log.info("Coletando line-up de Paranaguá...")
        url = PORTOS["paranagua"]["url_lineup"]

        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # A tabela do APPA tem id específico — ajustar conforme inspeção do HTML
            tabela = soup.find("table", {"id": "dgLineUp"})
            if not tabela:
                # Fallback: pega a primeira tabela grande da página
                tabela = soup.find("table")

            if not tabela:
                log.warning("Tabela de line-up não encontrada em Paranaguá. Usando Selenium.")
                return self._coletar_paranagua_selenium()

            linhas = tabela.find_all("tr")
            dados = []
            for linha in linhas[1:]:  # Pula o cabeçalho
                cols = [td.get_text(strip=True) for td in linha.find_all("td")]
                if len(cols) >= 5:
                    dados.append({
                        "porto": "Paranaguá",
                        "navio": cols[0],
                        "bandeira": cols[1] if len(cols) > 1 else "",
                        "tipo_carga": cols[2] if len(cols) > 2 else "",
                        "tev": cols[3] if len(cols) > 3 else "",
                        "eta": cols[4] if len(cols) > 4 else "",
                        "status": cols[5] if len(cols) > 5 else "",
                        "coletado_em": datetime.now(),
                    })

            df = pd.DataFrame(dados)
            log.info(f"Paranaguá: {len(df)} navios coletados.")
            return df

        except Exception as e:
            log.error(f"Erro ao coletar Paranaguá via requests: {e}")
            return self._coletar_paranagua_selenium()

    def _coletar_paranagua_selenium(self) -> pd.DataFrame:
        """Fallback com Selenium para sites que renderizam tabelas via JavaScript."""
        driver = self._get_driver()
        dados = []
        try:
            driver.get(PORTOS["paranagua"]["url_lineup"])
            # Aguarda a tabela aparecer na tela (timeout 20s)
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            soup = BeautifulSoup(driver.page_source, "html.parser")
            tabela = soup.find("table")
            if tabela:
                for linha in tabela.find_all("tr")[1:]:
                    cols = [td.get_text(strip=True) for td in linha.find_all("td")]
                    if len(cols) >= 5:
                        dados.append({
                            "porto": "Paranaguá",
                            "navio": cols[0],
                            "bandeira": cols[1] if len(cols) > 1 else "",
                            "tipo_carga": cols[2] if len(cols) > 2 else "",
                            "tev": cols[3] if len(cols) > 3 else "",
                            "eta": cols[4] if len(cols) > 4 else "",
                            "status": cols[5] if len(cols) > 5 else "",
                            "coletado_em": datetime.now(),
                        })
        finally:
            driver.quit()

        df = pd.DataFrame(dados)
        log.info(f"Paranaguá (Selenium): {len(df)} navios coletados.")
        return df

    # ------------------------------------------------------------------
    # Santos — site público, estrutura HTML diferente
    # ------------------------------------------------------------------
    def coletar_santos(self) -> pd.DataFrame:
        """
        Coleta o line-up do Porto de Santos.
        O site de Santos pode mudar layout — inspecionar o HTML e ajustar
        o seletor CSS conforme necessário.
        """
        log.info("Coletando line-up de Santos...")
        url = PORTOS["santos"]["url_lineup"]

        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            tabela = soup.find("table")
            dados = []
            if tabela:
                for linha in tabela.find_all("tr")[1:]:
                    cols = [td.get_text(strip=True) for td in linha.find_all("td")]
                    if len(cols) >= 3:
                        dados.append({
                            "porto": "Santos",
                            "navio": cols[0],
                            "bandeira": "",
                            "tipo_carga": cols[1] if len(cols) > 1 else "",
                            "tev": "",
                            "eta": cols[2] if len(cols) > 2 else "",
                            "status": cols[3] if len(cols) > 3 else "",
                            "coletado_em": datetime.now(),
                        })

            df = pd.DataFrame(dados)
            log.info(f"Santos: {len(df)} navios coletados.")
            return df

        except Exception as e:
            log.error(f"Erro ao coletar Santos: {e}")
            return pd.DataFrame()

    def coletar_todos(self) -> pd.DataFrame:
        """Agrega os line-ups de todos os portos em um único DataFrame."""
        frames = [self.coletar_paranagua(), self.coletar_santos()]
        df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
        return df


# ---------------------------------------------------------------------------
# MÓDULO 2 — COLETA DE CLIMA (OpenWeatherMap)
# ---------------------------------------------------------------------------

class ColetaClima:
    """
    Consulta a API do OpenWeatherMap para obter previsão de 5 dias
    (de 3 em 3 horas) para cada porto.

    Documentação da API: https://openweathermap.org/forecast5
    Plano gratuito: 1.000 chamadas/dia — suficiente para o MVP.
    """

    BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"

    def coletar_previsao(self, porto_key: str) -> pd.DataFrame:
        """
        Retorna DataFrame com previsão horária de:
          chuva_mm, temperatura, umidade, velocidade_vento, descricao
        para os próximos 5 dias no porto informado.
        """
        porto = PORTOS[porto_key]
        log.info(f"Coletando clima para {porto['nome']}...")

        params = {
            "lat": porto["lat"],
            "lon": porto["lon"],
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
            "lang": "pt_br",
        }

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            registros = []
            for item in data.get("list", []):
                registros.append({
                    "porto": porto["nome"],
                    "data_hora": datetime.fromtimestamp(item["dt"]),
                    "temperatura": item["main"]["temp"],
                    "umidade": item["main"]["humidity"],
                    "velocidade_vento": item["wind"]["speed"],
                    # Chuva pode não existir se não houver precipitação
                    "chuva_mm_3h": item.get("rain", {}).get("3h", 0.0),
                    "descricao": item["weather"][0]["description"],
                    "coletado_em": datetime.now(),
                })

            df = pd.DataFrame(registros)
            log.info(f"Clima {porto['nome']}: {len(df)} registros coletados.")
            return df

        except Exception as e:
            log.error(f"Erro ao coletar clima para {porto['nome']}: {e}")
            return pd.DataFrame()

    def coletar_todos(self) -> pd.DataFrame:
        frames = [self.coletar_previsao(k) for k in PORTOS]
        return pd.concat([f for f in frames if not f.empty], ignore_index=True)


# ---------------------------------------------------------------------------
# MÓDULO 3 — TELEMETRIA DA FROTA VIDAL (SQL interno)
# ---------------------------------------------------------------------------

class ColetaTelemetria:
    """
    Puxa do banco interno da Vidal os dados de telemetria dos caminhões.

    Supõe que a Vidal já tem um banco com tabela de posições dos caminhões.
    Ajustar query conforme o schema real do banco da empresa.

    Métricas coletadas:
      - Quantos caminhões estão a menos de 200 km de cada porto (em rota)
      - Quantos já estão na fila (parados próximo ao porto há > 1 hora)
    """

    def __init__(self):
        try:
            self.engine = create_engine(DB_FROTA_URL)
        except Exception as e:
            log.warning(f"Banco de telemetria indisponível: {e}")
            self.engine = None

    def coletar_caminhoes_em_rota(self) -> pd.DataFrame:
        """
        Retorna caminhões que estão se dirigindo para cada porto.

        Ajuste a query conforme o schema real da Vidal.
        """
        if not self.engine:
            log.warning("Sem conexão com banco de telemetria. Retornando DataFrame vazio.")
            return pd.DataFrame()

        query = text("""
            SELECT
                destino_porto,
                COUNT(*) AS caminhoes_em_rota,
                AVG(distancia_km_porto) AS distancia_media_km,
                SUM(CASE WHEN velocidade_kmh < 5 AND distancia_km_porto < 20
                         THEN 1 ELSE 0 END) AS caminhoes_na_fila,
                NOW() AS coletado_em
            FROM telemetria_posicoes
            WHERE
                data_hora >= NOW() - INTERVAL '2 hours'
                AND destino_porto IN ('Paranaguá', 'Santos')
            GROUP BY destino_porto
        """)

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)
            log.info(f"Telemetria: {len(df)} portos com dados de frota.")
            return df
        except Exception as e:
            log.error(f"Erro ao consultar telemetria: {e}")
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# MÓDULO 4 — SALVAR NO BANCO DE DADOS
# ---------------------------------------------------------------------------

class BancoDados:
    """Persiste os dados coletados no PostgreSQL."""

    def __init__(self):
        self.engine = create_engine(DB_URL)
        self._criar_tabelas()

    def _criar_tabelas(self):
        """Cria as tabelas se ainda não existirem."""
        ddl = """
        CREATE TABLE IF NOT EXISTS lineup_navios (
            id          SERIAL PRIMARY KEY,
            porto       VARCHAR(50),
            navio       VARCHAR(150),
            bandeira    VARCHAR(50),
            tipo_carga  VARCHAR(100),
            tev         VARCHAR(30),
            eta         VARCHAR(50),
            status      VARCHAR(80),
            coletado_em TIMESTAMP DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS previsao_clima (
            id                SERIAL PRIMARY KEY,
            porto             VARCHAR(50),
            data_hora         TIMESTAMP,
            temperatura       FLOAT,
            umidade           INT,
            velocidade_vento  FLOAT,
            chuva_mm_3h       FLOAT,
            descricao         VARCHAR(100),
            coletado_em       TIMESTAMP DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS telemetria_frota (
            id                    SERIAL PRIMARY KEY,
            destino_porto         VARCHAR(50),
            caminhoes_em_rota     INT,
            distancia_media_km    FLOAT,
            caminhoes_na_fila     INT,
            coletado_em           TIMESTAMP DEFAULT NOW()
        );
        """
        try:
            with self.engine.connect() as conn:
                for stmt in ddl.strip().split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        conn.execute(text(stmt))
                conn.commit()
            log.info("Tabelas verificadas/criadas com sucesso.")
        except Exception as e:
            log.error(f"Erro ao criar tabelas: {e}")

    def salvar(self, df: pd.DataFrame, tabela: str):
        """Insere o DataFrame na tabela informada."""
        if df.empty:
            log.warning(f"DataFrame vazio — nada salvo em '{tabela}'.")
            return
        try:
            df.to_sql(tabela, self.engine, if_exists="append", index=False)
            log.info(f"{len(df)} registros salvos em '{tabela}'.")
        except Exception as e:
            log.error(f"Erro ao salvar em '{tabela}': {e}")


# ---------------------------------------------------------------------------
# MÓDULO 5 — ORQUESTRADOR (coleta tudo de uma vez)
# ---------------------------------------------------------------------------

def executar_coleta():
    """
    Função principal que orquestra todos os coletores.
    Chamada pelo agendador toda madrugada ou manualmente via terminal.
    """
    log.info("=" * 60)
    log.info(f"INICIANDO COLETA — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    db = BancoDados()

    # 1. Line-up de navios
    log.info("--- Line-up de navios ---")
    lineup = ScraperLineup().coletar_todos()
    db.salvar(lineup, "lineup_navios")

    # 2. Clima
    log.info("--- Previsão climática ---")
    clima = ColetaClima().coletar_todos()
    db.salvar(clima, "previsao_clima")

    # 3. Telemetria
    log.info("--- Telemetria da frota ---")
    frota = ColetaTelemetria().coletar_caminhoes_em_rota()
    db.salvar(frota, "telemetria_frota")

    log.info("COLETA FINALIZADA.")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# MÓDULO 6 — AGENDADOR (roda toda madrugada automaticamente)
# ---------------------------------------------------------------------------

def iniciar_agendador():
    """
    Agenda a coleta para rodar:
      - Todo dia às 02:00 (linha-up completo)
      - A cada 6 horas (clima atualizado)

    Para rodar em produção:
      python fase1_coleta_dados.py
    """
    scheduler = BlockingScheduler(timezone="America/Sao_Paulo")

    # Coleta completa todo dia às 2h da manhã
    scheduler.add_job(
        executar_coleta,
        trigger="cron",
        hour=2,
        minute=0,
        id="coleta_diaria",
    )

    # Atualização de clima a cada 6 horas
    scheduler.add_job(
        lambda: BancoDados().salvar(ColetaClima().coletar_todos(), "previsao_clima"),
        trigger="interval",
        hours=6,
        id="atualizacao_clima",
    )

    log.info("Agendador iniciado. Pressione Ctrl+C para parar.")
    log.info("Próxima coleta completa: 02:00 (horário de Brasília)")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Agendador encerrado.")


# ---------------------------------------------------------------------------
# PONTO DE ENTRADA
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--agora":
        # Execução imediata para teste: python fase1_coleta_dados.py --agora
        executar_coleta()
    else:
        # Modo produção: agenda e fica rodando em background
        iniciar_agendador()
