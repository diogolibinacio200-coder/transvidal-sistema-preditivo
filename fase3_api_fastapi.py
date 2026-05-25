"""
=============================================================================
FASE 3 — DEPLOY E INTEGRAÇÃO (API REST)
Sistema Preditivo de Gargalos Portuários | Vidal Transportes
=============================================================================

Endpoints:
  GET  /health                   → Status da API e modelos carregados
  GET  /portos                   → Lista portos disponíveis
  GET  /previsao/{porto}         → Previsão usando dados do banco (produção)
  POST /previsao/{porto}/manual  → Previsão com features enviadas no body
  GET  /historico/{porto}        → Últimas 30 previsões geradas
  GET  /alerta/ativos            → Todos os alertas LARANJA ou VERMELHO

Rodar localmente:
  uvicorn fase3_api_fastapi:app --reload --port 8000

Documentação interativa:
  http://localhost:8000/docs

Deploy em produção (Gunicorn + Uvicorn):
  gunicorn fase3_api_fastapi:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

Variáveis de ambiente adicionais (.env):
  API_SECRET_KEY=chave_secreta_para_jwt
  ALERT_WEBHOOK_URL=https://hooks.slack.com/... (ou Teams, etc.)
=============================================================================
"""

import os
import json
import logging
import httpx
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pathlib import Path as FilePath
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# Importa as classes da Fase 2
from fase2_modelo_lstm import (
    PreditorOnline,
    CONFIG,
    FEATURES,
    MODELOS_DIR,
)

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

API_SECRET_KEY    = os.getenv("API_SECRET_KEY", "dev-key-trocar-em-producao")
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")

# ---------------------------------------------------------------------------
# INICIALIZAÇÃO DO FASTAPI
# ---------------------------------------------------------------------------

_api_key_scheme = APIKeyHeader(name="x-api-key", description="Chave de autenticação da API")

app = FastAPI(
    title       = "Vidal Portos — API Preditiva de Gargalos",
    description = (
        "Previsão de congestionamento portuário baseada em LSTM. "
        "Integra line-up de navios, clima e telemetria da frota Vidal."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # restringir em produção ao domínio do painel
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ---------------------------------------------------------------------------
# CACHE DE MODELOS (carregados uma vez no startup)
# ---------------------------------------------------------------------------

_modelos: dict[str, PreditorOnline] = {}
_historico_previsoes: dict[str, list] = {}  # cache simples em memória

# Portos ativos para predição (subset — todos com modelo treinado)
# Portos principais com volume de dados suficiente para LSTM
PORTOS_DISPONIVEIS = [
    # Sul
    "paranagua", "rio_grande", "itajai", "sao_francisco_sul",
    # Sudeste
    "santos", "rio_de_janeiro", "vitoria",
    # Nordeste
    "suape", "fortaleza", "itaqui",
    # Norte (Arco Norte)
    "vila_do_conde", "santarem",
    # Terminais interiores
    "rondonopolis", "ponta_grossa",
]

# Mapa completo de informações dos locais (para o dashboard)
MAPA_LOCAIS = {
    "paranagua":        {"nome":"Paranaguá","uf":"PR","regiao":"Sul","lat":-25.5169,"lon":-48.5133,"tipo":"porto","cargas":["soja","milho","açúcar"]},
    "rio_grande":       {"nome":"Rio Grande","uf":"RS","regiao":"Sul","lat":-32.0350,"lon":-52.0986,"tipo":"porto","cargas":["soja","milho","arroz"]},
    "itajai":           {"nome":"Itajaí","uf":"SC","regiao":"Sul","lat":-26.9078,"lon":-48.6700,"tipo":"porto","cargas":["contêineres","papel"]},
    "sao_francisco_sul":{"nome":"S. Francisco Sul","uf":"SC","regiao":"Sul","lat":-26.2430,"lon":-48.6350,"tipo":"porto","cargas":["grãos","fertilizantes"]},
    "imbituba":         {"nome":"Imbituba","uf":"SC","regiao":"Sul","lat":-28.2330,"lon":-48.6560,"tipo":"porto","cargas":["carvão","potássio"]},
    "santos":           {"nome":"Santos","uf":"SP","regiao":"Sudeste","lat":-23.9608,"lon":-46.3336,"tipo":"porto","cargas":["soja","milho","açúcar","contêineres"]},
    "rio_de_janeiro":   {"nome":"Rio de Janeiro","uf":"RJ","regiao":"Sudeste","lat":-22.8942,"lon":-43.1729,"tipo":"porto","cargas":["contêineres","combustíveis"]},
    "vitoria":          {"nome":"Vitória","uf":"ES","regiao":"Sudeste","lat":-20.3222,"lon":-40.3381,"tipo":"porto","cargas":["minério","celulose","grãos"]},
    "suape":            {"nome":"Suape","uf":"PE","regiao":"Nordeste","lat":-8.4030,"lon":-34.9700,"tipo":"porto","cargas":["contêineres","grãos"]},
    "fortaleza":        {"nome":"Fortaleza","uf":"CE","regiao":"Nordeste","lat":-3.7160,"lon":-38.4830,"tipo":"porto","cargas":["granéis sólidos","combustíveis"]},
    "salvador":         {"nome":"Salvador","uf":"BA","regiao":"Nordeste","lat":-12.9714,"lon":-38.5124,"tipo":"porto","cargas":["veículos","contêineres"]},
    "itaqui":           {"nome":"Itaqui (São Luís)","uf":"MA","regiao":"Nordeste","lat":-2.5847,"lon":-44.3628,"tipo":"porto","cargas":["soja","milho","minério"]},
    "vila_do_conde":    {"nome":"Vila do Conde","uf":"PA","regiao":"Norte","lat":-1.5167,"lon":-48.6333,"tipo":"porto","cargas":["soja","alumínio"]},
    "santarem":         {"nome":"Santarém","uf":"PA","regiao":"Norte","lat":-2.4430,"lon":-54.7080,"tipo":"terminal","cargas":["soja","milho"]},
    "belem":            {"nome":"Belém","uf":"PA","regiao":"Norte","lat":-1.4558,"lon":-48.4902,"tipo":"porto","cargas":["grãos","madeira"]},
    "porto_velho":      {"nome":"Porto Velho","uf":"RO","regiao":"Norte","lat":-8.7612,"lon":-63.9004,"tipo":"terminal_hidroviario","cargas":["soja","combustíveis"]},
    "rondonopolis":     {"nome":"Rondonópolis","uf":"MT","regiao":"Centro-Oeste","lat":-16.4726,"lon":-54.6358,"tipo":"terminal_interior","cargas":["soja","milho","algodão"]},
    "sorriso":          {"nome":"Sorriso","uf":"MT","regiao":"Centro-Oeste","lat":-12.5438,"lon":-55.7212,"tipo":"terminal_interior","cargas":["soja","milho"]},
    "ponta_grossa":     {"nome":"Ponta Grossa","uf":"PR","regiao":"Sul","lat":-25.0945,"lon":-50.1633,"tipo":"terminal_interior","cargas":["soja","fertilizantes"]},
    "maringa":          {"nome":"Maringá","uf":"PR","regiao":"Sul","lat":-23.4253,"lon":-51.9381,"tipo":"terminal_interior","cargas":["soja","milho"]},
    "cascavel":         {"nome":"Cascavel","uf":"PR","regiao":"Sul","lat":-24.9578,"lon":-53.4595,"tipo":"terminal_interior","cargas":["soja","milho"]},
    "ribeirao_preto":   {"nome":"Ribeirão Preto","uf":"SP","regiao":"Sudeste","lat":-21.1767,"lon":-47.8208,"tipo":"terminal_interior","cargas":["açúcar","etanol"]},
    "uberlandia":       {"nome":"Uberlândia","uf":"MG","regiao":"Sudeste","lat":-18.9186,"lon":-48.2772,"tipo":"terminal_interior","cargas":["soja","grãos"]},
    "rio_verde":        {"nome":"Rio Verde","uf":"GO","regiao":"Centro-Oeste","lat":-17.7983,"lon":-50.9269,"tipo":"terminal_interior","cargas":["soja","milho","carne"]},
}

BRS_MONITORADAS = {
    "br_163":{"nome":"BR-163","trecho":"Cuiabá → Santarém","extensao_km":1780,"carga":"soja"},
    "br_116":{"nome":"BR-116","trecho":"RJ → Porto Alegre","extensao_km":4500,"carga":"geral"},
    "br_101":{"nome":"BR-101","trecho":"RN → RS (litorânea)","extensao_km":4630,"carga":"geral"},
    "br_364":{"nome":"BR-364","trecho":"SP → Porto Velho","extensao_km":3100,"carga":"soja/milho"},
    "br_153":{"nome":"BR-153","trecho":"Belém → Anápolis","extensao_km":2120,"carga":"soja"},
    "br_277":{"nome":"BR-277","trecho":"Paranaguá → Foz do Iguaçu","extensao_km":730,"carga":"soja"},
    "br_376":{"nome":"BR-376","trecho":"Apucarana → Garuva","extensao_km":400,"carga":"soja/milho"},
    "br_060":{"nome":"BR-060","trecho":"Brasília → Corumbá","extensao_km":1500,"carga":"geral"},
}


@app.on_event("startup")
async def carregar_modelos():
    """Carrega todos os modelos LSTM na inicialização da API."""
    for porto in PORTOS_DISPONIVEIS:
        meta_path = MODELOS_DIR / f"meta_{porto}.json"
        if meta_path.exists():
            try:
                _modelos[porto]            = PreditorOnline(porto)
                _historico_previsoes[porto] = []
                log.info(f"✓ Modelo '{porto}' carregado.")
            except Exception as e:
                log.warning(f"✗ Falha ao carregar modelo '{porto}': {e}")
        else:
            log.warning(f"✗ Modelo '{porto}' não encontrado — treine antes de usar a API.")


# ---------------------------------------------------------------------------
# SCHEMAS PYDANTIC (contrato da API)
# ---------------------------------------------------------------------------

class FeaturesDia(BaseModel):
    """Features de um único dia para previsão manual."""
    navios_aguardando       : int   = Field(..., ge=0,   le=200,  example=18)
    chuva_mm_acumulada_24h  : float = Field(..., ge=0.0, le=200.0, example=12.5)
    chuva_mm_prev_72h       : float = Field(..., ge=0.0, le=300.0, example=45.0)
    velocidade_vento_media  : float = Field(..., ge=0.0, le=80.0,  example=18.0)
    caminhoes_em_rota       : int   = Field(..., ge=0,   le=1000, example=130)
    caminhoes_na_fila       : int   = Field(..., ge=0,   le=500,  example=22)
    data                    : date  = Field(..., example="2025-08-15")


class PrevisaoManualRequest(BaseModel):
    """Body para POST /previsao/{porto}/manual"""
    historico: list[FeaturesDia] = Field(
        ...,
        min_length = CONFIG["janela_entrada"],
        description=f"Lista de {CONFIG['janela_entrada']} dias de histórico, em ordem cronológica.",
    )

    @field_validator("historico")
    @classmethod
    def validar_ordem_cronologica(cls, v):
        datas = [d.data for d in v]
        if datas != sorted(datas):
            raise ValueError("Os registros de 'historico' devem estar em ordem cronológica.")
        return v


class DiaPrevisto(BaseModel):
    data          : str
    horas_espera  : float
    descricao     : str


class PrevisaoResponse(BaseModel):
    porto          : str
    gerado_em      : str
    horizonte_dias : int
    previsao       : list[DiaPrevisto]
    alerta         : str   # VERDE | AMARELO | LARANJA | VERMELHO
    risco_pct      : float
    modelo_mae_h   : Optional[float]
    recomendacao   : str


class AlertaAtivo(BaseModel):
    porto      : str
    alerta     : str
    risco_pct  : float
    max_horas  : float
    gerado_em  : str


# ---------------------------------------------------------------------------
# AUTENTICAÇÃO SIMPLES (token estático — substituir por JWT em produção)
# ---------------------------------------------------------------------------

def verificar_token(x_api_key: str = Security(_api_key_scheme)):
    if x_api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Token inválido.")
    return x_api_key


# ---------------------------------------------------------------------------
# UTILITÁRIOS
# ---------------------------------------------------------------------------

def _gerar_recomendacao(alerta: str, porto: str, previsao: list[dict]) -> str:
    """Gera texto de recomendação operacional baseado no nível de alerta."""
    max_dia = max(previsao, key=lambda x: x["horas_espera"])
    porto_nome = porto.capitalize()

    if alerta == "VERDE":
        return f"Porto de {porto_nome} operando normalmente. Sem ações necessárias."

    elif alerta == "AMARELO":
        return (
            f"Atenção: fila prevista de até {max_dia['horas_espera']:.0f}h em {porto_nome} "
            f"em {max_dia['data']}. Monitorar situação e acionar despachantes."
        )

    elif alerta == "LARANJA":
        return (
            f"ALERTA: congestionamento significativo previsto em {porto_nome} "
            f"({max_dia['horas_espera']:.0f}h em {max_dia['data']}). "
            "Recomendar renegociação de janelas de entrega com as tradings."
        )

    else:  # VERMELHO
        return (
            f"🚨 CRÍTICO: travamento iminente em {porto_nome} "
            f"({max_dia['horas_espera']:.0f}h previstas em {max_dia['data']}). "
            "Redirecionar caminhões imediatamente para silos pulmão. "
            "Acionar diretoria e renegociar contratos de demurrage."
        )


def _features_para_dataframe(historico: list[FeaturesDia]) -> pd.DataFrame:
    """Converte lista de FeaturesDia para DataFrame indexado por data."""
    records = [h.model_dump() for h in historico]
    df = pd.DataFrame(records)
    df["data"] = pd.to_datetime(df["data"])
    df.set_index("data", inplace=True)
    # Remover 'data' duplicada se houver
    df = df[[c for c in df.columns if c in FEATURES or c in [
        "navios_aguardando", "chuva_mm_acumulada_24h", "chuva_mm_prev_72h",
        "velocidade_vento_media", "caminhoes_em_rota", "caminhoes_na_fila",
    ]]]
    return df


async def _enviar_webhook_alerta(resultado: dict):
    """Envia alerta crítico para Slack/Teams via webhook (fire-and-forget)."""
    if not ALERT_WEBHOOK_URL or resultado["alerta"] not in ("LARANJA", "VERMELHO"):
        return

    emoji = "🟠" if resultado["alerta"] == "LARANJA" else "🔴"
    payload = {
        "text": (
            f"{emoji} *ALERTA PORTUÁRIO — {resultado['porto'].upper()}*\n"
            f"Risco: {resultado['risco_pct']}% | Nível: {resultado['alerta']}\n"
            f"{resultado['recomendacao']}"
        )
    }

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(ALERT_WEBHOOK_URL, json=payload)
    except Exception as e:
        log.warning(f"Falha ao enviar webhook: {e}")


def _obter_preditor(porto: str) -> PreditorOnline:
    """Retorna o preditor carregado ou lança 404/503."""
    if porto not in PORTOS_DISPONIVEIS:
        raise HTTPException(
            status_code=404,
            detail=f"Porto '{porto}' não reconhecido. Disponíveis: {PORTOS_DISPONIVEIS}",
        )
    if porto not in _modelos:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo para '{porto}' não treinado. Execute: python fase2_modelo_lstm.py --treinar --porto {porto}",
        )
    return _modelos[porto]


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["Infraestrutura"], include_in_schema=False)
async def dashboard():
    """Painel visual da Torre de Controle Transvidal."""
    # Tenta encontrar o dashboard.html relativo ao arquivo Python ou ao CWD
    for candidate in [
        FilePath(__file__).parent / "dashboard.html",
        FilePath("dashboard.html"),
        FilePath("/app/dashboard.html"),
    ]:
        if candidate.exists():
            return HTMLResponse(content=candidate.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1 style='font-family:sans-serif;padding:40px;color:#ef4444'>dashboard.html não encontrado</h1>")

@app.get("/health", tags=["Infraestrutura"])
async def health():
    """Verifica se a API e os modelos estão operacionais."""
    return {
        "status"          : "ok",
        "timestamp"       : datetime.now().isoformat(),
        "modelos_ativos"  : list(_modelos.keys()),
        "modelos_ausentes": [p for p in PORTOS_DISPONIVEIS if p not in _modelos],
        "pytorch_version" : torch.__version__,
    }


@app.get("/portos", tags=["Portos"])
async def listar_portos():
    """Retorna lista de portos disponíveis e status de cada modelo."""
    resultado = []
    for porto in PORTOS_DISPONIVEIS:
        meta_path = MODELOS_DIR / f"meta_{porto}.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            resultado.append({
                "porto"       : porto,
                "disponivel"  : porto in _modelos,
                "treinado_em" : meta.get("treinado_em", "")[:10],
                "mae_horas"   : meta.get("metricas_teste", {}).get("MAE"),
            })
        else:
            resultado.append({"porto": porto, "disponivel": False, "treinado_em": None})
    return resultado


@app.get("/mapa", tags=["Portos"])
async def mapa_locais():
    """Retorna todos os locais monitorados no Brasil com coordenadas e tipo."""
    return {"locais": MAPA_LOCAIS, "total": len(MAPA_LOCAIS)}


@app.get("/brs", tags=["Portos"])
async def brs_monitoradas():
    """Retorna as rodovias federais monitoradas pela Vidal."""
    return {"rodovias": BRS_MONITORADAS, "total": len(BRS_MONITORADAS)}


@app.get(
    "/previsao/{porto}",
    response_model=PrevisaoResponse,
    tags=["Previsão"],
)
async def previsao_automatica(
    porto: str,
    background_tasks: BackgroundTasks,
    _: str = Depends(verificar_token),
):
    """
    **Endpoint de produção**: busca automaticamente os últimos
    `janela_entrada` dias do banco e retorna a previsão para os próximos
    `horizonte` dias.

    Requer que a Fase 1 (coleta de dados) esteja rodando regularmente.
    """
    preditor = _obter_preditor(porto)

    # Busca dados recentes do banco via Extrator da Fase 2
    try:
        from fase2_modelo_lstm import Extrator
        df = Extrator(porto).extrair()
        df_recente = df.tail(CONFIG["janela_entrada"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar dados do banco: {e}")

    try:
        resultado = preditor.prever(df_recente)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na previsão: {e}")

    resultado["recomendacao"] = _gerar_recomendacao(
        resultado["alerta"], porto, resultado["previsao"]
    )

    # Salva no histórico em memória
    _historico_previsoes[porto].append(resultado)
    _historico_previsoes[porto] = _historico_previsoes[porto][-30:]  # mantém 30

    # Webhook assíncrono (não bloqueia a resposta)
    background_tasks.add_task(_enviar_webhook_alerta, resultado)

    return resultado


@app.post(
    "/previsao/{porto}/manual",
    response_model=PrevisaoResponse,
    tags=["Previsão"],
)
async def previsao_manual(
    porto: str,
    body: PrevisaoManualRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(verificar_token),
):
    """
    **Endpoint manual**: recebe features diretamente no body da requisição.

    Útil para simulações (ex: "o que acontece se chover 80mm amanhã?")
    ou quando o banco ainda não tem dados suficientes.

    Exemplo de body:
    ```json
    {
      "historico": [
        {
          "data": "2025-08-09",
          "navios_aguardando": 25,
          "chuva_mm_acumulada_24h": 0,
          "chuva_mm_prev_72h": 60,
          "velocidade_vento_media": 20,
          "caminhoes_em_rota": 180,
          "caminhoes_na_fila": 35
        }
        // ... 6 dias mais
      ]
    }
    ```
    """
    preditor = _obter_preditor(porto)
    df = _features_para_dataframe(body.historico)

    try:
        resultado = preditor.prever(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Erro na previsão: {e}")

    resultado["recomendacao"] = _gerar_recomendacao(
        resultado["alerta"], porto, resultado["previsao"]
    )

    background_tasks.add_task(_enviar_webhook_alerta, resultado)
    return resultado


@app.get("/historico/{porto}", tags=["Histórico"])
async def historico_previsoes(
    porto: str,
    limite: int = 10,
    _: str = Depends(verificar_token),
):
    """
    Retorna as últimas `limite` previsões geradas para o porto.
    (Cache em memória — reiniciado com a API. Para persistência, salvar no banco.)
    """
    if porto not in PORTOS_DISPONIVEIS:
        raise HTTPException(status_code=404, detail=f"Porto '{porto}' não reconhecido.")

    historico = _historico_previsoes.get(porto, [])
    return {
        "porto"     : porto,
        "total"     : len(historico),
        "previsoes" : historico[-limite:][::-1],  # mais recentes primeiro
    }


@app.get("/alerta/ativos", response_model=list[AlertaAtivo], tags=["Alertas"])
async def alertas_ativos(_: str = Depends(verificar_token)):
    """
    Retorna alertas LARANJA ou VERMELHO em todos os portos monitorados.
    Endpoint ideal para o painel da Torre de Controle da Vidal.
    """
    alertas = []
    for porto, historico in _historico_previsoes.items():
        if not historico:
            continue
        ultima = historico[-1]
        if ultima["alerta"] in ("LARANJA", "VERMELHO"):
            alertas.append(AlertaAtivo(
                porto      = porto,
                alerta     = ultima["alerta"],
                risco_pct  = ultima["risco_pct"],
                max_horas  = max(d["horas_espera"] for d in ultima["previsao"]),
                gerado_em  = ultima["gerado_em"],
            ))

    return sorted(alertas, key=lambda x: x.risco_pct, reverse=True)


# ---------------------------------------------------------------------------
# HANDLER DE ERROS GLOBAL
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def handler_geral(request, exc):
    log.error(f"Erro não tratado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Erro interno. Verifique os logs da API."},
    )


# ---------------------------------------------------------------------------
# PONTO DE ENTRADA (dev)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fase3_api_fastapi:app", host="0.0.0.0", port=8000, reload=True)
