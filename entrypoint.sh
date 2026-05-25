#!/bin/bash
# =============================================================================
# entrypoint.sh — Sistema Preditivo Transvidal
# =============================================================================
# Ordem:
#   1. Aguarda PostgreSQL (pula se DB_URL não apontar para host 'postgres')
#   2. Inicializa tabelas
#   3. Treina 17 modelos LSTM com dados sintéticos (se banco vazio)
#   4. Sobe API REST + serve landing page em /
# =============================================================================

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${BLUE}[BOOT]${NC} $1"; }
ok()   { echo -e "${GREEN}[OK]${NC}   $1"; }
warn() { echo -e "${YELLOW}[AVISO]${NC} $1"; }
err()  { echo -e "${RED}[ERRO]${NC}  $1"; }

echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  🚢  Sistema Preditivo Transvidal                          ${NC}"
echo -e "${BOLD}       Torre de Controle Logística                          ${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""

# --- Passo 1: PostgreSQL (somente se DB aponta para host 'postgres') ---------
if echo "$DB_URL" | grep -q "@postgres"; then
    log "Aguardando PostgreSQL..."
    DB_HOST=$(echo "$DB_URL" | sed 's/.*@\([^:]*\).*/\1/')
    DB_PORT=$(echo "$DB_URL" | sed 's/.*:\([0-9]*\)\/.*/\1/')
    MAX=30; T=0
    until nc -z "$DB_HOST" "$DB_PORT" 2>/dev/null; do
        T=$((T+1))
        [ "$T" -ge "$MAX" ] && { err "PostgreSQL não respondeu. Abortando."; exit 1; }
        log "  Tentativa $T/$MAX — aguardando 2s..."
        sleep 2
    done
    ok "PostgreSQL pronto em $DB_HOST:$DB_PORT"
    sleep 2

    log "Inicializando tabelas..."
    python - <<'PYEOF'
import os
from sqlalchemy import create_engine, text
engine = create_engine(os.getenv("DB_URL"))
ddl = """
CREATE TABLE IF NOT EXISTS lineup_navios (
    id SERIAL PRIMARY KEY, porto VARCHAR(50), navio VARCHAR(150),
    bandeira VARCHAR(50), tipo_carga VARCHAR(100), tev VARCHAR(30),
    eta VARCHAR(50), status VARCHAR(80), coletado_em TIMESTAMP DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS previsao_clima (
    id SERIAL PRIMARY KEY, porto VARCHAR(50), data_hora TIMESTAMP,
    temperatura FLOAT, umidade INT, velocidade_vento FLOAT,
    chuva_mm_3h FLOAT, descricao VARCHAR(100), coletado_em TIMESTAMP DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS telemetria_frota (
    id SERIAL PRIMARY KEY, destino_porto VARCHAR(50),
    caminhoes_em_rota INT, distancia_media_km FLOAT,
    caminhoes_na_fila INT, coletado_em TIMESTAMP DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS horas_espera_historico (
    id SERIAL PRIMARY KEY, porto VARCHAR(50),
    horas_espera FLOAT, data_registro TIMESTAMP DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS historico_previsoes (
    id SERIAL PRIMARY KEY, porto VARCHAR(50),
    gerado_em TIMESTAMP DEFAULT NOW(), alerta VARCHAR(20),
    risco_pct FLOAT, payload TEXT
);
"""
with engine.connect() as conn:
    for stmt in ddl.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(text(stmt))
    conn.commit()
print("Tabelas criadas.")
PYEOF
    ok "Banco inicializado."
else
    warn "DB_URL sem host 'postgres' — modo standalone (dados sintéticos)."
fi

# --- Passo 2: Treina modelos LSTM -------------------------------------------
if [ "${TREINAR_MODELOS:-true}" = "true" ]; then
    echo ""
    log "Verificando/Treinando modelos LSTM para todos os portos..."
    warn "Usa dados sintéticos automaticamente se banco estiver vazio."
    echo ""

    PORTOS_DEFAULT="paranagua,santos,rio_grande,itajai,sao_francisco_sul,rio_de_janeiro,vitoria,salvador,suape,fortaleza,itaqui,vila_do_conde,santarem,rondonopolis,sorriso,miritituba,ponta_grossa"
    IFS=',' read -ra LISTA <<< "${PORTOS_TREINAR:-$PORTOS_DEFAULT}"

    TREINADOS=0; PULADOS=0
    for PORTO in "${LISTA[@]}"; do
        PORTO=$(echo "$PORTO" | tr -d ' ')
        [ -z "$PORTO" ] && continue
        if [ -f "modelos_salvos/meta_${PORTO}.json" ] && \
           [ -f "modelos_salvos/modelo_lstm_${PORTO}.pt" ]; then
            ok "  v $PORTO ja treinado."
            PULADOS=$((PULADOS+1))
            continue
        fi
        log "  -> Treinando: $PORTO..."
        if python fase2_modelo_lstm.py --treinar --porto "$PORTO"; then
            ok "  v $PORTO treinado."
            TREINADOS=$((TREINADOS+1))
        else
            err "  x Falha em $PORTO — seguindo."
        fi
    done
    ok "Treino: $TREINADOS treinado(s), $PULADOS ja existiam."
else
    warn "TREINAR_MODELOS=false — treino pulado."
fi

# --- Passo 3: API REST -------------------------------------------------------
echo ""
echo -e "${BOLD}============================================================${NC}"
ok "Iniciando API REST na porta 8000..."
echo -e "  Dashboard: http://localhost:8000/"
echo -e "  API Docs:  http://localhost:8000/docs"
echo -e "  Health:    http://localhost:8000/health"
echo -e "  API Key:   ${API_SECRET_KEY:-vidal-demo-key-2025}"
echo -e "${BOLD}============================================================${NC}"
echo ""

exec uvicorn fase3_api_fastapi:app --host 0.0.0.0 --port 8000 --log-level info
