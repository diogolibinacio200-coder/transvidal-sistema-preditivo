# 🚢 Sistema Preditivo de Gargalos Portuários
**Vidal Transportes — Torre de Controle Logístico**

Prevê o congestionamento nos Portos de Paranaguá e Santos com até **3 dias de antecedência**, combinando line-up de navios, previsão climática e telemetria da frota.

---

## Arquitetura Geral

```
┌─────────────────────────────────────────────────────────────────┐
│                        FASE 1 — COLETA                          │
│                                                                 │
│  [Sites dos Portos] ──scraping──► ScraperLineup                 │
│  [OpenWeatherMap]  ──API REST──► ColetaClima                    │
│  [Banco Interno]   ──SQL──────► ColetaTelemetria                │
│                                     │                           │
│                          Orquestrador (cron 02h)                │
│                                     │                           │
│                          PostgreSQL (vidal_portos)              │
└─────────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                      FASE 2 — MODELO I.A.                        │
│                                                                 │
│  PostgreSQL ──► Extrator ──► Preprocessador                     │
│                                    │                            │
│                          Janelas Deslizantes                    │
│                          (7 dias → prever 3)                    │
│                                    │                            │
│                       LSTM Bidirecional (PyTorch)               │
│                       hidden=128 | layers=2                     │
│                                    │                            │
│                    Early Stopping + HuberLoss                   │
│                                    │                            │
│                    modelo_lstm_paranagua.pt                     │
│                    scaler_x_paranagua.pkl                       │
│                    meta_paranagua.json                          │
└─────────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                      FASE 3 — API FASTAPI                        │
│                                                                 │
│  GET  /previsao/{porto}          → Previsão automática          │
│  POST /previsao/{porto}/manual   → Simulação de cenários        │
│  GET  /alerta/ativos             → Painel da Torre de Controle  │
│  GET  /historico/{porto}         → Histórico de alertas         │
│  GET  /health                    → Status dos modelos           │
│                                                                 │
│  Webhook → Slack/Teams (alertas críticos)                       │
└─────────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────────────────┐
                    │  PAINEL DA TORRE     │
                    │  "Risco 80% em      │
                    │  Paranaguá — Redir.  │
                    │  50 caminhões para  │
                    │  silo Ponta Grossa" │
                    └──────────────────────┘
```

---

## Estrutura de Arquivos

```
projeto/
├── fase1_coleta_dados.py      # Scraping + APIs + SQL → PostgreSQL
├── fase2_modelo_lstm.py       # Treino LSTM + avaliação + exportação
├── fase3_api_fastapi.py       # API REST (FastAPI)
├── requirements.txt
├── .env.exemplo               # Copiar para .env e preencher
└── modelos_salvos/            # Criado automaticamente
    ├── modelo_lstm_paranagua.pt
    ├── scaler_x_paranagua.pkl
    ├── scaler_y_paranagua.pkl
    ├── meta_paranagua.json
    ├── diagnostico_modelo.png
    └── curva_aprendizado.png
```

---

## Setup Rápido

### 1. Pré-requisitos
```bash
python 3.11+
PostgreSQL 14+
Google Chrome (para Selenium)
```

### 2. Instalar dependências
```bash
pip install -r requirements.txt
```

### 3. Configurar variáveis de ambiente
```bash
cp .env.exemplo .env
# Editar .env com suas credenciais
```

### 4. Criar banco de dados
```sql
CREATE DATABASE vidal_portos;
```

---

## Execução

### Fase 1 — Iniciar coleta de dados
```bash
# Executar UMA VEZ para testar
python fase1_coleta_dados.py --agora

# Modo produção (agenda coleta diária às 02h)
python fase1_coleta_dados.py
```

### Fase 2 — Treinar o modelo
```bash
# Treinar para Paranaguá (recomendado ter 6+ meses de dados)
python fase2_modelo_lstm.py --treinar --porto paranagua

# Treinar para Santos
python fase2_modelo_lstm.py --treinar --porto santos

# Fazer previsão de demonstração (após treino)
python fase2_modelo_lstm.py --prever paranagua
```

### Fase 3 — Subir a API
```bash
# Desenvolvimento
uvicorn fase3_api_fastapi:app --reload --port 8000

# Produção
gunicorn fase3_api_fastapi:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## Exemplos de Uso da API

### Verificar saúde
```bash
curl http://localhost:8000/health
```

### Previsão automática (usa dados do banco)
```bash
curl -H "x-api-key: sua_chave_aqui" \
     http://localhost:8000/previsao/paranagua
```

### Resposta esperada
```json
{
  "porto": "paranagua",
  "gerado_em": "2025-08-15T03:00:12",
  "horizonte_dias": 3,
  "previsao": [
    { "data": "2025-08-16", "horas_espera": 8.2,  "descricao": "Atenção (6–12h)" },
    { "data": "2025-08-17", "horas_espera": 41.5, "descricao": "Crítico (24–48h)" },
    { "data": "2025-08-18", "horas_espera": 38.1, "descricao": "Crítico (24–48h)" }
  ],
  "alerta": "LARANJA",
  "risco_pct": 43.2,
  "modelo_mae_h": 4.1,
  "recomendacao": "ALERTA: congestionamento significativo previsto em Paranagua (41.5h em 2025-08-17). Recomendar renegociação de janelas de entrega com as tradings."
}
```

### Simulação manual (cenário de chuva extrema)
```bash
curl -X POST http://localhost:8000/previsao/paranagua/manual \
  -H "x-api-key: sua_chave_aqui" \
  -H "Content-Type: application/json" \
  -d '{
    "historico": [
      {"data":"2025-08-09","navios_aguardando":20,"chuva_mm_acumulada_24h":0,"chuva_mm_prev_72h":0,"velocidade_vento_media":15,"caminhoes_em_rota":120,"caminhoes_na_fila":18},
      {"data":"2025-08-10","navios_aguardando":22,"chuva_mm_acumulada_24h":5,"chuva_mm_prev_72h":10,"velocidade_vento_media":16,"caminhoes_em_rota":130,"caminhoes_na_fila":20},
      {"data":"2025-08-11","navios_aguardando":25,"chuva_mm_acumulada_24h":30,"chuva_mm_prev_72h":80,"velocidade_vento_media":25,"caminhoes_em_rota":150,"caminhoes_na_fila":35},
      {"data":"2025-08-12","navios_aguardando":28,"chuva_mm_acumulada_24h":60,"chuva_mm_prev_72h":120,"velocidade_vento_media":32,"caminhoes_em_rota":160,"caminhoes_na_fila":50},
      {"data":"2025-08-13","navios_aguardando":30,"chuva_mm_acumulada_24h":45,"chuva_mm_prev_72h":90,"velocidade_vento_media":28,"caminhoes_em_rota":155,"caminhoes_na_fila":60},
      {"data":"2025-08-14","navios_aguardando":33,"chuva_mm_acumulada_24h":20,"chuva_mm_prev_72h":70,"velocidade_vento_media":22,"caminhoes_em_rota":165,"caminhoes_na_fila":72},
      {"data":"2025-08-15","navios_aguardando":35,"chuva_mm_acumulada_24h":10,"chuva_mm_prev_72h":80,"velocidade_vento_media":20,"caminhoes_em_rota":180,"caminhoes_na_fila":80}
    ]
  }'
```

---

## Níveis de Alerta

| Alerta    | Horas de Fila | Ação Recomendada |
|-----------|--------------|-----------------|
| 🟢 VERDE   | < 12h        | Operação normal |
| 🟡 AMARELO | 12–36h       | Monitorar, acionar despachantes |
| 🟠 LARANJA | 36–72h       | Renegociar janelas com tradings |
| 🔴 VERMELHO| > 72h        | Redirecionar caminhões, acionar diretoria |

---

## Features do Modelo

| Feature | Fonte | Descrição |
|---|---|---|
| `navios_aguardando` | Scraping APPA/Santos | Navios no line-up |
| `chuva_mm_acumulada_24h` | OpenWeatherMap | Chuva real últimas 24h |
| `chuva_mm_prev_72h` | OpenWeatherMap | Previsão 72h |
| `velocidade_vento_media` | OpenWeatherMap | Vento médio (afeta atracação) |
| `caminhoes_em_rota` | Telemetria Vidal (SQL) | Caminhões a caminho |
| `caminhoes_na_fila` | Telemetria Vidal (SQL) | Caminhões já parados |
| `dia_semana` | Calendário | Sazonalidade semanal |
| `semana_do_ano` | Calendário | Sazonalidade de safra |
| `mes` | Calendário | Sazonalidade mensal |

---

## Próximos Passos (Roadmap)

- [ ] Dashboard web (React) integrado à API
- [ ] Adicionar porto de Arco Norte (Vila do Conde)
- [ ] Modelo de otimização de rotas (PuLP/OR-Tools) baseado nas previsões
- [ ] Integração com sistema de NF-e (volume de notas emitidas por destino)
- [ ] Retreino automático mensal com novos dados
- [ ] Autenticação JWT completa
- [ ] Containerização com Docker + docker-compose
