"""
=============================================================================
FASE 2 — MODELAGEM PREDITIVA (I.A.)
Sistema Preditivo de Gargalos Portuários | Vidal Transportes
=============================================================================

Pipeline completo:
  1. Extrator         → Puxa dados do PostgreSQL e monta séries temporais
  2. Preprocessador   → Normaliza, codifica e cria janelas deslizantes
  3. ModeloLSTM       → Rede Neural Recorrente (PyTorch) — prevê horas de fila
  4. Treinador        → Loop de treino com early stopping e validação
  5. Avaliador        → Métricas RMSE, MAE e gráficos de diagnóstico
  6. Exportador       → Salva modelo .pt e scaler .pkl prontos para a API
  7. PreditorOnline   → Interface simples: recebe features → retorna previsão

Rodar o treino:
  python fase2_modelo_lstm.py --treinar

Fazer previsão rápida (após treino):
  python fase2_modelo_lstm.py --prever paranagua

Dependências:
  pip install torch scikit-learn pandas sqlalchemy psycopg2-binary
              matplotlib seaborn python-dotenv numpy joblib
=============================================================================
"""

import os
import logging
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------------------------

DB_URL        = os.getenv("DB_URL", "postgresql://user:pass@localhost:5432/vidal_portos")
MODELOS_DIR   = Path("modelos_salvos")
MODELOS_DIR.mkdir(exist_ok=True)

# Hiperparâmetros do modelo
CONFIG = {
    "janela_entrada"  : 7,      # dias de histórico usados como input
    "horizonte"       : 3,      # dias à frente para prever
    "hidden_size"     : 128,    # neurônios por camada LSTM
    "num_layers"      : 2,      # camadas LSTM empilhadas
    "dropout"         : 0.2,    # dropout para regularização
    "batch_size"      : 32,
    "learning_rate"   : 1e-3,
    "epochs"          : 100,
    "patience"        : 15,     # early stopping
    "val_split"       : 0.15,
    "test_split"      : 0.10,
}

# Features usadas como input do modelo
FEATURES = [
    "navios_aguardando",      # quantidade de navios no line-up
    "chuva_mm_acumulada_24h", # chuva nas últimas 24h no porto
    "chuva_mm_prev_72h",      # previsão de chuva nos próximos 3 dias
    "velocidade_vento_media", # vento médio (afeta atracação)
    "caminhoes_em_rota",      # caminhões Vidal a caminho
    "caminhoes_na_fila",      # caminhões já parados no porto
    "dia_semana",             # 0=segunda, 6=domingo (sazonalidade)
    "semana_do_ano",          # sazonalidade anual (safra)
    "mes",                    # mês do ano
]

TARGET = "horas_espera_media"  # variável alvo: horas de fila de descarga


# ---------------------------------------------------------------------------
# MÓDULO 1 — EXTRAÇÃO E MONTAGEM DA SÉRIE TEMPORAL
# ---------------------------------------------------------------------------

class Extrator:
    """
    Combina lineup_navios + previsao_clima + telemetria_frota em uma
    série temporal diária agregada por porto, pronta para o modelo.
    """

    def __init__(self, porto: str):
        self.porto  = porto
        self.engine = create_engine(DB_URL)

    def extrair(self) -> pd.DataFrame:
        """
        Retorna DataFrame indexado por data com todas as features + target.

        NOTA: A query pressupõe que a tabela 'horas_espera_historico' existe
        com registros reais de tempo de fila (alimentada pela telemetria).
        Se ainda não existir, use dados sintéticos via _gerar_dados_sinteticos().
        """
        log.info(f"Extraindo dados para porto: {self.porto}")

        min_registros = CONFIG["janela_entrada"] + CONFIG["horizonte"] + 10
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(
                    text(self._query_principal()),
                    conn,
                    params={"porto": self.porto},
                    parse_dates=["data"],
                )
            log.info(f"Extraídos {len(df)} registros do banco.")
            if len(df) < min_registros:
                log.warning(
                    f"Dados insuficientes no banco ({len(df)} registros, mínimo {min_registros}). "
                    f"Usando dados sintéticos para treino inicial."
                )
                return self._gerar_dados_sinteticos()
            return df
        except Exception as e:
            log.warning(f"Banco indisponível ({e}). Usando dados sintéticos para teste.")
            return self._gerar_dados_sinteticos()

    def _query_principal(self) -> str:
        return """
        WITH lineup AS (
            SELECT
                DATE(coletado_em)     AS data,
                porto,
                COUNT(DISTINCT navio) AS navios_aguardando
            FROM lineup_navios
            WHERE porto = :porto
            GROUP BY DATE(coletado_em), porto
        ),
        clima AS (
            SELECT
                DATE(data_hora)              AS data,
                porto,
                SUM(chuva_mm_3h)             AS chuva_mm_acumulada_24h,
                AVG(velocidade_vento)        AS velocidade_vento_media,
                -- Previsão dos próximos 3 dias (data_hora > NOW())
                SUM(CASE WHEN data_hora > NOW()
                         THEN chuva_mm_3h ELSE 0 END) AS chuva_mm_prev_72h
            FROM previsao_clima
            WHERE porto = :porto
              AND data_hora >= CURRENT_DATE - INTERVAL '365 days'
            GROUP BY DATE(data_hora), porto
        ),
        frota AS (
            SELECT
                DATE(coletado_em)   AS data,
                destino_porto       AS porto,
                AVG(caminhoes_em_rota)  AS caminhoes_em_rota,
                AVG(caminhoes_na_fila)  AS caminhoes_na_fila
            FROM telemetria_frota
            WHERE destino_porto = :porto
            GROUP BY DATE(coletado_em), destino_porto
        ),
        espera AS (
            SELECT
                DATE(data_registro) AS data,
                porto,
                AVG(horas_espera)   AS horas_espera_media
            FROM horas_espera_historico
            WHERE porto = :porto
            GROUP BY DATE(data_registro), porto
        )
        SELECT
            l.data,
            COALESCE(l.navios_aguardando, 0)      AS navios_aguardando,
            COALESCE(c.chuva_mm_acumulada_24h, 0) AS chuva_mm_acumulada_24h,
            COALESCE(c.chuva_mm_prev_72h, 0)      AS chuva_mm_prev_72h,
            COALESCE(c.velocidade_vento_media, 0) AS velocidade_vento_media,
            COALESCE(f.caminhoes_em_rota, 0)      AS caminhoes_em_rota,
            COALESCE(f.caminhoes_na_fila, 0)      AS caminhoes_na_fila,
            COALESCE(e.horas_espera_media, 0)     AS horas_espera_media
        FROM lineup l
        LEFT JOIN clima   c ON l.data = c.data
        LEFT JOIN frota   f ON l.data = f.data
        LEFT JOIN espera  e ON l.data = e.data
        ORDER BY l.data
        """

    def _gerar_dados_sinteticos(self, n_dias: int = 730) -> pd.DataFrame:
        """
        Gera 2 anos de dados realistas para desenvolvimento e testes.
        Simula sazonalidade de safra (pico jan-mar e jun-ago), efeito chuva
        e correlação com número de navios.
        """
        log.warning("ATENÇÃO: usando dados sintéticos — não usar em produção!")
        np.random.seed(42)
        datas = pd.date_range(end=datetime.today(), periods=n_dias, freq="D")

        dias_ano   = np.array([d.timetuple().tm_yday for d in datas])
        # Duas safras por ano (soja jan-mar, milho jun-ago)
        safra = (
            0.6 * np.sin(2 * np.pi * dias_ano / 365) +
            0.4 * np.sin(4 * np.pi * dias_ano / 365)
        )
        safra = (safra - safra.min()) / (safra.max() - safra.min())  # 0-1

        navios        = (8 + 22 * safra + np.random.normal(0, 2, n_dias)).clip(0, 40).astype(int)
        chuva_atual   = np.random.exponential(scale=5, size=n_dias).clip(0, 80)
        chuva_prev    = np.random.exponential(scale=8, size=n_dias).clip(0, 120)
        vento         = np.random.normal(15, 5, n_dias).clip(0, 40)
        caminhoes_rota = (50 + 150 * safra + np.random.normal(0, 15, n_dias)).clip(0).astype(int)
        caminhoes_fila = (caminhoes_rota * 0.15 + np.random.normal(0, 5, n_dias)).clip(0).astype(int)

        # Horas de espera: função dos navios + chuva + fila
        horas = (
            4 +
            navios * 1.2 +
            chuva_atual * 0.3 +
            chuva_prev * 0.15 +
            caminhoes_fila * 0.08 +
            np.random.normal(0, 3, n_dias)
        ).clip(1, 96)

        df = pd.DataFrame({
            "data"                   : datas,
            "navios_aguardando"      : navios,
            "chuva_mm_acumulada_24h" : chuva_atual,
            "chuva_mm_prev_72h"      : chuva_prev,
            "velocidade_vento_media" : vento,
            "caminhoes_em_rota"      : caminhoes_rota,
            "caminhoes_na_fila"      : caminhoes_fila,
            "horas_espera_media"      : horas,
        })
        df.set_index("data", inplace=True)
        return df


# ---------------------------------------------------------------------------
# MÓDULO 2 — PRÉ-PROCESSAMENTO
# ---------------------------------------------------------------------------

class Preprocessador:
    """
    Normaliza features, adiciona variáveis de calendário e cria
    janelas deslizantes (sliding windows) para o LSTM.
    """

    def __init__(self, janela: int = CONFIG["janela_entrada"],
                 horizonte: int = CONFIG["horizonte"]):
        self.janela    = janela
        self.horizonte = horizonte
        self.scaler_x  = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y  = MinMaxScaler(feature_range=(0, 1))

    def preparar(self, df: pd.DataFrame):
        """
        Recebe DataFrame bruto, retorna (X, y) como tensores PyTorch.
        X: (amostras, janela, n_features)
        y: (amostras, horizonte)
        """
        df = df.copy()
        df.index = pd.to_datetime(df.index)

        # Variáveis de calendário (sazonalidade)
        df["dia_semana"]   = df.index.dayofweek
        df["semana_do_ano"] = df.index.isocalendar().week.astype(int)
        df["mes"]          = df.index.month

        # Normalização
        X_cols = FEATURES
        y_col  = TARGET

        X_raw = df[X_cols].values.astype(np.float32)
        y_raw = df[[y_col]].values.astype(np.float32)

        X_norm = self.scaler_x.fit_transform(X_raw)
        y_norm = self.scaler_y.fit_transform(y_raw).flatten()

        # Janelas deslizantes
        Xs, ys = [], []
        for i in range(len(X_norm) - self.janela - self.horizonte + 1):
            Xs.append(X_norm[i : i + self.janela])
            ys.append(y_norm[i + self.janela : i + self.janela + self.horizonte])

        X = np.array(Xs, dtype=np.float32)
        y = np.array(ys, dtype=np.float32)

        log.info(f"Dataset: X={X.shape}, y={y.shape}")
        return torch.tensor(X), torch.tensor(y)

    def salvar(self, porto: str):
        joblib.dump(self.scaler_x, MODELOS_DIR / f"scaler_x_{porto}.pkl")
        joblib.dump(self.scaler_y, MODELOS_DIR / f"scaler_y_{porto}.pkl")
        log.info(f"Scalers salvos em {MODELOS_DIR}/")

    @classmethod
    def carregar(cls, porto: str):
        obj = cls.__new__(cls)
        obj.scaler_x = joblib.load(MODELOS_DIR / f"scaler_x_{porto}.pkl")
        obj.scaler_y = joblib.load(MODELOS_DIR / f"scaler_y_{porto}.pkl")
        obj.janela    = CONFIG["janela_entrada"]
        obj.horizonte = CONFIG["horizonte"]
        return obj


# ---------------------------------------------------------------------------
# MÓDULO 3 — DATASET PyTorch
# ---------------------------------------------------------------------------

class PortoDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# MÓDULO 4 — ARQUITETURA LSTM
# ---------------------------------------------------------------------------

class ModeloLSTM(nn.Module):
    """
    LSTM bidirecional com camadas densas de saída.

    Arquitetura:
      Input → LSTM(bidirectional) × num_layers → Dropout → Dense → Dense → Output

    Por que bidirecional?
      Permite que o modelo aprenda padrões tanto no sentido passado→futuro
      quanto em ciclos sazonais (ex: padrão de sábado influenciado pela segunda).
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int   = CONFIG["hidden_size"],
        num_layers: int    = CONFIG["num_layers"],
        dropout: float     = CONFIG["dropout"],
        horizonte: int     = CONFIG["horizonte"],
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        directions         = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size    = n_features,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            dropout       = dropout if num_layers > 1 else 0.0,
            batch_first   = True,
            bidirectional = bidirectional,
        )

        self.dropout = nn.Dropout(dropout)

        lstm_out = hidden_size * directions
        self.fc1 = nn.Linear(lstm_out, 64)
        self.fc2 = nn.Linear(64, horizonte)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        # Pega apenas o último timestep
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out  # (batch, horizonte)


# ---------------------------------------------------------------------------
# MÓDULO 5 — TREINADOR
# ---------------------------------------------------------------------------

class Treinador:

    def __init__(self, modelo: ModeloLSTM, device: str = "cpu"):
        self.modelo    = modelo.to(device)
        self.device    = device
        self.historico = {"treino": [], "val": []}

    def treinar(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        val_split:  float = CONFIG["val_split"],
        test_split: float = CONFIG["test_split"],
        epochs:     int   = CONFIG["epochs"],
        batch_size: int   = CONFIG["batch_size"],
        lr:         float = CONFIG["learning_rate"],
        patience:   int   = CONFIG["patience"],
    ):
        # Divisão cronológica (não aleatória — dados de série temporal!)
        n = len(X)
        n_test  = int(n * test_split)
        n_val   = int(n * val_split)
        n_train = n - n_val - n_test

        X_train, y_train = X[:n_train], y[:n_train]
        X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
        self.X_test = X[n_train+n_val:]
        self.y_test = y[n_train+n_val:]

        log.info(f"Treino: {n_train} | Validação: {n_val} | Teste: {len(self.X_test)}")

        loader_train = DataLoader(
            PortoDataset(X_train, y_train), batch_size=batch_size, shuffle=False
        )
        loader_val = DataLoader(
            PortoDataset(X_val, y_val), batch_size=batch_size, shuffle=False
        )

        criterion = nn.HuberLoss(delta=1.0)  # robusta a outliers
        optimizer = torch.optim.Adam(self.modelo.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)

        melhor_val = float("inf")
        sem_melhora = 0

        for epoch in range(1, epochs + 1):
            # --- Treino ---
            self.modelo.train()
            loss_treino = 0.0
            for xb, yb in loader_train:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred  = self.modelo(xb)
                loss  = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.modelo.parameters(), 1.0)
                optimizer.step()
                loss_treino += loss.item()
            loss_treino /= len(loader_train)

            # --- Validação ---
            self.modelo.eval()
            loss_val = 0.0
            with torch.no_grad():
                for xb, yb in loader_val:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    loss_val += criterion(self.modelo(xb), yb).item()
            loss_val /= len(loader_val)

            self.historico["treino"].append(loss_treino)
            self.historico["val"].append(loss_val)

            scheduler.step(loss_val)

            if epoch % 10 == 0 or epoch == 1:
                log.info(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Loss treino: {loss_treino:.4f} | "
                    f"Loss val: {loss_val:.4f}"
                )

            # Early Stopping
            if loss_val < melhor_val - 1e-4:
                melhor_val  = loss_val
                sem_melhora = 0
                torch.save(self.modelo.state_dict(), MODELOS_DIR / "_checkpoint_best.pt")
            else:
                sem_melhora += 1
                if sem_melhora >= patience:
                    log.info(f"Early stopping na epoch {epoch}.")
                    break

        # Carrega o melhor checkpoint
        self.modelo.load_state_dict(
            torch.load(MODELOS_DIR / "_checkpoint_best.pt", weights_only=True)
        )
        log.info(f"Melhor loss de validação: {melhor_val:.4f}")


# ---------------------------------------------------------------------------
# MÓDULO 6 — AVALIADOR
# ---------------------------------------------------------------------------

class Avaliador:

    def __init__(self, modelo: ModeloLSTM, prep: Preprocessador, device: str = "cpu"):
        self.modelo  = modelo.to(device)
        self.prep    = prep
        self.device  = device

    def avaliar(self, X_test: torch.Tensor, y_test: torch.Tensor) -> dict:
        self.modelo.eval()
        with torch.no_grad():
            preds_norm = self.modelo(X_test.to(self.device)).cpu().numpy()

        # Desnormalizar
        preds_h = self.prep.scaler_y.inverse_transform(
            preds_norm[:, 0:1]  # apenas o primeiro dia do horizonte para métrica
        ).flatten()
        real_h = self.prep.scaler_y.inverse_transform(
            y_test[:, 0:1].numpy()
        ).flatten()

        mae  = mean_absolute_error(real_h, preds_h)
        rmse = np.sqrt(mean_squared_error(real_h, preds_h))
        mape = np.mean(np.abs((real_h - preds_h) / (real_h + 1e-8))) * 100

        metricas = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE%": round(mape, 2)}
        log.info(f"Métricas no conjunto de teste: {metricas}")

        self._plot_predicao(real_h, preds_h)
        return metricas

    def _plot_predicao(self, real: np.ndarray, pred: np.ndarray):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        axes[0].plot(real,  label="Real",   color="#1f77b4", alpha=0.8)
        axes[0].plot(pred,  label="Previsto", color="#ff7f0e", linestyle="--", alpha=0.8)
        axes[0].set_title("Horas de Espera — Real vs Previsto (conjunto de teste)")
        axes[0].set_ylabel("Horas de fila")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        erro = real - pred
        axes[1].hist(erro, bins=40, color="#2ca02c", edgecolor="white", alpha=0.8)
        axes[1].axvline(0, color="red", linestyle="--")
        axes[1].set_title("Distribuição do Erro de Previsão")
        axes[1].set_xlabel("Erro (horas)")
        axes[1].set_ylabel("Frequência")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        caminho = MODELOS_DIR / "diagnostico_modelo.png"
        plt.savefig(caminho, dpi=150)
        log.info(f"Gráfico de diagnóstico salvo em {caminho}")
        plt.close()

    def plot_curva_aprendizado(self, historico: dict):
        plt.figure(figsize=(10, 4))
        plt.plot(historico["treino"], label="Loss Treino")
        plt.plot(historico["val"],   label="Loss Validação")
        plt.title("Curva de Aprendizado")
        plt.xlabel("Epoch")
        plt.ylabel("Huber Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        caminho = MODELOS_DIR / "curva_aprendizado.png"
        plt.savefig(caminho, dpi=150)
        log.info(f"Curva de aprendizado salva em {caminho}")
        plt.close()


# ---------------------------------------------------------------------------
# MÓDULO 7 — EXPORTADOR
# ---------------------------------------------------------------------------

class Exportador:
    """Salva modelo e metadados em formato pronto para a API (Fase 3)."""

    def salvar(self, modelo: ModeloLSTM, prep: Preprocessador,
               metricas: dict, porto: str):
        # Pesos do modelo
        torch.save(modelo.state_dict(), MODELOS_DIR / f"modelo_lstm_{porto}.pt")

        # Scalers
        prep.salvar(porto)

        # Metadados (usados pela API para validar compatibilidade)
        meta = {
            "porto"          : porto,
            "versao"         : "1.0.0",
            "treinado_em"    : datetime.now().isoformat(),
            "features"       : FEATURES,
            "target"         : TARGET,
            "janela_entrada" : CONFIG["janela_entrada"],
            "horizonte"      : CONFIG["horizonte"],
            "n_features"     : len(FEATURES),
            "hidden_size"    : CONFIG["hidden_size"],
            "num_layers"     : CONFIG["num_layers"],
            "metricas_teste" : metricas,
        }
        def converter_tipos(obj):
            import numpy as np
            if isinstance(obj, (np.float32, np.float64)): return float(obj)
            if isinstance(obj, (np.int32, np.int64)):     return int(obj)
            if isinstance(obj, np.ndarray):               return obj.tolist()
            raise TypeError(f"Tipo {type(obj)} não serializável")

        with open(MODELOS_DIR / f"meta_{porto}.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=converter_tipos)

        log.info(f"Modelo exportado com sucesso → {MODELOS_DIR}/modelo_lstm_{porto}.pt")


# ---------------------------------------------------------------------------
# MÓDULO 8 — PREDITOR ONLINE (usado pela API)
# ---------------------------------------------------------------------------

class PreditorOnline:
    """
    Carrega modelo salvo e responde previsões em tempo real.
    Interface simples para a Fase 3 (FastAPI).
    """

    def __init__(self, porto: str, device: str = "cpu"):
        self.porto  = porto
        self.device = device

        meta_path = MODELOS_DIR / f"meta_{porto}.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Modelo para '{porto}' não encontrado. Execute o treinamento primeiro."
            )

        with open(meta_path) as f:
            self.meta = json.load(f)

        self.prep = Preprocessador.carregar(porto)

        self.modelo = ModeloLSTM(
            n_features  = self.meta["n_features"],
            hidden_size = self.meta["hidden_size"],
            num_layers  = self.meta["num_layers"],
            horizonte   = self.meta["horizonte"],
        )
        self.modelo.load_state_dict(
            torch.load(
                MODELOS_DIR / f"modelo_lstm_{porto}.pt",
                map_location=device,
                weights_only=True,
            )
        )
        self.modelo.eval()
        log.info(f"Modelo '{porto}' carregado (treinado em {self.meta['treinado_em'][:10]}).")

    def prever(self, historico_features: pd.DataFrame) -> dict:
        """
        historico_features: DataFrame com CONFIG['janela_entrada'] linhas
                            e todas as colunas de FEATURES (sem o target).

        Retorna dict com previsão de horas de fila para cada dia do horizonte,
        o nível de alerta e a confiança do modelo.
        """
        if len(historico_features) < CONFIG["janela_entrada"]:
            raise ValueError(
                f"Necessário ao menos {CONFIG['janela_entrada']} dias de histórico."
            )

        df_prep = historico_features.copy()
        df_prep = df_prep.tail(CONFIG["janela_entrada"])
        df_prep["dia_semana"]    = pd.to_datetime(df_prep.index).dayofweek
        df_prep["semana_do_ano"] = pd.to_datetime(df_prep.index).isocalendar().week.astype(int)
        df_prep["mes"]           = pd.to_datetime(df_prep.index).month

        X_raw  = df_prep[FEATURES].values.astype(np.float32)
        X_norm = self.prep.scaler_x.transform(X_raw)
        X_t    = torch.tensor(X_norm).unsqueeze(0)  # (1, janela, features)

        with torch.no_grad():
            pred_norm = self.modelo(X_t.to(self.device)).cpu().numpy()

        # Desnormalizar para horas reais
        horas_prev = self.prep.scaler_y.inverse_transform(pred_norm).flatten()
        horas_prev = np.clip(horas_prev, 0, 168).tolist()  # cap em 7 dias

        # Determinar nível de alerta
        max_h = max(horas_prev)
        if max_h < 12:
            alerta, risco_pct = "VERDE", round(max_h / 96 * 100, 1)
        elif max_h < 36:
            alerta, risco_pct = "AMARELO", round(max_h / 96 * 100, 1)
        elif max_h < 72:
            alerta, risco_pct = "LARANJA", round(min(max_h / 96 * 100, 99), 1)
        else:
            alerta, risco_pct = "VERMELHO", 99.0

        hoje = datetime.today().date()
        return {
            "porto"         : self.porto,
            "gerado_em"     : datetime.now().isoformat(),
            "horizonte_dias": CONFIG["horizonte"],
            "previsao"      : [
                {
                    "data"           : str(hoje + timedelta(days=i + 1)),
                    "horas_espera"   : round(h, 1),
                    "descricao"      : _classificar_espera(h),
                }
                for i, h in enumerate(horas_prev)
            ],
            "alerta"        : alerta,
            "risco_pct"     : risco_pct,
            "modelo_mae_h"  : self.meta["metricas_teste"].get("MAE"),
        }


def _classificar_espera(horas: float) -> str:
    if horas < 6:   return "Normal (< 6h)"
    if horas < 12:  return "Atenção (6–12h)"
    if horas < 24:  return "Elevado (12–24h)"
    if horas < 48:  return "Crítico (24–48h)"
    return "Colapso (> 48h)"


# ---------------------------------------------------------------------------
# PIPELINE COMPLETO DE TREINAMENTO
# ---------------------------------------------------------------------------

def pipeline_treino(porto: str):
    log.info(f"{'='*60}")
    log.info(f"INICIANDO TREINO — Porto: {porto.upper()}")
    log.info(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Dispositivo: {device}")

    # 1. Extração
    df = Extrator(porto).extrair()

    # 2. Pré-processamento
    prep = Preprocessador()
    X, y = prep.preparar(df)

    # 3. Modelo
    n_features = X.shape[2]
    modelo = ModeloLSTM(n_features=n_features)
    log.info(f"Parâmetros do modelo: {sum(p.numel() for p in modelo.parameters()):,}")

    # 4. Treino
    treinador = Treinador(modelo, device)
    treinador.treinar(X, y)

    # 5. Avaliação
    avaliador = Avaliador(modelo, prep, device)
    metricas  = avaliador.avaliar(treinador.X_test, treinador.y_test)
    avaliador.plot_curva_aprendizado(treinador.historico)

    # 6. Exportação
    Exportador().salvar(modelo, prep, metricas, porto)

    log.info(f"TREINO CONCLUÍDO | MAE={metricas['MAE']}h | RMSE={metricas['RMSE']}h")
    return modelo, prep, metricas


# ---------------------------------------------------------------------------
# PONTO DE ENTRADA
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fase 2 — Modelo LSTM Portuário")
    parser.add_argument("--treinar", action="store_true", help="Treinar modelo")
    parser.add_argument("--prever",  type=str, metavar="PORTO",
                        help="Fazer previsão para o porto (ex: paranagua)")
    parser.add_argument("--porto",   type=str, default="paranagua",
                        help="Porto alvo do treino (default: paranagua)")
    args = parser.parse_args()

    if args.treinar:
        pipeline_treino(args.porto)

    elif args.prever:
        # Demonstração: gera features fictícias e faz previsão
        preditor = PreditorOnline(args.prever)
        datas    = pd.date_range(end=datetime.today(), periods=CONFIG["janela_entrada"])
        df_demo  = pd.DataFrame(
            {
                "navios_aguardando"      : np.random.randint(10, 30, CONFIG["janela_entrada"]),
                "chuva_mm_acumulada_24h" : np.random.uniform(0, 40, CONFIG["janela_entrada"]),
                "chuva_mm_prev_72h"      : np.random.uniform(20, 80, CONFIG["janela_entrada"]),
                "velocidade_vento_media" : np.random.uniform(10, 30, CONFIG["janela_entrada"]),
                "caminhoes_em_rota"      : np.random.randint(80, 200, CONFIG["janela_entrada"]),
                "caminhoes_na_fila"      : np.random.randint(5, 40, CONFIG["janela_entrada"]),
            },
            index=datas,
        )
        resultado = preditor.prever(df_demo)
        print(json.dumps(resultado, indent=2, ensure_ascii=False))

    else:
        parser.print_help()
