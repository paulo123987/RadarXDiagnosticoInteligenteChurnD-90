"""
modules/data_utils.py
Utilitários de dados: carregamento, validação, consolidação D-90 e timeline da jornada.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import streamlit as st

REQUIRED_COLS = ["ID_CLIENTE", "TRANSCRICAO_LIGACAO_CLIENTE", "DATETIME_TRANSCRICAO_LIGACAO"]

ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "iso-8859-1", "cp1252"]


def load_csv(uploaded_file) -> pd.DataFrame:
    """Carrega CSV com tratamento de encoding."""
    raw = uploaded_file.read()
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            return df
        except Exception:
            continue
    raise ValueError("Não foi possível decodificar o arquivo. Tente salvar como UTF-8.")


def validate_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Valida se as colunas obrigatórias existem."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return (len(missing) == 0), missing


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa e converte tipos do DataFrame."""
    df = df.copy()
    df["DATETIME_TRANSCRICAO_LIGACAO"] = pd.to_datetime(
        df["DATETIME_TRANSCRICAO_LIGACAO"], errors="coerce"
    )
    df = df.dropna(subset=["DATETIME_TRANSCRICAO_LIGACAO"])
    df["TRANSCRICAO_LIGACAO_CLIENTE"] = df["TRANSCRICAO_LIGACAO_CLIENTE"].fillna("").astype(str)
    df["ID_CLIENTE"] = df["ID_CLIENTE"].astype(str).str.strip()
    df = df.sort_values(["ID_CLIENTE", "DATETIME_TRANSCRICAO_LIGACAO"]).reset_index(drop=True)
    df["MES_ANO"] = df["DATETIME_TRANSCRICAO_LIGACAO"].dt.to_period("M").astype(str)
    df["ANO"] = df["DATETIME_TRANSCRICAO_LIGACAO"].dt.year
    df["MES"] = df["DATETIME_TRANSCRICAO_LIGACAO"].dt.month
    df["DIA_SEMANA"] = df["DATETIME_TRANSCRICAO_LIGACAO"].dt.day_name()
    df["N_TOKENS_EST"] = df["TRANSCRICAO_LIGACAO_CLIENTE"].apply(
        lambda x: max(1, int(len(x.split()) * 1.3))
    )
    return df


def consolidate_d90(df: pd.DataFrame, days: int = 90) -> pd.DataFrame:
    """
    Consolida todas as interações de cada cliente em 1 linha,
    filtrando apenas os últimos N dias antes da última interação (proxy de churn).
    """
    rows = []
    for cli_id, grp in df.groupby("ID_CLIENTE"):
        grp = grp.sort_values("DATETIME_TRANSCRICAO_LIGACAO")
        data_ultimo_contato = grp["DATETIME_TRANSCRICAO_LIGACAO"].max()
        data_corte = data_ultimo_contato - timedelta(days=days)
        grp_d90 = grp[grp["DATETIME_TRANSCRICAO_LIGACAO"] >= data_corte].copy()

        if grp_d90.empty:
            grp_d90 = grp.tail(1)

        transcricoes = grp_d90["TRANSCRICAO_LIGACAO_CLIENTE"].tolist()
        datas = grp_d90["DATETIME_TRANSCRICAO_LIGACAO"].tolist()

        # Empilhar transcrições com separador
        jornada_texto = "\n\n---\n".join(
            [f"[INTERAÇÃO {i+1} – {d.strftime('%d/%m/%Y %H:%M')}]\n{t}"
             for i, (t, d) in enumerate(zip(transcricoes, datas))]
        )

        # Timeline textual
        timeline = " → ".join([f"Ligação {i+1} ({d.strftime('%d/%m')})" for i, d in enumerate(datas)])

        # Features numéricas
        n_ligacoes = len(grp_d90)
        tokens_total = grp_d90["N_TOKENS_EST"].sum()
        tokens_medio = grp_d90["N_TOKENS_EST"].mean()
        span_dias = max(1, (datas[-1] - datas[0]).days)
        intervalo_medio = span_dias / max(1, n_ligacoes - 1) if n_ligacoes > 1 else span_dias

        rows.append({
            "ID_CLIENTE": cli_id,
            "N_LIGACOES_D90": n_ligacoes,
            "DATA_PRIMEIRA_LIGACAO": datas[0],
            "DATA_ULTIMA_LIGACAO": data_ultimo_contato,
            "SPAN_DIAS": span_dias,
            "INTERVALO_MEDIO_DIAS": round(intervalo_medio, 1),
            "TOKENS_TOTAL": tokens_total,
            "TOKENS_MEDIO": round(tokens_medio, 1),
            "JORNADA_TEXTO": jornada_texto,
            "TIMELINE": timeline,
            "TRANSCRICOES_LIST": transcricoes,
            "DATAS_LIST": datas,
        })

    return pd.DataFrame(rows)


def get_synthetic_csv_path() -> str:
    """Retorna o caminho do CSV sintético embarcado."""
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "dados_churn_sintetico.csv")


def get_top_journey_patterns(df_classified: pd.DataFrame, top_n: int = 3) -> list:
    """
    Identifica os padrões de jornada (sequência de macro motivos) mais frequentes.
    Ex: [Técnico, Técnico, Atendimento] -> 5 ocorrências.
    """
    # Suportar tanto MACRO_MOTIVO quanto PERFIL_RECLAMACAO
    motivo_col = None
    if "PERFIL_RECLAMACAO" in df_classified.columns:
        motivo_col = "PERFIL_RECLAMACAO"
    elif "MACRO_MOTIVO" in df_classified.columns:
        motivo_col = "MACRO_MOTIVO"
    
    if df_classified is None or df_classified.empty or motivo_col is None:
        return []
    
    # Agrupar por cliente e criar lista ordenada de motivos
    journeys = df_classified.sort_values(["ID_CLIENTE", "DATETIME_TRANSCRICAO_LIGACAO"]).groupby("ID_CLIENTE")[motivo_col].apply(lambda x: " → ".join(x))
    
    top_patterns = journeys.value_counts().head(top_n).reset_index()
    top_patterns.columns = ["JORNADA", "FREQUENCIA"]
    
    return top_patterns.to_dict("records")


def get_enriched_csv_path() -> str:
    """Retorna o caminho do CSV enriquecido com classificações."""
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "dados_churn_sintetico_enriquecido.csv")


def save_enriched_dataset(df: pd.DataFrame) -> None:
    """
    Salva DataFrame enriquecido com classificações (overwrite).
    Arquivo: dados_churn_sintetico_enriquecido.csv
    """
    path = get_enriched_csv_path()
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_enriched_dataset_if_exists(original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Carrega CSV enriquecido se existir e for compatível com o dataset original.
    Retorna None se:
    - Arquivo não existir
    - Estar corrompido
    - Não ter as colunas necessárias
    - Número de linhas não bater com o original
    """
    import os
    path = get_enriched_csv_path()
    
    if not os.path.exists(path):
        return None
    
    try:
        df_enriched = pd.read_csv(path)
        
        # Validar se tem as colunas necessárias de classificação
        required = ["PERFIL_RECLAMACAO", "CONFIDENCE_SCORE"]
        if not all(col in df_enriched.columns for col in required):
            return None
        
        # Validar se número de linhas bate
        if len(df_enriched) != len(original_df):
            return None
        
        # Converter datetime se necessário
        if "DATETIME_TRANSCRICAO_LIGACAO" in df_enriched.columns:
            df_enriched["DATETIME_TRANSCRICAO_LIGACAO"] = pd.to_datetime(
                df_enriched["DATETIME_TRANSCRICAO_LIGACAO"], errors="coerce"
            )
        
        return df_enriched
        
    except Exception as e:
        # Se houver qualquer erro ao carregar, retorna None
        return None
