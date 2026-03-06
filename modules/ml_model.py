"""
modules/ml_model.py
Módulo de Machine Learning preditivo de churn.
Modelos: RandomForest, XGBoost, Logistic Regression.
Feature Engineering expandida com frequência, recência, NLP e diagnóstico de causa raiz comportamental.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score,
)
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


COLORS = {
    "primary": "#E63946",
    "secondary": "#457B9D",
    "accent": "#F4A261",
    "success": "#2A9D8F",
    "warning": "#E9C46A",
}

PALETTE = [
    "#E63946", "#1D3557", "#457B9D", "#A8DADC",
    "#F4A261", "#2A9D8F", "#264653", "#6D597A",
]

# ─ Termos para NLP básico ────────────────────────────────────────────────────
CONCORRENTES = ["vivo", "claro", "oi", "tim", "starlink", "sky", "net", "nextel"]
TERMOS_CANCELAMENTO = ["cancelar", "cancelo", "cancelamento", "procon", "justiça",
                       "insatisfeito", "multa", "estorno", "rescisão", "desistir"]
TERMOS_PROBLEMA_TECNICO = ["lentidão", "queda", "sem sinal", "internet caindo",
                            "não funciona", "travando", "oscilação", "instável",
                            "técnico", "fibra", "modem", "roteador"]
TERMOS_PRAZOS = ["amanhã", "hoje", "urgente", "imediato", "agora", "3 dias",
                 "prazo", "deadline", "imediatamente"]


def build_features(df_d90: pd.DataFrame, df_classified: pd.DataFrame = None) -> pd.DataFrame:
    """
    Constrói feature matrix para o modelo preditivo.
    Features numéricas derivadas da jornada D-90.
    """
    feat = df_d90[["ID_CLIENTE", "N_LIGACOES_D90", "SPAN_DIAS",
                   "INTERVALO_MEDIO_DIAS", "TOKENS_TOTAL", "TOKENS_MEDIO"]].copy()

    # Feature: n_categorias_distintas e repeticao_motivo (se análise de agentes disponível)
    if df_classified is not None and "N_CATEGORIAS_DISTINTAS" in df_classified.columns:
        merge_cols = ["ID_CLIENTE", "N_CATEGORIAS_DISTINTAS", "REPETICAO_MOTIVO"]
        feat = feat.merge(
            df_classified[merge_cols],
            on="ID_CLIENTE", how="left"
        )
        feat["N_CATEGORIAS_DISTINTAS"] = feat["N_CATEGORIAS_DISTINTAS"].fillna(1)
        feat["REPETICAO_MOTIVO"] = feat["REPETICAO_MOTIVO"].fillna(0.5)
    else:
        feat["N_CATEGORIAS_DISTINTAS"] = 1
        feat["REPETICAO_MOTIVO"] = 0.5

    # Feature derivadas
    feat["TAXA_LIGACOES_DIA"] = feat["N_LIGACOES_D90"] / feat["SPAN_DIAS"].replace(0, 1)
    feat["SCORE_INTENSIDADE"] = (
        feat["N_LIGACOES_D90"] * 0.3
        + feat["TOKENS_TOTAL"] / 1000 * 0.2
        + feat["N_CATEGORIAS_DISTINTAS"] * 0.2
        + feat["REPETICAO_MOTIVO"] * 0.3
    )

    # Target: 1 = churn de alto risco (> percentil 60 no score de intensidade)
    threshold = feat["SCORE_INTENSIDADE"].quantile(0.4)
    feat["TARGET_CHURN"] = (feat["SCORE_INTENSIDADE"] >= threshold).astype(int)

    return feat.set_index("ID_CLIENTE")


def build_features_expanded(df: pd.DataFrame, df_d90: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói feature matrix expandida com:
    - Frequência em janelas temporais (7d, 30d, 90d)
    - Recência (dias desde última ligação)
    - Intervalos entre ligações (média, máx, mín, desvio padrão)
    - Intensidade de contato
    - NLP básico: menções a concorrentes, flag cancelamento, problemas técnicos, prazos
    - Evolução: taxa de aceleração de insatisfação
    """
    df_work = df.copy()
    df_work["DATETIME_TRANSCRICAO_LIGACAO"] = pd.to_datetime(df_work["DATETIME_TRANSCRICAO_LIGACAO"])
    df_work = df_work.sort_values(["ID_CLIENTE", "DATETIME_TRANSCRICAO_LIGACAO"])

    # Data de referência = dia após a última ligação do dataset
    data_ref = df_work["DATETIME_TRANSCRICAO_LIGACAO"].max() + pd.Timedelta(days=1)

    # Diferença de dias entre ligações consecutivas do mesmo cliente
    df_work["diff_days"] = (
        df_work.groupby("ID_CLIENTE")["DATETIME_TRANSCRICAO_LIGACAO"]
        .diff()
        .dt.total_seconds() / 86400
    )

    # ── Agregação base ────────────────────────────────────────────────────────
    agg = df_work.groupby("ID_CLIENTE").agg(
        total_ligacoes=("ID_CLIENTE", "count"),
        ultima_ligacao=("DATETIME_TRANSCRICAO_LIGACAO", "max"),
        primeira_ligacao=("DATETIME_TRANSCRICAO_LIGACAO", "min"),
        avg_interval_days=("diff_days", "mean"),
        max_interval_days=("diff_days", "max"),
        min_interval_days=("diff_days", "min"),
        std_interval_days=("diff_days", "std"),
        avg_token_count=("N_TOKENS_EST", "mean"),
        max_token_count=("N_TOKENS_EST", "max"),
        texto_consolidado=("TRANSCRICAO_LIGACAO_CLIENTE", lambda x: " ".join(x).lower())
    ).reset_index()

    # ── Recência ──────────────────────────────────────────────────────────────
    agg["days_since_last_call"] = (data_ref - agg["ultima_ligacao"]).dt.days

    # ── Intensidade semanal ───────────────────────────────────────────────────
    agg["semanas_ativas"] = (
        (agg["ultima_ligacao"] - agg["primeira_ligacao"]).dt.total_seconds() / (7 * 86400)
    ).clip(lower=1)
    agg["calls_per_week_avg"] = agg["total_ligacoes"] / agg["semanas_ativas"]

    # ── Frequências por janela temporal ──────────────────────────────────────
    def count_window(customer_id, days):
        cutoff = data_ref - pd.Timedelta(days=days)
        return len(df_work[
            (df_work["ID_CLIENTE"] == customer_id) &
            (df_work["DATETIME_TRANSCRICAO_LIGACAO"] >= cutoff)
        ])

    agg["freq_7d"] = agg["ID_CLIENTE"].apply(lambda x: count_window(x, 7))
    agg["freq_30d"] = agg["ID_CLIENTE"].apply(lambda x: count_window(x, 30))
    agg["freq_90d"] = agg["ID_CLIENTE"].apply(lambda x: count_window(x, 90))

    # ── Taxa de aceleração (30d vs 90d) ──────────────────────────────────────
    media_mensal_90d = agg["freq_90d"] / 3.0
    agg["increase_rate_30d_vs_90d"] = (
        agg["freq_30d"] / media_mensal_90d.replace(0, 0.01)
    ).round(2)

    # ── NLP básico ───────────────────────────────────────────────────────────
    agg["total_mentions_concorrente"] = agg["texto_consolidado"].apply(
        lambda x: sum(1 for c in CONCORRENTES if c in x)
    )
    agg["flag_ameaca_cancelamento"] = agg["texto_consolidado"].apply(
        lambda x: 1 if any(t in x for t in TERMOS_CANCELAMENTO) else 0
    )
    agg["total_mentions_problemas"] = agg["texto_consolidado"].apply(
        lambda x: sum(1 for t in TERMOS_PROBLEMA_TECNICO if t in x)
    )
    agg["total_mentions_prazos"] = agg["texto_consolidado"].apply(
        lambda x: sum(1 for t in TERMOS_PRAZOS if t in x)
    )

    # ── Merge com df_d90 para pegar N_LIGACOES_D90, SPAN_DIAS etc. ───────────
    feat = agg.merge(
        df_d90[["ID_CLIENTE", "N_LIGACOES_D90", "SPAN_DIAS",
                "INTERVALO_MEDIO_DIAS", "TOKENS_TOTAL"]],
        on="ID_CLIENTE", how="left"
    )

    # ── Feature derivada de intensidade ──────────────────────────────────────
    feat["TAXA_LIGACOES_DIA"] = feat["N_LIGACOES_D90"] / feat["SPAN_DIAS"].replace(0, 1)
    feat["SCORE_INTENSIDADE"] = (
        feat["N_LIGACOES_D90"] * 0.2
        + feat["TOKENS_TOTAL"].fillna(0) / 1000 * 0.1
        + feat["total_mentions_concorrente"] * 0.15
        + feat["flag_ameaca_cancelamento"] * 0.2
        + feat["total_mentions_problemas"] * 0.1
        + feat["increase_rate_30d_vs_90d"].clip(0, 5) * 0.1
        + feat["calls_per_week_avg"].clip(0, 10) * 0.15
    )

    # ── Target ───────────────────────────────────────────────────────────────
    threshold = feat["SCORE_INTENSIDADE"].quantile(0.4)
    feat["TARGET_CHURN"] = (feat["SCORE_INTENSIDADE"] >= threshold).astype(int)

    feat = feat.fillna(0)
    feat = feat.drop(columns=["texto_consolidado", "ultima_ligacao", "primeira_ligacao",
                               "semanas_ativas"], errors="ignore")

    return feat.set_index("ID_CLIENTE")


def train_models(feat: pd.DataFrame) -> dict:
    """
    Treina RandomForest, XGBoost (se disponível) e Logistic Regression.
    Retorna dicionário com resultados de cada modelo.
    """
    # Selecionar colunas disponíveis
    all_possible = [
        "N_LIGACOES_D90", "SPAN_DIAS", "INTERVALO_MEDIO_DIAS",
        "TOKENS_TOTAL", "TOKENS_MEDIO", "N_CATEGORIAS_DISTINTAS",
        "REPETICAO_MOTIVO", "TAXA_LIGACOES_DIA", "SCORE_INTENSIDADE",
        # Features expandidas (presentes se build_features_expanded foi usado)
        "freq_7d", "freq_30d", "freq_90d", "days_since_last_call",
        "avg_interval_days", "max_interval_days", "min_interval_days",
        "std_interval_days", "calls_per_week_avg", "avg_token_count",
        "max_token_count", "total_mentions_concorrente", "flag_ameaca_cancelamento",
        "total_mentions_problemas", "total_mentions_prazos",
        "increase_rate_30d_vs_90d",
    ]
    feature_cols = [c for c in all_possible if c in feat.columns]
    X = feat[feature_cols].fillna(0)
    y = feat["TARGET_CHURN"]

    if X.shape[0] < 10:
        return {}

    # Split
    test_size = min(0.3, max(0.2, 5 / len(X)))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models_def = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42, class_weight="balanced"
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=500, random_state=42, class_weight="balanced"
        ),
    }

    results = {}
    for name, mdl in models_def.items():
        try:
            use_scaled = name != "Random Forest" and not name.startswith("XGB")
            Xtr = X_train_s if use_scaled else X_train.values
            Xte = X_test_s if use_scaled else X_test.values

            mdl.fit(Xtr, y_train)
            preds = mdl.predict(Xte)
            proba = mdl.predict_proba(Xte)[:, 1] if hasattr(mdl, "predict_proba") else preds

            acc = accuracy_score(y_test, preds)
            auc = roc_auc_score(y_test, proba) if y_test.nunique() > 1 else 0.5
            report = classification_report(y_test, preds, output_dict=True, zero_division=0)

            # Feature importances
            importances = {}
            if hasattr(mdl, "feature_importances_"):
                importances = dict(zip(feature_cols, mdl.feature_importances_.tolist()))
            elif hasattr(mdl, "coef_"):
                importances = dict(zip(feature_cols, abs(mdl.coef_[0]).tolist()))

            # Probabilidades para todos os registros
            all_proba = mdl.predict_proba(
                scaler.transform(X.values) if use_scaled else X.values
            )[:, 1]

            results[name] = {
                "model": mdl,
                "accuracy": round(acc, 4),
                "auc": round(auc, 4),
                "report": report,
                "importance": importances,
                "all_prob": dict(zip(feat.index, all_proba.tolist())),
                "feature_cols": feature_cols,
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return results


def generate_root_cause_diagnosis(feat: pd.DataFrame, results: dict) -> dict:
    """
    Gera diagnóstico de causa raiz comportamental com base nas features e importância dos modelos.
    Usa Análise de Pareto (80/20) e segmentação de risco para identificar os principais drivers.
    
    Retorna dicionário com:
    - pareto_features: top features que explicam 80% da importância
    - segment_drivers: drivers por segmento de risco (Alto/Médio/Baixo)
    - narrativa: texto diagnóstico automático
    - perfil_risco: distribuição dos clientes por segmento
    """
    # Usar o melhor modelo disponível (RF > XGB > LR)
    best_model_name = None
    best_importance = {}
    for name in ["Random Forest", "XGBoost", "Logistic Regression"]:
        if name in results and "importance" in results[name] and results[name]["importance"]:
            best_model_name = name
            best_importance = results[name]["importance"]
            break

    if not best_importance:
        return {}

    # ── Análise de Pareto ────────────────────────────────────────────────────
    imp_df = pd.DataFrame({
        "Feature": list(best_importance.keys()),
        "Importância": list(best_importance.values())
    }).sort_values("Importância", ascending=False)

    total_imp = imp_df["Importância"].sum()
    imp_df["Importância_Pct"] = imp_df["Importância"] / total_imp * 100
    imp_df["Importância_Acum"] = imp_df["Importância_Pct"].cumsum()

    pareto_80 = imp_df[imp_df["Importância_Acum"] <= 80].copy()
    if pareto_80.empty:
        pareto_80 = imp_df.head(3).copy()

    # ── Segmentação de Risco ─────────────────────────────────────────────────
    # Usar as probabilidades do melhor modelo
    all_prob = results[best_model_name]["all_prob"]
    prob_df = pd.DataFrame({
        "ID_CLIENTE": list(all_prob.keys()),
        "PROB_CHURN": list(all_prob.values())
    })
    prob_df["SEGMENTO"] = pd.cut(
        prob_df["PROB_CHURN"],
        bins=[0, 0.4, 0.7, 1.01],
        labels=["🟢 Baixo Risco", "🟡 Médio Risco", "🔴 Alto Risco"]
    )

    perfil = prob_df["SEGMENTO"].value_counts().to_dict()

    # ── Tradução dos nomes das features para PT-BR ───────────────────────────
    FEATURE_LABELS = {
        "N_LIGACOES_D90": "Nº de Ligações (D-90)",
        "SPAN_DIAS": "Span Temporal (dias)",
        "INTERVALO_MEDIO_DIAS": "Intervalo Médio entre Ligações",
        "TOKENS_TOTAL": "Volume Total de Texto",
        "TOKENS_MEDIO": "Média de Tokens por Ligação",
        "TAXA_LIGACOES_DIA": "Taxa de Ligações por Dia",
        "SCORE_INTENSIDADE": "Score de Intensidade",
        "freq_7d": "Frequência Últimos 7 Dias",
        "freq_30d": "Frequência Últimos 30 Dias",
        "freq_90d": "Frequência Total (90 Dias)",
        "days_since_last_call": "Dias Desde Última Ligação",
        "avg_interval_days": "Intervalo Médio entre Ligações",
        "max_interval_days": "Maior Hiato entre Ligações",
        "min_interval_days": "Menor Intervalo entre Ligações",
        "std_interval_days": "Volatilidade de Contato",
        "calls_per_week_avg": "Média Semanal de Ligações",
        "avg_token_count": "Média de Palavras por Ligação",
        "max_token_count": "Pico de Palavras (Maior Ligação)",
        "total_mentions_concorrente": "Menções à Concorrência",
        "flag_ameaca_cancelamento": "Ameaça de Cancelamento",
        "total_mentions_problemas": "Menções a Problemas Técnicos",
        "total_mentions_prazos": "Urgência / Menções a Prazos",
        "increase_rate_30d_vs_90d": "Aceleração de Insatisfação (30d/90d)",
        "N_CATEGORIAS_DISTINTAS": "Diversidade de Motivos",
        "REPETICAO_MOTIVO": "Repetição do Mesmo Motivo",
    }

    top_features = pareto_80["Feature"].tolist()
    top_labels = [FEATURE_LABELS.get(f, f) for f in top_features]

    # ── Narrativa diagnóstica automática ────────────────────────────────────
    n_alto = sum(1 for _, row in prob_df.iterrows() if row["PROB_CHURN"] >= 0.7)
    n_medio = sum(1 for _, row in prob_df.iterrows() if 0.4 <= row["PROB_CHURN"] < 0.7)
    total = len(prob_df)

    principais_drivers = ", ".join(top_labels[:3]) if top_labels else "N/A"

    # Categorias de diagnóstico baseadas nos top drivers
    has_nlp_signal = any(f in top_features for f in [
        "total_mentions_concorrente", "flag_ameaca_cancelamento", "total_mentions_problemas"
    ])
    has_freq_signal = any(f in top_features for f in [
        "freq_7d", "freq_30d", "N_LIGACOES_D90", "calls_per_week_avg"
    ])
    has_aceleracao = "increase_rate_30d_vs_90d" in top_features

    conclusao_partes = []
    if has_freq_signal:
        conclusao_partes.append("alta frequência de contato indica insatisfação crônica não resolvida")
    if has_nlp_signal:
        conclusao_partes.append("sinais textuais revelam ameaças explícitas e menção à concorrência")
    if has_aceleracao:
        conclusao_partes.append("aceleração de ligações nos últimos 30 dias sinaliza deterioração aguda")

    conclusao = "; ".join(conclusao_partes) if conclusao_partes else \
        "comportamento de contato intensivo é o principal preditor de churn"

    narrativa = (
        f"**Modelo utilizado:** {best_model_name}\n\n"
        f"De um total de **{total} clientes**, **{n_alto} ({n_alto/max(total,1):.0%})** "
        f"apresentam risco **alto** de churn, **{n_medio} ({n_medio/max(total,1):.0%})** "
        f"risco **médio**.\n\n"
        f"**Os {len(top_features)} principais drivers de risco** — que explicam ~80% do poder preditivo "
        f"do modelo — são: **{', '.join(top_labels[:5])}**.\n\n"
        f"**Diagnóstico de causa raiz comportamental:** {conclusao.capitalize()}. "
        f"Clientes com perfil de alto risco devem ser priorizados em campanhas de retenção proativas."
    )

    return {
        "pareto_df": imp_df,
        "top_features": top_features,
        "top_labels": top_labels,
        "segment_df": prob_df,
        "perfil_risco": perfil,
        "narrativa": narrativa,
        "best_model": best_model_name,
    }


def chart_feature_importance(importance_dict: dict, model_name: str):
    """Gráfico de importância de features."""
    if not importance_dict:
        return None

    FEATURE_LABELS = {
        "N_LIGACOES_D90": "Nº Ligações (D-90)",
        "SPAN_DIAS": "Span Temporal",
        "INTERVALO_MEDIO_DIAS": "Intervalo Médio",
        "TOKENS_TOTAL": "Volume Texto Total",
        "TOKENS_MEDIO": "Média Tokens",
        "TAXA_LIGACOES_DIA": "Taxa Lig./Dia",
        "SCORE_INTENSIDADE": "Score Intensidade",
        "freq_7d": "Freq. 7 Dias",
        "freq_30d": "Freq. 30 Dias",
        "freq_90d": "Freq. 90 Dias",
        "days_since_last_call": "Dias Última Lig.",
        "avg_interval_days": "Intervalo Médio",
        "max_interval_days": "Maior Hiato",
        "min_interval_days": "Menor Intervalo",
        "std_interval_days": "Volatilidade Contato",
        "calls_per_week_avg": "Média Semanal Lig.",
        "avg_token_count": "Média Palavras",
        "max_token_count": "Pico Palavras",
        "total_mentions_concorrente": "Menções Concorrência",
        "flag_ameaca_cancelamento": "Ameaça Cancelamento",
        "total_mentions_problemas": "Menções Problemas",
        "total_mentions_prazos": "Menções Urgência",
        "increase_rate_30d_vs_90d": "Aceleração Insatisfação",
        "N_CATEGORIAS_DISTINTAS": "Diversidade Motivos",
        "REPETICAO_MOTIVO": "Repetição Motivo",
    }

    # ascending=False → menor importância no início do df → aparece na base do gráfico
    # ascending=True  → maior importância no início do df → aparece no TOPO do gráfico horizontal
    df_imp = pd.DataFrame(
        {"Feature": list(importance_dict.keys()), "Importância": list(importance_dict.values())}
    ).sort_values("Importância", ascending=True)  # ascending=True coloca o maior no TOPO do eixo Y
    df_imp["Label"] = df_imp["Feature"].map(lambda x: FEATURE_LABELS.get(x, x))

    fig = px.bar(
        df_imp, x="Importância", y="Label", orientation="h",
        title=f"🔍 Importância das Features — {model_name} (Ordenado por Relevância)",
        color="Importância",
        color_continuous_scale=[[0, "#457B9D"], [1, "#E63946"]],
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#000000"),
        title_font_size=16,
        xaxis=dict(gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0.1)",
            title="",
            categoryorder="total ascending",  # garante ordenação por valor no eixo Y
        ),
        showlegend=False,
        height=max(300, len(df_imp) * 28),
    )
    return fig


def _get_feature_labels():
    """Retorna dicionário de tradução de nomes de features para PT-BR."""
    return {
        "N_LIGACOES_D90": "Nº Ligações D-90", "SPAN_DIAS": "Span Temporal",
        "INTERVALO_MEDIO_DIAS": "Intervalo Médio", "TOKENS_TOTAL": "Volume Texto",
        "TOKENS_MEDIO": "Média Tokens", "TAXA_LIGACOES_DIA": "Taxa Lig./Dia",
        "SCORE_INTENSIDADE": "Score Intensidade", "freq_7d": "Freq. 7d",
        "freq_30d": "Freq. 30d", "freq_90d": "Freq. 90d",
        "days_since_last_call": "Dias Ult. Lig.", "avg_interval_days": "Intervalo Médio",
        "max_interval_days": "Maior Hiato", "min_interval_days": "Menor Intervalo",
        "std_interval_days": "Volatilidade", "calls_per_week_avg": "Média Sem. Lig.",
        "avg_token_count": "Média Palavras", "max_token_count": "Pico Palavras",
        "total_mentions_concorrente": "Menções Concorrência",
        "flag_ameaca_cancelamento": "Ameaça Cancelamento",
        "total_mentions_problemas": "Problemas Técnicos",
        "total_mentions_prazos": "Menções Urgência",
        "increase_rate_30d_vs_90d": "Aceleração Insatisfação",
        "N_CATEGORIAS_DISTINTAS": "Diversidade Motivos",
        "REPETICAO_MOTIVO": "Repetição Motivo",
    }


def chart_pareto_features(pareto_df: pd.DataFrame):
    """
    [BACKWARD COMPAT] Retorna apenas o Pareto principal (≤80%) — gráfico vermelho.
    Use chart_pareto_features_80 e chart_pareto_features_rest para a visualização separada.
    """
    return chart_pareto_features_80(pareto_df)


def chart_pareto_features_80(pareto_df: pd.DataFrame):
    """
    Pareto 1 (vermelho): features que explicam até 80% do poder preditivo.
    Barras vermelhas + linha cumulativa sobreposta.
    """
    if pareto_df is None or pareto_df.empty:
        return None

    FEATURE_LABELS = _get_feature_labels()

    # Só os que ficam dentro dos 80%
    df_all = pareto_df.head(15).copy()
    df_all["Label"] = df_all["Feature"].map(lambda x: FEATURE_LABELS.get(x, x))
    df_80 = df_all[df_all["Importância_Acum"] <= 80].copy()
    if df_80.empty:
        df_80 = df_all.head(3).copy()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_80["Label"],
        y=df_80["Importância_Pct"],
        name="Importância Top-80% (%)",
        marker_color="#E63946",
        text=[f"{v:.1f}%" for v in df_80["Importância_Pct"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Importância: %{y:.1f}%<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df_80["Label"],
        y=df_80["Importância_Acum"],
        mode="lines+markers",
        name="% Acumulado",
        line=dict(color="#1D3557", width=2, dash="dot"),
        marker=dict(size=6, color="#1D3557"),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Acumulado: %{y:.1f}%<extra></extra>",
    ))

    fig.add_hline(
        y=80, line_dash="dash", line_color="#F4A261",
        annotation_text="Threshold 80%", annotation_position="right",
        yref="y2"
    )

    fig.update_layout(
        title="<b>🔴 Pareto 1 — Principais Drivers de Churn (explicam ≈80% do risco)</b>",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#000000"),
        title_font_size=15,
        xaxis=dict(tickangle=-35, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(title="Importância (%)", gridcolor="rgba(0,0,0,0.1)"),
        yaxis2=dict(title="% Acumulado", overlaying="y", side="right",
                    range=[0, 110], gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
    )
    return fig


def chart_pareto_features_rest(pareto_df: pd.DataFrame):
    """
    Pareto 2 (azul): features secundárias que completam os 20% restantes do poder preditivo.
    Barras azuis — menor relevância individual, mas contribuem para 100%.
    """
    if pareto_df is None or pareto_df.empty:
        return None

    FEATURE_LABELS = _get_feature_labels()

    df_all = pareto_df.head(15).copy()
    df_all["Label"] = df_all["Feature"].map(lambda x: FEATURE_LABELS.get(x, x))

    # Determinar o índice de corte (após 80%)
    idx_80 = df_all[df_all["Importância_Acum"] <= 80].shape[0]
    df_rest = df_all.iloc[idx_80:].copy()

    if df_rest.empty:
        return None  # Todos os features já estão no pareto principal

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_rest["Label"],
        y=df_rest["Importância_Pct"],
        name="Importância Secundária (%)",
        marker_color="#A8DADC",
        text=[f"{v:.1f}%" for v in df_rest["Importância_Pct"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Importância: %{y:.1f}%<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df_rest["Label"],
        y=df_rest["Importância_Acum"],
        mode="lines+markers",
        name="% Acumulado",
        line=dict(color="#457B9D", width=2, dash="dot"),
        marker=dict(size=6, color="#457B9D"),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Acumulado: %{y:.1f}%<extra></extra>",
    ))

    fig.add_hline(
        y=100, line_dash="dash", line_color="#2A9D8F",
        annotation_text="100%", annotation_position="right",
        yref="y2"
    )

    fig.update_layout(
        title="<b>🔵 Pareto 2 — Drivers Secundários (complementam os 20% restantes)</b>",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#000000"),
        title_font_size=15,
        xaxis=dict(tickangle=-35, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(title="Importância (%)", gridcolor="rgba(0,0,0,0.1)"),
        yaxis2=dict(title="% Acumulado", overlaying="y", side="right",
                    range=[0, 110], gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
    )
    return fig


def chart_risk_scores(all_prob: dict, title: str = "Score de Risco por Cliente"):
    """Gráfico de score de risco por cliente."""
    df_r = pd.DataFrame(
        {"ID_CLIENTE": list(all_prob.keys()), "SCORE_RISCO": list(all_prob.values())}
    ).sort_values("SCORE_RISCO", ascending=False)

    df_r["COR"] = df_r["SCORE_RISCO"].apply(
        lambda x: "#E63946" if x > 0.7 else ("#F4A261" if x > 0.4 else "#2A9D8F")
    )

    fig = go.Figure(go.Bar(
        x=df_r["ID_CLIENTE"],
        y=df_r["SCORE_RISCO"],
        marker_color=df_r["COR"].tolist(),
        text=[f"{v:.0%}" for v in df_r["SCORE_RISCO"]],
        textposition="outside",
        textfont=dict(color="#000000"),
        hovertemplate="<b>%{x}</b><br>Score: %{y:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title=f"🎯 {title}",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#000000"),
        title_font_size=16,
        xaxis=dict(tickangle=-45, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0.1)", tickformat=".0%", range=[0, 1.15]),
    )
    return fig


# ─── Journey String ─────────────────────────────────────────────────────────


def build_churn_journey_string(df_work: pd.DataFrame, client_id: str, max_steps: int = 6) -> str:
    """
    Constrói a string de jornada de um cliente:
    "Motivo (dd/mm) → Motivo2 (dd/mm) → ..."
    """
    try:
        client_df = df_work[df_work["ID_CLIENTE"].astype(str) == str(client_id)].copy()
        client_df["DATETIME_TRANSCRICAO_LIGACAO"] = pd.to_datetime(
            client_df["DATETIME_TRANSCRICAO_LIGACAO"], errors="coerce"
        )
        client_df = client_df.sort_values("DATETIME_TRANSCRICAO_LIGACAO")
        total = len(client_df)
        parts = []
        for _, row in client_df.head(max_steps).iterrows():
            try:
                date_str = row["DATETIME_TRANSCRICAO_LIGACAO"].strftime("%d/%m")
            except Exception:
                date_str = "?"
            motivo = str(row.get("PERFIL_RECLAMACAO", "?"))
            parts.append(f"{motivo} ({date_str})")
        if total > max_steps:
            parts.append(f"... +{total - max_steps} mais")
        return " → ".join(parts) if parts else "—"
    except Exception:
        return "—"


# ─── Sankey: Jornada de Motivos → Segmento de Risco ─────────────────────────


def chart_sankey_risk_journey(df_work: pd.DataFrame, seg_df: pd.DataFrame) -> go.Figure:
    """
    Sankey multi-etapa: Motivo (Etapa 1) → Motivo (Etapa 2) → ... → Segmento de Risco.
    Cada link é colorido pelo segmento final do cliente.
    Labels dos nós incluem o nome do motivo + contagem de clientes naquela etapa.
    """
    # ── Segmentos limpos ──────────────────────────────────────────────────────
    seg_clean = seg_df[["ID_CLIENTE", "SEGMENTO"]].copy()
    seg_clean["ID_CLIENTE"] = seg_clean["ID_CLIENTE"].astype(str)

    def _safe_seg(val):
        try:
            s = str(val.item() if hasattr(val, "item") else val)
            return s if s not in ("nan", "NaN", "None", "") else "⚪ Indefinido"
        except Exception:
            return "⚪ Indefinido"

    seg_clean["SEGMENTO"] = seg_clean["SEGMENTO"].apply(_safe_seg)
    seg_dict = dict(zip(seg_clean["ID_CLIENTE"], seg_clean["SEGMENTO"]))

    # ── Jornadas por cliente ──────────────────────────────────────────────────
    df_w = df_work[["ID_CLIENTE", "DATETIME_TRANSCRICAO_LIGACAO", "PERFIL_RECLAMACAO"]].copy()
    df_w["ID_CLIENTE"] = df_w["ID_CLIENTE"].astype(str)
    df_w["DATETIME_TRANSCRICAO_LIGACAO"] = pd.to_datetime(
        df_w["DATETIME_TRANSCRICAO_LIGACAO"], errors="coerce"
    )
    df_w = df_w.sort_values(["ID_CLIENTE", "DATETIME_TRANSCRICAO_LIGACAO"])

    MAX_STEPS = 4  # número máximo de etapas no Sankey

    # (src_label, tgt_label, segmento) → contagem de clientes
    transitions: dict = {}
    # Clientes por nó (para o label de contagem)
    node_client_counts: dict = {}

    for client_id, group in df_w.groupby("ID_CLIENTE"):
        motivos = group["PERFIL_RECLAMACAO"].tolist()
        segment = seg_dict.get(str(client_id), "⚪ Indefinido")
        steps = motivos[:MAX_STEPS]
        n = len(steps)

        # Contabilizar clientes por nó (etapas)
        for i, m in enumerate(steps):
            nk = f"Etapa {i+1}|{m}"
            node_client_counts[nk] = node_client_counts.get(nk, 0) + 1
        # Nó do segmento final
        node_client_counts[segment] = node_client_counts.get(segment, 0) + 1

        # Transições entre etapas consecutivas
        for i in range(n - 1):
            src = f"Etapa {i+1}|{steps[i]}"
            tgt = f"Etapa {i+2}|{steps[i+1]}"
            key = (src, tgt, segment)
            transitions[key] = transitions.get(key, 0) + 1

        # Última etapa → Segmento de Risco
        if steps:
            src = f"Etapa {n}|{steps[-1]}"
            tgt = segment
            key = (src, tgt, segment)
            transitions[key] = transitions.get(key, 0) + 1

    if not transitions:
        return None

    # ── Nós únicos ─────────────────────────────────────────────────────────────
    all_nodes_set: set = set()
    for src, tgt, _ in transitions.keys():
        all_nodes_set.add(src)
        all_nodes_set.add(tgt)

    # Ordenar: etapas primeiro (por número e motivo), depois segmentos
    etapa_nodes = sorted(
        [n for n in all_nodes_set if n.startswith("Etapa")],
        key=lambda x: (int(x.split("|")[0].split()[1]), x.split("|")[1])
    )
    seg_nodes = sorted([n for n in all_nodes_set if not n.startswith("Etapa")])
    all_nodes = etapa_nodes + seg_nodes
    node_idx = {node: i for i, node in enumerate(all_nodes)}

    # ── Labels dos nós ─────────────────────────────────────────────────────────
    def make_label(node: str) -> str:
        cnt = node_client_counts.get(node, 0)
        if "|" in node:
            step_part, motivo = node.split("|", 1)
            return f"{step_part.strip()}: {motivo} ({cnt})"
        return f"{node} ({cnt})"

    node_labels = [make_label(n) for n in all_nodes]

    # ── Cores ──────────────────────────────────────────────────────────────────
    SEG_COLORS = {
        "🔴 Alto Risco":  "#E63946",
        "🟡 Médio Risco": "#F4A261",
        "🟢 Baixo Risco": "#2A9D8F",
        "⚪ Indefinido":  "#AAAAAA",
    }
    STEP_COLORS = {
        "Etapa 1": "#1D3557",
        "Etapa 2": "#2E6B97",
        "Etapa 3": "#457B9D",
        "Etapa 4": "#6D97B8",
    }
    LINK_SEG_COLORS = {
        "🔴 Alto Risco":  "rgba(230, 57,  70,  0.40)",
        "🟡 Médio Risco": "rgba(244, 162,  97,  0.40)",
        "🟢 Baixo Risco": "rgba( 42, 157, 143,  0.40)",
        "⚪ Indefinido":  "rgba(180, 180, 180,  0.30)",
    }

    node_colors = []
    for node in all_nodes:
        if node in SEG_COLORS:
            node_colors.append(SEG_COLORS[node])
        elif "|" in node:
            step_key = node.split("|")[0].strip()
            node_colors.append(STEP_COLORS.get(step_key, "#888888"))
        else:
            node_colors.append("#888888")

    sources, targets, values, link_colors = [], [], [], []
    for (src, tgt, seg), val in transitions.items():
        sources.append(node_idx[src])
        targets.append(node_idx[tgt])
        values.append(val)
        link_colors.append(LINK_SEG_COLORS.get(seg, "rgba(150,150,150,0.35)"))

    # ── Figura ─────────────────────────────────────────────────────────────────
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18,
            thickness=22,
            line=dict(color="rgba(0,0,0,0.25)", width=0.5),
            label=node_labels,
            color=node_colors,
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate=(
                "De <b>%{source.label}</b><br>"
                "Para <b>%{target.label}</b><br>"
                "Clientes: <b>%{value}</b><extra></extra>"
            ),
        ),
    )])

    fig.update_layout(
        title=dict(
            text="<b>🌊 Fluxo da Jornada — Evolução de Motivos → Segmento de Risco</b>",
            font=dict(size=16, color="#000000"),
        ),
        font=dict(size=11, color="#000000"),
        height=560,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, l=20, r=20, b=20),
    )
    return fig
