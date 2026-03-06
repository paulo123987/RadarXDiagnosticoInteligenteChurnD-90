"""
modules/eda.py
Análise Exploratória de Dados (EDA) com Plotly.
Versão Light: Branco, Cinza Claro, Preto e Vermelho.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ─── Paleta de cores (Light) ──────────────────────────────────────────────────
COLORS = {
    "primary": "#E63946",  # Vermelho
    "secondary": "#457B9D", # Azul (usado como secundário discreto)
    "accent": "#F4A261",    # Laranja
    "bg": "#FFFFFF",        # Branco
    "surface": "#F0F2F6",   # Cinza Claro
    "text": "#000000",      # Preto
}

PALETTE = [
    "#E63946", "#1D3557", "#457B9D", "#A8DADC",
    "#F4A261", "#2A9D8F", "#264653", "#6D597A",
]


def update_fig_layout(fig, title: str):
    """Padroniza o layout das figuras para o tema claro."""
    fig.update_layout(
        title=f"<b>{title}</b>",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        title_font_size=18,
        xaxis=dict(gridcolor="rgba(0,0,0,0.1)", tickfont=dict(color=COLORS["text"])),
        yaxis=dict(gridcolor="rgba(0,0,0,0.1)", tickfont=dict(color=COLORS["text"])),
        legend=dict(font=dict(color=COLORS["text"])),
    )
    return fig


def chart_volume_por_mes(df: pd.DataFrame):
    """Volume de transcrições por mês."""
    agg = df.groupby("MES_ANO").size().reset_index(name="VOLUME")
    agg = agg.sort_values("MES_ANO")
    fig = px.bar(
        agg, x="MES_ANO", y="VOLUME",
        color_discrete_sequence=[COLORS["primary"]],
        labels={"MES_ANO": "Mês/Ano", "VOLUME": "Qtd. Ligações"},
    )
    return update_fig_layout(fig, "📅 Volume de Ligações por Mês")


def chart_ligacoes_por_cliente(df: pd.DataFrame):
    """Quantidade de ligações por cliente."""
    agg = df.groupby("ID_CLIENTE").size().reset_index(name="N_LIGACOES")
    agg = agg.sort_values("N_LIGACOES", ascending=False)
    fig = px.bar(
        agg, x="ID_CLIENTE", y="N_LIGACOES",
        color="N_LIGACOES",
        color_continuous_scale=[[0, COLORS["secondary"]], [1, COLORS["primary"]]],
        labels={"ID_CLIENTE": "Cliente", "N_LIGACOES": "Ligações"},
    )
    fig = update_fig_layout(fig, "👤 Ligações por Cliente")
    fig.update_coloraxes(showscale=False)
    return fig


def chart_distribuicao_motivos(df_classified: pd.DataFrame, title="🎯 Distribuição de Macro Motivos"):
    """Distribuição de macro motivos classificados."""
    # Suportar tanto MACRO_MOTIVO quanto PERFIL_RECLAMACAO
    motivo_col = None
    if "PERFIL_RECLAMACAO" in df_classified.columns:
        motivo_col = "PERFIL_RECLAMACAO"
    elif "MACRO_MOTIVO" in df_classified.columns:
        motivo_col = "MACRO_MOTIVO"
    
    if df_classified is None or motivo_col is None:
        return None
    
    agg = df_classified[motivo_col].value_counts().reset_index()
    agg.columns = ["MOTIVO", "COUNT"]
    fig = px.pie(
        agg, names="MOTIVO", values="COUNT",
        color_discrete_sequence=PALETTE,
        hole=0.45,
    )
    return update_fig_layout(fig, title)


def chart_heatmap_dia_hora(df: pd.DataFrame):
    """Heatmap de ligações por dia da semana x hora."""
    df2 = df.copy()
    df2["HORA"] = df2["DATETIME_TRANSCRICAO_LIGACAO"].dt.hour
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pt = df2.pivot_table(index="DIA_SEMANA", columns="HORA", aggfunc="size", fill_value=0)
    pt = pt.reindex([d for d in order if d in pt.index])

    fig = go.Figure(data=go.Heatmap(
        z=pt.values,
        x=[f"{h}h" for h in pt.columns],
        y=pt.index.tolist(),
        colorscale=[[0, "#F0F2F6"], [0.5, "#457B9D"], [1, "#E63946"]],
        showscale=True,
    ))
    return update_fig_layout(fig, "🕐 Concentração de Ligações (Dia × Hora)")


def chart_sankey_jornada(df_classified: pd.DataFrame):
    """Visualização Sankey do fluxo de motivos na jornada."""
    if df_classified is None or "MACRO_MOTIVO" not in df_classified.columns:
        return None
    
    # Gerar pares Origem -> Destino na sequência do cliente
    df_sorted = df_classified.sort_values(["ID_CLIENTE", "DATETIME_TRANSCRICAO_LIGACAO"])
    df_sorted["NEXT_MOTIVO"] = df_sorted.groupby("ID_CLIENTE")["MACRO_MOTIVO"].shift(-1)
    
    # Adicionar evento final de Churn para a última ligação
    df_sorted["NEXT_MOTIVO"] = df_sorted["NEXT_MOTIVO"].fillna("CHURN")
    
    flows = df_sorted.groupby(["MACRO_MOTIVO", "NEXT_MOTIVO"]).size().reset_index(name="VALUE")
    
    all_nodes = list(set(flows["MACRO_MOTIVO"]) | set(flows["NEXT_MOTIVO"]))
    node_map = {node: i for i, node in enumerate(all_nodes)}
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=all_nodes,
            color=[COLORS["primary"] if n == "CHURN" else COLORS["secondary"] for n in all_nodes]
        ),
        link=dict(
            source=flows["MACRO_MOTIVO"].map(node_map),
            target=flows["NEXT_MOTIVO"].map(node_map),
            value=flows["VALUE"]
        )
    )])
    return update_fig_layout(fig, "🌊 Fluxo da Jornada (Motivo 1 → Motivo 2 → Churn)")


def chart_sankey_dinamico(
    df_classified: pd.DataFrame,
    filtro_motivos: list = None,
    conf_minima: int = 0,
    periodo: tuple = None
):
    """
    Sankey dinâmico do fluxo completo da jornada, responsivo aos filtros do Dashboard.
    Mostra Motivo1 → Motivo2 → ... → Churn com todos os motivos intermediários.
    """
    # Suportar tanto MACRO_MOTIVO quanto PERFIL_RECLAMACAO
    motivo_col = None
    if "PERFIL_RECLAMACAO" in df_classified.columns:
        motivo_col = "PERFIL_RECLAMACAO"
    elif "MACRO_MOTIVO" in df_classified.columns:
        motivo_col = "MACRO_MOTIVO"
    
    if df_classified is None or motivo_col is None:
        return None

    df_f = df_classified.copy()

    # Aplicar filtro de confiança (suportar ambos nomes)
    conf_col = "CONFIDENCE_SCORE" if "CONFIDENCE_SCORE" in df_f.columns else "CONFIDENCE"
    if conf_col in df_f.columns:
        df_f = df_f[df_f[conf_col] >= conf_minima]

    # Aplicar filtro de período
    if periodo and len(periodo) == 2:
        dt_min = pd.Timestamp(periodo[0])
        dt_max = pd.Timestamp(periodo[1]) + pd.Timedelta(days=1)
        df_f = df_f[
            (df_f["DATETIME_TRANSCRICAO_LIGACAO"] >= dt_min) &
            (df_f["DATETIME_TRANSCRICAO_LIGACAO"] <= dt_max)
        ]

    # Filtrar clientes cujo último motivo está na lista selecionada
    if filtro_motivos and "Todos" not in filtro_motivos and filtro_motivos:
        clientes_validos = (
            df_f.sort_values(["ID_CLIENTE", "DATETIME_TRANSCRICAO_LIGACAO"])
            .groupby("ID_CLIENTE")[motivo_col]
            .last()
        )
        clientes_validos = clientes_validos[clientes_validos.isin(filtro_motivos)].index
        df_f = df_f[df_f["ID_CLIENTE"].isin(clientes_validos)]

    if df_f.empty:
        return None

    # Construir pares de fluxo com posição na jornada
    df_sorted = df_f.sort_values(["ID_CLIENTE", "DATETIME_TRANSCRICAO_LIGACAO"])
    df_sorted["NEXT_MOTIVO"] = df_sorted.groupby("ID_CLIENTE")[motivo_col].shift(-1)
    df_sorted["NEXT_MOTIVO"] = df_sorted["NEXT_MOTIVO"].fillna("⛔ CHURN")

    # Adicionar posição sequencial (Mot.1 → Mot.2 etc.)
    df_sorted["POS"] = df_sorted.groupby("ID_CLIENTE").cumcount()
    df_sorted["SRC_NODE"] = df_sorted.apply(
        lambda r: f"{r[motivo_col]} [{r['POS']+1}]", axis=1
    )
    df_sorted["TGT_NODE"] = df_sorted.apply(
        lambda r: f"{r['NEXT_MOTIVO']} [{r['POS']+2}]" if r["NEXT_MOTIVO"] != "⛔ CHURN" else "⛔ CHURN",
        axis=1
    )

    flows = df_sorted.groupby(["SRC_NODE", "TGT_NODE"]).size().reset_index(name="VALUE")

    all_nodes = list(dict.fromkeys(list(flows["SRC_NODE"]) + list(flows["TGT_NODE"])))
    node_map = {node: i for i, node in enumerate(all_nodes)}

    node_colors = [
        COLORS["primary"] if "CHURN" in n
        else COLORS["accent"] if "[1]" in n
        else COLORS["secondary"]
        for n in all_nodes
    ]

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20, thickness=22,
            line=dict(color="rgba(0,0,0,0.3)", width=0.5),
            label=all_nodes,
            color=node_colors,
            hovertemplate="%{label}<br>Fluxo total: %{value}<extra></extra>"
        ),
        link=dict(
            source=[node_map[s] for s in flows["SRC_NODE"]],
            target=[node_map[t] for t in flows["TGT_NODE"]],
            value=flows["VALUE"],
            color="rgba(69,123,157,0.3)",
            hovertemplate="%{source.label} → %{target.label}<br>Clientes: %{value}<extra></extra>"
        )
    )])
    fig.update_layout(height=480)
    return update_fig_layout(fig, "🌊 Fluxo da Jornada Completa (Motivo 1 → Motivo 2 → ... → Churn)")


def chart_bubble_causaraiz(df_classified: pd.DataFrame):
    """
    Gráfico de bolhas mostrando a distribuição percentual das causas raiz predominantes.
    Tamanho e cor da bolha = % de cada causa raiz no total.
    """
    # Suportar tanto MACRO_MOTIVO quanto PERFIL_RECLAMACAO
    motivo_col = None
    if "PERFIL_RECLAMACAO" in df_classified.columns:
        motivo_col = "PERFIL_RECLAMACAO"
    elif "MACRO_MOTIVO" in df_classified.columns:
        motivo_col = "MACRO_MOTIVO"
    
    if df_classified is None or motivo_col is None:
        return None

    agg = df_classified[motivo_col].value_counts().reset_index()
    agg.columns = ["CAUSA_RAIZ", "CONTAGEM"]
    total = agg["CONTAGEM"].sum()
    agg["PCT"] = (agg["CONTAGEM"] / total * 100).round(1)
    agg["LABEL"] = agg["CAUSA_RAIZ"] + "<br>" + agg["PCT"].astype(str) + "%"

    fig = go.Figure()
    for i, row in agg.iterrows():
        fig.add_trace(go.Scatter(
            x=[i % 4],
            y=[i // 4],
            mode="markers+text",
            marker=dict(
                size=max(40, row["PCT"] * 3.5),
                color=PALETTE[i % len(PALETTE)],
                opacity=0.85,
                line=dict(width=2, color="white")
            ),
            text=row["LABEL"],
            textposition="middle center",
            textfont=dict(color="white", size=11, family="Segoe UI, Arial"),
            name=row["CAUSA_RAIZ"],
            hovertemplate=(
                f"<b>{row['CAUSA_RAIZ']}</b><br>"
                f"Clientes: {row['CONTAGEM']}<br>"
                f"Participação: {row['PCT']}%<extra></extra>"
            ),
            showlegend=True,
        ))

    fig.update_layout(
        xaxis=dict(visible=False, range=[-0.8, 3.8]),
        yaxis=dict(visible=False, range=[-0.8, max(1, (len(agg)//4) + 0.5)]),
        height=380,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.25,
            xanchor="center", x=0.5,
            font=dict(color="#000000", size=10)
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        title=dict(text="<b>🫧 Causa Raiz Predominante (%)</b>", font=dict(size=18, color=COLORS["text"]))
    )
    return fig


def chart_correlacao_churn(df_d90: pd.DataFrame):
    """Correlação entre Nº de ligações e Score de Risco/Churn."""
    try:
        # Tenta gerar com trendline (requer statsmodels)
        fig = px.scatter(
            df_d90, x="N_LIGACOES_D90", y="TOKENS_TOTAL",
            size="SPAN_DIAS", color="N_LIGACOES_D90",
            color_continuous_scale="Reds",
            labels={"N_LIGACOES_D90": "Nº Ligações", "TOKENS_TOTAL": "Volume Termos"},
            trendline="ols", trendline_color_override=COLORS["primary"]
        )
    except Exception:
        # Fallback sem trendline se statsmodels falhar/não existir
        fig = px.scatter(
            df_d90, x="N_LIGACOES_D90", y="TOKENS_TOTAL",
            size="SPAN_DIAS", color="N_LIGACOES_D90",
            color_continuous_scale="Reds",
            labels={"N_LIGACOES_D90": "Nº Ligações", "TOKENS_TOTAL": "Volume Termos"}
        )
    return update_fig_layout(fig, "📊 Correlação: Frequência vs Volume de Texto")


def wordcloud_fig(df: pd.DataFrame, motive: str = None):
    """Gera nuvem de palavras das transcrições, opcionalmente filtrada por motivo."""
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import io as _io
        from PIL import Image

        # Stopwords expandidas em português
        STOPWORDS = {
            # Palavras específicas a remover
            "atendente", "cliente",
            # Artigos
            "o", "a", "os", "as", "um", "uma", "uns", "umas",
            # Preposições
            "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas",
            "por", "para", "com", "sem", "sob", "sobre", "perante", "ante",
            "ao", "aos", "à", "às",
            # Pronomes
            "eu", "tu", "ele", "ela", "nós", "vós", "eles", "elas",
            "me", "te", "se", "lhe", "nos", "vos", "lhes", "o", "a",
            "meu", "minha", "meus", "minhas", "seu", "sua", "seus", "suas",
            "nosso", "nossa", "nossos", "nossas",
            # Verbos auxiliares e comuns
            "é", "são", "ser", "foi", "eram", "era", "seja", "sejam",
            "estar", "está", "estou", "estão", "estava", "estavam",
            "ter", "tem", "temos", "têm", "teve", "tinha", "tinham",
            "haver", "há", "houve", "houver",
            "fazer", "faz", "fez", "fazem",
            # Conjunções
            "e", "ou", "mas", "porém", "contudo", "todavia", "entretanto",
            "que", "se", "porque", "pois", "como",
            # Advérbios
            "não", "sim", "já", "mais", "muito", "também", "quando", "onde",
            "aqui", "lá", "ali", "aí", "agora", "hoje", "ontem", "amanhã",
            # Outros
            "isso", "isto", "esse", "essa", "este", "esta", "aquele", "aquela",
            "pelo", "pela", "pelos", "pelas",
            "vocês", "ligando", "ligação", "chamado", "protocolo",
            "posso", "pode", "podem", "ajudar", "ver", "verificar",
            "favor", "obrigado", "obrigada", "desculpa", "desculpe"
        }

        # Suportar tanto MACRO_MOTIVO quanto PERFIL_RECLAMACAO
        motivo_col = None
        if "PERFIL_RECLAMACAO" in df.columns:
            motivo_col = "PERFIL_RECLAMACAO"
        elif "MACRO_MOTIVO" in df.columns:
            motivo_col = "MACRO_MOTIVO"

        if motive and motivo_col and motivo_col in df.columns:
            subset = df[df[motivo_col] == motive]
        else:
            subset = df

        if subset.empty: return None

        texto = " ".join(subset["TRANSCRICAO_LIGACAO_CLIENTE"].dropna().tolist())
        
        # Limpar "Atendente:" e "Cliente:" do texto
        texto = texto.replace("Atendente:", "").replace("Cliente:", "")
        texto = texto.replace("atendente:", "").replace("cliente:", "")
        
        if not texto.strip(): return None

        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            colormap="Reds",
            stopwords=STOPWORDS,
            max_words=100,
            collocations=False,
        ).generate(texto)

        buf = _io.BytesIO()
        plt.figure(figsize=(10, 5), facecolor="white")
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close()
        buf.seek(0)
        return buf
    except Exception as e:
        return None
