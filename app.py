"""
app.py
Radar X – Diagnóstico Inteligente de Churn (D-90)
Aplicação Streamlit com Foco Executivo, Dashboards de Jornada e IA (LangGraph).
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Importação dos módulos locais
from modules.data_utils import (
    load_csv, validate_columns, preprocess, consolidate_d90, 
    get_synthetic_csv_path, get_top_journey_patterns,
    get_enriched_csv_path, save_enriched_dataset, load_enriched_dataset_if_exists
)
from modules.eda import (
    chart_volume_por_mes, chart_ligacoes_por_cliente, chart_distribuicao_motivos,
    chart_heatmap_dia_hora, chart_sankey_jornada, chart_sankey_dinamico,
    chart_bubble_causaraiz, chart_correlacao_churn, wordcloud_fig,
    COLORS as EDA_COLORS
)
from modules.agents import run_langgraph_pipeline, MACRO_MOTIVOS, batch_classify_all_transcriptions
from modules.ml_model import (
    build_features, build_features_expanded, train_models,
    generate_root_cause_diagnosis,
    chart_feature_importance, chart_risk_scores,
    chart_pareto_features, chart_pareto_features_80, chart_pareto_features_rest,
    build_churn_journey_string, chart_sankey_risk_journey
)

# ─── Configuração da Página ──────────────────────────────────────────────────

st.set_page_config(
    page_title="Radar X – Diagnóstico Churn",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS para estética Light (Branco, Cinza, Preto, Vermelho)
st.markdown(f"""
<style>
    .main {{ background-color: #FFFFFF; color: #000000; }}
    [data-testid="stSidebar"] {{ background-color: #F8F9FA; border-right: 1px solid #E0E0E0; }}
    .stMetric {{ 
        background-color: #FFFFFF; 
        padding: 20px; 
        border-radius: 8px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 5px solid #E63946; 
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; border-bottom: 2px solid #F0F2F6; }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px; background-color: #F8F9FA; border-radius: 4px 4px 0 0; 
        color: #666666; border: 1px solid #F0F2F6; margin-bottom: -2px;
    }}
    .stTabs [aria-selected="true"] {{ 
        background-color: #FFFFFF !important; 
        color: #E63946 !important; 
        border-bottom: 2px solid #E63946 !important;
        font-weight: bold;
    }}
    h1, h2, h3 {{ color: #000000; font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }}
    .stButton>button {{
        background-color: #E63946; color: white; border-radius: 4px; border: none;
        padding: 0.5rem 1rem; transition: all 0.3s;
    }}
    .stButton>button:hover {{ background-color: #C1121F; color: white; border: none; }}
    .diag-card {{
        background: #F8F9FA;
        border-left: 5px solid #E63946;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }}
    .diag-title {{
        font-size: 13px;
        color: #666;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }}
    .diag-value {{
        font-size: 16px;
        color: #000;
        font-weight: 500;
    }}
    .diag-conf {{
        font-size: 12px;
        color: #E63946;
        font-weight: 700;
        margin-top: 4px;
    }}
    .resumo-card {{
        background: linear-gradient(135deg, #1D3557 0%, #457B9D 100%);
        color: white;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 18px;
        box-shadow: 0 4px 12px rgba(29,53,87,0.15);
    }}
</style>
""", unsafe_allow_html=True)

# ─── Estado da Sessão ────────────────────────────────────────────────────────

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = pd.DataFrame()

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<h1 style='color: #E63946; margin-bottom: 0;'>RADAR X</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #666; margin-top: 0;'>Diagnóstico de Churn D-90</p>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("### 📂 Fonte de Dados")
    source_choice = st.radio("Selecione:", ("Base Sintética (Radar X)", "Carregar CSV Próprio"))
    
    df_raw = None
    if source_choice == "Carregar CSV Próprio":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            try:
                df_raw = load_csv(uploaded_file)
            except Exception as e:
                st.error(f"Erro no carregamento: {e}")
    else:
        path = get_synthetic_csv_path()
        if os.path.exists(path):
            df_raw = pd.read_csv(path)
        else:
            st.warning("Executor de dados sintéticos não encontrado.")

    st.divider()
    
    st.markdown("### 🤖 Motor de IA")
    provider = st.selectbox("Provedor", ["OpenAI", "Groq"])
    
    if provider == "OpenAI":
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        model = st.selectbox("Modelo", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    else:
        api_key = st.secrets.get("GROQ_API_KEY", "")
        model = st.selectbox("Modelo", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])
        
    threshold = st.slider("Corte de Confiança (%)", 0, 100, 90)
    
    st.divider()
    if st.button("🗑️ Limpar Cache de IA"):
        st.session_state.analysis_results = pd.DataFrame()
        st.rerun()

# ─── Função auxiliar: renderizar storytelling do diagnóstico ─────────────────

def _render_diagnostico_storytelling(diag: dict, cliente_id: str, row):
    """Renderiza o storytelling aprimorado do diagnóstico IA."""
    st.markdown(f"### 🩺 Diagnóstico — Cliente `{cliente_id}`")
    st.caption(f"Jornada: {row.get('N_LIGACOES_D90', '?')} ligações em {row.get('SPAN_DIAS', '?')} dias")
    
    # ── Resumo Sequencial ────────────────────────────────────────────────────
    resumo = diag.get("resumo_sequencial") or diag.get("diagnostico_sequencial", "—")
    st.markdown(
        f"""<div class="resumo-card">
            <div style="font-size:11px; text-transform:uppercase; letter-spacing:1px; opacity:0.7; margin-bottom:8px;">📖 RESUMO SEQUENCIAL DA JORNADA</div>
            <div style="font-size:15px; line-height:1.6;">{resumo}</div>
        </div>""",
        unsafe_allow_html=True
    )

    # ── Métricas de diagnóstico em 2 colunas ─────────────────────────────────
    col1, col2 = st.columns(2)
    
    with col1:
        causa = diag.get("causa_raiz_predominante") or diag.get("causa_raiz", "—")
        causa_score = diag.get("causa_raiz_score", diag.get("confianca_geral", diag.get("confianca", "—")))
        st.markdown(
            f"""<div class="diag-card">
                <div class="diag-title">🌱 Causa Provável da Raiz da Jornada</div>
                <div class="diag-value">{causa}</div>
                <div class="diag-conf">Confiança da IA = {causa_score}%</div>
            </div>""",
            unsafe_allow_html=True
        )
        sentimento = diag.get("sentimento_jornada", "—")
        sentimento_score = diag.get("sentimento_score", "—")
        st.markdown(
            f"""<div class="diag-card">
                <div class="diag-title">💬 Sentimento da Jornada</div>
                <div class="diag-value">{sentimento}</div>
                <div class="diag-conf">Confiança da IA = {sentimento_score}%</div>
            </div>""",
            unsafe_allow_html=True
        )

    with col2:
        gatilho = diag.get("evento_gatilho") or diag.get("ruptura", "—")
        gatilho_score = diag.get("evento_gatilho_score", "—")
        st.markdown(
            f"""<div class="diag-card">
                <div class="diag-title">⚡ Evento Gatilho do Churn / Ruptura</div>
                <div class="diag-value">{gatilho}</div>
                <div class="diag-conf">Confiança da IA = {gatilho_score}%</div>
            </div>""",
            unsafe_allow_html=True
        )
        padroes = diag.get("padroes_detectados") or diag.get("padroes_recorrentes", "—")
        padroes_score = diag.get("padroes_score", "—")
        st.markdown(
            f"""<div class="diag-card">
                <div class="diag-title">🔍 Padrões Detectados na Jornada</div>
                <div class="diag-value">{padroes}</div>
                <div class="diag-conf">Confiança da IA = {padroes_score}%</div>
            </div>""",
            unsafe_allow_html=True
        )
    
    # ── Indicadores adicionais ────────────────────────────────────────────────
    conf_geral = diag.get("confianca_geral", diag.get("confianca", "—"))
    escalada = diag.get("escalada", False)
    ind_col1, ind_col2, ind_col3 = st.columns(3)
    ind_col1.metric("🎯 Confiança Geral da IA", f"{conf_geral}%")
    ind_col2.metric("📈 Escalada Detectada", "✅ Sim" if escalada else "❌ Não")
    ruptura = diag.get("ruptura", "—")
    ind_col3.metric("🔀 Ponto de Ruptura", ruptura[:40] + "..." if len(str(ruptura)) > 40 else ruptura)


# ─── Lógica Principal ────────────────────────────────────────────────────────

if df_raw is not None:
    valid, missing = validate_columns(df_raw)
    if not valid:
        st.error(f"Seu CSV precisa das colunas: {missing}")
        st.stop()
    
    # Processamento base
    df = preprocess(df_raw)
    df_d90 = consolidate_d90(df)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NOVO SISTEMA: Carregar classificações salvas ou criar mock temporário
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Tentar carregar classificações previamente salvas
    df_enriched = load_enriched_dataset_if_exists(df)
    
    if df_enriched is not None:
        # Classificações reais existem!
        st.session_state.classification_completed = True
        df_work = df_enriched
    else:
        # Não existe classificação - usar mock temporário
        st.session_state.classification_completed = False
        df_work = df.copy()
        
        # Função mock simples para visualização temporária
        def mock_classify(text):
            text = str(text).lower()
            if "técnico" in text or "internet" in text or "lenta" in text or "caiu" in text or "cai" in text: 
                return "Problema Técnico"
            if "fatura" in text or "cobrança" in text or "preço" in text or "valor" in text or "mensalidade" in text: 
                return "Financeiro"
            if "concorr" in text or "operadora" in text or "outra" in text: 
                return "Concorrência"
            if "atend" in text or "grosseiro" in text or "supervisor" in text or "espera" in text: 
                return "Atendimento"
            if "mudar" in text or "mudança" in text or "endereço" in text or "cep" in text: 
                return "Mudança de Endereço/Localidade"
            if "desemp" in text or "dificuld" in text or "financeira" in text:
                return "Pessoal"
            return "Não identificado"
        
        df_work["PERFIL_RECLAMACAO"] = df_work["TRANSCRICAO_LIGACAO_CLIENTE"].apply(mock_classify)
        df_work["CONFIDENCE_SCORE"] = np.random.randint(60, 80, size=len(df_work))

    # ─── TABS DO APP ──────────────────────────────────────────────────────────
    
    t_dash, t_jornadas, t_classif, t_ia, t_ml = st.tabs([
        "📈 Dashboard Executivo", "🌊 Jornadas", "📋 Classificações", "🤖 Diagnóstico IA", "🧬 Predição Risco"
    ])
    
    # ─── 1. Dashboard Executivo ───
    with t_dash:
        st.title("📊 Raio-X Executivo da Jornada de Churn")
        
        # KPIs no Topo
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Clientes Analisados", df["ID_CLIENTE"].nunique())
        
        main_motive = df_work["PERFIL_RECLAMACAO"].value_counts().index[0]
        main_motive_pct = (df_work["PERFIL_RECLAMACAO"] == main_motive).mean()
        kpi2.metric("Principal Motivo", f"{main_motive}")
        kpi3.metric(f"% Churn – {main_motive}", f"{main_motive_pct:.0%}")
        
        avg_int = len(df) / df["ID_CLIENTE"].nunique()
        kpi4.metric("Ticket Médio Interações", f"{avg_int:.1f}")
        
        st.divider()
        
        # ═══════════════════════════════════════════════════════════════════════
        # SEÇÃO DE CLASSIFICAÇÃO AUTOMÁTICA
        # ═══════════════════════════════════════════════════════════════════════
        
        st.markdown("### 🤖 Classificação Automática de Perfil de Reclamação")
        
        llm_p_config = {
            "provider": provider,
            "api_key": api_key,
            "model": model,
            "threshold": threshold
        }
        
        if not st.session_state.get("classification_completed", False):
            # Ainda não foi classificado
            st.warning("⚠️ **As transcrições ainda não foram classificadas pela IA.** Execute a classificação para obter insights mais precisos.")
            
            col_info, col_btn = st.columns([3, 1])
            
            with col_info:
                st.info(f"""
                📊 **Total de transcrições**: {len(df)}  
                🧠 **Sistema**: Arquitetura Multiagentes LangGraph  
                ⏱️ **Tempo estimado**: ~{int(len(df) * 1.5)} segundos  
                💾 **Cache**: Resultados salvos em `dados_churn_sintetico_enriquecido.csv`
                """)
            
            with col_btn:
                if st.button("🚀 Gerar Classificações", type="primary", use_container_width=True, key="btn_classify_main"):
                    if not api_key:
                        st.error("❌ Configure a API Key no sidebar primeiro!")
                    else:
                        with st.spinner(f"🔄 Classificando {len(df)} transcrições com IA..."):
                            try:
                                df_classified = batch_classify_all_transcriptions(df, llm_p_config)
                                save_enriched_dataset(df_classified)
                                st.session_state.classification_completed = True
                                st.success(f"✅ {len(df_classified)} transcrições classificadas com sucesso!")
                                st.balloons()
                                import time
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Erro na classificação: {e}")
                                st.exception(e)
        
        else:
            # Classificações concluídas
            st.success("✅ **Classificações por IA concluídas!** Os dados abaixo refletem análises reais.")
            
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("📊 Transcrições", len(df_work))
            col_b.metric("🎯 Perfis Únicos", df_work["PERFIL_RECLAMACAO"].nunique())
            col_c.metric("🔍 Confiança Média", f"{df_work['CONFIDENCE_SCORE'].mean():.0f}%")
            
            with col_d:
                if st.button("🔄 Reclassificar Tudo", key="btn_reclassify"):
                    if os.path.exists(get_enriched_csv_path()):
                        os.remove(get_enriched_csv_path())
                    st.session_state.classification_completed = False
                    st.rerun()
        
        st.divider()
        
        # Gráficos Principais: Bolha (esq) + Rosca (dir)
        col_l, col_r = st.columns([2, 1])
        with col_l:
            bubble_fig = chart_bubble_causaraiz(df_work)
            if bubble_fig:
                st.plotly_chart(bubble_fig, use_container_width=True)
        with col_r:
            st.plotly_chart(chart_distribuicao_motivos(df_work, "Causa Raiz Predominante"), use_container_width=True)
            
        st.divider()
        
        # Filtros de Visualização
        st.subheader("🔍 Filtros de Visualização")
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            f_motive = st.multiselect("Motivo", ["Todos"] + list(MACRO_MOTIVOS), default="Todos")
        with f_col2:
            f_conf = st.slider("Confiança Mínima IA (%)", 0, 100, 0)
        with f_col3:
            f_date = st.date_input(
                "Período Churn",
                [df["DATETIME_TRANSCRICAO_LIGACAO"].min(), df["DATETIME_TRANSCRICAO_LIGACAO"].max()]
            )
        
        # Filtragem do DataFrame de Jornadas (df_d90)
        df_d90_dash = df_d90.merge(
            df_work.groupby("ID_CLIENTE")["PERFIL_RECLAMACAO"].last().reset_index(),
            on="ID_CLIENTE", how="left"
        )
        if "CONFIDENCE_SCORE" not in df_d90_dash.columns:
            df_d90_dash["CONFIDENCE_SCORE"] = np.random.randint(85, 99, size=len(df_d90_dash))
            
        # Aplicar filtros
        if "Todos" not in f_motive and f_motive:
            df_d90_dash = df_d90_dash[df_d90_dash["PERFIL_RECLAMACAO"].isin(f_motive)]
        df_d90_dash = df_d90_dash[df_d90_dash["CONFIDENCE_SCORE"] >= f_conf]
        
        st.markdown("### Clientes sob Diagnóstico")
        
        # Validação para prevenir erro React
        if not df_d90_dash.empty:
            df_d90_display = df_d90_dash[["ID_CLIENTE", "TIMELINE", "PERFIL_RECLAMACAO", "CONFIDENCE_SCORE", "N_LIGACOES_D90"]].fillna("").rename(
                columns={"TIMELINE": "Fluxo da Jornada", "PERFIL_RECLAMACAO": "Causa Raiz", "CONFIDENCE_SCORE": "IA Conf (%)"}
            )
            st.dataframe(df_d90_display, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum cliente encontrado para os filtros selecionados.")
        st.markdown("---")
        st.markdown("### 🌊 Fluxo da Jornada – Narrativa Completa")
        st.caption("Visualiza o caminho de cada motivo de insatisfação até o Churn, filtrado pelos critérios acima.")
        sankey_fig = chart_sankey_dinamico(
            df_work,
            filtro_motivos=f_motive,
            conf_minima=f_conf,
            periodo=f_date if len(f_date) == 2 else None
        )
        if sankey_fig:
            st.plotly_chart(sankey_fig, use_container_width=True)
        else:
            st.info("Nenhum dado disponível para os filtros selecionados.")

    # ─── 2. Jornadas ───
    with t_jornadas:
        st.header("🌊 Padrões e Experiência do Cliente")
        
        col_patterns, col_corr = st.columns([1, 1])
        with col_patterns:
            patterns = get_top_journey_patterns(df_work)
            st.markdown("#### 🏆 Top 3 Jornadas de Churn")
            if patterns:
                for i, p in enumerate(patterns):
                    st.warning(f"**#{i+1}:** {p['JORNADA']} ({p['FREQUENCIA']} casos)")
            else:
                st.info("Aguardando classificação real das interações.")
        
        with col_corr:
            st.plotly_chart(chart_correlacao_churn(df_d90), use_container_width=True)
            
        st.divider()
        st.markdown("### Detalhamento das Jornadas")
        for idx, row in df_d90.head(10).iterrows():
            cli_data = df_work[df_work["ID_CLIENTE"] == row["ID_CLIENTE"]]
            motive_label = cli_data["PERFIL_RECLAMACAO"].iloc[-1] if not cli_data.empty else "Não identificado"
            
            with st.expander(f"👤 {row['ID_CLIENTE']} | {row['N_LIGACOES_D90']} ligações | Causa: {motive_label}"):
                st.write(f"**Resumo da Jornada:** {row['TIMELINE']}")
                st.markdown("**Interações Originais:**")
                st.code(row["JORNADA_TEXTO"])

    # ─── 3. Classificações ───
    with t_classif:
        st.header("📋 Auditoria de Interações")
        st.plotly_chart(chart_volume_por_mes(df), use_container_width=True)
        
        sel_motivo = st.selectbox("Filtrar Nuvem de Palavras por Motivo:", ["Todos"] + list(MACRO_MOTIVOS))
        wc = wordcloud_fig(df_work, None if sel_motivo == "Todos" else sel_motivo)
        if wc: st.image(wc)
        
        st.divider()
        
        # Aplicar o mesmo filtro à lista de classificações
        if sel_motivo == "Todos":
            df_display = df_work.copy()
        else:
            df_display = df_work[df_work["PERFIL_RECLAMACAO"] == sel_motivo].copy()
        
        st.subheader(f"Lista de Interações Classificadas ({len(df_display)} de {len(df_work)} transcrições)")
        
        # Validar e preparar dados para exibição (prevenir erro React)
        df_show = df_display[["ID_CLIENTE", "DATETIME_TRANSCRICAO_LIGACAO", "PERFIL_RECLAMACAO", "CONFIDENCE_SCORE", "TRANSCRICAO_LIGACAO_CLIENTE"]].fillna("")
        
        if not df_show.empty:
            st.dataframe(df_show, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhuma transcrição encontrada para o filtro selecionado.")

    # ─── 4. Diagnóstico IA ───
    with t_ia:
        st.header("🤖 Diagnóstico Profundo por IA (LangGraph)")
        st.info("A análise utiliza o sistema multiagente (Auditor, Resumidor, Classificador, Diagnóstico) para gerar storytelling de causa raiz com scores de confiança individuais.")

        llm_p_config = {
            "provider": provider,
            "api_key": api_key,
            "model": model,
            "threshold": threshold
        }

        st.markdown("#### 🔎 Escolha o Modo de Diagnóstico")
        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # LAYOUT LADO A LADO: Cliente Individual (esq) | Causa Raiz (dir)
        # ══════════════════════════════════════════════════════════════════════
        
        col_cliente, col_causa = st.columns(2)
        
        # ────────────────────────────────────────────────────────────────
        # COLUNA ESQUERDA: Por Cliente Individual
        # ────────────────────────────────────────────────────────────────
        with col_cliente:
            st.markdown("##### 👤 Raio-X de Cliente Individual")
            target_client = st.selectbox(
                "Escolha um Cliente para Raio-X Detalhado:",
                df_d90["ID_CLIENTE"].unique(),
                key="sel_cliente_ia"
            )
            
            if st.button("🚀 Executar", key="btn_exec_cliente", use_container_width=True, type="primary"):
                st.session_state["diagnostico_ativo"] = "cliente"
                st.session_state["cliente_selecionado"] = target_client

        # ────────────────────────────────────────────────────────────────
        # COLUNA DIREITA: Por Causa Raiz Predominante
        # ────────────────────────────────────────────────────────────────
        with col_causa:
            st.markdown("##### 🎯 Raio-X por Causa Raiz Predominante")
            causas_disponiveis = sorted(df_work["PERFIL_RECLAMACAO"].dropna().unique().tolist())
            target_causa = st.selectbox(
                "Escolha uma Causa Raiz Predominante:",
                causas_disponiveis,
                key="sel_causa_ia"
            )
            
            # Clientes com essa causa raiz
            clientes_da_causa = (
                df_work[df_work["PERFIL_RECLAMACAO"] == target_causa]["ID_CLIENTE"]
                .unique().tolist()
            )
            
            if clientes_da_causa:
                cliente_rep = st.selectbox(
                    "Cliente representativo:",
                    clientes_da_causa,
                    key="sel_cliente_rep_ia"
                )
                
                if st.button("🚀 Executar", key="btn_exec_causa", use_container_width=True, type="primary"):
                    st.session_state["diagnostico_ativo"] = "causa"
                    st.session_state["causa_selecionada"] = target_causa
                    st.session_state["cliente_rep_selecionado"] = cliente_rep
            else:
                st.warning("Nenhum cliente encontrado.")

        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # ÁREA DE RESULTADOS (Full Width)
        # ══════════════════════════════════════════════════════════════════════
        
        if st.session_state.get("diagnostico_ativo") == "cliente":
            target_client = st.session_state.get("cliente_selecionado")
            row = df_d90[df_d90["ID_CLIENTE"] == target_client].iloc[0]
            
            with st.spinner(f"🔄 Analistas IA processando a jornada de {target_client}..."):
                try:
                    meta = {"n_ligacoes": row["N_LIGACOES_D90"], "span_dias": row["SPAN_DIAS"]}
                    result = run_langgraph_pipeline(row["TRANSCRICOES_LIST"], meta, llm_p_config)
                    
                    res_rows = []
                    for r in result["results_per_call"]:
                        res_rows.append({
                            "ID_CLIENTE": target_client,
                            "DATETIME_TRANSCRICAO_LIGACAO": row["DATAS_LIST"][r["index"]],
                            "TRANSCRICAO_LIGACAO_CLIENTE": row["TRANSCRICOES_LIST"][r["index"]],
                            "MACRO_MOTIVO": r["classificacao"].get("macro_motivo"),
                            "CONFIDENCE": r["classificacao"].get("score")
                        })
                    new_analysis = pd.DataFrame(res_rows)
                    st.session_state.analysis_results = pd.concat([
                        st.session_state.analysis_results, new_analysis
                    ]).drop_duplicates(subset=["ID_CLIENTE", "DATETIME_TRANSCRICAO_LIGACAO"])

                    st.success("✅ Diagnóstico Gerado com Sucesso!")
                    _render_diagnostico_storytelling(result["diagnostico_final"], target_client, row)
                    
                    with st.expander("🔬 Ver Auditoria Detalhada dos Agentes"):
                        st.json(result["results_per_call"])
                        
                except Exception as e:
                    st.error(f"Erro na execução da IA: {e}. Verifique suas chaves de API.")
                    
            # Limpar flag após processar
            st.session_state["diagnostico_ativo"] = None

        elif st.session_state.get("diagnostico_ativo") == "causa":
            target_causa = st.session_state.get("causa_selecionada")
            cliente_rep = st.session_state.get("cliente_rep_selecionado")
            row = df_d90[df_d90["ID_CLIENTE"] == cliente_rep].iloc[0]
            
            # Mostrar métricas do grupo
            clientes_da_causa = df_work[df_work["PERFIL_RECLAMACAO"] == target_causa]["ID_CLIENTE"].unique().tolist()
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Clientes neste grupo", len(clientes_da_causa))
            c_m2.metric("% do total", f"{len(clientes_da_causa) / max(1, df['ID_CLIENTE'].nunique()):.0%}")
            grupo_conf = df_work[df_work["PERFIL_RECLAMACAO"] == target_causa]["CONFIDENCE_SCORE"].mean()
            c_m3.metric("Confiança Média IA", f"{grupo_conf:.1f}%")
            
            st.markdown("---")
            
            with st.spinner(f"🔄 Analisando jornada de {cliente_rep} (Causa: {target_causa})..."):
                try:
                    meta = {"n_ligacoes": row["N_LIGACOES_D90"], "span_dias": row["SPAN_DIAS"]}
                    result = run_langgraph_pipeline(row["TRANSCRICOES_LIST"], meta, llm_p_config)
                    
                    res_rows = []
                    for r in result["results_per_call"]:
                        res_rows.append({
                            "ID_CLIENTE": cliente_rep,
                            "DATETIME_TRANSCRICAO_LIGACAO": row["DATAS_LIST"][r["index"]],
                            "TRANSCRICAO_LIGACAO_CLIENTE": row["TRANSCRICOES_LIST"][r["index"]],
                            "MACRO_MOTIVO": r["classificacao"].get("macro_motivo"),
                            "CONFIDENCE": r["classificacao"].get("score")
                        })
                    new_analysis = pd.DataFrame(res_rows)
                    st.session_state.analysis_results = pd.concat([
                        st.session_state.analysis_results, new_analysis
                    ]).drop_duplicates(subset=["ID_CLIENTE", "DATETIME_TRANSCRICAO_LIGACAO"])

                    st.success(f"✅ Diagnóstico para '{target_causa}' gerado!")
                    _render_diagnostico_storytelling(result["diagnostico_final"], cliente_rep, row)
                    
                    with st.expander("🔬 Ver Auditoria Detalhada dos Agentes"):
                        st.json(result["results_per_call"])

                except Exception as e:
                    st.error(f"Erro na execução da IA: {e}. Verifique suas chaves de API.")
                    
            # Limpar flag após processar
            st.session_state["diagnostico_ativo"] = None



    # ─── 5. Predição Risco ───
    with t_ml:
        st.header("🧬 Modelagem de Risco Preditivo")
        st.write("O modelo utiliza variáveis comportamentais da jornada D-90 para estimar a probabilidade de churn futuro.")

        # Tentar features expandidas se tivermos N_TOKENS_EST no df
        use_expanded = "N_TOKENS_EST" in df.columns
        if use_expanded:
            feat = build_features_expanded(df, df_d90)
            st.success("✅ **Feature Engineering Expandida ativa** — usando frequência, recência, NLP e intervalos.")
        else:
            feat = build_features(df_d90)
            st.info("Feature engineering básica ativa.")
        
        if st.button("⚙️ Treinar e Avaliar Modelos"):
            with st.spinner("Treinando RandomForest e XGBoost..."):
                results = train_models(feat)
                if results:
                    # ── Métricas dos modelos ──────────────────────────────────
                    for name, res in results.items():
                        if "error" in res:
                            st.error(f"Modelo {name}: {res['error']}")
                            continue
                        st.subheader(f"Modelo: {name}")
                        m1, m2 = st.columns(2)
                        m1.metric("Acurácia", f"{res['accuracy']:.2%}")
                        m2.metric("AUC ROC", f"{res['auc']:.3f}")
                        
                        st.plotly_chart(chart_risk_scores(res["all_prob"], f"Score de Risco - {name}"), use_container_width=True)
                        
                        # Mostrar Feature Importance para todos os modelos
                        st.plotly_chart(chart_feature_importance(res["importance"], name), use_container_width=True)
                        
                        st.divider()

                    # ── Diagnóstico de Causa Raiz Comportamental ──────────────
                    st.markdown("## 🔬 Diagnóstico de Causa Raiz Comportamental")
                    st.caption("Baseado na análise de importância das features dos modelos treinados acima.")
                    
                    diag_comp = generate_root_cause_diagnosis(feat, results)
                    
                    if diag_comp:
                        # Narrativa automática
                        st.markdown(diag_comp["narrativa"])
                        
                        st.markdown("---")
                        
                        # Gráfico de Pareto — Pareto 1 (vermelho): drivers principais
                        st.markdown("#### 🔴 Análise de Pareto — Drivers Principais (explicam ~80% do risco)")
                        pareto_fig_80 = chart_pareto_features_80(diag_comp["pareto_df"])
                        if pareto_fig_80:
                            st.plotly_chart(pareto_fig_80, use_container_width=True)

                        # Gráfico de Pareto — Pareto 2 (azul): drivers secundários
                        st.markdown("#### 🔵 Análise de Pareto — Drivers Secundários (20% restantes)")
                        pareto_fig_rest = chart_pareto_features_rest(diag_comp["pareto_df"])
                        if pareto_fig_rest:
                            st.plotly_chart(pareto_fig_rest, use_container_width=True)
                        else:
                            st.info("Todos os drivers estão concentrados no Pareto Principal (acima).")

                        # Distribuição de Risco
                        st.markdown("#### 📊 Distribuição de Clientes por Segmento de Risco")
                        seg_df = diag_comp["segment_df"].copy()
                        seg_df["PROB_PCT"] = (seg_df["PROB_CHURN"] * 100).round(1)
                        
                        dist_col1, dist_col2 = st.columns([1, 2])
                        with dist_col1:
                            perf = diag_comp["perfil_risco"]
                            for seg, cnt in sorted(perf.items(), key=lambda x: str(x[0])):
                                st.metric(str(seg), cnt)
                        with dist_col2:
                            # Usa st.table() (HTML estático) para evitar React error #185
                            if not seg_df.empty:
                                # Pré-computar jornadas por cliente (uma só vez, fora do loop)
                                client_ids_seg = seg_df["ID_CLIENTE"].astype(str).tolist()
                                journey_map = {
                                    cid: build_churn_journey_string(df_work, cid)
                                    for cid in client_ids_seg
                                }

                                seg_display_data = []
                                for _, r in seg_df.iterrows():
                                    # try/except blindado contra Categorical
                                    try:
                                        cliente = str(r["ID_CLIENTE"])
                                    except Exception:
                                        cliente = "?"
                                    try:
                                        score = round(float(r["PROB_PCT"]), 1)
                                    except Exception:
                                        score = 0.0
                                    try:
                                        raw_seg = r["SEGMENTO"]
                                        seg_str = str(raw_seg.item() if hasattr(raw_seg, "item") else raw_seg)
                                        segmento = seg_str if seg_str not in ("nan", "NaN", "", "None") else "⚪ Indefinido"
                                    except Exception:
                                        segmento = "⚪ Indefinido"
                                    jornada = journey_map.get(cliente, "—")
                                    seg_display_data.append({
                                        "Cliente": cliente,
                                        "Score Risco (%)": score,
                                        "Segmento": segmento,
                                        "Jornadas de Churn": jornada,
                                    })
                                seg_display = pd.DataFrame(seg_display_data)
                                seg_display = seg_display.sort_values("Score Risco (%)", ascending=False)
                                seg_display = seg_display.reset_index(drop=True)
                                # st.table() renderiza HTML puro — imune ao React #185
                                st.table(seg_display)
                            else:
                                st.warning("Nenhum dado disponível para segmentação de risco.")

                        # ── Sankey: Fluxo de Jornada → Segmento de Risco ──────────────────────────────
                        st.markdown("---")
                        st.markdown("#### 🌊 Fluxo da Jornada – Narrativa Completa")
                        st.caption(
                            "Visualiza a evolução sequencial dos motivos de insatisfação de cada cliente, "
                            "colorida pelo segmento final de risco (🔴 Alto / 🟡 Médio / 🟢 Baixo). "
                            "Cada nó exibe o nome do motivo e a quantidade de clientes naquela etapa."
                        )
                        with st.spinner("🔄 Calculando fluxo de jornada..."):
                            sankey_risk = chart_sankey_risk_journey(df_work, seg_df)
                        if sankey_risk:
                            st.plotly_chart(sankey_risk, use_container_width=True)
                        else:
                            st.info("Dados insuficientes para gerar o fluxo de jornada.")
                else:
                    st.error("Dados insuficientes para treino.")
else:
    st.title("🎯 Radar X — Inteligência Analítica em Churn")
    st.markdown("""
    Bem-vindo ao **Radar X**. Esta plataforma utiliza Inteligência Artificial avançada para extrair o máximo valor das suas interações de clientes.
    
    ### Como começar:
    1. Escolha a **Fonte de Dados** no menu lateral.
    2. Visualize os padrões na aba **Dashboard Executivo**.
    3. Realize diagnósticos profundos na aba **Diagnóstico IA**.
    """)
    st.image("https://images.unsplash.com/photo-1460925895917-afdab827c52f?q=80&w=1000", caption="Radar X Analysis")
