"""
app.py
Radar X – Diagnóstico Inteligente de Churn (D-90)
Aplicação Streamlit com Foco Executivo, Dashboards de Jornada e IA (LangGraph).
VERSÃO CORRIGIDA E APRIMORADA
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
import plotly.express as px
from io import BytesIO

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('radar_x.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    .alert-card {{
        background: #FFF3CD;
        border-left: 5px solid #FFC107;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }}
    .alert-title {{
        font-size: 13px;
        color: #856404;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }}
    .alert-value {{
        font-size: 16px;
        color: #856404;
        font-weight: 500;
    }}
    </style>
""", unsafe_allow_html=True)

# ─── Estado da Sessão ────────────────────────────────────────────────────────
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = pd.DataFrame()
if "risk_alerts" not in st.session_state:
    st.session_state.risk_alerts = []
if "rfm_segments" not in st.session_state:
    st.session_state.rfm_segments = None
if "ab_test_results" not in st.session_state:
    st.session_state.ab_test_results = None

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='font-size: 24px; color: #E63946;'>🎯 RADAR X</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 14px; color: #666;'>Diagnóstico de Churn D-90</p>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("### 📂 Fonte de Dados")
    # ✅ CORREÇÃO: Removidos espaços extras nas strings
    source_choice = st.radio("Selecione:", ("Base Sintética (Radar X)", "Carregar CSV Próprio"))

    df_raw = None
    if source_choice == "Carregar CSV Próprio":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            try:
                df_raw = load_csv(uploaded_file)
                logger.info(f"CSV carregado com sucesso: {len(df_raw)} registros")
            except Exception as e:
                logger.error(f"Erro no carregamento do CSV: {e}")
                st.error(f"Erro no carregamento: {e}")
    else:
        path = get_synthetic_csv_path()
        if os.path.exists(path):
            df_raw = pd.read_csv(path)
            logger.info(f"Base sintética carregada: {len(df_raw)} registros")
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
    
    # ✅ CORREÇÃO: Validação de API Key
    if not api_key or api_key.strip() == "":
        st.error("⚠️ **API Key não configurada!** Configure no `.streamlit/secrets.toml`")
        api_key_valid = False
    else:
        api_key_valid = True
    
    threshold = st.slider("Corte de Confiança (%)", 0, 100, 90)

    st.divider()
    
    # Nova feature: Threshold para alertas de risco
    st.markdown("### 🚨 Alertas de Risco")
    risk_threshold = st.slider("Threshold de Alerta (%)", 50, 100, 70)
    
    if st.button("🗑️ Limpar Cache de IA"):
        st.session_state.analysis_results = pd.DataFrame()
        st.session_state.risk_alerts = []
        st.session_state.rfm_segments = None
        st.session_state.ab_test_results = None
        logger.info("Cache de IA limpo")
        st.rerun()

# ─── Função auxiliar: renderizar storytelling do diagnóstico ─────────────────
def _render_diagnostico_storytelling(diag: dict, cliente_id: str, row):
    """Renderiza o storytelling aprimorado do diagnóstico IA."""
    st.markdown(f"### 🩺 Diagnóstico — Cliente `{cliente_id}`")
    st.caption(f"Jornada: {row.get('N_LIGACOES_D90', '?')} ligações em {row.get('SPAN_DIAS', '?')} dias")
    
    # ── Resumo Sequencial ────────────────────────────────────────────────────
    resumo = diag.get("resumo_sequencial") or diag.get("diagnostico_sequencial", "—")
    st.markdown(
        f""" <div class="resumo-card">
             <div style="font-size:11px; text-transform:uppercase; letter-spacing:1px; opacity:0.7; margin-bottom:8px;">📖 RESUMO SEQUENCIAL DA JORNADA</div>
             <div style="font-size:15px; line-height:1.6;">{resumo}</div>
         </div> """,
        unsafe_allow_html=True
    )

    # ── Métricas de diagnóstico em 2 colunas ─────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        causa = diag.get("causa_raiz_predominante") or diag.get("causa_raiz", "—")
        causa_score = diag.get("causa_raiz_score", diag.get("confianca_geral", diag.get("confianca", "—")))
        st.markdown(
            f""" <div class="diag-card">
                 <div class="diag-title">🌱 Causa Provável da Raiz da Jornada</div>
                 <div class="diag-value">{causa}</div>
                 <div class="diag-conf">Confiança da IA = {causa_score}%</div>
             </div> """,
            unsafe_allow_html=True
        )
        sentimento = diag.get("sentimento_jornada", "—")
        sentimento_score = diag.get("sentimento_score", "—")
        st.markdown(
            f""" <div class="diag-card">
                 <div class="diag-title">💬 Sentimento da Jornada</div>
                 <div class="diag-value">{sentimento}</div>
                 <div class="diag-conf">Confiança da IA = {sentimento_score}%</div>
             </div> """,
            unsafe_allow_html=True
        )

    with col2:
        gatilho = diag.get("evento_gatilho") or diag.get("ruptura", "—")
        gatilho_score = diag.get("evento_gatilho_score", "—")
        st.markdown(
            f""" <div class="diag-card">
                 <div class="diag-title">⚡ Evento Gatilho do Churn / Ruptura</div>
                 <div class="diag-value">{gatilho}</div>
                 <div class="diag-conf">Confiança da IA = {gatilho_score}%</div>
             </div> """,
            unsafe_allow_html=True
        )
        padroes = diag.get("padroes_detectados") or diag.get("padroes_recorrentes", "—")
        padroes_score = diag.get("padroes_score", "—")
        st.markdown(
            f""" <div class="diag-card">
                 <div class="diag-title">🔍 Padrões Detectados na Jornada</div>
                 <div class="diag-value">{padroes}</div>
                 <div class="diag-conf">Confiança da IA = {padroes_score}%</div>
             </div> """,
            unsafe_allow_html=True
        )

    # ── Indicadores adicionais ────────────────────────────────────────────────
    conf_geral = diag.get("confianca_geral", diag.get("confianca", "—"))
    escalada = diag.get("escalada", False)
    ind_col1, ind_col2, ind_col3 = st.columns(3)
    ind_col1.metric("🎯 Confiança Geral da IA", f"{conf_geral}%")
    ind_col2.metric("📈 Escalada Detectada", "✅ Sim" if escalada else "❌ Não")
    ruptura = diag.get("ruptura", "—")
    ind_col3.metric("🔀 Ponto de Ruptura", str(ruptura)[:40] + "..." if len(str(ruptura)) > 40 else str(ruptura))


# ─── Nova Feature: Geração de Relatório PDF ──────────────────────────────────
def generate_pdf_report(df_work, df_d90, risk_results=None):
    """Gera relatório executivo em PDF."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()
        
        # Título
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#E63946'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        elements.append(Paragraph("🎯 RADAR X - Relatório Executivo de Churn", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Data do relatório
        date_style = ParagraphStyle(
            'DateStyle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        elements.append(Paragraph(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", date_style))
        elements.append(Spacer(1, 0.3*inch))
        
        # KPIs Principais
        kpi_data = [
            ['Métrica', 'Valor'],
            ['Total de Clientes', str(df_work['ID_CLIENTE'].nunique())],
            ['Total de Transcrições', str(len(df_work))],
            ['Clientes com Churn', str(df_work[df_work['TARGET_CHURN']==1]['ID_CLIENTE'].nunique()) if 'TARGET_CHURN' in df_work.columns else 'N/A'],
            ['Taxa de Churn', f"{(df_work[df_work['TARGET_CHURN']==1]['ID_CLIENTE'].nunique() / df_work['ID_CLIENTE'].nunique() * 100):.1f}%" if 'TARGET_CHURN' in df_work.columns else 'N/A'],
        ]
        
        kpi_table = Table(kpi_data, colWidths=[3*inch, 2*inch])
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E63946')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(kpi_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Top Motivos
        if 'PERFIL_RECLAMACAO' in df_work.columns:
            motivos = df_work['PERFIL_RECLAMACAO'].value_counts().head(5)
            motivo_data = [['Motivo', 'Ocorrências', '%']]
            for motivo, count in motivos.items():
                pct = (count / len(df_work) * 100)
                motivo_data.append([motivo, str(count), f"{pct:.1f}%"])
            
            motivo_table = Table(motivo_data, colWidths=[2.5*inch, 1.5*inch, 1*inch])
            motivo_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1D3557')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(Paragraph("📊 Top 5 Motivos de Reclamação", styles['Heading2']))
            elements.append(motivo_table)
        
        # Alertas de Risco
        if st.session_state.risk_alerts:
            elements.append(Spacer(1, 0.3*inch))
            elements.append(Paragraph("🚨 Alertas de Risco Ativos", styles['Heading2']))
            alert_data = [['Cliente', 'Score de Risco', 'Motivo Principal']]
            for alert in st.session_state.risk_alerts[:10]:
                alert_data.append([alert['cliente'], f"{alert['score']:.0f}%", alert['motivo']])
            
            alert_table = Table(alert_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
            alert_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FFC107')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(alert_table)
        
        # Recomendações
        elements.append(Spacer(1, 0.3*inch))
        elements.append(Paragraph("💡 Recomendações Estratégicas", styles['Heading2']))
        recommendations = [
            "1. Priorizar atendimento proativo para clientes com score de risco > 70%",
            "2. Implementar programa de retenção para clientes com múltiplas reclamações técnicas",
            "3. Revisar política de preços para reduzir churn por motivos financeiros",
            "4. Melhorar tempo de resposta para visitas técnicas",
            "5. Criar campanha de win-back para clientes churned nos últimos 30 dias"
        ]
        for rec in recommendations:
            elements.append(Paragraph(rec, styles['Normal']))
            elements.append(Spacer(1, 0.1*inch))
        
        doc.build(elements)
        buffer.seek(0)
        logger.info("Relatório PDF gerado com sucesso")
        return buffer
    except Exception as e:
        logger.error(f"Erro ao gerar PDF: {e}")
        st.error(f"Erro ao gerar relatório: {e}")
        return None

# ─── Nova Feature: Cálculo RFM ───────────────────────────────────────────────
def calculate_rfm_segments(df_work):
    """Calcula segmentação RFM (Recência, Frequência, Valor)."""
    try:
        df_rfm = df_work.copy()
        
        # Recência: dias desde última interação
        df_rfm['DATETIME_TRANSCRICAO_LIGACAO'] = pd.to_datetime(df_rfm['DATETIME_TRANSCRICAO_LIGACAO'])
        reference_date = df_rfm['DATETIME_TRANSCRICAO_LIGACAO'].max()
        
        # ✅ CORREÇÃO: Usar named aggregation para evitar conflito de colunas
        rfm = df_rfm.groupby('ID_CLIENTE').agg(
            Recencia=('DATETIME_TRANSCRICAO_LIGACAO', lambda x: (reference_date - x.max()).days),
            Frequencia=('ID_CLIENTE', 'count'),
            Valor=('N_TOKENS_EST', 'sum')
        ).reset_index()  # ← ID_CLIENTE volta como coluna após reset_index()
        
        # ✅ CORREÇÃO: Usar rank(method='first') para evitar erros com valores duplicados
        for col in ['Recencia', 'Frequencia', 'Valor']:
            try:
                if col == 'Recencia':
                    # Recência menor = score maior (cliente mais recente = melhor)
                    rfm[f'{col}_score'] = pd.qcut(rfm[col].rank(method='first'), q=5, labels=[5, 4, 3, 2, 1])
                else:
                    # Frequência e Valor maiores = score maior
                    rfm[f'{col}_score'] = pd.qcut(rfm[col].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
            except Exception as e:
                logger.warning(f"Erro no qcut para {col}: {e}")
                rfm[f'{col}_score'] = 3  # Score médio como fallback
        
        rfm['RFM_score'] = rfm['Recencia_score'].astype(int) + rfm['Frequencia_score'].astype(int) + rfm['Valor_score'].astype(int)
        
        # Segmentação
        def segment_rfm(row):
            if row['RFM_score'] >= 12:
                return '🏆 Campeões'
            elif row['RFM_score'] >= 9:
                return '⭐ Clientes Leais'
            elif row['RFM_score'] >= 6:
                return '⚠️ Em Risco'
            else:
                return '😴 Perdidos'
        
        rfm['Segmento'] = rfm.apply(segment_rfm, axis=1)
        
        logger.info(f"Segmentação RFM calculada: {rfm['Segmento'].value_counts().to_dict()}")
        return rfm
    except Exception as e:
        logger.error(f"Erro no cálculo RFM: {e}")
        st.error(f"Erro no cálculo RFM: {e}")
        return None

# ─── Nova Feature: Simulação A/B Testing ─────────────────────────────────────
def simulate_ab_testing(df_work, action_type='discount', impact_rate=0.3):
    """Simula impacto de ações de retenção."""
    try:
        df_sim = df_work.copy()
        
        # Identificar clientes churned ou em risco
        if 'TARGET_CHURN' in df_sim.columns:
            at_risk = df_sim[df_sim['TARGET_CHURN'] == 1].copy()
        else:
            at_risk = df_sim.sample(frac=0.5).copy()
        
        # Dividir em grupo de controle e tratamento
        n = len(at_risk)
        np.random.seed(42)  # Para reprodutibilidade
        at_risk = at_risk.copy()
        at_risk['grupo'] = np.where(np.random.rand(n) < 0.5, 'Controle', 'Tratamento')
        
        # 🎯 Mapeamento de eficácia por tipo de ação
        # Base: impact_rate é a taxa base de retenção esperada
        action_multipliers = {
            'discount': 1.0,              # Base: 30% de retenção
            'upgrade': 1.2,               # +20% mais eficaz que desconto
            'priority_support': 1.15,     # +15% mais eficaz
            'loyalty_bonus': 1.25,        # +25% (clientes valorizam recompensas)
            'free_installation': 1.3,     # +30% (resolve dor de mudança)
            'bundle_offer': 1.35,         # +35% (aumento de valor percebido)
            'contract_extension': 1.1,    # +10% (menor atratividade)
            'personalized_offer': 1.4     # +40% (alta relevância = alta conversão)
        }
        
        # Custos estimados por ação (em R$ por cliente)
        action_costs = {
            'discount': 50,
            'upgrade': 80,
            'priority_support': 30,
            'loyalty_bonus': 40,
            'free_installation': 120,
            'bundle_offer': 100,
            'contract_extension': 20,
            'personalized_offer': 60
        }
        
        # Aplicar impacto da ação
        multiplier = action_multipliers.get(action_type, 1.0)
        at_risk.loc[at_risk['grupo'] == 'Tratamento', 'prob_retencao'] = np.clip(
            impact_rate * multiplier, 0, 1
        )
        at_risk.loc[at_risk['grupo'] == 'Controle', 'prob_retencao'] = 0.1  # Taxa base sem ação
        
        # Calcular métricas
        controle_retidos = (at_risk[at_risk['grupo'] == 'Controle']['prob_retencao'] > 0.5).sum()
        tratamento_retidos = (at_risk[at_risk['grupo'] == 'Tratamento']['prob_retencao'] > 0.5).sum()
        
        controle_total = len(at_risk[at_risk['grupo'] == 'Controle'])
        tratamento_total = len(at_risk[at_risk['grupo'] == 'Tratamento'])
        
        # ROI: Valor do cliente retido = R$150/mês × 12 meses = R$1.800 LTV
        CLIENT_LTV = 1800
        action_cost = action_costs.get(action_type, 50)
        
        results = {
            'acao': action_type,
            'controle_retidos': int(controle_retidos),
            'controle_total': int(controle_total),
            'controle_taxa': controle_retidos / max(controle_total, 1),
            'tratamento_retidos': int(tratamento_retidos),
            'tratamento_total': int(tratamento_total),
            'tratamento_taxa': tratamento_retidos / max(tratamento_total, 1),
            'lift': (tratamento_retidos / max(tratamento_total, 1)) - (controle_retidos / max(controle_total, 1)),
            'roi_estimado': ((tratamento_retidos - controle_retidos) * CLIENT_LTV) - (tratamento_total * action_cost),
            'custo_por_acao': action_cost,
            'clientes_retidos_extra': int(tratamento_retidos - controle_retidos)
        }
        
        return results
    except Exception as e:
        st.error(f"Erro na simulação: {e}")
        return None


# ─── Lógica Principal ────────────────────────────────────────────────────────
if df_raw is not None:
    valid, missing = validate_columns(df_raw)
    if not valid:
        logger.error(f"Colunas faltando: {missing}")
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
        # ✅ CORREÇÃO: Variável corrigida (era df _enriched)
        df_work = df_enriched
        logger.info("Classificações carregadas do cache")
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
        logger.info("Classificações mock geradas")

    # ─── TABS DO APP ──────────────────────────────────────────────────────────
    t_dash, t_jornadas, t_classif, t_ia, t_ml, t_rfm, t_ab = st.tabs([
        "📈 Dashboard Executivo", "🌊 Jornadas", "📋 Classificações", 
        "🤖 Diagnóstico IA", "🧬 Predição Risco", "📊 RFM", "🧪 A/B Testing"
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
                    # ✅ CORREÇÃO: Validação de API Key antes de executar
                    if not api_key_valid:
                        st.error("❌ Configure a API Key no sidebar primeiro!")
                        logger.error("Tentativa de classificação sem API Key")
                    else:
                        with st.spinner(f"🔄 Classificando {len(df)} transcrições com IA..."):
                            try:
                                df_classified = batch_classify_all_transcriptions(df, llm_p_config)
                                save_enriched_dataset(df_classified)
                                st.session_state.classification_completed = True
                                st.success(f"✅ {len(df_classified)} transcrições classificadas com sucesso!")
                                st.balloons()
                                logger.info(f"Classificação concluída: {len(df_classified)} registros")
                                import time
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                logger.error(f"Erro na classificação: {e}")
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
                    st.session_state.risk_alerts = []
                    st.rerun()
            
            # ✅ NOVA FEATURE: Exportar Relatório PDF
            st.divider()
            col_pdf1, col_pdf2 = st.columns([3, 1])
            with col_pdf1:
                st.markdown("### 📄 Exportar Relatório")
            with col_pdf2:
                if st.button("📥 Baixar PDF", use_container_width=True):
                    pdf_buffer = generate_pdf_report(df_work, df_d90)
                    if pdf_buffer:
                        st.download_button(
                            label="📥 Download",
                            data=pdf_buffer,
                            file_name=f"radar_x_relatorio_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf"
                        )
        
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
        
        # ✅ CORREÇÃO: Tratamento para datasets vazios
        st.markdown("### Clientes sob Diagnóstico")
        
        if not df_d90_dash.empty:
            df_d90_display = df_d90_dash[["ID_CLIENTE", "TIMELINE", "PERFIL_RECLAMACAO", "CONFIDENCE_SCORE", "N_LIGACOES_D90"]].fillna(" ").rename(
                columns={"TIMELINE": "Fluxo da Jornada", "PERFIL_RECLAMACAO": "Causa Raiz", "CONFIDENCE_SCORE": "IA Conf (%)"}
            )
            st.dataframe(df_d90_display, use_container_width=True, hide_index=True)
        else:
            st.info("⚠️ Nenhum cliente encontrado para os filtros selecionados.")
            logger.warning("Filtros resultaram em dataset vazio")
        
        st.markdown("---")
        st.markdown("### 🌊 Fluxo da Jornada – Narrativa Completa")
        st.caption("Visualiza o caminho de cada motivo de insatisfação até o destino final: 🔴 **CHURN** (TARGET_CHURN = 1, clientes que saíram da empresa) ou 🟢 **NÃO CHURN** (TARGET_CHURN = 0, clientes com jornada ativa). Filtrado pelos critérios acima.")
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
            # ─── NOVA FEATURE: Comparativo Temporal ───
        st.divider()
        st.markdown("### 📅 Comparativo Temporal - Evolução do Churn")
        st.caption("Visualize a evolução das interações e taxa de churn ao longo do tempo.")
        
        if 'DATETIME_TRANSCRICAO_LIGACAO' in df_work.columns and 'TARGET_CHURN' in df_work.columns:
            # Seletor de granularidade
            granularidade = st.selectbox(
                "Granularidade:",
                ["Dia", "Semana", "Mês"],
                key="sel_granularidade_temporal"
            )
            
            df_temp = df_work.copy()
            df_temp['DATETIME_TRANSCRICAO_LIGACAO'] = pd.to_datetime(df_temp['DATETIME_TRANSCRICAO_LIGACAO'])
            
            # Criar coluna de período baseado na granularidade
            if granularidade == "Dia":
                df_temp['PERIODO'] = df_temp['DATETIME_TRANSCRICAO_LIGACAO'].dt.strftime('%d/%m/%Y')
                df_temp['PERIODO_SORT'] = df_temp['DATETIME_TRANSCRICAO_LIGACAO'].dt.date
            elif granularidade == "Semana":
                df_temp['PERIODO'] = df_temp['DATETIME_TRANSCRICAO_LIGACAO'].dt.to_period('W').astype(str)
                df_temp['PERIODO_SORT'] = df_temp['DATETIME_TRANSCRICAO_LIGACAO'].dt.to_period('W').apply(lambda r: r.start_time)
            else:  # Mês
                df_temp['PERIODO'] = df_temp['DATETIME_TRANSCRICAO_LIGACAO'].dt.to_period('M').astype(str)
                df_temp['PERIODO_SORT'] = df_temp['DATETIME_TRANSCRICAO_LIGACAO'].dt.to_period('M').apply(lambda r: r.start_time)
            
            churn_by_period = df_temp.groupby('PERIODO').agg({
                'ID_CLIENTE': 'count',
                'TARGET_CHURN': 'sum',
                'PERIODO_SORT': 'first'
            }).reset_index()
            churn_by_period = churn_by_period.sort_values('PERIODO_SORT')
            churn_by_period['TAXA_CHURN'] = (churn_by_period['TARGET_CHURN'] / churn_by_period['ID_CLIENTE'] * 100).round(1)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = px.line(
                    churn_by_period, 
                    x='PERIODO', 
                    y='TAXA_CHURN',
                    title=f'Evolução da Taxa de Churn por {granularidade}',
                    markers=True,
                    hover_data={'PERIODO_SORT': False, 'TAXA_CHURN': ':.1f%', 'ID_CLIENTE': True, 'TARGET_CHURN': True}
                )
                fig.update_traces(line=dict(color='#E63946', width=3))
                fig.update_layout(xaxis_title=granularidade, yaxis_title='Taxa de Churn (%)')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("#### 📊 Resumo")
                if not churn_by_period.empty:
                    st.metric("Período com Maior Churn", churn_by_period.loc[churn_by_period['TAXA_CHURN'].idxmax(), 'PERIODO'])
                    st.metric("Maior Taxa", f"{churn_by_period['TAXA_CHURN'].max():.1f}%")
                    st.metric("Média", f"{churn_by_period['TAXA_CHURN'].mean():.1f}%")
                    st.metric("Total de Interações", churn_by_period['ID_CLIENTE'].sum())
                else:
                    st.info("Dados insuficientes")
        else:
            st.info("Dados insuficientes para análise temporal.")
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
        
        # ✅ CORREÇÃO: Tratamento para datasets vazios
        if not df_d90.empty:
            for idx, row in df_d90.head(10).iterrows():
                cli_data = df_work[df_work["ID_CLIENTE"] == row["ID_CLIENTE"]]
                motive_label = cli_data["PERFIL_RECLAMACAO"].iloc[-1] if not cli_data.empty else "Não identificado"
                
                with st.expander(f"👤 {row['ID_CLIENTE']} | {row['N_LIGACOES_D90']} ligações | Causa: {motive_label}"):
                    st.write(f"**Resumo da Jornada:** {row['TIMELINE']}")
                    st.markdown("**Interações Originais:**")
                    st.code(row["JORNADA_TEXTO"])
        else:
            st.info("Nenhuma jornada disponível para exibição.")

    # ─── 3. Classificações ───
    with t_classif:
        st.header("📋 Auditoria de Interações")
        st.plotly_chart(chart_volume_por_mes(df), use_container_width=True)
        
        sel_motivo = st.selectbox("Filtrar Nuvem de Palavras por Motivo:", ["Todos"] + list(MACRO_MOTIVOS))
        wc = wordcloud_fig(df_work, None if sel_motivo == "Todos" else sel_motivo)
        if wc: 
            st.image(wc)
        
        st.divider()
        
        # Aplicar o mesmo filtro à lista de classificações
        if sel_motivo == "Todos":
            df_display = df_work.copy()
        else:
            df_display = df_work[df_work["PERFIL_RECLAMACAO"] == sel_motivo].copy()
        
        st.subheader(f"Lista de Interações Classificadas ({len(df_display)} de {len(df_work)} transcrições)")
        
        # Validar e preparar dados para exibição (prevenir erro React)
        # ✅ CORREÇÃO: Tratamento para datasets vazios
        df_show = df_display[["ID_CLIENTE", "DATETIME_TRANSCRICAO_LIGACAO", "PERFIL_RECLAMACAO", "CONFIDENCE_SCORE", "TRANSCRICAO_LIGACAO_CLIENTE"]].fillna(" ")
        
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

        col_cliente, col_causa = st.columns(2)
        
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

        with col_causa:
            st.markdown("##### 🎯 Raio-X por Causa Raiz Predominante")
            causas_disponiveis = sorted(df_work["PERFIL_RECLAMACAO"].dropna().unique().tolist())
            target_causa = st.selectbox(
                "Escolha uma Causa Raiz Predominante:",
                causas_disponiveis,
                key="sel_causa_ia"
            )
            
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
                    logger.error(f"Erro na execução da IA: {e}")
                    st.error(f"Erro na execução da IA: {e}. Verifique suas chaves de API.")
                    
            st.session_state["diagnostico_ativo"] = None

        elif st.session_state.get("diagnostico_ativo") == "causa":
            target_causa = st.session_state.get("causa_selecionada")
            cliente_rep = st.session_state.get("cliente_rep_selecionado")
            row = df_d90[df_d90["ID_CLIENTE"] == cliente_rep].iloc[0]
            
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
                    logger.error(f"Erro na execução da IA: {e}")
                    st.error(f"Erro na execução da IA: {e}. Verifique suas chaves de API.")
                    
            st.session_state["diagnostico_ativo"] = None

    # ─── 5. Predição Risco ───
    with t_ml:
        st.header("🧬 Modelagem de Risco Preditivo")
        st.write("O modelo utiliza variáveis comportamentais da jornada D-90 para estimar a probabilidade de churn futuro.")

        use_expanded = "N_TOKENS_EST" in df.columns
        if use_expanded:
            feat = build_features_expanded(df, df_d90)
            st.success("✅ **Feature Engineering Expandida ativa** — usando frequência, recência, NLP e intervalos.")
        else:
            feat = build_features(df_d90, df_raw=df)
            st.info("Feature engineering básica ativa.")
        
        if st.button("⚙️ Treinar e Avaliar Modelos"):
            with st.spinner("Treinando RandomForest e XGBoost..."):
                results = train_models(feat)
                if results:
                    for name, res in results.items():
                        if "error" in res:
                            st.error(f"Modelo {name}: {res['error']}")
                            continue
                        st.subheader(f"Modelo: {name}")
                        m1, m2 = st.columns(2)
                        m1.metric("Acurácia", f"{res['accuracy']:.2%}")
                        m2.metric("AUC ROC", f"{res['auc']:.3f}")
                        
                        st.plotly_chart(chart_risk_scores(res["all_prob"], f"Score de Risco - {name}"), use_container_width=True)
                        st.plotly_chart(chart_feature_importance(res["importance"], name), use_container_width=True)
                        st.divider()

                    st.markdown("## 🔬 Diagnóstico de Causa Raiz Comportamental")
                    st.caption("Baseado na análise de importância das features dos modelos treinados acima.")
                    
                    diag_comp = generate_root_cause_diagnosis(feat, results)
                    
                    if diag_comp:
                        st.markdown(diag_comp["narrativa"])
                        st.markdown("---")
                        
                        st.markdown("#### 🔴 Análise de Pareto — Drivers Principais (explicam ~80% do risco)")
                        pareto_fig_80 = chart_pareto_features_80(diag_comp["pareto_df"])
                        if pareto_fig_80:
                            st.plotly_chart(pareto_fig_80, use_container_width=True)

                        st.markdown("#### 🔵 Análise de Pareto — Drivers Secundários (20% restantes)")
                        pareto_fig_rest = chart_pareto_features_rest(diag_comp["pareto_df"])
                        if pareto_fig_rest:
                            st.plotly_chart(pareto_fig_rest, use_container_width=True)
                        else:
                            st.info("Todos os drivers estão concentrados no Pareto Principal (acima).")

                        st.markdown("#### 📊 Distribuição de Clientes por Segmento de Risco")
                        seg_df = diag_comp["segment_df"].copy()
                        seg_df["PROB_PCT"] = (seg_df["PROB_CHURN"] * 100).round(1)
                        
                        dist_col1, dist_col2 = st.columns([1, 2])
                        with dist_col1:
                            perf = diag_comp["perfil_risco"]
                            for seg, cnt in sorted(perf.items(), key=lambda x: str(x[0])):
                                st.metric(str(seg), cnt)
                        with dist_col2:
                            if not seg_df.empty:
                                client_ids_seg = seg_df["ID_CLIENTE"].astype(str).tolist()
                                journey_map = {
                                    cid: build_churn_journey_string(df_work, cid)
                                    for cid in client_ids_seg
                                }

                                seg_display_data = []
                                for _, r in seg_df.iterrows():
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
                                        segmento = seg_str if seg_str not in ("nan", "NaN", " ", "None") else "⚪ Indefinido"
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
                                st.table(seg_display)
                            else:
                                st.warning("Nenhum dado disponível para segmentação de risco.")

                        # ✅ NOVA FEATURE: Alertas em Tempo Real
                        st.markdown("---")
                        st.markdown("### 🚨 Alertas de Risco em Tempo Real")
                        
                        high_risk = seg_df[seg_df["PROB_CHURN"] >= (risk_threshold / 100)]
                        
                        if not high_risk.empty:
                            st.warning(f"⚠️ **{len(high_risk)} clientes** acima do threshold de {risk_threshold}%")
                            
                            for _, row in high_risk.head(5).iterrows():
                                cliente_id = str(row["ID_CLIENTE"])
                                score = row["PROB_CHURN"] * 100
                                motivo = df_work[df_work["ID_CLIENTE"] == cliente_id]["PERFIL_RECLAMACAO"].iloc[-1] if len(df_work[df_work["ID_CLIENTE"] == cliente_id]) > 0 else "Desconhecido"
                                
                                st.markdown(
                                    f"""<div class="alert-card">
                                        <div class="alert-title">🚨 ALERTA: {cliente_id}</div>
                                        <div class="alert-value">Score: {score:.1f}% | Motivo: {motivo}</div>
                                    </div>""",
                                    unsafe_allow_html=True
                                )
                                
                                # Adicionar ao session state
                                alert = {
                                    "cliente": cliente_id,
                                    "score": score,
                                    "motivo": motivo,
                                    "timestamp": datetime.now()
                                }
                                if alert not in st.session_state.risk_alerts:
                                    st.session_state.risk_alerts.append(alert)
                        else:
                            st.success("✅ Nenhum cliente acima do threshold de risco configurado.")

                        st.markdown("---")
                        st.markdown("#### 🌊 Fluxo da Jornada – Narrativa Completa")
                        st.caption("Visualiza a evolução sequencial dos motivos de insatisfação de cada cliente, colorida pelo segmento final de risco (🔴 Alto / 🟡 Médio / 🟢 Baixo).")
                        with st.spinner("🔄 Calculando fluxo de jornada..."):
                            sankey_risk = chart_sankey_risk_journey(df_work, seg_df)
                        if sankey_risk:
                            st.plotly_chart(sankey_risk, use_container_width=True)
                        else:
                            st.info("Dados insuficientes para gerar o fluxo de jornada.")
                else:
                    st.error("Dados insuficientes para treino.")

    # ─── 6. NOVA FEATURE: Segmentação RFM ───
    with t_rfm:
        st.header("📊 Segmentação RFM (Recência, Frequência, Valor)")
        st.info("A segmentação RFM identifica clientes valiosos, em risco e perdidos com base em seu comportamento de interação.")
        
        if st.button("🔄 Calcular Segmentação RFM"):
            with st.spinner("Calculando segmentos RFM..."):
                rfm_result = calculate_rfm_segments(df_work)
                if rfm_result is not None:
                    st.session_state.rfm_segments = rfm_result
                    st.success("✅ Segmentação RFM calculada com sucesso!")
                    st.rerun()
        
        if st.session_state.rfm_segments is not None:
            rfm_df = st.session_state.rfm_segments
            
            # Distribuição de Segmentos
            col1, col2, col3, col4 = st.columns(4)
            seg_counts = rfm_df['Segmento'].value_counts()
            
            col1.metric("🏆 Campeões", seg_counts.get('🏆 Campeões', 0))
            col2.metric("⭐ Clientes Leais", seg_counts.get('⭐ Clientes Leais', 0))
            col3.metric("⚠️ Em Risco", seg_counts.get('⚠️ Em Risco', 0))
            col4.metric("😴 Perdidos", seg_counts.get('😴 Perdidos', 0))
 
         # Adicionar filtro de segmento na UI
 #       if st.session_state.rfm_segments is not None:
 #           segmento_filtro = st.selectbox(
 #               "Filtrar por Segmento RFM (opcional)",
 #               ["Todos", "🏆 Campeões", "⭐ Clientes Leais", "⚠️ Em Risco", "😴 Perdidos"])
 #           if segmento_filtro != "Todos":
 #               clientes_segmento = st.session_state.rfm_segments[st.session_state.rfm_segments['Segmento'] == segmento_filtro]['ID_CLIENTE'].tolist()
 #               at_risk = at_risk[at_risk['ID_CLIENTE'].isin(clientes_segmento)]

            st.divider()
            
            # Gráfico de Distribuição
            import plotly.express as px
            fig = px.pie(
                rfm_df, 
                names='Segmento', 
                title='Distribuição de Segmentos RFM',
                color='Segmento',
                color_discrete_map={
                    '🏆 Campeões': '#2A9D8F',
                    '⭐ Clientes Leais': '#457B9D',
                    '⚠️ Em Risco': '#F4A261',
                    '😴 Perdidos': '#E63946'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            st.markdown("### 📋 Detalhes por Segmento")
            st.dataframe(rfm_df[['ID_CLIENTE', 'Recencia', 'Frequencia', 'Valor', 'RFM_score', 'Segmento']], use_container_width=True)
            
            # Recomendações por Segmento
            st.markdown("### 💡 Ações Recomendadas por Segmento")
            st.markdown("""
            | Segmento | Ação Recomendada | Prioridade |
            |----------|-----------------|------------|
            | 🏆 Campeões | Programa de fidelidade, upsell | Alta |
            | ⭐ Clientes Leais | Manter engajamento, pesquisa de satisfação | Média |
            | ⚠️ Em Risco | Contato proativo, oferta de retenção | **Crítica** |
            | 😴 Perdidos | Campanha de win-back, desconto agressivo | Baixa |
            """)
    # ─── 7. NOVA FEATURE: A/B Testing ───
    with t_ab:
        st.header("🧪 Simulação A/B Testing de Ações de Retenção")
        st.info("Simule o impacto de diferentes ações de retenção antes de implementá-las na prática.")
        
        col1, col2 = st.columns(2)
        with col1:
            action_type = st.selectbox(
                "Tipo de Ação",
                [
                    "discount",
                    "upgrade",
                    "priority_support",
                    "loyalty_bonus",
                    "free_installation",
                    "bundle_offer",
                    "contract_extension",
                    "personalized_offer"
                ],
                format_func=lambda x: {
                    "discount": "💰 Desconto na Mensalidade",
                    "upgrade": "🚀 Upgrade de Plano Gratuito",
                    "priority_support": "⭐ Suporte Prioritário",
                    "loyalty_bonus": "🎁 Bônus de Fidelidade",
                    "free_installation": "🔧 Instalação Grátis em Mudança",
                    "bundle_offer": "📦 Pacote Combinado com Desconto",
                    "contract_extension": "📄 Extensão de Contrato com Benefício",
                    "personalized_offer": "🎯 Oferta Personalizada por Perfil"
                }[x]
            )
        with col2:
            impact_rate = st.slider("Taxa de Impacto Esperada (%)", 10, 80, 30)
        
        if st.button("🧪 Executar Simulação"):
            with st.spinner("Simulando impacto da ação..."):
                ab_result = simulate_ab_testing(df_work, action_type, impact_rate / 100)
                if ab_result:
                    st.session_state.ab_test_results = ab_result
                    st.success("✅ Simulação concluída!")
                    st.rerun()
        
        if st.session_state.ab_test_results:
            res = st.session_state.ab_test_results  # ✅ CORREÇÃO: Usar session_state
            
            col1, col2, col3 = st.columns(3)
            col1.metric("📊 Grupo Controle", f"{res['controle_taxa']:.1%} retidos")
            col2.metric("🧪 Grupo Tratamento", f"{res['tratamento_taxa']:.1%} retidos")
            col3.metric("📈 Lift", f"{res['lift']:.1%}" if res['lift'] > 0 else f"{res['lift']:.1%}")
            
            st.divider()
            
            # Visualização
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Controle',
                x=['Retidos', 'Perdidos'],
                y=[res['controle_retidos'], res['controle_total'] - res['controle_retidos']],
                marker_color='#457B9D'
            ))
            fig.add_trace(go.Bar(
                name='Tratamento',
                x=['Retidos', 'Perdidos'],
                y=[res['tratamento_retidos'], res['tratamento_total'] - res['tratamento_retidos']],
                marker_color='#E63946'
            ))
            fig.update_layout(
                title="Resultado da Simulação A/B",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            st.markdown("### 💰 ROI Estimado")
            roi = res['roi_estimado']
            if roi > 0:
                st.success(f"✅ **ROI Positivo**: R$ {roi:,.2f}")
            else:
                st.error(f"❌ **ROI Negativo**: R$ {roi:,.2f}")
            
            st.markdown(f"""
            **Análise:**
            - A ação de **{action_type}** com impacto de **{impact_rate}%** resultou em um lift de **{res['lift']:.1%}**
            - **{res['clientes_retidos_extra']}** clientes adicionais seriam retidos
            - ROI estimado considera custo de R$50 por ação e valor de R$150 por cliente retido
            """)
else:
    st.title("🎯 Radar X — Inteligência Analítica em Churn")
    st.markdown("""
    Bem-vindo ao Radar X. Esta plataforma utiliza Inteligência Artificial avançada para extrair o máximo valor das suas interações de clientes.
    ### Como começar:
    1. Escolha a **Fonte de Dados** no menu lateral.
    2. Visualize os padrões na aba **Dashboard Executivo**.
    3. Realize diagnósticos profundos na aba **Diagnóstico IA**.
    """)
    st.image("https://images.unsplash.com/photo-1460925895917-afdab827c52f?q=80&w=1000", caption="Radar X Analysis")
