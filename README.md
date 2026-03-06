# Radar X – Diagnóstico Inteligente de Churn (D-90)

Plataforma avançada de análise de churn que combina **IA Generativa (Multiagentes)**, **LangGraph** e **Machine Learning** para identificar a causa raiz do cancelamento a partir de jornadas D-90 (90 dias antes do churn).

---

## 🚀 Como Executar

```bash
# 1. Criar e ativar ambiente
conda create --name churn python=3.11
source activate churn

# 2. Instalar pip e dependências
conda install pip
pip install -r requirements.txt

# 3. (Opcional) Gerar base sintética
python generate_dataset.py

# 4. Configurar API Keys (.streamlit/secrets.toml)
# OPENAI_API_KEY = "sk-..."
# GROQ_API_KEY   = "gsk-..."

# 5. Iniciar o app
streamlit run app.py
```

---

## 🗂️ Estrutura do Projeto

```
projeto_churn_jornada/
├── app.py                              # App principal Streamlit
├── dados_churn_sintetico.csv           # Base de dados para demo
├── dados_churn_sintetico_enriquecido.csv  # Cache de classificações IA (gerado automaticamente)
├── generate_dataset.py                 # Gerador de dados sintéticos
├── requirements.txt
└── modules/
    ├── agents.py      # Multiagentes LangGraph (Auditor, Resumidor, Classificador, Diagnóstico)
    ├── data_utils.py  # Carregamento, validação e consolidação D-90
    ├── eda.py         # Visualizações Plotly (Sankey, Bolha, Rosca, Heatmap, WordCloud)
    └── ml_model.py    # Feature Engineering (24 features) + Modelos ML + Diagnóstico Pareto + Sankey de Risco
```

---

## 📊 Abas do Dashboard

### 📈 Dashboard Executivo
- **KPIs** — Clientes analisados, principal motivo, % churn e ticket médio de interações
- **Classificação Automática** — Botão para classificar todas as transcrições via IA (LangGraph); resultados em cache (`dados_churn_sintetico_enriquecido.csv`)
- **Gráfico de Bolhas** — Distribuição percentual das causas raiz predominantes
- **Gráfico de Rosca** — Causa raiz predominante detalhada
- **Filtros de Visualização** — Motivo, Confiança Mínima IA (%) e Período Churn
- **Tabela "Clientes sob Diagnóstico"** — Filtrável pelos critérios acima
- **Sankey Dinâmico** — Fluxo completo da jornada (Motivo 1 → Motivo 2 → ⛔ CHURN), reativo aos filtros

### 🌊 Jornadas
- Top 3 padrões de jornada mais frequentes
- Correlação: Frequência de ligações × Volume de Texto
- Detalhamento expandível por cliente (até 10 clientes)

### 📋 Classificações
- Volume de ligações por mês
- Nuvem de palavras por motivo
- Lista completa de interações classificadas (filtrável por motivo)

### 🤖 Diagnóstico IA
Dois modos de diagnóstico lado a lado:

| Modo | Descrição |
|---|---|
| **👤 Por Cliente Individual** | Raio-X detalhado de um cliente específico |
| **🎯 Por Causa Raiz Predominante** | Diagnóstico por grupo (causa raiz) + cliente representativo |

**Storytelling gerado pela IA (LangGraph):**

| Campo | Descrição |
|---|---|
| 📖 Resumo Sequencial | Narrativa cronológica da jornada completa |
| 🌱 Causa Raiz Predominante | Causa raiz + Confiança da IA (%) |
| ⚡ Evento Gatilho / Ruptura | Momento de virada + Confiança da IA (%) |
| 💬 Sentimento da Jornada | Sentimento predominante + Confiança da IA (%) |
| 🔍 Padrões Detectados | Padrões comportamentais + Confiança da IA (%) |

### 🧬 Predição de Risco
Treina **Random Forest** e **Logistic Regression** sobre as 24 features comportamentais.

**Saídas por modelo:**
- Acurácia e AUC ROC
- Score de Risco por cliente (gráfico de barras colorido por risco)
- Importância das Features ordenada por relevância (da mais para a menos importante)

**🔬 Diagnóstico de Causa Raiz Comportamental:**

1. **Narrativa diagnóstica automática** — interpreta os principais drivers e o perfil de risco
2. **🔴 Pareto 1 — Drivers Principais** — features que explicam ~80% do poder preditivo (barras vermelhas)
3. **🔵 Pareto 2 — Drivers Secundários** — features que completam os 20% restantes (barras azuis)
4. **📊 Distribuição de Clientes por Segmento de Risco** — tabela com Score Risco (%), Segmento e **Jornadas de Churn** (sequência de motivos com datas, ex: `Atendimento (01/09) → Financeiro (15/10) → ...`)
5. **🌊 Fluxo da Jornada — Narrativa Completa** — Sankey multi-etapa mostrando a evolução sequencial de motivos até o segmento de risco final, com cores por destino (🔴 Alto / 🟡 Médio / 🟢 Baixo)

---

## 🤖 Arquitetura Multiagente (LangGraph)

```
Transcrição → [Auditor] → [Resumidor] → [Classificador] → loop por ligação
                                                              ↓
                                                        [Diagnóstico D-90]
                                                              ↓
                                                    Storytelling + Causa Raiz
```

| Agente | Função |
|---|---|
| **Auditor** | Valida qualidade e ruído da transcrição |
| **Resumidor** | Extrai resumo, sentimento e evento-chave |
| **Classificador** | Categoriza macro motivo com few-shot |
| **Diagnóstico** | Analisa a jornada completa e gera storytelling |

Suporta **OpenAI** (GPT-4o, GPT-4o-mini, GPT-3.5-turbo) e **Groq** (LLaMA 3.1-8b, LLaMA 3.3-70b).

---

## 🧬 Feature Engineering (24 Features)

| Categoria | Features |
|---|---|
| Frequência | `freq_7d`, `freq_30d`, `freq_90d` |
| Recência | `days_since_last_call` |
| Intervalos | `avg_interval_days`, `max_interval_days`, `min_interval_days`, `std_interval_days` |
| Intensidade | `calls_per_week_avg`, `TAXA_LIGACOES_DIA`, `SCORE_INTENSIDADE` |
| NLP | `avg_token_count`, `max_token_count`, `total_mentions_concorrente`, `flag_ameaca_cancelamento`, `total_mentions_problemas`, `total_mentions_prazos` |
| Evolução | `increase_rate_30d_vs_90d` |
| Base D-90 | `N_LIGACOES_D90`, `SPAN_DIAS`, `INTERVALO_MEDIO_DIAS`, `TOKENS_TOTAL` |

---

## 🌊 Sankey de Jornada → Risco (novo)

O gráfico Sankey de risco exibe:

- **Nós de Etapa** (Etapa 1 → Etapa 4): motivo de insatisfação em cada passo da jornada, com contagem de clientes
- **Nós finais**: segmento de risco (🔴 Alto / 🟡 Médio / 🟢 Baixo)
- **Links coloridos** pelo segmento de destino do cliente
- **Hover**: mostra origem, destino e número de clientes no fluxo

Exemplo de fluxo:
```
Atendimento (45) → Financeiro (30) → 🔴 Alto Risco (18)
Atendimento (45) → Financeiro (30) → 🟡 Médio Risco (12)
Problema Técnico (20)              → 🟢 Baixo Risco (20)
```

---

## ⚙️ Ajustes Recentes

| # | Ajuste | Descrição |
|---|---|---|
| 1 | Features ordenadas por importância | `chart_feature_importance` ordena da mais relevante para a menos relevante no eixo Y |
| 2 | Correção React error #185 | Substituição de `st.dataframe()` por `st.table()` e conversão defensiva de `Categorical` → tipos nativos |
| 3 | Pareto separado em 2 gráficos | Pareto 1 (🔴 vermelho, 80% do risco) + Pareto 2 (🔵 azul, 20% restantes) renderizados sequencialmente |
| 4 | Sankey de Risco | Novo Sankey: `Etapa N | Motivo → Segmento de Risco`, colorido por destino, com contagem por nó |
| 5 | Coluna "Jornadas de Churn" | Adicionada à tabela de distribuição de risco com a sequência de motivos e datas de cada cliente |

---

## ☁️ Deploy no Streamlit Cloud

1. Suba o repositório no GitHub
2. Em **Settings › Secrets**, cole o conteúdo do `secrets.toml`
3. Aponte o deploy para `app.py`

---

**Radar X — Inteligência na Retenção de Clientes**
