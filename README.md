# Radar X – Diagnóstico Inteligente de Churn (D-90)

Plataforma avançada de análise de churn que combina **IA Generativa Multiagentes (LangGraph)**, **Machine Learning** e **Feature Engineering comportamental** para identificar a causa raiz do cancelamento a partir de jornadas D-90 (90 dias antes do churn).

> **LGPD:** a variável `NOME_CLIENTE` é excluída de toda análise preditiva. Apenas comportamentos e transcrições anonimizadas são utilizados.

---

## 🚀 Como Executar

```bash
# 1. Criar e ativar ambiente
conda create --name churn python=3.11
source activate churn
conda activate churn

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
├── app.py                                 # App principal Streamlit (5 abas)
├── dados_churn_sintetico.csv              # Base de dados com TARGET_CHURN real (0/1)
├── dados_churn_sintetico_enriquecido.csv  # Cache de classificações IA (gerado automaticamente)
├── generate_dataset.py                    # Gerador de dados sintéticos
├── requirements.txt
└── modules/
    ├── agents.py      # Multiagentes LangGraph (Auditor, Resumidor, Classificador, Diagnóstico)
    ├── data_utils.py  # Carregamento, validação e consolidação D-90
    ├── eda.py         # Visualizações Plotly (Sankey, Bolha, Rosca, Heatmap, WordCloud)
    └── ml_model.py    # Feature Engineering (24 features) + Modelos ML + Diagnóstico Pareto + Sankey de Risco
```

---

## 🎯 TARGET_CHURN — Os Dois Públicos

O dataset (`dados_churn_sintetico.csv`) contém a coluna **`TARGET_CHURN`** que define o status real de churn de cada cliente:

| Valor | Público | Definição |
|-------|---------|-----------|
| `0` | 🟢 **NÃO CHURN** | Clientes com jornada ativa — **ainda sem churn registrado** (foco em retenção preventiva) |
| `1` | 🔴 **CHURN** | Clientes que **saíram da empresa** — churn confirmado (análise post-mortem de causa raiz) |

Toda a modelagem preditiva (Random Forest, Logistic Regression) e os gráficos de jornada refletem essa divisão real — **nenhum target sintético é criado**.

---

## 📊 Abas do Dashboard

### 📈 Dashboard Executivo
- **KPIs** — Clientes analisados, principal motivo, % churn e ticket médio de interações
- **Classificação Automática via IA** — Classifica todas as transcrições com LangGraph; resultados em cache (`dados_churn_sintetico_enriquecido.csv`)
- **Gráfico de Bolhas** — Distribuição percentual das causas raiz predominantes
- **Gráfico de Rosca** — Causa raiz predominante detalhada
- **Filtros de Visualização** — Motivo, Confiança Mínima IA (%) e Período
- **Tabela "Clientes sob Diagnóstico"** — Filtrável pelos critérios acima
- **🌊 Sankey Dinâmico (atualizado)** — Fluxo completo da jornada:
  - Motivo 1 → Motivo 2 → ... → **🔴 CHURN** (TARGET=1) ou **🟢 NÃO CHURN** (TARGET=0)
  - Links em **vermelho** para jornadas que terminam em CHURN
  - Links em **verde** para jornadas que terminam em NÃO CHURN
  - Links em **azul** para etapas intermediárias
  - Reativo aos filtros de motivo, confiança e período

### 🌊 Jornadas
- Top 3 padrões de jornada mais frequentes (sequência de macro motivos)
- Correlação: Frequência de ligações × Volume de Texto
- Detalhamento expandível por cliente (até 10 clientes)

### 📋 Classificações
- Volume de ligações por mês
- Nuvem de palavras por motivo (filtrável)
- Lista completa de interações classificadas (filtrável por motivo)

### 🤖 Diagnóstico IA (LangGraph)
Dois modos de diagnóstico lado a lado:

| Modo | Descrição |
|---|---|
| **👤 Por Cliente Individual** | Raio-X detalhado de um cliente específico |
| **🎯 Por Causa Raiz Predominante** | Diagnóstico por grupo + cliente representativo |

**Storytelling gerado pela IA:**

| Campo | Descrição |
|---|---|
| 📖 Resumo Sequencial | Narrativa cronológica da jornada completa |
| 🌱 Causa Raiz Predominante | Causa raiz + Confiança da IA (%) |
| ⚡ Evento Gatilho / Ruptura | Momento de virada + Confiança da IA (%) |
| 💬 Sentimento da Jornada | Sentimento predominante + Confiança da IA (%) |
| 🔍 Padrões Detectados | Padrões comportamentais + Confiança da IA (%) |

### 🧬 Predição de Risco
Treina **Random Forest** e **Logistic Regression** com as 24 features comportamentais extraídas da jornada D-90. O **TARGET_CHURN real do CSV** é utilizado (0 = sem churn, 1 = com churn), sem proxy sintético.

**Saídas por modelo:**
- Acurácia e AUC ROC
- Score de Risco por cliente (gráfico de barras colorido: 🔴 Alto / 🟡 Médio / 🟢 Baixo)
- Importância das Features ordenada da mais relevante para a menos relevante

**🔬 Diagnóstico de Causa Raiz Comportamental:**

1. **Narrativa diagnóstica automática** — tabela com os dois públicos (TARGET=0 e TARGET=1) + interpretação dos principais drivers e perfil de risco
2. **🔴 Pareto 1 — Drivers Principais** — features que explicam ~80% do poder preditivo (barras vermelhas + linha cumulativa)
3. **🔵 Pareto 2 — Drivers Secundários** — features que completam os 20% restantes (barras azuis)
4. **📊 Distribuição de Clientes por Segmento de Risco** — tabela com Score Risco (%), Segmento e **Jornadas de Churn** (sequência de motivos com datas)
5. **🌊 Fluxo da Jornada → Segmento de Risco** — Sankey multi-etapa: Motivo 1 → ... → 🔴 Alto / 🟡 Médio / 🟢 Baixo Risco, colorido por destino, com contagem por nó

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

> **Nota:** `SCORE_INTENSIDADE` é mantido como **feature de entrada** do modelo — não é mais usado como proxy de target.

---

## ⚙️ Histórico de Melhorias

| # | Melhoria | Descrição |
|---|---|---|
| 1 | Features ordenadas por importância | `chart_feature_importance` ordena da mais relevante para a menos relevante no eixo Y |
| 2 | Correção React error #185 | Substituição de `st.dataframe()` por `st.table()` com conversão defensiva de `Categorical` |
| 3 | Pareto separado em 2 gráficos | Pareto 1 (🔴 80% risco) + Pareto 2 (🔵 20% restantes) renderizados sequencialmente |
| 4 | Sankey de Risco (aba ML) | Sankey: `Etapa N \| Motivo → Segmento de Risco`, colorido por destino, com contagem por nó |
| 5 | Coluna "Jornadas de Churn" | Adicionada à tabela de distribuição de risco com sequência de motivos e datas |
| 6 | **TARGET_CHURN real do CSV** | `build_features` e `build_features_expanded` leem o `TARGET_CHURN` diretamente do dataset — TARGET=0 (sem churn) / TARGET=1 (com churn). Eliminado o target sintético por percentil |
| 7 | **Sankey Dinâmico com destino real** | No Dashboard Executivo, o Sankey termina em 🔴 CHURN (TARGET=1) ou 🟢 NÃO CHURN (TARGET=0) com links coloridos por destino |
| 8 | **Narrativa com dois públicos** | `generate_root_cause_diagnosis` exibe tabela com contagem dos dois públicos e diferencia ações de retenção preventiva (TARGET=0) vs. análise pós-churn (TARGET=1) |

---

## ☁️ Deploy no Streamlit Cloud

1. Suba o repositório no GitHub
2. Em **Settings › Secrets**, cole o conteúdo do `secrets.toml`
3. Aponte o deploy para `app.py`

---

**Radar X — Inteligência na Retenção de Clientes**
