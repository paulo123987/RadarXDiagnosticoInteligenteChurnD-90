"""
modules/agents.py
Arquitetura Multiagentes para análise de transcrições de churn usando LangGraph.
"""

import json
import re
import time
from typing import TypedDict, List, Optional, Annotated
import operator

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ─── Configurações e Prompts ──────────────────────────────────────────────────

MACRO_MOTIVOS = [
    "Problema Técnico",
    "Concorrência",
    "Financeiro",
    "Mudança de Endereço/Localidade",
    "Atendimento",
    "Pessoal",
    "Não identificado",
]

FEW_SHOT_CLASSIFICACAO = """
Exemplos de classificação (few-shot):

TRANSCRIÇÃO: "Minha internet fica caindo toda hora, já mandaram técnico e não resolveu."
MACRO_MOTIVO: Problema Técnico
EVIDÊNCIA: "internet fica caindo toda hora"
SCORE: 97

TRANSCRIÇÃO: "A outra operadora está me oferecendo o dobro da velocidade pelo mesmo preço."
MACRO_MOTIVO: Concorrência
EVIDÊNCIA: "outra operadora está me oferecendo o dobro da velocidade"
SCORE: 95
"""

# ─── Definição do Estado do LangGraph ──────────────────────────────────────────

class AgentState(TypedDict):
    """Estado do grafo para processar a jornada de um cliente."""
    transcriptions: List[str]
    current_index: int
    results_per_call: Annotated[List[dict], operator.add]
    journey_metadata: dict  # n_ligacoes, span_dias
    diagnostico_final: dict
    llm_p_config: dict # Renamed to avoid collision with LangGraph 'config'

# ─── Utilitários ─────────────────────────────────────────────────────────────

def _get_llm(llm_p_config: dict):
    provider = llm_p_config.get("provider", "OpenAI")
    api_key = llm_p_config.get("api_key", "")
    model_name = llm_p_config.get("model", "gpt-4o-mini") # Default to 4o-mini
    
    if provider == "OpenAI":
        # api_key is the standard for langchain-openai >= 0.1.0
        return ChatOpenAI(api_key=api_key, model=model_name, temperature=0.2)
    elif provider == "Groq":
        # groq_api_key is standard for langchain-groq
        return ChatGroq(groq_api_key=api_key, model_name=model_name, temperature=0.2)
    return None

def _parse_json_safe(text: str) -> dict:
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return {}

# ─── Nós do Grafo (Agentes) ───────────────────────────────────────────────────

def node_process_call(state: AgentState):
    """Processa uma ligação individual usando Auditor, Resumidor e Classificador."""
    idx = state["current_index"]
    transcr = state["transcriptions"][idx]
    llm = _get_llm(state["llm_p_config"])
    threshold = state["llm_p_config"].get("threshold", 90)
    
    if not llm:
        return {"current_index": idx + 1}

    # Auditor
    p1 = f"Auditor de qualidade (JSON): {transcr}"
    r1 = llm.invoke([SystemMessage(content="Responda JSON: {\"score\":int, \"ruido\":bool}"), HumanMessage(content=p1)])
    auditoria = _parse_json_safe(r1.content)
    
    # Resumidor
    p2 = f"Resumo estruturado (JSON): {transcr}"
    r2 = llm.invoke([SystemMessage(content="Responda JSON: {\"resumo\":str, \"sentimento\":str, \"evento\":str}"), HumanMessage(content=p2)])
    resumo = _parse_json_safe(r2.content)
    
    # Classificador
    motivos_str = ", ".join(MACRO_MOTIVOS)
    p3 = f"{FEW_SHOT_CLASSIFICACAO}\nClassifique (JSON): {transcr}\nMotivos: {motivos_str}"
    r3 = llm.invoke([SystemMessage(content="Responda JSON: {\"macro_motivo\":str, \"score\":int}"), HumanMessage(content=p3)])
    classif = _parse_json_safe(r3.content)
    
    if classif.get("score", 0) < threshold:
        classif["macro_motivo"] = "Não identificado"
        
    call_result = {
        "index": idx,
        "auditoria": auditoria,
        "resumo": resumo,
        "classificacao": classif
    }
    
    return {
        "results_per_call": [call_result],
        "current_index": idx + 1
    }

def node_diagnostician(state: AgentState):
    """Agente 4: Diagnóstico de Jornada (D-90) com storytelling aprimorado."""
    llm = _get_llm(state["llm_p_config"])
    meta = state["journey_metadata"]
    
    if not llm:
        return {"diagnostico_final": {"error": "LLM not configured"}}

    # Consolidar texto para o diagnóstico
    jornada_texto = "\n".join([
        f"Ligação {r['index']+1}: RESUMO={r['resumo'].get('resumo', '')} | "
        f"SENTIMENTO={r['resumo'].get('sentimento', '')} | "
        f"EVENTO={r['resumo'].get('evento', '')} | "
        f"MOTIVO={r['classificacao'].get('macro_motivo', '')} | "
        f"CONF={r['classificacao'].get('score', 0)}%"
        for r in state["results_per_call"]
    ])
    
    prompt = f"""Você é um especialista em análise de churn de clientes de telecomunicações.
Analise a jornada completa de um cliente que cancelou o serviço nos últimos 90 dias.

Total de ligações: {meta['n_ligacoes']} em {meta['span_dias']} dias.
Jornada cronológica:
{jornada_texto}

Sua tarefa é gerar um diagnóstico narrativo rico e preciso.

Responda SOMENTE em JSON com esta estrutura exata:
{{
  "resumo_sequencial": "Narrativa fluida e detalhada que descreve cronologicamente como a insatisfação evoluiu ao longo da jornada, incluindo os fatos principais de cada ligação e como eles se conectam causalmente até o churn.",
  "causa_raiz_predominante": "Nome da causa raiz principal que iniciou a jornada de insatisfação",
  "causa_raiz_score": <número 0-100 representando confiança nesta análise>,
  "evento_gatilho": "Descrição do evento ou momento específico que foi o ponto de ruptura / virada decisiva para o churn",
  "evento_gatilho_score": <número 0-100>,
  "sentimento_jornada": "Sentimento predominante ao longo da jornada (ex: frustração crescente, decepção, raiva, resignação)",
  "sentimento_score": <número 0-100>,
  "padroes_detectados": "Descrição dos padrões comportamentais recorrentes detectados na jornada (ex: escalada progressiva, mesmo problema não resolvido, múltiplos canais tentados)",
  "padroes_score": <número 0-100>,
  "ruptura": "Momento específico de virada",
  "escalada": <true ou false>,
  "confianca_geral": <número 0-100>
}}"""
    
    res = llm.invoke([HumanMessage(content=prompt)])
    diagnostico = _parse_json_safe(res.content)
    return {"diagnostico_final": diagnostico}


# ─── Construção do Grafo ──────────────────────────────────────────────────────

def create_churn_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("process_call", node_process_call)
    workflow.add_node("diagnostician", node_diagnostician)
    
    workflow.set_entry_point("process_call")
    
    def route_after_call(state):
        if state["current_index"] < len(state["transcriptions"]):
            return "process_call"
        return "diagnostician"
    
    workflow.add_conditional_edges(
        "process_call",
        route_after_call,
        {
            "process_call": "process_call",
            "diagnostician": "diagnostician"
        }
    )
    
    workflow.add_edge("diagnostician", END)
    
    return workflow.compile()

# ─── Funções de Classificação Individual e Batch ──────────────────────────────

def classify_single_transcription(transcricao: str, llm_p_config: dict) -> dict:
    """
    Classifica UMA transcrição usando apenas o agente Classificador.
    Retorna: {"macro_motivo": str, "score": int, "evidencia": str}
    """
    llm = _get_llm(llm_p_config)
    threshold = llm_p_config.get("threshold", 90)
    
    if not llm:
        return {
            "macro_motivo": "Erro - LLM não configurado",
            "score": 0,
            "evidencia": "Configuração de API inválida"
        }
    
    # Prompt de classificação
    motivos_str = ", ".join(MACRO_MOTIVOS)
    prompt = f"{FEW_SHOT_CLASSIFICACAO}\n\nClassifique a seguinte transcrição:\n{transcricao}\n\nMotivos possíveis: {motivos_str}"
    
    try:
        response = llm.invoke([
            SystemMessage(content='Responda APENAS em JSON com esta estrutura: {"macro_motivo": str, "score": int, "evidencia": str}'),
            HumanMessage(content=prompt)
        ])
        
        result = _parse_json_safe(response.content)
        
        # Aplicar threshold
        if result.get("score", 0) < threshold:
            result["macro_motivo"] = "Não identificado"
        
        # Garantir estrutura mínima
        if "macro_motivo" not in result:
            result["macro_motivo"] = "Não identificado"
        if "score" not in result:
            result["score"] = 0
        if "evidencia" not in result:
            result["evidencia"] = ""
            
        return result
        
    except Exception as e:
        return {
            "macro_motivo": "Erro na classificação",
            "score": 0,
            "evidencia": f"Erro: {str(e)}"
        }


def batch_classify_all_transcriptions(df, llm_p_config: dict) -> "pd.DataFrame":
    """
    Classifica TODAS as transcrições do DataFrame.
    Adiciona colunas: PERFIL_RECLAMACAO, CONFIDENCE_SCORE, EVIDENCIA
    """
    import pandas as pd
    import time
    
    df_result = df.copy()
    results = []
    
    total = len(df)
    
    for idx, row in df.iterrows():
        try:
            transcricao = row.get("TRANSCRICAO_LIGACAO_CLIENTE", "")
            
            result = classify_single_transcription(transcricao, llm_p_config)
            
            results.append({
                "PERFIL_RECLAMACAO": result.get("macro_motivo", "Não identificado"),
                "CONFIDENCE_SCORE": result.get("score", 0),
                "EVIDENCIA": result.get("evidencia", "")
            })
            
            # Pequeno delay para evitar rate limit
            time.sleep(0.1)
            
        except Exception as e:
            results.append({
                "PERFIL_RECLAMACAO": "Erro na classificação",
                "CONFIDENCE_SCORE": 0,
                "EVIDENCIA": str(e)
            })
    
    # Adicionar colunas ao DataFrame
    df_result["PERFIL_RECLAMACAO"] = [r["PERFIL_RECLAMACAO"] for r in results]
    df_result["CONFIDENCE_SCORE"] = [r["CONFIDENCE_SCORE"] for r in results]
    df_result["EVIDENCIA"] = [r["EVIDENCIA"] for r in results]
    
    return df_result


# ─── Função de Interface para o App ───────────────────────────────────────────

def run_langgraph_pipeline(
    transcriptions: List[str],
    journey_metadata: dict,
    llm_p_config: dict
):
    """Executa o grafo do LangGraph e retorna o estado final."""
    app = create_churn_graph()
    
    initial_state = {
        "transcriptions": transcriptions,
        "current_index": 0,
        "results_per_call": [],
        "journey_metadata": journey_metadata,
        "diagnostico_final": {},
        "llm_p_config": llm_p_config
    }
    
    final_output = app.invoke(initial_state)
    return final_output
