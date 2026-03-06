"""
Script para gerar dataset sintético de call center - Radar X Churn
Volume: 1000 registros com diálogos (Atendente vs Cliente)
Colunas: ID_CLIENTE, NOME_CLIENTE, TRANSCRICAO_LIGACAO_CLIENTE, DATETIME_TRANSCRICAO_LIGACAO
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configuração de sementes para reprodutibilidade
random.seed(42)
np.random.seed(42)

# --- Listas auxiliares para geração dinâmica de nomes ---
NOMES = ["Ana", "Roberto", "Carla", "Marcos", "Juliana", "Fernando", "Patrícia", "Diego", "Beatriz", "Ricardo", 
         "Vanessa", "André", "Mônica", "Thiago", "Luciana", "Eduardo", "Camila", "Bruno", "Renata", "Gabriel", 
         "Priscila", "Felipe", "Adriana", "Leandro", "Fernanda", "Gustavo", "Larissa", "Rafael", "Sabrina", "Marcelo"]

SOBRENOMES = ["Mendes", "Alves", "Souza", "Lima", "Costa", "Rocha", "Nunes", "Martins", "Freitas", "Santos", 
              "Torres", "Oliveira", "Barbosa", "Pereira", "Ferreira", "Gomes", "Ribeiro", "Carvalho", "Azevedo", "Silva"]

# ─── Templates de transcrições em formato de Diálogo ────────────────────────
TRANSCRICOES = {
    "tecnico": [
        "Atendente: Central de Relacionamento, como posso ajudar? Cliente: Estou com problema na minha internet, está muito lenta. Atendente: Entendo, o senhor já reiniciou o modem? Cliente: Já fiz de tudo e nada resolve, faz uma semana que está assim.",
        "Atendente: Suporte técnico, boa tarde. Cliente: Minha internet caiu de novo. Atendente: Vou verificar o sinal na sua região. Cliente: Não adianta verificar, eu quero que alguém venha aqui resolver de vez, não aguento mais essa instabilidade.",
        "Atendente: Como posso auxiliar hoje? Cliente: É a terceira vez que ligo pelo mesmo motivo técnico. Atendente: Sinto muito pelo transtorno, vamos abrir um novo chamado. Cliente: Se não resolverem hoje, eu vou cancelar o contrato.",
        "Atendente: Boa tarde, em que posso ajudar? Cliente: Minha conexão está caindo toda hora durante o meu home office. Atendente: Posso agendar uma visita técnica? Cliente: Já vieram aqui e disseram que estava ok, mas continua caindo. Estou sendo prejudicado no trabalho."
    ],
    "financeiro": [
        "Atendente: Setor financeiro, boa tarde. Cliente: Recebi minha fatura com um valor absurdo este mês. Atendente: Vou verificar os detalhes. Cliente: Tem cobranças de serviços que eu nunca contratei, exijo o estorno.",
        "Atendente: Como posso ajudar com sua conta? Cliente: Minha mensalidade subiu do nada. Atendente: Houve um reajuste anual previsto em contrato. Cliente: Ninguém me avisou de 30% de aumento. Outras operadoras são bem mais baratas.",
        "Atendente: Central de cobrança, bom dia. Cliente: Estou ligando porque minha fatura veio errada de novo. Atendente: Deixe-me analisar o histórico. Cliente: Já é o segundo mês seguido com o mesmo erro. Assim fica difícil continuar com vocês.",
        "Atendente: Boa tarde, como posso ajudar? Cliente: Quero cancelar o plano pois o valor está muito alto. Atendente: Podemos ver um desconto? Cliente: Não quero desconto, quero que parem de cobrar taxas indevidas que eu não solicitei."
    ],
    "atendimento": [
        "Atendente: Boa tarde, como posso ajudar? Cliente: Estou ligando para reclamar do atendimento que recebi ontem. Atendente: O que aconteceu? Cliente: O atendente foi grosseiro e desligou o telefone na minha cara.",
        "Atendente: Central de atendimento, em que posso ser útil? Cliente: Estou há mais de uma hora esperando para ser transferido. Atendente: Pedimos desculpas pela demora. Cliente: É um desrespeito com o consumidor, parece que vocês não querem resolver o problema.",
        "Atendente: Olá, como posso auxiliar? Cliente: Preciso falar com um supervisor urgente. Atendente: Posso tentar resolver primeiro? Cliente: Não, já expliquei meu caso para cinco pessoas diferentes hoje e ninguém faz nada.",
        "Atendente: Boa tarde, suporte ao cliente. Cliente: Vocês nunca registram o que eu falo no histórico. Atendente: Vou verificar as notas aqui. Cliente: Toda vez que ligo tenho que explicar tudo do zero. É um desgaste total."
    ],
    "concorrencia": [
        "Atendente: Como posso ajudar? Cliente: Recebi uma proposta da concorrência com o dobro da velocidade. Atendente: Qual o valor oferecido? Cliente: Eles oferecem fibra por um preço muito menor do que eu pago aqui há anos.",
        "Atendente: Central de relacionamento, boa tarde. Cliente: Quero cancelar meu plano hoje. Atendente: Qual seria o motivo? Cliente: A outra operadora instalou fibra na minha rua e o serviço deles é muito superior ao de vocês.",
        "Atendente: Boa tarde, em que posso ser útil? Cliente: Vou trocar de operadora. Atendente: Podemos tentar cobrir a oferta? Cliente: Não tenho mais interesse, já agendei a instalação com a concorrente para amanhã.",
        "Atendente: Olá, como posso ajudar? Cliente: Estou insatisfeito com o custo-benefício daqui. Atendente: Posso oferecer um upgrade? Cliente: A operadora vizinha me ofereceu 6 meses de desconto e instalação grátis. Não tem como competir."
    ],
    "mudanca": [
        "Atendente: Boa tarde, como posso ajudar? Cliente: Vou me mudar e quero levar meu plano. Atendente: Qual o novo CEP? Cliente: É no interior, vocês atendem lá? Atendente: Infelizmente não temos cobertura nessa região. Cliente: Então vou ter que cancelar infelizmente.",
        "Atendente: Central de vendas, boa tarde. Cliente: Preciso transferir meu endereço. Atendente: Verificando disponibilidade... Cliente: E se não tiver cobertura? Atendente: Nesse caso, procedemos com o cancelamento. Cliente: Poxa, eu gostava do serviço, mas preciso da internet no novo local."
    ],
    "pessoal": [
        "Atendente: Como posso auxiliar hoje? Cliente: Estou passando por dificuldades e preciso reduzir minha conta. Atendente: Temos planos mais básicos. Cliente: No momento nem o básico eu consigo pagar, estou desempregado.",
        "Atendente: Boa tarde, em que posso ajudar? Cliente: Preciso cancelar os canais extras. Atendente: Algum motivo específico? Cliente: Corte de gastos em casa, a situação está apertada e preciso economizar."
    ],
}

# ─── Configurações de Geração ───────────────────────────────────────────────
TOTAL_REGISTROS_DESEJADOS = 200
registros = []
data_churn_base = datetime(2025, 11, 30)
contador_global = 0
cliente_id_num = 1

# ─── Execução do Loop de Geração ────────────────────────────────────────────
while contador_global < TOTAL_REGISTROS_DESEJADOS:
    id_cliente = f"CLI{cliente_id_num:03d}"
    nome_cliente = f"{random.choice(NOMES)} {random.choice(SOBRENOMES)}"
    perfil = random.choice(list(TRANSCRICOES.keys()))
    
    # Número de ligações (2 a 7)
    n_ligacoes = random.randint(2, 7)
    
    if contador_global + n_ligacoes > TOTAL_REGISTROS_DESEJADOS:
        n_ligacoes = TOTAL_REGISTROS_DESEJADOS - contador_global
        
    dias_churn = random.randint(0, 29)
    data_churn = data_churn_base - timedelta(days=dias_churn)
    
    datas_ligacoes = sorted([
        data_churn - timedelta(days=random.randint(1, 89))
        for _ in range(n_ligacoes)
    ])
    
    templates = TRANSCRICOES[perfil]
    
    for i, data in enumerate(datas_ligacoes):
        dialogo_base = templates[i % len(templates)]
        
        # Variação final do diálogo para personalizar
        variacao_final = random.choice([
            " Cliente: Vou pensar melhor no que fazer.",
            " Atendente: Posso ajudar em algo mais? Cliente: Por enquanto é só.",
            " Cliente: Aguardo uma solução até amanhã.",
            " Atendente: O protocolo é 202511098. Cliente: Ok, anotado.",
            " Cliente: Isso é o que todos dizem.",
            ""
        ])
        
        dt_formatado = data.replace(
            hour=random.randint(8, 19), 
            minute=random.randint(0, 59), 
            second=random.randint(0, 59)
        )

        registros.append({
            "ID_CLIENTE": id_cliente,
            "NOME_CLIENTE": nome_cliente,
            "TRANSCRICAO_LIGACAO_CLIENTE": (dialogo_base + variacao_final).strip(),
            "DATETIME_TRANSCRICAO_LIGACAO": dt_formatado.strftime("%Y-%m-%d %H:%M:%S"),
        })
        contador_global += 1
    
    cliente_id_num += 1

# ─── Criação do DataFrame e Salvamento ───────────────────────────────────────
df = pd.DataFrame(registros)
df = df.sort_values(["ID_CLIENTE", "DATETIME_TRANSCRICAO_LIGACAO"]).reset_index(drop=True)

output_path = "dados_churn_sintetico_1000.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"✅ Dataset gerado com sucesso!")
print(f"   Total de registros: {len(df)}")
print(f"   Formato: Diálogo (Atendente: / Cliente:)")
print(f"   Arquivo: {output_path}")
print("\nExemplo da primeira linha:")
print(df.iloc[0]["TRANSCRICAO_LIGACAO_CLIENTE"])
