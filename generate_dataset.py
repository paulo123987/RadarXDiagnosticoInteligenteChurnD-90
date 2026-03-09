"""
Script para gerar dataset sintético de call center - Radar X Churn
Volume: 2000 registros (1000 Churn / 1000 Retenção)
Colunas: ID_CLIENTE, TRANSCRICAO_LIGACAO_CLIENTE, DATETIME_TRANSCRICAO_LIGACAO, TARGET_CHURN
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configuração de sementes para reprodutibilidade
random.seed(42)
np.random.seed(42)

# ─── Templates de transcrições por Target ────────────────────────────────────

# Diálogos para clientes que SAÍRAM (Churn = 1) - Baseado em pesquisas de mercado (High Effort, Bill Shock, Competitor Bundles)
TRANSCRICOES_CHURN = {
    "tecnico_quebra_sla": [
        "Atendente: Suporte técnico, boa tarde. Cliente: Eu agendei uma visita técnica para ontem à tarde, fiquei em casa esperando e ninguém apareceu. Atendente: Peço desculpas, houve um imprevisto na rota. Cliente: Isso é um desrespeito. É a segunda vez que faltam. Quero cancelar.",
        "Atendente: Central de Relacionamento. Cliente: Minha internet está caindo toda hora faz duas semanas. Atendente: O sistema mostra que sua região está passando por instabilidade. Cliente: Vocês sempre dizem isso e nunca resolvem. Eu trabalho de home office e estou perdendo dinheiro.",
        "Atendente: Como posso auxiliar? Cliente: Já trocamos o modem, já veio técnico, e minha internet não passa de 50 mega, sendo que pago 500. Atendente: Vou realizar um novo teste de sinal. Cliente: Não adianta testar mais nada, vocês não entregam o que vendem. Vou mudar de operadora."
    ],
    "financeiro_bill_shock": [
        "Atendente: Setor financeiro. Cliente: Minha conta dobrou de valor esse mês! De 99 foi para 199. Atendente: Senhor, o seu período promocional de 12 meses acabou. Cliente: Ninguém me avisou que ia dobrar! Isso é propaganda enganosa, quero cancelar agora.",
        "Atendente: Como posso ajudar com sua fatura? Cliente: Tem uma cobrança de 'serviços digitais e antivírus' que eu nunca pedi. Atendente: Esses serviços vêm inclusos no pacote base. Cliente: Eu não fui informado no momento da venda. É uma venda casada e eu não aceito. Podem cancelar meu plano.",
        "Atendente: Central de cobrança. Cliente: Vocês cortaram meu sinal, mas eu já paguei a conta faz três dias! Atendente: O banco pode levar até 72 horas úteis para compensar. Cliente: Nas outras operadoras libera na hora com o comprovante. O serviço de vocês é péssimo."
    ],
    "atendimento_alto_esforco": [
        "Atendente: Boa tarde, com quem falo? Cliente: Sério? É a quarta vez que me transferem e eu tenho que confirmar meu CPF e explicar o problema tudo de novo. Atendente: Senhor, por favor, confirme os dados. Cliente: Não, eu cansei. Passa para o setor de cancelamento agora.",
        "Atendente: Suporte ao cliente, pois não? Cliente: Estou ligando referente ao protocolo que abri semana passada. Ficaram de me retornar em 48 horas e nada. Atendente: Consta aqui que o setor responsável ainda está analisando. Cliente: O descaso de vocês é absurdo. Não quero mais o serviço."
    ],
    "concorrencia_bundles": [
        "Atendente: Central de relacionamento. Cliente: Quero cancelar. Atendente: Posso saber o motivo? Cliente: A operadora concorrente me ofereceu a mesma velocidade, pela metade do preço, e ainda me deram assinatura da Netflix e Max de graça. Vocês não têm como cobrir isso.",
        "Atendente: Como posso ajudar? Cliente: Já instalei a internet da outra empresa. Pode cancelar a de vocês. Atendente: O senhor teve algum problema conosco? Cliente: A fibra deles chega direto no modem, a de vocês ainda é cabo coaxial antigo. A tecnologia de vocês ficou para trás."
    ],
    "mudanca_sem_cobertura": [
        "Atendente: Central de mudanças, boa tarde. Cliente: Vou me mudar para o interior e preciso transferir a linha. Atendente: Qual o novo CEP? Cliente: 18130-000. Atendente: Infelizmente, nossa fibra não chega nesse endereço. Cliente: Que pena, fui cliente por 5 anos. Terei que cancelar."
    ]
}

# Diálogos para clientes que FICARAM (Churn = 0) - Baseado em pesquisas (FCR, Retenção Ativa, Empatia, Transparência)
TRANSCRICOES_RETENCAO = {
    "tecnico_fcr": [ # First Contact Resolution
        "Atendente: Suporte técnico, como posso ajudar? Cliente: Minha internet está muito lenta hoje. Atendente: Vejo aqui que seu roteador está há muito tempo sem atualização. Vou enviar um comando de reset remoto, aguarde na linha. Cliente: Opa, as luzes piscaram... Voltou! Agora sim está rápido. Muito obrigado pela agilidade.",
        "Atendente: Boa tarde, em que posso ser útil? Cliente: O fio da internet rompeu porque o caminhão passou e puxou. Atendente: Nossa, entendo a urgência. Vou mandar um técnico de emergência agora à tarde, sem custo adicional. Cliente: Perfeito, muito obrigado pela compreensão, aguardo ele.",
        "Atendente: Central de ajuda. Cliente: Comprei uma TV nova e não consigo conectar no Wi-Fi. Atendente: Fique tranquilo, vou te guiar passo a passo. O senhor está com o controle da TV em mãos? Cliente: Sim... Ah, consegui achar a rede! Deu certo, o atendimento de vocês é ótimo."
    ],
    "financeiro_transparencia": [
        "Atendente: Setor financeiro. Cliente: Minha primeira conta veio muito mais alta que o combinado! Atendente: Entendo sua preocupação. Deixe-me explicar: como o senhor instalou no dia 15, essa conta cobra os 15 dias do mês passado mais o mês atual inteiro. É o pagamento pró-rata. As próximas virão no valor normal de 99 reais. Cliente: Ah, entendi! O instalador não tinha me explicado isso. Sendo assim, tudo bem, vou pagar.",
        "Atendente: Como posso ajudar? Cliente: Paguei minha conta em duplicidade sem querer. Atendente: Não se preocupe. O sistema já identificou o pagamento a maior. O valor ficará como crédito e abaterá automaticamente a fatura do mês que vem. Cliente: Que alívio! Muito prático, obrigado.",
        "Atendente: Boa tarde. Cliente: Preciso da segunda via do boleto. Atendente: Claro, acabei de enviar para o seu WhatsApp cadastrado e também te ensinei como pegar direto pelo nosso app para as próximas vezes. Cliente: Chegou aqui no WhatsApp, muito fácil. Valeu!"
    ],
    "fidelizacao_sucesso": [ # Reversão de Churn
        "Atendente: Relacionamento. Cliente: Quero cancelar, vi uma promoção da concorrência por 79 reais. Atendente: O senhor é nosso cliente há 3 anos e não queremos perdê-lo. Se o senhor continuar conosco, eu cubro a oferta para 79 reais e ainda dobro sua velocidade de 250 para 500 mega sem custo de instalação. Cliente: Bom, se dobrar a velocidade e manter esse preço, eu não preciso ter a dor de cabeça de trocar de operadora. Eu aceito.",
        "Atendente: Boa tarde, posso ajudar? Cliente: Estou desempregado e preciso cancelar, não consigo pagar os 150 mensais. Atendente: Sinto muito pela situação. Temos um plano 'Essencial' de 50 reais que mantém o senhor conectado até as coisas melhorarem. Podemos migrar temporariamente? Cliente: Isso ajudaria demais, eu preciso da internet para mandar currículos. Pode alterar para esse de 50."
    ],
    "suporte_proativo": [
        "Atendente: Olá, aqui é da operadora. Identificamos uma falha no equipamento da sua rua e a internet oscilou durante a madrugada. Já consertamos, mas ligamos para confirmar se está tudo ok e informar que daremos desconto de 1 dia na sua fatura. Cliente: Nossa, eu nem tinha percebido a queda, estava dormindo! Mas achei muito bacana vocês avisarem e darem o desconto. Parabéns pelo serviço."
    ]
}

# ─── Função de Geração de Dados ─────────────────────────────────────────────

def gerar_dados(meta_registros, target_valor, transcricoes_dict, inicio_id):
    registros = []
    contador = 0
    id_num = inicio_id
    data_base = datetime(2025, 11, 30)

    while contador < meta_registros:
        id_cliente = f"CLI{id_num:04d}"
        perfil = random.choice(list(transcricoes_dict.keys()))
        n_ligacoes = random.randint(2, 5)
        
        if contador + n_ligacoes > meta_registros:
            n_ligacoes = meta_registros - contador
            
        # Datas nos últimos 60 dias
        datas = sorted([data_base - timedelta(days=random.randint(1, 60)) for _ in range(n_ligacoes)])
        templates = transcricoes_dict[perfil]

        for i, dt in enumerate(datas):
            dialogo = templates[i % len(templates)]
            dt_com_hora = dt.replace(hour=random.randint(8, 18), minute=random.randint(0, 59))
            
            registros.append({
                "ID_CLIENTE": id_cliente,
                "TRANSCRICAO_LIGACAO_CLIENTE": dialogo,
                "DATETIME_TRANSCRICAO_LIGACAO": dt_com_hora.strftime("%Y-%m-%d %H:%M:%S"),
                "TARGET_CHURN": target_valor
            })
            contador += 1
        id_num += 1
        
    return registros, id_num

# ─── Execução ───────────────────────────────────────────────────────────────

# Gerar 1000 registros de Churn (Target 1)
dados_churn, proximo_id = gerar_dados(1000, 1, TRANSCRICOES_CHURN, 1)

# Gerar 1000 registros de Retenção (Target 0)
dados_retencao, _ = gerar_dados(1000, 0, TRANSCRICOES_RETENCAO, proximo_id)

# Unificar e Salvar
df_final = pd.DataFrame(dados_churn + dados_retencao)
df_final = df_final.sort_values(["TARGET_CHURN", "ID_CLIENTE", "DATETIME_TRANSCRICAO_LIGACAO"]).reset_index(drop=True)

output_path = "dataset_churn_2000_balanceado.csv"
df_final.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"✅ Dataset gerado com sucesso!")
print(f"   Total de registros: {len(df_final)}")
print(f"   Churn (1): {len(df_final[df_final['TARGET_CHURN'] == 1])}")
print(f"   Retenção (0): {len(df_final[df_final['TARGET_CHURN'] == 0])}")
print(f"   Arquivo: {output_path}")

