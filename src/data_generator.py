"""
Gerador de feedbacks sintéticos para e-commerce
Segmentos: Eletrônicos, Beleza/Cosméticos, Roupas/Fashion
"""
import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta
import os
from utils import CONFIG

# Configurar faker para português
fake = Faker('pt_BR')
random.seed(CONFIG['random_seed'])

# Templates de feedbacks por segmento
FEEDBACK_TEMPLATES = {
    'eletronicos': {
        'produtos': [
            'smartphone', 'celular', 'iPhone', 'Samsung', 'fone de ouvido', 
            'notebook', 'laptop', 'TV', 'televisão', 'tablet', 'smartwatch',
            'carregador', 'cabo USB', 'mouse', 'teclado', 'monitor', 'caixa de som'
        ],
        'positivos': [
            "O {produto} é excelente, bateria dura o dia todo!",
            "{produto} com ótima qualidade de imagem, recomendo",
            "Som perfeito do {produto}, superou expectativas",
            "{produto} chegou rapidinho, bem embalado",
            "Performance incrível do {produto}, vale cada centavo",
            "Design lindo do {produto}, muito elegante",
            "{produto} funciona perfeitamente, sem travamentos",
            "Excelente custo-benefício do {produto}",
            "Tela do {produto} é maravilhosa, cores vivas",
            "{produto} carrega super rápido, adorei"
        ],
        'negativos': [
            "{produto} chegou com defeito, tela riscada",
            "Bateria do {produto} dura muito pouco, decepcionante",
            "Som do {produto} é terrível, chiado constante",
            "{produto} trava toda hora, muito lento",
            "Entrega demorou 15 dias, {produto} chegou sujo",
            "{produto} esquenta demais durante uso",
            "Qualidade péssima do {produto}, plástico frágil",
            "{produto} não funciona conforme anunciado",
            "Preço alto para qualidade do {produto}",
            "{produto} quebrou em uma semana de uso"
        ]
    },
    'beleza': {
        'produtos': [
            'base', 'corretivo', 'batom', 'gloss', 'rímel', 'máscara de cílios',
            'sombra', 'blush', 'pó compacto', 'shampoo', 'condicionador', 
            'creme facial', 'protetor solar', 'perfume', 'esmalte', 'hidratante',
            'sérum', 'primer', 'delineador', 'lápis de olho'
        ],
        'positivos': [
            "A {produto} tem cobertura incrível, durou o dia todo",
            "{produto} deixou minha pele muito macia e hidratada",
            "Cor linda do {produto}, combinou perfeitamente comigo",
            "Fragrância do {produto} é maravilhosa, recebo elogios",
            "{produto} não resseca, textura perfeita",
            "Resultado incrível com o {produto}, pele radiante",
            "Embalagem linda do {produto}, produto de qualidade",
            "{produto} fixou super bem, não borrou",
            "Entrega rápida, {produto} bem protegido",
            "Preço justo pelo {produto}, vale muito a pena"
        ],
        'negativos': [
            "O {produto} ressecou muito minha pele, horrível",
            "{produto} saiu muito facilmente, não durou nada",
            "Cor do {produto} não combinou, muito diferente da foto",
            "Cheiro forte do {produto}, deu alergia",
            "{produto} deixou oleosa, textura péssima",
            "Embalagem do {produto} vazou, chegou toda suja",
            "{produto} não fez efeito nenhum, dinheiro jogado fora",
            "Qualidade ruim do {produto}, muito líquido",
            "{produto} causou irritação na pele",
            "Preço caro demais para qualidade do {produto}"
        ]
    },
    'roupas': {
        'produtos': [
            'camiseta', 'blusa', 'vestido', 'calça jeans', 'shorts', 'saia',
            'jaqueta', 'casaco', 'tênis', 'sapato', 'sandália', 'bolsa',
            'mochila', 'cinto', 'boné', 'sutiã', 'calcinha', 'pijama',
            'moletom', 'regata'
        ],
        'positivos': [
            "A {produto} tem tecido de ótima qualidade, muito confortável",
            "{produto} ficou perfeita, tamanho certinho",
            "Cor linda da {produto}, exatamente como na foto",
            "Entrega super rápida da {produto}, bem embalada",
            "{produto} é muito estilosa, recebo muitos elogios",
            "Caimento perfeito da {produto}, modelagem linda",
            "Material da {produto} não desbota, já lavei várias vezes",
            "{produto} chegou antes do prazo, qualidade excelente",
            "Preço ótimo da {produto}, super em conta",
            "Acabamento impecável da {produto}, costura perfeita"
        ],
        'negativos': [
            "A {produto} veio com tamanho errado, muito grande",
            "{produto} desbotou na primeira lavagem, qualidade ruim",
            "Tecido da {produto} é muito fino, transparente",
            "Cor diferente da foto, {produto} decepcionante",
            "{produto} rasgou facilmente, material frágil",
            "Entrega demorou muito, {produto} chegou amassada",
            "Tamanho da {produto} não confere com tabela",
            "{produto} encolheu após lavagem, não serve mais",
            "Costura da {produto} está soltando, péssima qualidade",
            "Cheiro forte da {produto}, difícil de sair"
        ]
    }
}

def generate_feedback(segmento, sentimento):
    """
    Gera um feedback individual
    """
    templates = FEEDBACK_TEMPLATES[segmento][sentimento + 's']
    produtos = FEEDBACK_TEMPLATES[segmento]['produtos']
    
    template = random.choice(templates)
    produto = random.choice(produtos)
    
    return template.format(produto=produto)

def generate_date():
    """
    Gera data aleatória nos últimos 6 meses com tendências
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=CONFIG['date_range_months'] * 30)
    
    # Gerar data aleatória
    random_date = fake.date_between(start_date=start_date, end_date=end_date)
    return random_date

def create_synthetic_dataset():
    """
    Cria dataset sintético completo
    """
    print("🚀 Gerando dataset sintético de feedbacks...")
    
    feedbacks = []
    segmentos = ['eletronicos', 'beleza', 'roupas']
    
    # Distribuição: 70% positivos, 30% negativos (mais realista)
    sentimentos = ['positivo'] * 7 + ['negativo'] * 3
    
    for i in range(CONFIG['n_feedbacks']):
        segmento = random.choice(segmentos)
        sentimento = random.choice(sentimentos)
        
        feedback = {
            'id': i + 1,
            'texto': generate_feedback(segmento, sentimento),
            'data': generate_date(),
            'segmento': segmento,
            'sentimento_real': sentimento  # Para validação posterior
        }
        
        feedbacks.append(feedback)
        
        if (i + 1) % 100 == 0:
            print(f"✅ Gerados {i + 1}/{CONFIG['n_feedbacks']} feedbacks")
    
    # Criar DataFrame
    df = pd.DataFrame(feedbacks)
    
    # Ordenar por data
    df = df.sort_values('data').reset_index(drop=True)
    
    return df

def save_dataset(df):
    """
    Salva dataset no formato CSV
    """
    # Criar diretório data se não existir
    os.makedirs('data', exist_ok=True)
    
    # Salvar arquivo principal
    filepath = 'data/feedbacks_sample.csv'
    df.to_csv(filepath, index=False, encoding='utf-8')
    
    print(f"💾 Dataset salvo em: {filepath}")
    
    # Estatísticas
    print("\n📊 Estatísticas do dataset:")
    print(f"Total de feedbacks: {len(df)}")
    print(f"Período: {df['data'].min()} até {df['data'].max()}")
    print("\nDistribuição por segmento:")
    print(df['segmento'].value_counts())
    print("\nDistribuição por sentimento:")
    print(df['sentimento_real'].value_counts())
    
    return filepath

def main():
    """
    Função principal
    """
    print("🎯 Gerando feedbacks para segmentos: Eletrônicos, Beleza, Roupas")
    
    # Gerar dataset
    df = create_synthetic_dataset()
    
    # Salvar
    filepath = save_dataset(df)
    
    # Mostrar exemplos
    print("\n📝 Exemplos de feedbacks gerados:")
    print("-" * 50)
    for i, row in df.sample(5).iterrows():
        print(f"[{row['segmento']}] {row['texto']}")
    
    print(f"\n🎉 Dataset criado com sucesso! Execute: head data/feedbacks_sample.csv")

if __name__ == "__main__":
    main()