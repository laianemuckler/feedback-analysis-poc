"""
Utilidades e configurações do projeto
"""
import re
import nltk 
from nltk.corpus import stopwords

# Baixar stopwords se necessário
def download_nltk_data():
    """Download dados do NLTK se necessário"""
    try:
        nltk.data.find('corpora/stopwords')
        print("✅ Stopwords NLTK já disponíveis")
    except LookupError:
        print("📥 Baixando stopwords do NLTK...")
        nltk.download('stopwords', quiet=True)
        print("✅ Download concluído!")

# Executar download
download_nltk_data()

# Stopwords customizadas para análise de feedbacks
CUSTOM_STOPWORDS = set(stopwords.words('portuguese')) | {
    'produto', 'item', 'compra', 'comprei', 'recebi', 'pedido', 
    'site', 'loja', 'vendedor', 'muito', 'bem', 'mal', 'bom', 'boa',
    'ruim', 'melhor', 'pior', 'gostei', 'adorei', 'odiei', 'recomendo',
    'super', 'mega', 'demais', 'bastante', 'pouco', 'meio', 'ate', 'ate',
    'so', 'nao', 'sim', 'tambem', 'ja', 'ainda', 'depois', 'antes'
}

def clean_text(text):
    """
    Limpa e normaliza texto para processamento
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Converter para minúsculas
    text = text.lower()
    
    # Remover caracteres especiais, manter apenas letras, números e espaços
    text = re.sub(r'[^a-záàâãéèêíïóôõöúçñ\s]', ' ', text)
    
    # Remover espaços extras
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remover stopwords
    words = text.split()
    words = [word for word in words if word not in CUSTOM_STOPWORDS and len(word) > 2]
    
    return ' '.join(words)

def remove_stopwords(text):
    """
    Remove apenas stopwords, mantém outras palavras
    """
    words = text.split()
    return ' '.join([word for word in words if word not in CUSTOM_STOPWORDS])

# Configurações do projeto
CONFIG = {
    'random_seed': 42,
    'n_feedbacks': 800,
    'date_range_months': 6,
    'min_words_per_feedback': 3,
    'max_words_per_feedback': 15,
    'tfidf_max_features': 1000,
    'tfidf_min_df': 2,
    'tfidf_max_df': 0.8
}