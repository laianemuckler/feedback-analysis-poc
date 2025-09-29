"""
Pré-processamento de texto para análise de feedbacks
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from utils import clean_text, CONFIG

class FeedbackPreprocessor:
    """
    Classe para pré-processamento de feedbacks
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=CONFIG['tfidf_max_features'],
            min_df=CONFIG['tfidf_min_df'], 
            max_df=CONFIG['tfidf_max_df'],
            ngram_range=(1, 2),  # Unigramas e bigramas
            lowercase=True,
            stop_words=None  # Já removemos na limpeza
        )
        self.is_fitted = False
        
    def clean_texts(self, texts):
        """
        Limpa lista de textos
        """
        print("🧹 Limpando textos...")
        cleaned = []
        
        for i, text in enumerate(texts):
            cleaned_text = clean_text(text)
            cleaned.append(cleaned_text)
            
            if (i + 1) % 100 == 0:
                print(f"✅ Limpeza: {i + 1}/{len(texts)} textos processados")
        
        return cleaned
    
    def fit_transform(self, texts):
        """
        Treina o vectorizador e transforma os textos
        """
        print("🔢 Vectorizando textos com TF-IDF...")
        
        # Limpar textos
        clean_texts = self.clean_texts(texts)
        
        # Remover textos vazios
        valid_texts = [text for text in clean_texts if text.strip()]
        
        if len(valid_texts) == 0:
            raise ValueError("Nenhum texto válido após limpeza!")
        
        # Treinar e transformar
        tfidf_matrix = self.vectorizer.fit_transform(valid_texts)
        self.is_fitted = True
        
        print(f"✅ Vectorização completa!")
        print(f"📊 Matriz TF-IDF: {tfidf_matrix.shape[0]} documentos x {tfidf_matrix.shape[1]} features")
        print(f"🎯 Features mais importantes: {self.get_top_features(10)}")
        
        return tfidf_matrix, valid_texts
    
    def transform(self, texts):
        """
        Transforma novos textos (vectorizador já treinado)
        """
        if not self.is_fitted:
            raise ValueError("Vectorizador não foi treinado! Use fit_transform primeiro.")
        
        clean_texts = self.clean_texts(texts)
        valid_texts = [text for text in clean_texts if text.strip()]
        
        return self.vectorizer.transform(valid_texts), valid_texts
    
    def get_feature_names(self):
        """
        Retorna nomes das features
        """
        if not self.is_fitted:
            return []
        return self.vectorizer.get_feature_names_out()
    
    def get_top_features(self, n=10):
        """
        Retorna top N features por importância média
        """
        if not self.is_fitted:
            return []
        
        feature_names = self.get_feature_names()
        return list(feature_names[:n])
    
    def save_model(self, filepath='data/preprocessor_model.pkl'):
        """
        Salva o modelo treinado
        """
        if not self.is_fitted:
            print("⚠️ Modelo não foi treinado ainda!")
            return False
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.vectorizer, filepath)
        print(f"💾 Modelo salvo em: {filepath}")
        return True
    
    def load_model(self, filepath='data/preprocessor_model.pkl'):
        """
        Carrega modelo pré-treinado
        """
        try:
            self.vectorizer = joblib.load(filepath)
            self.is_fitted = True
            print(f"📂 Modelo carregado de: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            return False

def preprocess_feedback_data(csv_path='data/feedbacks_sample.csv'):
    """
    Função principal para processar dados de feedback
    """
    print("🚀 Iniciando pré-processamento dos feedbacks...")
    
    # Carregar dados
    try:
        df = pd.read_csv(csv_path)
        print(f"📊 Dataset carregado: {len(df)} feedbacks")
    except Exception as e:
        print(f"❌ Erro ao carregar dataset: {e}")
        return None, None
    
    # Verificar coluna de texto
    if 'texto' not in df.columns:
        print("❌ Coluna 'texto' não encontrada!")
        return None, None
    
    # Inicializar pré-processador
    preprocessor = FeedbackPreprocessor()
    
    # Processar textos
    try:
        tfidf_matrix, clean_texts = preprocessor.fit_transform(df['texto'].tolist())
        
        # Salvar modelo
        preprocessor.save_model()
        
        # Criar DataFrame processado
        processed_df = df.copy()
        processed_df['texto_limpo'] = clean_texts
        
        # Salvar dados processados
        output_path = 'data/feedbacks_processed.csv'
        processed_df.to_csv(output_path, index=False)
        print(f"💾 Dados processados salvos em: {output_path}")
        
        return tfidf_matrix, processed_df, preprocessor
        
    except Exception as e:
        print(f"❌ Erro durante processamento: {e}")
        return None, None, None

def demo_preprocessing():
    """
    Demonstração do pré-processamento
    """
    print("🎯 DEMO: Pré-processamento de Texto")
    print("=" * 50)
    
    # Textos exemplo
    exemplos = [
        "O smartphone é EXCELENTE!!! Bateria dura muito!",
        "Base ficou oleosa na minha pele, não gostei :(",  
        "Camiseta chegou com tamanho errado, muito grande",
        "Adorei o perfume, cheiro maravilhoso!!!"
    ]
    
    print("📝 Textos originais:")
    for i, texto in enumerate(exemplos, 1):
        print(f"{i}. {texto}")
    
    print("\n🧹 Após limpeza:")
    for i, texto in enumerate(exemplos, 1):
        limpo = clean_text(texto)
        print(f"{i}. {limpo}")
    
    print("\n" + "=" * 50)

def main():
    """
    Função principal - executa pré-processamento completo
    """
    # Demo primeiro
    demo_preprocessing()
    
    # Verificar se dataset existe
    if not os.path.exists('data/feedbacks_sample.csv'):
        print("❌ Dataset não encontrado! Execute primeiro: python src/data_generator.py")
        return
    
    # Processar dados completos
    results = preprocess_feedback_data()
    
    if results[0] is not None:
        tfidf_matrix, processed_df, preprocessor = results
        print(f"\n🎉 Pré-processamento concluído com sucesso!")
        print(f"📊 Matriz TF-IDF: {tfidf_matrix.shape}")
        print(f"🎯 Pronto para clustering (Fase 3)!")
    else:
        print("\n❌ Falha no pré-processamento!")

if __name__ == "__main__":
    main()