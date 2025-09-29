"""
Agrupamento temático de feedbacks usando K-Means
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from collections import Counter
from preprocessor import FeedbackPreprocessor
from utils import CONFIG

class FeedbackClusterer:
    """
    Classe para clustering de feedbacks
    """
    
    def __init__(self, n_clusters=None, random_state=CONFIG['random_seed']):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.preprocessor = None
        self.cluster_labels = None
        self.cluster_centers = None
        self.silhouette_avg = None
        self.is_fitted = False
        
    def find_optimal_clusters(self, tfidf_matrix, max_k=10):
        """
        Encontra número ótimo de clusters usando método do cotovelo e silhouette
        """
        print("🔍 Procurando número ótimo de clusters...")
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            # Treinar K-Means
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(tfidf_matrix)
            
            # Calcular métricas
            inertia = kmeans.inertia_
            sil_score = silhouette_score(tfidf_matrix, labels)
            
            inertias.append(inertia)
            silhouette_scores.append(sil_score)
            
            print(f"K={k}: Inertia={inertia:.2f}, Silhouette={sil_score:.3f}")
        
        # Plotar gráficos
        self._plot_cluster_metrics(k_range, inertias, silhouette_scores)
        
        # Escolher melhor K (maior silhouette score)
        best_k = k_range[np.argmax(silhouette_scores)]
        best_silhouette = max(silhouette_scores)
        
        print(f"\n🎯 Melhor número de clusters: {best_k} (Silhouette: {best_silhouette:.3f})")
        
        return best_k, silhouette_scores
    
    def _plot_cluster_metrics(self, k_range, inertias, silhouette_scores):
        """
        Plota gráficos de métricas dos clusters
        """
        plt.figure(figsize=(15, 5))
        
        # Método do cotovelo
        plt.subplot(1, 2, 1)
        plt.plot(k_range, inertias, 'bo-')
        plt.xlabel('Número de Clusters (K)')
        plt.ylabel('Inércia')
        plt.title('Método do Cotovelo')
        plt.grid(True, alpha=0.3)
        
        # Silhouette Score
        plt.subplot(1, 2, 2)
        plt.plot(k_range, silhouette_scores, 'ro-')
        plt.xlabel('Número de Clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score por K')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salvar gráfico
        os.makedirs('assets', exist_ok=True)
        plt.savefig('assets/cluster_metrics.png', dpi=300, bbox_inches='tight')
        print("📊 Gráfico de métricas salvo em: assets/cluster_metrics.png")
        plt.show()
    
    def fit(self, tfidf_matrix, texts, auto_k=True):
        """
        Treina o modelo de clustering
        """
        print("🚀 Iniciando clustering dos feedbacks...")
        
        # Encontrar K ótimo se necessário
        if auto_k or self.n_clusters is None:
            self.n_clusters, _ = self.find_optimal_clusters(tfidf_matrix)
        
        # Treinar K-Means final
        print(f"🎯 Treinando K-Means com {self.n_clusters} clusters...")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=self.random_state,
            n_init=20,
            max_iter=300
        )
        
        # Fazer clustering
        self.cluster_labels = self.kmeans.fit_predict(tfidf_matrix)
        self.cluster_centers = self.kmeans.cluster_centers_
        
        # Calcular silhouette score
        self.silhouette_avg = silhouette_score(tfidf_matrix, self.cluster_labels)
        
        self.is_fitted = True
        
        print(f"✅ Clustering concluído!")
        print(f"📊 Silhouette Score: {self.silhouette_avg:.3f}")
        print(f"📈 Distribuição dos clusters: {Counter(self.cluster_labels)}")
        
        return self.cluster_labels
    
    def extract_cluster_keywords(self, preprocessor, n_keywords=10):
        """
        Extrai palavras-chave mais importantes de cada cluster
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado!")
        
        print("🔑 Extraindo palavras-chave dos clusters...")
        
        feature_names = preprocessor.get_feature_names()
        cluster_keywords = {}
        
        for i in range(self.n_clusters):
            # Pegar centroide do cluster
            centroid = self.cluster_centers[i]
            
            # Índices das features mais importantes
            top_indices = centroid.argsort()[-n_keywords:][::-1]
            
            # Extrair palavras correspondentes
            keywords = []
            for idx in top_indices:
                if idx < len(feature_names):
                    word = feature_names[idx]
                    weight = centroid[idx]
                    keywords.append((word, weight))
            
            cluster_keywords[i] = keywords
            
            # Mostrar resultados
            words_str = ', '.join([word for word, _ in keywords[:5]])
            print(f"📋 Cluster {i}: {words_str}")
        
        return cluster_keywords
    
    def assign_cluster_themes(self, cluster_keywords):
        """
        Atribui nomes temáticos aos clusters baseado nas palavras-chave
        """
        theme_mapping = {}
        
        for cluster_id, keywords in cluster_keywords.items():
            top_words = [word for word, _ in keywords[:3]]
            
            # Lógica simples para identificar temas
            if any(word in ['bateria', 'carregador', 'smartphone', 'tela'] for word in top_words):
                theme = 'Eletrônicos - Funcionalidades'
            elif any(word in ['entrega', 'chegou', 'pedido', 'prazo'] for word in top_words):
                theme = 'Logística e Entrega'
            elif any(word in ['pele', 'hidratada', 'macia', 'oleosa'] for word in top_words):
                theme = 'Beleza - Efeitos na Pele'
            elif any(word in ['tamanho', 'grande', 'pequeno', 'tecido'] for word in top_words):
                theme = 'Roupas - Tamanho e Material'
            elif any(word in ['qualidade', 'preço', 'caro', 'barato'] for word in top_words):
                theme = 'Preço e Qualidade'
            elif any(word in ['cor', 'cores', 'linda', 'bonita'] for word in top_words):
                theme = 'Aparência e Design'
            else:
                theme = f'Tema {cluster_id + 1}'
            
            theme_mapping[cluster_id] = theme
            print(f"🏷️ Cluster {cluster_id} → {theme}")
        
        return theme_mapping
    
    def visualize_clusters(self, tfidf_matrix, texts, theme_mapping=None):
        """
        Visualiza clusters em 2D usando PCA
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado!")
        
        print("📊 Criando visualização dos clusters...")
        
        # Redução de dimensionalidade para visualização
        pca = PCA(n_components=2, random_state=self.random_state)
        tfidf_2d = pca.fit_transform(tfidf_matrix.toarray())
        
        # Criar gráfico
        plt.figure(figsize=(12, 8))
        
        # Cores para clusters
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_clusters))
        
        for i in range(self.n_clusters):
            cluster_points = tfidf_2d[self.cluster_labels == i]
            theme_name = theme_mapping.get(i, f'Cluster {i}') if theme_mapping else f'Cluster {i}'
            
            plt.scatter(
                cluster_points[:, 0], cluster_points[:, 1],
                c=[colors[i]], label=theme_name, alpha=0.6, s=50
            )
        
        plt.title(f'Visualização dos Clusters (PCA)\nSilhouette Score: {self.silhouette_avg:.3f}')
        plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]:.1%} da variância)')
        plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]:.1%} da variância)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Salvar gráfico
        plt.tight_layout()
        plt.savefig('assets/clusters_visualization.png', dpi=300, bbox_inches='tight')
        print("📊 Visualização salva em: assets/clusters_visualization.png")
        plt.show()
    
    def create_cluster_summary(self, df, cluster_keywords, theme_mapping):
        """
        Cria resumo detalhado dos clusters
        """
        print("\n📋 RESUMO DOS CLUSTERS ENCONTRADOS")
        print("=" * 60)
        
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = self.cluster_labels
        
        for i in range(self.n_clusters):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == i]
            theme = theme_mapping.get(i, f'Cluster {i}')
            keywords = cluster_keywords[i]
            
            print(f"\n🎯 {theme.upper()}")
            print(f"   📊 Quantidade: {len(cluster_data)} feedbacks")
            print(f"   🔑 Palavras-chave: {', '.join([word for word, _ in keywords[:5]])}")
            print(f"   📈 Segmentos: {cluster_data['segmento'].value_counts().to_dict()}")
            print(f"   😊 Sentimento: {cluster_data['sentimento_real'].value_counts().to_dict()}")
            
            # Exemplos de feedbacks
            print(f"   📝 Exemplos:")
            examples = cluster_data['texto'].head(3).tolist()
            for j, example in enumerate(examples, 1):
                example_short = example[:60] + "..." if len(example) > 60 else example
                print(f"      {j}. {example_short}")
        
        return df_with_clusters
    
    def save_model(self, filepath='data/clustering_model.pkl'):
        """
        Salva modelo treinado
        """
        if not self.is_fitted:
            print("⚠️ Modelo não foi treinado ainda!")
            return False
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.kmeans, filepath)
        print(f"💾 Modelo de clustering salvo em: {filepath}")
        return True

def main():
    """
    Função principal - executa clustering completo
    """
    print("🚀 FASE 3: CLUSTERING TEMÁTICO DE FEEDBACKS")
    print("=" * 50)
    
    # Verificar se dados processados existem
    if not os.path.exists('data/feedbacks_processed.csv'):
        print("❌ Dados processados não encontrados! Execute primeiro as fases anteriores.")
        return
    
    # Carregar dados
    df = pd.read_csv('data/feedbacks_processed.csv')
    print(f"📊 Dados carregados: {len(df)} feedbacks")
    
    # Carregar pré-processador
    preprocessor = FeedbackPreprocessor()
    if not preprocessor.load_model():
        print("❌ Erro ao carregar pré-processador!")
        return
    
    # Vetorizar textos limpos
    tfidf_matrix, clean_texts = preprocessor.transform(df['texto_limpo'].tolist())
    
    # Inicializar clusterer
    clusterer = FeedbackClusterer()
    
    # Fazer clustering
    labels = clusterer.fit(tfidf_matrix, clean_texts)
    
    # Extrair palavras-chave
    cluster_keywords = clusterer.extract_cluster_keywords(preprocessor)
    
    # Atribuir temas
    theme_mapping = clusterer.assign_cluster_themes(cluster_keywords)
    
    # Visualizar
    clusterer.visualize_clusters(tfidf_matrix, clean_texts, theme_mapping)
    
    # Criar resumo
    df_final = clusterer.create_cluster_summary(df, cluster_keywords, theme_mapping)
    
    # Salvar resultados
    clusterer.save_model()
    df_final.to_csv('data/feedbacks_clustered.csv', index=False)
    print(f"\n💾 Resultados salvos em: data/feedbacks_clustered.csv")
    
    print(f"\n🎉 Clustering concluído! Encontrados {clusterer.n_clusters} temas principais.")

if __name__ == "__main__":
    main()