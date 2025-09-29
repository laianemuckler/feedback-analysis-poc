"""
Visualizações para análise de clusters de feedbacks
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import numpy as np
from collections import Counter
import os

# Configurar estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ClusterVisualizer:
    """
    Classe para visualizações de clusters
    """
    
    def __init__(self):
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                      '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
    
    def plot_cluster_distribution(self, df, theme_mapping=None, save_path='assets/cluster_distribution.png'):
        """
        Gráfico de distribuição dos clusters
        """
        print("📊 Criando gráfico de distribuição dos clusters...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distribuição por cluster
        cluster_counts = df['cluster'].value_counts().sort_index()
        if theme_mapping:
            labels = [theme_mapping.get(i, f'Cluster {i}') for i in cluster_counts.index]
        else:
            labels = [f'Cluster {i}' for i in cluster_counts.index]
        
        axes[0,0].pie(cluster_counts.values, labels=labels, autopct='%1.1f%%', 
                     colors=self.colors[:len(cluster_counts)])
        axes[0,0].set_title('Distribuição dos Feedbacks por Tema')
        
        # 2. Clusters por segmento
        cluster_segment = pd.crosstab(df['cluster'], df['segmento'])
        cluster_segment.plot(kind='bar', stacked=True, ax=axes[0,1], 
                           color=['#FF9999', '#66B2FF', '#99FF99'])
        axes[0,1].set_title('Clusters por Segmento')
        axes[0,1].set_xlabel('Cluster')
        axes[0,1].legend(title='Segmento')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Sentimento por cluster
        sentiment_cluster = pd.crosstab(df['cluster'], df['sentimento_real'])
        sentiment_cluster.plot(kind='bar', ax=axes[1,0], 
                              color=['#FF6B6B', '#4ECDC4'])
        axes[1,0].set_title('Sentimento por Cluster')
        axes[1,0].set_xlabel('Cluster')
        axes[1,0].legend(title='Sentimento')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Timeline dos clusters
        df['data'] = pd.to_datetime(df['data'])
        df['mes'] = df['data'].dt.to_period('M')
        timeline = df.groupby(['mes', 'cluster']).size().unstack(fill_value=0)
        timeline.plot(kind='area', stacked=True, ax=axes[1,1], 
                     color=self.colors[:len(timeline.columns)])
        axes[1,1].set_title('Evolução Temporal dos Clusters')
        axes[1,1].set_xlabel('Mês')
        axes[1,1].legend(title='Cluster', bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico salvo em: {save_path}")
        plt.show()
    
    def create_wordclouds(self, df, cluster_keywords, theme_mapping=None, 
                         save_dir='assets/wordclouds'):
        """
        Cria wordclouds para cada cluster
        """
        print("☁️ Criando wordclouds por cluster...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        n_clusters = df['cluster'].nunique()
        cols = 3
        rows = (n_clusters + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = [axes] if n_clusters == 1 else axes
        else:
            axes = axes.flatten()
        
        for i in range(n_clusters):
            # Textos do cluster
            cluster_texts = df[df['cluster'] == i]['texto_limpo'].tolist()
            all_text = ' '.join(cluster_texts)
            
            if all_text.strip():
                # Criar wordcloud
                wordcloud = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    max_words=50,
                    colormap='viridis',
                    relative_scaling=0.5,
                    random_state=42
                ).generate(all_text)
                
                # Plotar
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].axis('off')
                
                theme_name = theme_mapping.get(i, f'Cluster {i}') if theme_mapping else f'Cluster {i}'
                axes[i].set_title(f'{theme_name}\n({len(cluster_texts)} feedbacks)', 
                                fontsize=12, pad=10)
                
                # Salvar individual
                wordcloud.to_file(f'{save_dir}/cluster_{i}_wordcloud.png')
            else:
                axes[i].text(0.5, 0.5, 'Sem dados', ha='center', va='center')
                axes[i].set_title(f'Cluster {i} (Vazio)')
        
        # Remover subplots extras
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/all_wordclouds.png', dpi=300, bbox_inches='tight')
        print(f"☁️ Wordclouds salvos em: {save_dir}/")
        plt.show()
    
    def create_interactive_visualization(self, df, tfidf_matrix=None, 
                                       theme_mapping=None, save_path='assets/interactive_clusters.html'):
        """
        Cria visualização interativa com Plotly
        """
        print("🎮 Criando visualização interativa...")
        
        # Preparar dados
        df_viz = df.copy()
        if theme_mapping:
            df_viz['tema'] = df_viz['cluster'].map(theme_mapping)
        else:
            df_viz['tema'] = 'Cluster ' + df_viz['cluster'].astype(str)
        
        # Criar subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribuição por Tema', 'Timeline dos Temas', 
                          'Sentimento vs Segmento', 'Clusters por Segmento'),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Pizza dos temas
        theme_counts = df_viz['tema'].value_counts()
        fig.add_trace(
            go.Pie(labels=theme_counts.index, values=theme_counts.values,
                  name="Distribuição", hovertemplate="<b>%{label}</b><br>" +
                  "Quantidade: %{value}<br>Percentual: %{percent}<extra></extra>"),
            row=1, col=1
        )
        
        # 2. Timeline
        df_viz['data'] = pd.to_datetime(df_viz['data'])
        df_viz['semana'] = df_viz['data'].dt.to_period('W')
        timeline_data = df_viz.groupby(['semana', 'tema']).size().reset_index(name='count')
        
        for tema in timeline_data['tema'].unique():
            tema_data = timeline_data[timeline_data['tema'] == tema]
            fig.add_trace(
                go.Scatter(x=tema_data['semana'].astype(str), y=tema_data['count'],
                          mode='lines+markers', name=tema, showlegend=False),
                row=1, col=2
            )
        
        # 3. Sentimento vs Segmento
        sentiment_segment = pd.crosstab(df_viz['segmento'], df_viz['sentimento_real'])
        fig.add_trace(
            go.Bar(x=sentiment_segment.index, y=sentiment_segment['positivo'],
                  name='Positivo', marker_color='lightblue'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=sentiment_segment.index, y=sentiment_segment['negativo'],
                  name='Negativo', marker_color='lightcoral'),
            row=2, col=1
        )
        
        # 4. Clusters por Segmento
        cluster_segment = pd.crosstab(df_viz['tema'], df_viz['segmento'])
        for segmento in cluster_segment.columns:
            fig.add_trace(
                go.Bar(x=cluster_segment.index, y=cluster_segment[segmento],
                      name=segmento, showlegend=False),
                row=2, col=2
            )
        
        # Layout
        fig.update_layout(
            height=800,
            title_text="Dashboard Interativo - Análise de Clusters de Feedbacks",
            title_x=0.5,
            showlegend=True
        )
        
        # Salvar
        fig.write_html(save_path)
        print(f"🎮 Visualização interativa salva em: {save_path}")
        
        return fig
    
    def create_cluster_comparison_table(self, df, cluster_keywords, theme_mapping=None):
        """
        Cria tabela comparativa dos clusters
        """
        print("📋 Criando tabela comparativa dos clusters...")
        
        comparison_data = []
        
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster_id]
            keywords = cluster_keywords.get(cluster_id, [])
            
            comparison_data.append({
                'Cluster': cluster_id,
                'Tema': theme_mapping.get(cluster_id, f'Cluster {cluster_id}') if theme_mapping else f'Cluster {cluster_id}',
                'Quantidade': len(cluster_data),
                'Percentual': f"{len(cluster_data)/len(df)*100:.1f}%",
                'Palavras-chave': ', '.join([word for word, _ in keywords[:5]]),
                'Segmento Principal': cluster_data['segmento'].mode()[0],
                'Sentimento Dominante': cluster_data['sentimento_real'].mode()[0],
                'Período Principal': f"{cluster_data['data'].min()} a {cluster_data['data'].max()}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Salvar tabela
        comparison_df.to_csv('data/cluster_comparison.csv', index=False)
        print("📋 Tabela salva em: data/cluster_comparison.csv")
        
        # Mostrar tabela formatada
        print("\n" + "="*100)
        print("COMPARAÇÃO DOS CLUSTERS ENCONTRADOS")
        print("="*100)
        print(comparison_df.to_string(index=False))
        
        return comparison_df

def main():
    """
    Função principal - gera todas as visualizações
    """
    print("🎨 CRIANDO VISUALIZAÇÕES DOS CLUSTERS")
    print("=" * 50)
    
    # Verificar se dados existem
    if not os.path.exists('data/feedbacks_clustered.csv'):
        print("❌ Dados clusterizados não encontrados! Execute primeiro o clustering.")
        return
    
    # Carregar dados
    df = pd.read_csv('data/feedbacks_clustered.csv')
    print(f"📊 Dados carregados: {len(df)} feedbacks, {df['cluster'].nunique()} clusters")
    
    # Criar visualizador
    visualizer = ClusterVisualizer()
    
    # Mapping básico (seria carregado do clustering)
    theme_mapping = {}
    for i in range(df['cluster'].nunique()):
        theme_mapping[i] = f"Tema {i+1}"
    
    # Gerar visualizações
    visualizer.plot_cluster_distribution(df, theme_mapping)
    
    # Mock de cluster keywords para wordcloud
    cluster_keywords = {}
    for i in range(df['cluster'].nunique()):
        cluster_texts = df[df['cluster'] == i]['texto_limpo'].tolist()
        all_words = ' '.join(cluster_texts).split()
        word_freq = Counter(all_words)
        cluster_keywords[i] = [(word, freq) for word, freq in word_freq.most_common(10)]
    
    visualizer.create_wordclouds(df, cluster_keywords, theme_mapping)
    visualizer.create_cluster_comparison_table(df, cluster_keywords, theme_mapping)

    visualizer.create_interactive_visualization(df, theme_mapping=theme_mapping, save_path='assets/interactive_clusters.html')

    
    print("\n🎉 Todas as visualizações foram criadas!")

if __name__ == "__main__":
    main()