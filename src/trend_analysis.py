"""
Análise temporal de tendências dos clusters de feedbacks
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TrendAnalyzer:
    """
    Classe para análise temporal de clusters
    """
    
    def __init__(self, df_path='data/feedbacks_clustered.csv'):
        self.df = pd.read_csv(df_path)
        self.df['data'] = pd.to_datetime(self.df['data'])
        
        # Mapeamento de temas (baseado nos resultados da Fase 3)
        self.theme_mapping = {
            0: 'Custo-Benefício Premium',
            1: 'Excelência e Qualidade', 
            2: 'Logística e Entrega',
            3: 'Tamanho e Ajuste',
            4: 'Aparência e Design',
            5: 'Material e Conforto',
            6: 'Durabilidade',
            7: 'Fixação de Produtos',
            8: 'Carregamento Rápido',
            9: 'Satisfação e Elogios'
        }
        
        print(f"📊 Dados carregados: {len(self.df)} feedbacks")
        print(f"📅 Período: {self.df['data'].min().date()} até {self.df['data'].max().date()}")
        print(f"🎯 Clusters: {self.df['cluster'].nunique()}")
    
    def prepare_temporal_data(self, freq='W'):
        """
        Prepara dados temporais agrupados por frequência
        freq: 'D' (daily), 'W' (weekly), 'M' (monthly)
        """
        print(f"📈 Preparando dados temporais (frequência: {freq})...")
        
        # Agrupar por período e cluster
        df_temporal = self.df.copy()
        
        if freq == 'W':
            df_temporal['periodo'] = df_temporal['data'].dt.to_period('W')
        elif freq == 'M':
            df_temporal['periodo'] = df_temporal['data'].dt.to_period('M')
        else:
            df_temporal['periodo'] = df_temporal['data'].dt.to_period('D')
        
        # Contar feedbacks por período e cluster
        temporal_counts = df_temporal.groupby(['periodo', 'cluster']).size().unstack(fill_value=0)
        
        # Converter período para string para facilitar plotagem
        temporal_counts.index = temporal_counts.index.astype(str)
        
        # Adicionar nomes dos temas
        temporal_counts.columns = [self.theme_mapping.get(col, f'Cluster {col}') 
                                  for col in temporal_counts.columns]
        
        return temporal_counts
    
    def analyze_growth_trends(self, temporal_data, window=3):
        """
        Analisa tendências de crescimento usando média móvel
        """
        print(f"📊 Analisando tendências de crescimento (janela: {window} períodos)...")
        
        # Calcular médias móveis
        rolling_data = temporal_data.rolling(window=window, center=True).mean()
        
        # Calcular taxa de crescimento (diferença percentual)
        growth_rates = rolling_data.pct_change(periods=window).fillna(0) * 100
        
        # Estatísticas por tema
        growth_stats = {}
        
        for tema in temporal_data.columns:
            if temporal_data[tema].sum() > 5:  # Só analisar temas com dados suficientes
                recent_avg = temporal_data[tema].tail(window).mean()
                early_avg = temporal_data[tema].head(window).mean()
                
                if early_avg > 0:
                    total_growth = ((recent_avg - early_avg) / early_avg) * 100
                else:
                    total_growth = 0
                
                growth_stats[tema] = {
                    'crescimento_total': total_growth,
                    'media_recente': recent_avg,
                    'media_inicial': early_avg,
                    'volatilidade': temporal_data[tema].std(),
                    'tendencia': 'Crescimento' if total_growth > 10 else 'Declínio' if total_growth < -10 else 'Estável'
                }
        
        return growth_stats, growth_rates
    
    def detect_seasonality_patterns(self, temporal_data):
        """
        Detecta padrões sazonais básicos
        """
        print("🔍 Detectando padrões sazonais...")
        
        try:
            # Método mais simples - usar os dados originais agrupados por mês
            # Voltar aos dados originais para análise sazonal
            df_seasonal = self.df.copy()
            df_seasonal['mes'] = df_seasonal['data'].dt.month
            df_seasonal['mes_nome'] = df_seasonal['data'].dt.month_name()
            
            # Analisar por mês se temos dados suficientes
            if len(df_seasonal['mes'].unique()) >= 2:  # Pelo menos 2 meses diferentes
                monthly_patterns = {}
                
                for cluster_id in df_seasonal['cluster'].unique():
                    tema = self.theme_mapping.get(cluster_id, f'Cluster {cluster_id}')
                    cluster_data = df_seasonal[df_seasonal['cluster'] == cluster_id]
                    
                    if len(cluster_data) > 10:  # Só analisar temas com dados suficientes
                        # Contar feedbacks por mês
                        monthly_counts = cluster_data.groupby(['mes', 'mes_nome']).size().reset_index(name='count')
                        
                        if len(monthly_counts) > 1:
                            # Encontrar pico e baixa
                            max_row = monthly_counts.loc[monthly_counts['count'].idxmax()]
                            min_row = monthly_counts.loc[monthly_counts['count'].idxmin()]
                            
                            monthly_patterns[tema] = {
                                'pico_mes': max_row['mes'],
                                'pico_mes_nome': max_row['mes_nome'],
                                'baixa_mes': min_row['mes'],
                                'baixa_mes_nome': min_row['mes_nome'],
                                'variacao_sazonal': monthly_counts['count'].max() - monthly_counts['count'].min(),
                                'meses_analisados': len(monthly_counts)
                            }
                
                return monthly_patterns
            
            else:
                print("⚠️ Dados insuficientes para análise sazonal (menos de 2 meses)")
                return {}
                
        except Exception as e:
            print(f"⚠️ Erro na análise sazonal: {e}")
            print("📊 Continuando sem análise sazonal...")
            return {}
    
    def create_trend_visualizations(self, temporal_data, growth_stats, save_dir='assets'):
        """
        Cria visualizações das tendências
        """
        print("📊 Criando visualizações de tendências...")
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        colors = plt.cm.tab10(np.linspace(0, 1, len(temporal_data.columns)))
        
        # 1. Gráfico de evolução temporal de todos os temas
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1a. Evolução temporal completa
        ax1 = axes[0, 0]
        for i, tema in enumerate(temporal_data.columns):
            if temporal_data[tema].sum() > 5:  # Só plotar temas com dados
                ax1.plot(temporal_data.index, temporal_data[tema], 
                        marker='o', label=tema, color=colors[i], linewidth=2)
        
        ax1.set_title('Evolução Temporal dos Temas', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Período (Semanas)')
        ax1.set_ylabel('Número de Feedbacks')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 1b. Top 5 temas mais ativos
        top_themes = temporal_data.sum().nlargest(5)
        ax2 = axes[0, 1]
        
        for tema in top_themes.index:
            ax2.plot(temporal_data.index, temporal_data[tema], 
                    marker='s', label=tema, linewidth=3, markersize=6)
        
        ax2.set_title('Top 5 Temas Mais Ativos', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Período (Semanas)')
        ax2.set_ylabel('Número de Feedbacks')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # 1c. Gráfico de barras - crescimento total
        ax3 = axes[1, 0]
        themes = list(growth_stats.keys())
        growth_values = [growth_stats[tema]['crescimento_total'] for tema in themes]
        
        colors_growth = ['green' if x > 0 else 'red' for x in growth_values]
        bars = ax3.barh(themes, growth_values, color=colors_growth, alpha=0.7)
        
        ax3.set_title('Taxa de Crescimento por Tema', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Crescimento (%)')
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Adicionar valores nas barras
        for i, (bar, value) in enumerate(zip(bars, growth_values)):
            ax3.text(bar.get_width() + (1 if value > 0 else -1), bar.get_y() + bar.get_height()/2, 
                    f'{value:.1f}%', ha='left' if value > 0 else 'right', va='center')
        
        # 1d. Heatmap de atividade semanal
        ax4 = axes[1, 1]
        
        # Preparar dados para heatmap (últimas semanas vs temas principais)
        heatmap_data = temporal_data[top_themes.index].tail(8).T
        
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', 
                   ax=ax4, cbar_kws={'label': 'Feedbacks'})
        ax4.set_title('Atividade Recente por Tema', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Semanas Recentes')
        ax4.set_ylabel('Temas')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/trends_analysis.png', dpi=300, bbox_inches='tight')
        print(f"💾 Gráfico de tendências salvo em: {save_dir}/trends_analysis.png")
        plt.show()
        
        # 2. Gráfico individual dos temas mais importantes
        self._create_individual_trend_plots(temporal_data, top_themes.index[:4], save_dir)
    
    def _create_individual_trend_plots(self, temporal_data, top_themes, save_dir):
        """
        Cria gráficos individuais para os principais temas
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, tema in enumerate(top_themes):
            ax = axes[i]
            data = temporal_data[tema]
            
            # Plotar linha principal
            ax.plot(data.index, data.values, marker='o', linewidth=3, 
                   markersize=8, color='steelblue', label='Feedbacks')
            
            # Adicionar média móvel
            rolling_mean = data.rolling(window=3, center=True).mean()
            ax.plot(data.index, rolling_mean, '--', linewidth=2, 
                   color='red', alpha=0.8, label='Tendência (MA3)')
            
            # Destacar picos
            max_idx = data.idxmax()
            max_val = data.max()
            ax.annotate(f'Pico: {max_val}', xy=(max_idx, max_val), 
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            ax.set_title(f'{tema}\n(Total: {data.sum()} feedbacks)', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Período')
            ax.set_ylabel('Feedbacks')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.tick_params(axis='x', rotation=45)
            
            # Adicionar área sob a curva
            ax.fill_between(data.index, 0, data.values, alpha=0.3, color='lightblue')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/individual_trends.png', dpi=300, bbox_inches='tight')
        print(f"💾 Gráficos individuais salvos em: {save_dir}/individual_trends.png")
        plt.show()
    
    def generate_insights_summary(self, growth_stats, seasonal_patterns):
        """
        Gera resumo de insights descobertos
        """
        print("\n" + "="*60)
        print("📋 INSIGHTS DA ANÁLISE TEMPORAL")
        print("="*60)
        
        # Temas em crescimento
        growing_themes = [tema for tema, stats in growth_stats.items() 
                         if stats['crescimento_total'] > 10]
        
        # Temas em declínio
        declining_themes = [tema for tema, stats in growth_stats.items() 
                           if stats['crescimento_total'] < -10]
        
        # Temas estáveis
        stable_themes = [tema for tema, stats in growth_stats.items() 
                        if -10 <= stats['crescimento_total'] <= 10]
        
        print(f"\n🚀 TEMAS EM CRESCIMENTO ({len(growing_themes)}):")
        for tema in growing_themes:
            growth = growth_stats[tema]['crescimento_total']
            print(f"   📈 {tema}: +{growth:.1f}%")
        
        print(f"\n⚠️  TEMAS EM DECLÍNIO ({len(declining_themes)}):")
        for tema in declining_themes:
            growth = growth_stats[tema]['crescimento_total']
            print(f"   📉 {tema}: {growth:.1f}%")
        
        print(f"\n⚖️  TEMAS ESTÁVEIS ({len(stable_themes)}):")
        for tema in stable_themes:
            growth = growth_stats[tema]['crescimento_total']
            print(f"   📊 {tema}: {growth:+.1f}%")
        
        # Padrões sazonais
        if seasonal_patterns:
            print(f"\n🌐 PADRÕES SAZONAIS DETECTADOS:")
            for tema, pattern in seasonal_patterns.items():
                print(f"   📅 {tema}: Pico em {pattern['pico_mes_nome']} ({pattern['pico_mes']}), "
                      f"baixa em {pattern['baixa_mes_nome']} ({pattern['baixa_mes']})")
                print(f"      Variação: {pattern['variacao_sazonal']} feedbacks entre pico e baixa")
        else:
            print(f"\n🌐 PADRÕES SAZONAIS: Período muito curto para análise sazonal consistente")
        # Recomendações
        print(f"\n💡 RECOMENDAÇÕES:")
        
        if growing_themes:
            print(f"   ✅ Investir em melhorias nos temas em crescimento")
            print(f"   ✅ Monitorar de perto: {', '.join(growing_themes[:2])}")
        
        if declining_themes:
            print(f"   🔧 Ação urgente necessária nos temas em declínio")
            print(f"   🔧 Foco prioritário: {', '.join(declining_themes[:2])}")
        
        print(f"   📊 Continuar monitoramento dos temas estáveis")
        
        return {
            'growing_themes': growing_themes,
            'declining_themes': declining_themes,
            'stable_themes': stable_themes,
            'seasonal_patterns': seasonal_patterns
        }

def main():
    """
    Executa análise temporal completa
    """
    print("🚀 FASE 4A: ANÁLISE TEMPORAL DE TENDÊNCIAS")
    print("=" * 50)
    
    # Verificar se dados existem
    if not os.path.exists('data/feedbacks_clustered.csv'):
        print("❌ Dados clusterizados não encontrados!")
        return
    
    # Inicializar analisador
    analyzer = TrendAnalyzer()
    
    # Preparar dados temporais
    temporal_data = analyzer.prepare_temporal_data(freq='W')
    
    # Analisar tendências
    growth_stats, growth_rates = analyzer.analyze_growth_trends(temporal_data)
    
    # Detectar sazonalidade
    seasonal_patterns = analyzer.detect_seasonality_patterns(temporal_data)
    
    # Criar visualizações
    analyzer.create_trend_visualizations(temporal_data, growth_stats)
    
    # Gerar insights
    insights = analyzer.generate_insights_summary(growth_stats, seasonal_patterns)
    
    # Salvar dados para próxima fase
    temporal_data.to_csv('data/temporal_trends.csv')
    print(f"\n💾 Dados temporais salvos em: data/temporal_trends.csv")
    
    print(f"\n🎉 Análise temporal concluída!")
    
    return temporal_data, growth_stats, insights

if __name__ == "__main__":
    import os
    main()