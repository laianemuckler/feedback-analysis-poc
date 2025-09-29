"""
Análise estratégica com foco em insights acionáveis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class StrategicAnalyzer:
    """
    Análise estratégica com foco em valor de negócio
    """
    
    def __init__(self, df_path='data/feedbacks_clustered.csv'):
        self.df = pd.read_csv(df_path)
        self.df['data'] = pd.to_datetime(self.df['data'])
        
        # Mapeamento dos temas
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
        
        self.df['tema'] = self.df['cluster'].map(self.theme_mapping)
        print(f"📊 Análise estratégica iniciada: {len(self.df)} feedbacks")
    
    def calculate_business_impact(self):
        """
        Calcula impacto real no negócio por tema
        """
        print("💰 Calculando impacto no negócio...")
        
        impact_analysis = []
        
        for tema in self.df['tema'].unique():
            tema_data = self.df[self.df['tema'] == tema]
            
            # Métricas de impacto
            total_feedbacks = len(tema_data)
            negative_ratio = len(tema_data[tema_data['sentimento_real'] == 'negativo']) / total_feedbacks
            market_share = total_feedbacks / len(self.df) * 100
            
            # Calcular urgência (volume + negatividade)
            urgency_score = (market_share * 0.6) + (negative_ratio * 100 * 0.4)
            
            # Potencial de melhoria
            improvement_potential = negative_ratio * total_feedbacks
            
            # Classificação de prioridade
            if urgency_score > 25:
                priority = "🚨 Crítica"
            elif urgency_score > 15:
                priority = "⚠️ Alta"
            elif urgency_score > 8:
                priority = "📊 Média"
            else:
                priority = "✅ Baixa"
            
            impact_analysis.append({
                'tema': tema,
                'total_feedbacks': total_feedbacks,
                'participacao_mercado': market_share,
                'taxa_negatividade': negative_ratio * 100,
                'score_urgencia': urgency_score,
                'potencial_melhoria': improvement_potential,
                'prioridade': priority,
                'roi_estimado': self._estimate_roi(market_share, negative_ratio)
            })
        
        return sorted(impact_analysis, key=lambda x: x['score_urgencia'], reverse=True)
    
    def _estimate_roi(self, market_share, negative_ratio):
        """
        Estima ROI de investimento no tema
        """
        # Lógica simplificada de ROI
        base_roi = market_share * 2  # Maior participação = maior ROI potencial
        negative_multiplier = 1 + (negative_ratio * 2)  # Mais negativos = maior ROI ao resolver
        
        return base_roi * negative_multiplier
    
    def segment_performance_analysis(self):
        """
        Análise de performance por segmento
        """
        print("🎯 Analisando performance por segmento...")
        
        segment_analysis = []
        
        for segmento in self.df['segmento'].unique():
            seg_data = self.df[self.df['segmento'] == segmento]
            
            # Métricas por segmento
            total = len(seg_data)
            positive_rate = len(seg_data[seg_data['sentimento_real'] == 'positivo']) / total * 100
            
            # Temas principais no segmento
            top_themes = seg_data['tema'].value_counts().head(3)
            main_issues = seg_data[seg_data['sentimento_real'] == 'negativo']['tema'].value_counts().head(2)
            
            # Benchmarking
            if positive_rate >= 80:
                benchmark = "🌟 Excelente"
            elif positive_rate >= 70:
                benchmark = "✅ Bom"
            elif positive_rate >= 60:
                benchmark = "⚠️ Médio"
            else:
                benchmark = "🚨 Crítico"
            
            segment_analysis.append({
                'segmento': segmento.title(),
                'total_feedbacks': total,
                'taxa_satisfacao': positive_rate,
                'benchmark': benchmark,
                'temas_principais': list(top_themes.index[:2]),
                'problemas_principais': list(main_issues.index[:2]) if len(main_issues) > 0 else [],
                'recomendacao': self._generate_segment_recommendation(positive_rate, main_issues)
            })
        
        return sorted(segment_analysis, key=lambda x: x['taxa_satisfacao'], reverse=True)
    
    def _generate_segment_recommendation(self, satisfaction_rate, main_issues):
        """
        Gera recomendação específica por segmento
        """
        if satisfaction_rate >= 80:
            return "Manter estratégia atual e usar como benchmark"
        elif satisfaction_rate >= 70:
            return "Pequenos ajustes para chegar a excelência"
        elif len(main_issues) > 0:
            top_issue = main_issues.index[0]
            return f"Focar em resolver: {top_issue}"
        else:
            return "Revisão completa da estratégia necessária"
    
    def create_priority_matrix(self):
        """
        Cria matriz de priorização Impacto x Esforço
        """
        print("📊 Criando matriz de priorização...")
        
        impact_data = self.calculate_business_impact()
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Dados para scatter plot
        x_values = []  # Facilidade (inverso do esforço)
        y_values = []  # Impacto
        labels = []
        colors = []
        
        for data in impact_data:
            # Facilidade baseada no volume (menos feedbacks = mais fácil resolver)
            facilidade = max(0, 100 - data['total_feedbacks'])
            impacto = data['roi_estimado']
            
            x_values.append(facilidade)
            y_values.append(impacto)
            labels.append(data['tema'])
            
            # Cor baseada na prioridade
            if "Crítica" in data['prioridade']:
                colors.append('red')
            elif "Alta" in data['prioridade']:
                colors.append('orange')
            elif "Média" in data['prioridade']:
                colors.append('yellow')
            else:
                colors.append('green')
        
        # Criar scatter plot
        scatter = ax.scatter(x_values, y_values, c=colors, s=200, alpha=0.7, edgecolors='black')
        
        # Adicionar labels
        for i, label in enumerate(labels):
            ax.annotate(label, (x_values[i], y_values[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold')
        
        # Linhas divisórias
        ax.axhline(y=np.mean(y_values), color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=np.mean(x_values), color='gray', linestyle='--', alpha=0.5)
        
        # Quadrantes
        ax.text(0.75, 0.95, 'QUICK WINS\n(Alto Impacto, Fácil)', transform=ax.transAxes, 
                fontsize=12, fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        ax.text(0.25, 0.95, 'PROJETOS GRANDES\n(Alto Impacto, Difícil)', transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.text(0.75, 0.05, 'FILL-INS\n(Baixo Impacto, Fácil)', transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='center', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        ax.text(0.25, 0.05, 'THANKLESS TASKS\n(Baixo Impacto, Difícil)', transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='center', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        ax.set_xlabel('Facilidade de Implementação →', fontsize=12)
        ax.set_ylabel('Impacto no Negócio →', fontsize=12)
        ax.set_title('Matriz de Priorização - Temas por Impacto x Esforço', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('assets/priority_matrix.png', dpi=300, bbox_inches='tight')
        print("💾 Matriz salva em: assets/priority_matrix.png")
        plt.show()
    
    def generate_executive_summary(self):
        """
        Gera resumo executivo com insights acionáveis
        """
        impact_data = self.calculate_business_impact()
        segment_data = self.segment_performance_analysis()
        
        print("\n" + "="*80)
        print("📋 RESUMO EXECUTIVO - ANÁLISE ESTRATÉGICA DE FEEDBACKS")
        print("="*80)
        
        # Top insights
        critical_themes = [d for d in impact_data if "Crítica" in d['prioridade']]
        high_roi_themes = sorted(impact_data, key=lambda x: x['roi_estimado'], reverse=True)[:3]
        
        print(f"\n🎯 PRINCIPAIS DESCOBERTAS:")
        print(f"   • Total de {len(self.df)} feedbacks analisados em {len(self.df['tema'].unique())} temas")
        print(f"   • {len(critical_themes)} temas com prioridade crítica identificados")
        print(f"   • Segmento mais satisfeito: {segment_data[0]['segmento']} ({segment_data[0]['taxa_satisfacao']:.1f}%)")
        print(f"   • Segmento com mais oportunidade: {segment_data[-1]['segmento']} ({segment_data[-1]['taxa_satisfacao']:.1f}%)")
        
        print(f"\n🚨 AÇÕES PRIORITÁRIAS (próximos 30 dias):")
        for i, theme in enumerate(critical_themes[:3], 1):
            print(f"   {i}. {theme['tema']}: {theme['total_feedbacks']} feedbacks, "
                  f"{theme['taxa_negatividade']:.1f}% negativos")
            print(f"      → ROI estimado: {theme['roi_estimado']:.1f}x do investimento")
        
        print(f"\n📈 OPORTUNIDADES DE CRESCIMENTO:")
        for i, theme in enumerate(high_roi_themes[:3], 1):
            if theme['taxa_negatividade'] < 30:  # Oportunidades positivas
                print(f"   {i}. {theme['tema']}: {theme['participacao_mercado']:.1f}% do mercado, "
                      f"{theme['taxa_negatividade']:.1f}% negativos")
        
        print(f"\n🎯 RECOMENDAÇÕES POR SEGMENTO:")
        for seg in segment_data:
            print(f"   📦 {seg['segmento']}: {seg['recomendacao']}")
        
        # ROI calculation
        total_negative = len(self.df[self.df['sentimento_real'] == 'negativo'])
        estimated_savings = total_negative * 0.3 * 50  # 30% redução x R$50 por feedback
        
        print(f"\n💰 IMPACTO FINANCEIRO ESTIMADO:")
        print(f"   • {total_negative} feedbacks negativos atualmente")
        print(f"   • Potencial economia: R$ {estimated_savings:,.0f}/mês com melhorias")
        print(f"   • Investimento sugerido: R$ {estimated_savings * 0.4:,.0f} (payback 2.5 meses)")
        
        return {
            'critical_themes': critical_themes,
            'segment_performance': segment_data,
            'roi_themes': high_roi_themes,
            'estimated_savings': estimated_savings
        }

def main():
    """
    Executa análise estratégica completa
    """
    print("🚀 ANÁLISE ESTRATÉGICA COM FOCO EM VALOR DE NEGÓCIO")
    print("=" * 60)
    
    analyzer = StrategicAnalyzer()
    
    # Análises principais
    impact_data = analyzer.calculate_business_impact()
    segment_data = analyzer.segment_performance_analysis()
    
    # Visualizações
    analyzer.create_priority_matrix()
    
    # Resumo executivo
    executive_summary = analyzer.generate_executive_summary()
    
    print(f"\n🎉 Análise estratégica concluída!")
    print(f"💡 Foco: Insights acionáveis, não predições vazias")

if __name__ == "__main__":
    main()