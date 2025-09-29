"""
Gerador de relatório HTML com principais insights
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os

class HTMLReportGenerator:
    """
    Gera relatório HTML executivo com insights principais
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
    
    def calculate_key_metrics(self):
        """
        Calcula métricas principais para o relatório
        """
        # Métricas gerais
        total_feedbacks = len(self.df)
        period_start = self.df['data'].min().strftime('%d/%m/%Y')
        period_end = self.df['data'].max().strftime('%d/%m/%Y')
        
        # Análise por tema
        theme_analysis = []
        for tema in self.df['tema'].unique():
            tema_data = self.df[self.df['tema'] == tema]
            total = len(tema_data)
            negative_count = len(tema_data[tema_data['sentimento_real'] == 'negativo'])
            negative_rate = (negative_count / total) * 100
            market_share = (total / total_feedbacks) * 100
            
            # Calcular prioridade
            urgency_score = (market_share * 0.6) + (negative_rate * 0.4)
            
            if urgency_score > 25:
                priority = "🚨 Crítica"
                priority_class = "critical"
            elif urgency_score > 15:
                priority = "⚠️ Alta"
                priority_class = "high"
            elif urgency_score > 8:
                priority = "📊 Média"
                priority_class = "medium"
            else:
                priority = "✅ Baixa"
                priority_class = "low"
            
            theme_analysis.append({
                'tema': tema,
                'total': total,
                'negative_count': negative_count,
                'negative_rate': negative_rate,
                'market_share': market_share,
                'priority': priority,
                'priority_class': priority_class,
                'urgency_score': urgency_score
            })
        
        # Ordenar por urgência
        theme_analysis.sort(key=lambda x: x['urgency_score'], reverse=True)
        
        # Análise por segmento
        segment_analysis = []
        for segmento in self.df['segmento'].unique():
            seg_data = self.df[self.df['segmento'] == segmento]
            total = len(seg_data)
            positive_rate = (len(seg_data[seg_data['sentimento_real'] == 'positivo']) / total) * 100
            
            segment_analysis.append({
                'segmento': segmento.title(),
                'total': total,
                'satisfaction_rate': positive_rate
            })
        
        segment_analysis.sort(key=lambda x: x['satisfaction_rate'], reverse=True)
        
        return {
            'total_feedbacks': total_feedbacks,
            'period_start': period_start,
            'period_end': period_end,
            'themes': theme_analysis,
            'segments': segment_analysis
        }
    
    def get_top_examples(self, tema, sentiment='negativo', limit=3):
        """
        Busca exemplos representativos de um tema
        """
        tema_data = self.df[(self.df['tema'] == tema) & (self.df['sentimento_real'] == sentiment)]
        if len(tema_data) == 0:
            return []
        
        examples = tema_data['texto'].head(limit).tolist()
        return [ex[:80] + "..." if len(ex) > 80 else ex for ex in examples]
    
    def generate_html_report(self, output_path='dashboard_report.html'):
        """
        Gera relatório HTML completo
        """
        print("📊 Gerando relatório HTML executivo...")
        
        metrics = self.calculate_key_metrics()
        
        # Template HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard - Análise de Feedbacks</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }}
        
        .metric-label {{
            font-size: 1.1em;
            color: #666;
        }}
        
        .section {{
            background: white;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .section-header {{
            background: #f8f9fa;
            padding: 20px 25px;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .section-header h2 {{
            color: #495057;
            font-size: 1.5em;
        }}
        
        .section-content {{
            padding: 25px;
        }}
        
        .theme-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 0;
            border-bottom: 1px solid #f0f0f0;
        }}
        
        .theme-item:last-child {{
            border-bottom: none;
        }}
        
        .theme-info {{
            flex: 1;
        }}
        
        .theme-name {{
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 5px;
        }}
        
        .theme-stats {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .priority-badge {{
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }}
        
        .priority-badge.critical {{
            background-color: #dc3545;
            color: white;
        }}
        
        .priority-badge.high {{
            background-color: #fd7e14;
            color: white;
        }}
        
        .priority-badge.medium {{
            background-color: #ffc107;
            color: #212529;
        }}
        
        .priority-badge.low {{
            background-color: #28a745;
            color: white;
        }}
        
        .examples {{
            margin-top: 15px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }}
        
        .examples h4 {{
            margin-bottom: 10px;
            color: #495057;
        }}
        
        .example-item {{
            margin: 8px 0;
            padding: 8px;
            background: white;
            border-radius: 3px;
            font-size: 0.9em;
            color: #666;
        }}
        
        .segment-bar {{
            display: flex;
            align-items: center;
            margin: 10px 0;
        }}
        
        .segment-name {{
            min-width: 120px;
            font-weight: bold;
        }}
        
        .progress-bar {{
            flex: 1;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            margin: 0 15px;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            transition: width 0.3s ease;
        }}
        
        .segment-value {{
            font-weight: bold;
            color: #495057;
        }}
        
        .recommendations {{
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin-top: 30px;
        }}
        
        .recommendations h2 {{
            margin-bottom: 20px;
        }}
        
        .recommendation-item {{
            margin: 15px 0;
            padding: 15px;
            background: rgba(255,255,255,0.1);
            border-radius: 5px;
            border-left: 4px solid white;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            border-top: 1px solid #dee2e6;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Dashboard - Análise de Feedbacks</h1>
            <p class="subtitle">Relatório Executivo | Período: {metrics['period_start']} - {metrics['period_end']}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{metrics['total_feedbacks']}</div>
                <div class="metric-label">Total de Feedbacks</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(metrics['themes'])}</div>
                <div class="metric-label">Temas Identificados</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len([t for t in metrics['themes'] if 'Crítica' in t['priority']])}</div>
                <div class="metric-label">Prioridade Crítica</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['segments'][0]['satisfaction_rate']:.1f}%</div>
                <div class="metric-label">Melhor Satisfação</div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-header">
                <h2>🎯 Temas por Prioridade</h2>
            </div>
            <div class="section-content">
"""
        
        # Adicionar análise por tema
        for theme in metrics['themes'][:8]:  # Top 8 temas
            examples = self.get_top_examples(theme['tema'], 'negativo', 2)
            examples_html = ""
            if examples:
                examples_html = f"""
                <div class="examples">
                    <h4>Exemplos de feedback:</h4>
                    {"".join([f'<div class="example-item">• {ex}</div>' for ex in examples])}
                </div>
                """
            
            html_content += f"""
                <div class="theme-item">
                    <div class="theme-info">
                        <div class="theme-name">{theme['tema']}</div>
                        <div class="theme-stats">
                            {theme['total']} feedbacks • {theme['market_share']:.1f}% do total • {theme['negative_rate']:.1f}% negativos
                        </div>
                        {examples_html}
                    </div>
                    <div class="priority-badge {theme['priority_class']}">{theme['priority']}</div>
                </div>
            """
        
        # Análise por segmento
        html_content += f"""
            </div>
        </div>
        
        <div class="section">
            <div class="section-header">
                <h2>📈 Performance por Segmento</h2>
            </div>
            <div class="section-content">
"""
        
        for segment in metrics['segments']:
            html_content += f"""
                <div class="segment-bar">
                    <div class="segment-name">{segment['segmento']}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {segment['satisfaction_rate']}%"></div>
                    </div>
                    <div class="segment-value">{segment['satisfaction_rate']:.1f}%</div>
                </div>
            """
        
        # Recomendações
        critical_themes = [t for t in metrics['themes'] if 'Crítica' in t['priority']]
        high_themes = [t for t in metrics['themes'] if 'Alta' in t['priority']]
        
        html_content += f"""
            </div>
        </div>
        
        <div class="recommendations">
            <h2>💡 Recomendações Estratégicas</h2>
"""
        
        if critical_themes:
            html_content += f"""
            <div class="recommendation-item">
                <strong>🚨 AÇÃO IMEDIATA:</strong> Focar em {critical_themes[0]['tema']} 
                ({critical_themes[0]['total']} feedbacks, {critical_themes[0]['negative_rate']:.1f}% negativos)
            </div>
            """
        
        if high_themes:
            html_content += f"""
            <div class="recommendation-item">
                <strong>📈 PRÓXIMAS AÇÕES:</strong> Melhorar {', '.join([t['tema'] for t in high_themes[:2]])}
            </div>
            """
        
        best_segment = metrics['segments'][0]
        worst_segment = metrics['segments'][-1]
        
        html_content += f"""
            <div class="recommendation-item">
                <strong>🎯 BENCHMARKING:</strong> Usar estratégia de {best_segment['segmento']} 
                ({best_segment['satisfaction_rate']:.1f}% satisfação) para melhorar {worst_segment['segmento']} 
                ({worst_segment['satisfaction_rate']:.1f}% satisfação)
            </div>
            
            <div class="recommendation-item">
                <strong>💰 IMPACTO ESTIMADO:</strong> Resolução dos temas críticos pode reduzir 
                {sum([t['negative_count'] for t in critical_themes])} feedbacks negativos/mês
            </div>
        </div>
        
        <div class="footer">
            <p>Relatório gerado automaticamente em {datetime.now().strftime('%d/%m/%Y às %H:%M')}</p>
            <p>Sistema de Análise de Feedbacks - PoC | Desenvolvido com ajuda de IA</p>
        </div>
    </div>
</body>
</html>
        """
        
        # Salvar arquivo
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Relatório HTML gerado: {output_path}")
        print(f"🌐 Abra no navegador para visualizar")
        
        return output_path

def main():
    """
    Gera relatório HTML executivo
    """
    print("📊 GERANDO DASHBOARD DE RELATÓRIO EXECUTIVO")
    print("=" * 50)
    
    # Verificar dados
    if not os.path.exists('data/feedbacks_clustered.csv'):
        print("❌ Dados não encontrados! Execute primeiro as análises anteriores.")
        return
    
    # Gerar relatório
    generator = HTMLReportGenerator()
    report_path = generator.generate_html_report()
    
    print(f"\n🎉 Dashboard pronto!")
    print(f"📂 Arquivo: {report_path}")
    print(f"💻 Para visualizar: abra o arquivo no seu navegador")

if __name__ == "__main__":
    main()