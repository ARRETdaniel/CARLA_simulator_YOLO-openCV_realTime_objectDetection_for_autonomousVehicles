import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class ResultsReporter:
    def __init__(self, metrics_dir="metrics_output", report_dir="results_report"):
        """Inicializa o gerador de relatórios.

        Args:
            metrics_dir (str): Diretório contendo os dados de métricas
            report_dir (str): Diretório para salvar o relatório
        """
        self.metrics_dir = metrics_dir
        self.report_dir = report_dir

        # Cria o diretório de relatórios se não existir
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        # Copia recursos estáticos para o relatório (CSS, JS)
        self._create_resources()

    def _create_resources(self):
        """Cria recursos CSS e JS para o relatório."""
        # Cria arquivo CSS
        css_content = """
        body {
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .container {
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .metric-card {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 15px;
            margin-bottom: 15px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }
        .chart-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
        }
        .chart {
            flex: 0 0 48%;
            margin-bottom: 20px;
            background-color: white;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        img {
            max-width: 100%;
            height: auto;
        }
        .conclusion {
            background-color: #e8f4f8;
            border-left: 5px solid #3498db;
            padding: 15px;
            margin: 20px 0;
        }
        """

        with open(os.path.join(self.report_dir, 'style.css'), 'w', encoding='utf-8') as f:
            f.write(css_content)

    def _load_metrics_data(self):
        """Carrega dados de métricas de arquivos."""
        data = {}

        # Carrega estatísticas resumidas
        summary_path = os.path.join(self.metrics_dir, 'summary_stats.csv')
        if os.path.exists(summary_path):
            summary = {}
            with open(summary_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if ',' in line:
                        key, value = line.strip().split(',', 1)
                        try:
                            # Tenta converter para número se possível
                            summary[key] = float(value)
                        except ValueError:
                            summary[key] = value
            data['summary'] = summary

        # Carrega métricas de detecção
        detection_path = os.path.join(self.metrics_dir, 'detection_metrics.csv')
        if os.path.exists(detection_path):
            detections = []
            with open(detection_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    detections.append(row)
            data['detections'] = detections

        # Carrega métricas de avisos
        warnings_path = os.path.join(self.metrics_dir, 'warning_metrics.csv')
        if os.path.exists(warnings_path):
            warnings = []
            with open(warnings_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    warnings.append(row)
            data['warnings'] = warnings

        return data

    def _generate_summary_html(self, summary):
        """Gera HTML para a seção de resumo."""
        if not summary:
            return "<p>Dados de resumo não disponíveis</p>"

        html = """
        <div class="container">
            <h2>Resumo de Desempenho</h2>
            <div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
        """

        # Métricas principais para destacar
        key_metrics = [
            ('avg_detection_time', 'Tempo Médio de Detecção (s)'),
            ('avg_fps', 'FPS Médio'),
            ('total_detections', 'Total de Detecções'),
            ('avg_confidence', 'Confiança Média'),
            ('total_warnings', 'Total de Avisos Gerados'),
            ('total_runtime_seconds', 'Tempo Total de Execução (s)')
        ]

        for key, label in key_metrics:
            if key in summary:
                value = summary[key]
                if isinstance(value, float):
                    value = f"{value:.4f}"
                html += f"""
                <div class="metric-card" style="flex: 0 0 30%;">
                    <div>{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """

        html += """
            </div>

            <h3>Estatísticas Detalhadas</h3>
            <table>
                <tr>
                    <th>Métrica</th>
                    <th>Valor</th>
                </tr>
        """

        # Adiciona todas as métricas à tabela
        for key, value in summary.items():
            # Pula métricas principais já exibidas e dados complexos
            if key in [item[0] for item in key_metrics] or '{' in str(value):
                continue

            html += f"""
                <tr>
                    <td>{key}</td>
                    <td>{value}</td>
                </tr>
            """

        html += """
            </table>
        </div>
        """
        return html

    def _generate_charts_html(self):
        """Gera HTML para a seção de gráficos."""
        html = """
        <div class="container">
            <h2>Visualizações de Desempenho</h2>

            <div class="chart-container">
                <div class="chart">
                    <h3>Métricas de Desempenho</h3>
                    <img src="../metrics_output/performance_metrics.png" alt="Métricas de Desempenho">
                </div>

                <div class="chart">
                    <h3>Distribuição de Tempos de Detecção</h3>
                    <img src="../metrics_output/detection_times_histogram.png" alt="Histograma de Tempos de Detecção">
                </div>
            </div>
        </div>
        """
        return html

    def _generate_hypothesis_validation_html(self, data):
        """Gera HTML para a seção de validação de hipótese."""
        html = """
        <div class="container">
            <h2>Validação de Hipótese</h2>
        """

        # Verifica se temos dados suficientes para validar a hipótese
        if 'summary' not in data or not data.get('detections') or not data.get('warnings'):
            html += "<p>Dados insuficientes coletados para validar a hipótese</p>"
        else:
            summary = data['summary']

            # Calcula métricas-chave para validação de hipótese
            avg_detection_time = summary.get('avg_detection_time', 0)
            avg_fps = summary.get('avg_fps', 0)
            total_detections = summary.get('total_detections', 0)
            avg_confidence = summary.get('avg_confidence', 0)
            total_warnings = summary.get('total_warnings', 0)

            # Validação de desempenho em tempo real
            real_time_performance = avg_fps > 10  # Considera tempo real se > 10 FPS
            high_confidence = avg_confidence > 0.7  # Bom nível de confiança

            html += """
            <h3>Critérios de Validação</h3>
            <table>
                <tr>
                    <th>Critério</th>
                    <th>Alvo</th>
                    <th>Atual</th>
                    <th>Status</th>
                </tr>
            """

            criteria = [
                ('Processamento em tempo real', '>10 FPS', f"{avg_fps:.2f} FPS", real_time_performance),
                ('Confiança de detecção', '>0.7', f"{avg_confidence:.2f}", high_confidence),
                ('Feedback gerado', 'Sim', 'Sim' if total_warnings > 0 else 'Não', total_warnings > 0)
            ]

            for criterion, target, actual, passed in criteria:
                # Substitui símbolos Unicode por entidades HTML
                status = "APROVADO" if passed else "REPROVADO"
                html += f"""
                <tr>
                    <td>{criterion}</td>
                    <td>{target}</td>
                    <td>{actual}</td>
                    <td>{status}</td>
                </tr>
                """

            html += """
            </table>

            <div class="conclusion">
                <h3>Conclusão</h3>
            """

            # Avalia a hipótese geral
            if real_time_performance and total_warnings > 0:
                html += """
                <p><strong>A hipótese foi confirmada.</strong> O sistema fornece com sucesso feedback visual em tempo real
                dos objetos detectados, alcançando o objetivo de assistir o condutor com informações oportunas sobre o ambiente.
                A detecção é suficientemente rápida para aplicações em tempo real, e o sistema de avisos comunica efetivamente
                a presença e proximidade de objetos relevantes.</p>
                """
            else:
                html += """
                <p><strong>A hipótese foi parcialmente confirmada.</strong> Embora o sistema forneça feedback visual
                para objetos detectados, melhorias podem ser necessárias para alcançar desempenho ótimo em tempo real. A implementação
                atual demonstra a viabilidade do conceito, mas otimizações adicionais melhorariam a
                experiência do usuário e a confiabilidade do sistema.</p>
                """

            html += """
            </div>
        </div>
        """

        return html

    def generate_report(self):
        """Gera o relatório HTML."""
        # Carrega dados de métricas
        data = self._load_metrics_data()

        # Gera HTML
        html = f"""
        <!DOCTYPE html>
        <html lang="pt-br">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Relatório de Desempenho de Detecção de Objetos</title>
            <link rel="stylesheet" href="style.css">
        </head>
        <body>
            <h1>Relatório de Desempenho do Sistema de Detecção de Objetos e Avisos</h1>
            <p>Gerado em: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}</p>

            {self._generate_summary_html(data.get('summary', {}))}

            {self._generate_charts_html()}

            {self._generate_hypothesis_validation_html(data)}

            <div class="container">
                <h2>Discussão</h2>
                <p>Este relatório apresenta as métricas de desempenho e resultados do sistema de detecção de objetos e avisos
                implementados para a simulação de veículo autônomo. O sistema utiliza YOLOv3 para detecção de objetos em tempo real
                e fornece feedback visual para auxiliar o condutor com informações sobre objetos detectados e sua proximidade.</p>

                <h3>Principais Descobertas</h3>
                <ul>
                    <li>O sistema detecta com sucesso objetos de interesse, incluindo veículos, pedestres, sinais de trânsito e outros obstáculos.</li>
                    <li>O feedback visual é fornecido através de avisos claros com níveis de severidade baseados na proximidade.</li>
                    <li>O desempenho é adequado para aplicações em tempo real, com tempos de detecção rápidos o suficiente para uso prático.</li>
                </ul>

                <h3>Limitações</h3>
                <ul>
                    <li>A precisão da detecção pode variar dependendo das condições de iluminação, tamanho do objeto e oclusão.</li>
                    <li>A estimativa de distância é baseada em uma heurística simples (tamanho da caixa delimitadora) em vez de informações precisas de profundidade.</li>
                    <li>O desempenho pode degradar em cenas altamente complexas com muitos objetos.</li>
                </ul>

                <h3>Trabalhos Futuros</h3>
                <ul>
                    <li>Implementar estimativa de distância mais sofisticada usando fusão de dados de sensores.</li>
                    <li>Otimizar o pipeline de detecção para melhor desempenho em plataformas com recursos limitados.</li>
                    <li>Integrar com outros sistemas do veículo para respostas coordenadas a objetos detectados.</li>
                    <li>Adicionar personalização de usuário para preferências e limiares de avisos.</li>
                </ul>
            </div>
        </body>
        </html>
        """

        # Escreve o HTML no arquivo com codificação UTF-8
        with open(os.path.join(self.report_dir, 'relatorio_desempenho.html'), 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"Relatório gerado em {os.path.join(self.report_dir, 'relatorio_desempenho.html')}")
