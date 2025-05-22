import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt

def benchmark_yolo_models(image_path):
    """
    Compara o desempenho de diferentes modelos YOLO usando uma imagem real
    e gera gráficos com os resultados.
    """
    print(f"Benchmarking YOLO models on image: {image_path}")

    # Verificar se a imagem existe
    if not os.path.exists(image_path):
        print(f"Erro: Imagem não encontrada em {image_path}")
        return

    # Carregar a imagem
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro: Não foi possível ler a imagem em {image_path}")
        return

    # Obter dimensões da imagem
    height, width = img.shape[:2]

    # Carregar as labels
    labels_path = '../yolov3-coco/coco-labels'
    if not os.path.exists(labels_path):
        print(f"Erro: Arquivo de labels não encontrado em {labels_path}")
        return

    labels = open(labels_path).read().strip().split('\n')
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Definir modelos a serem testados
    models = [
        {
            'name': 'YOLOv3',
            'cfg': '../yolov3-coco/yolov3.cfg',
            'weights': '../yolov3-coco/yolov3.weights',
            'color': '#3498db',  # Azul
            'input_size': (416, 416)
        },
        {
            'name': 'YOLOv4',
            'cfg': '../yolov4-coco/yolov4.cfg',
            'weights': '../yolov4-coco/yolov4.weights',
            'color': '#e74c3c',  # Vermelho
            'input_size': (416, 416)
        },
        {
            'name': 'YOLOv3-tiny',
            'cfg': '../yolov3-coco/yolov3-tiny.cfg',
            'weights': '../yolov3-coco/yolov3-tiny.weights',
            'color': '#2ecc71',  # Verde
            'input_size': (416, 416)
        }
    ]

    # Resultados
    results = {}
    result_images = {}

    # Executar testes para cada modelo
    for model in models:
        print(f"\nTestando modelo {model['name']}...")

        # Verificar se os arquivos do modelo existem
        if not os.path.exists(model['cfg']) or not os.path.exists(model['weights']):
            print(f"Erro: Arquivos do modelo {model['name']} não encontrados")
            continue

        # Carregar o modelo
        net = cv2.dnn.readNetFromDarknet(model['cfg'], model['weights'])

        # Tentar usar GPU aceleração
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            # Check if CUDA was actually set by trying to get the backend info
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                backend = "CUDA"
            else:
                backend = "CPU"
                print(f"CUDA não disponível. Usando CPU para {model['name']}")
        except Exception as e:
            backend = "CPU"
            print(f"Usando CPU para {model['name']}: {str(e)}")

        # Obter camadas de saída
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

        # Executar detecção e medir tempo
        img_copy = img.copy()
        blob = cv2.dnn.blobFromImage(img_copy, 1/255.0, model['input_size'],
                                   swapRB=True, crop=False)
        net.setInput(blob)

        # Medir tempo de processamento
        start_time = time.time()
        outs = net.forward(output_layers)
        inference_time = time.time() - start_time

        # Check for warning signs that CUDA wasn't used
        if backend == "CUDA" and inference_time > 0.1:  # If inference is slow despite CUDA setting
            print(f"Aviso: Tempo de inferência sugere que CUDA não foi usado efetivamente")
            backend = "CPU"  # Override the backend label

        # Processar detecções
        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:  # Limiar de confiança
                    # Coordenadas da caixa (formato YOLO: centro_x, centro_y, largura, altura)
                    box = detection[0:4] * np.array([width, height, width, height])
                    (center_x, center_y, box_width, box_height) = box.astype("int")

                    # Converter para coordenadas de canto (x, y, largura, altura)
                    x = int(center_x - (box_width / 2))
                    y = int(center_y - (box_height / 2))

                    boxes.append([x, y, int(box_width), int(box_height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Aplicar Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Desenhar caixas na imagem
        if len(indices) > 0:
            for i in indices.flatten():
                # Get the bounding box coordinates
                (x, y, w, h) = boxes[i]

                # Make sure class_id is within range of labels
                class_id = class_ids[i]
                if class_id < len(labels):
                    label_text = f"{labels[class_id]}: {confidences[i]:.2f}"
                else:
                    label_text = f"Class {class_id}: {confidences[i]:.2f}"

                # Get color (ensure it's within range)
                color_index = class_id % len(colors)
                color = [int(c) for c in colors[color_index]]

                # Draw rectangle and label
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img_copy, label_text, (x, y - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Adicionar informações de desempenho à imagem
        fps = 1.0 / inference_time
        cv2.putText(img_copy, f"{model['name']} - {backend}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img_copy, f"FPS: {fps:.2f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img_copy, f"Deteccoes: {len(indices) if len(indices) > 0 else 0}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Guardar resultados
        results[model['name']] = {
            'inference_time': inference_time,
            'fps': fps,
            'detections': len(indices) if len(indices) > 0 else 0,
            'backend': backend
        }
        result_images[model['name']] = img_copy

        # Exibir resultados no console
        print(f"Tempo de inferência: {inference_time:.4f} segundos")
        print(f"FPS: {fps:.2f}")
        print(f"Objetos detectados: {len(indices) if len(indices) > 0 else 0}")

    # Salvar as imagens resultado
    output_dir = "benchmark_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Criar comparação visual
    img_height, img_width = img.shape[:2]
    max_height = img_height
    total_width = img_width * len(result_images)

    # Criar imagem combinada
    comparison_img = np.zeros((max_height, total_width, 3), dtype=np.uint8)

    # Combinar imagens lado a lado
    x_offset = 0
    for model_name, img in result_images.items():
        comparison_img[0:img.shape[0], x_offset:x_offset+img.shape[1], :] = img
        x_offset += img.shape[1]

    # Salvar a comparação visual
    cv2.imwrite(os.path.join(output_dir, "yolo_comparison.png"), comparison_img)

    # Gerar gráficos
    generate_performance_graphs(results, output_dir)

    return results

def generate_performance_graphs(results, output_dir):
    """
    Gera gráficos comparativos de desempenho dos modelos YOLO.
    """
    # Extrair dados
    models = list(results.keys())
    inference_times = [results[model]['inference_time'] for model in models]
    fps_values = [results[model]['fps'] for model in models]
    detection_counts = [results[model]['detections'] for model in models]
    backends = [results[model]['backend'] for model in models]

    # Cores para cada modelo
    model_colors = {
        'YOLOv3': '#3498db',     # Azul
        'YOLOv4': '#e74c3c',     # Vermelho
        'YOLOv3-tiny': '#2ecc71' # Verde
    }
    colors = [model_colors.get(model, '#95a5a6') for model in models]

    # Precisão estimada (baseado em benchmarks gerais)
    # mAP (@0.5 IoU) no dataset COCO
    accuracy = {
        'YOLOv3': 55.3,
        'YOLOv4': 62.8,
        'YOLOv3-tiny': 33.1
    }

    # Figura 1: Tempo de Processamento vs FPS
    plt.figure(figsize=(14, 10))

    # Gráfico de barras para tempo de inferência
    plt.subplot(2, 1, 1)
    bars = plt.bar(models, inference_times, color=colors, alpha=0.8)
    plt.title('Tempo de Processamento por Frame', fontsize=14, fontweight='bold')
    plt.ylabel('Tempo (segundos)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adicionar valores nas barras
    for bar, time_val in zip(bars, inference_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{time_val:.4f}s', ha='center', fontweight='bold')

    # Adicionar informação sobre backend (CPU/GPU)
    for i, backend in enumerate(backends):
        plt.text(i, 0.01, backend, ha='center', color='white',
                fontweight='bold', bbox=dict(facecolor='black', alpha=0.7))

    # Gráfico de barras para FPS
    plt.subplot(2, 1, 2)
    bars = plt.bar(models, fps_values, color=colors, alpha=0.8)
    plt.title('Taxa de Frames por Segundo (FPS)', fontsize=14, fontweight='bold')
    plt.ylabel('FPS', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adicionar valores nas barras
    for bar, fps in zip(bars, fps_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{fps:.2f} FPS', ha='center', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('Comparação de Desempenho entre Modelos YOLO', fontsize=16, fontweight='bold')

    # Salvar figura 1
    plt.savefig(os.path.join(output_dir, 'yolo_performance_time_fps.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'yolo_performance_time_fps.png'), format='png', dpi=300, bbox_inches='tight')

    # Figura 2: Comparação entre Precisão (mAP), Velocidade (FPS) e Número de Detecções
    plt.figure(figsize=(14, 10))

    # Subplot para FPS vs mAP
    plt.subplot(2, 1, 1)
    for model in models:
        plt.scatter(accuracy.get(model, 0), results[model]['fps'],
                  s=100, color=model_colors.get(model, '#95a5a6'), label=model, alpha=0.8)
        plt.annotate(model,
                   (accuracy.get(model, 0), results[model]['fps']),
                   xytext=(5, 5), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))

    plt.title('Relação entre Precisão e Velocidade', fontsize=14, fontweight='bold')
    plt.xlabel('Precisão (mAP@0.5)', fontsize=12)
    plt.ylabel('Velocidade (FPS)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Subplot para Número de Detecções
    plt.subplot(2, 1, 2)
    bars = plt.bar(models, detection_counts, color=colors, alpha=0.8)
    plt.title('Número de Objetos Detectados', fontsize=14, fontweight='bold')
    plt.ylabel('Quantidade', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adicionar valores nas barras
    for bar, count in zip(bars, detection_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{count}', ha='center', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('Análise de Eficácia dos Modelos YOLO', fontsize=16, fontweight='bold')

    # Salvar figura 2
    plt.savefig(os.path.join(output_dir, 'yolo_performance_accuracy.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'yolo_performance_accuracy.png'), format='png', dpi=300, bbox_inches='tight')

    # Informar sobre os arquivos gerados
    print(f"\nGráficos de desempenho salvos em {output_dir}/")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark de modelos YOLO com imagem real")
    parser.add_argument("-i", "--image", default="../_out/episode_3360/frame_camera/000960.png",
                      help="Caminho para imagem de teste")
    args = parser.parse_args()

    benchmark_yolo_models(args.image)
