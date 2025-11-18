import cv2
import time
from collections import deque
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QGroupBox, QLineEdit, QHBoxLayout
from PyQt5.QtCore import Qt
import numpy as np
import pyqtgraph as pg


class StrideAnalysis:
    def __init__(self, parent_layout, mp_pose, max_points=100, initial_frames=30):
        self.parent_layout = parent_layout
        self.mp_pose = mp_pose
        self.max_points = max_points
        self.initial_frames = initial_frames

        # Labels para exibir as informações
        self.cadence_label = None
        self.speed_label = None
        self.stride_length_label = None

        # Variáveis para calcular a cadência
        self.step_times = deque(maxlen=10)  # Armazena os tempos dos últimos 10 passos
        self.last_step_time = None
        self.cadence = 0  # passos por minuto

        # Variáveis para análise de pisada
        self.strike_types = []  # Armazena o tipo de pisada (1, 2, 3)
        self.strike_times = []  # Armazena o timestamp de cada pisada
        self.strike_graph = None

        # Variáveis para cálculo do comprimento da passada
        self.speed_input = None  # Campo de entrada para a velocidade da esteira
        self.speed = 0  # km/h
        self.stride_length = 0  # cm

    def setup_ui(self):
        """Configura os componentes da UI para a análise de passada."""
        # Estilo para os labels
        label_style = """
        QLabel {
            background-color: #f0f0f0;
            border: 1px solid #cccccc;
            border-radius: 5px;
            padding: 10px;
            font-size: 20px;
            font-weight: bold;
            color: #333333;
        }
        """

        # Inicialização das labels
        self.cadence_label = QLabel("Cadência: 0 passos/min")
        self.cadence_label.setAlignment(Qt.AlignCenter)
        self.cadence_label.setStyleSheet(label_style)

        self.speed_label = QLabel("Velocidade Estimada: 0 km/h")
        self.speed_label.setAlignment(Qt.AlignCenter)
        self.speed_label.setStyleSheet(label_style)

        self.stride_length_label = QLabel("Comprimento da Passada: 0 cm")
        self.stride_length_label.setAlignment(Qt.AlignCenter)
        self.stride_length_label.setStyleSheet(label_style)

        # Campo de entrada para a velocidade da esteira
        self.speed_input = QLineEdit()
        self.speed_input.setPlaceholderText("Velocidade da Esteira (km/h)")
        self.speed_input.setFixedWidth(200)

        # Layout para a entrada
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Velocidade da Esteira (km/h):"))
        input_layout.addWidget(self.speed_input)

        # Grupo para o gráfico do tipo de pisada
        self.strike_group = QGroupBox("Tipo de Pisada ao Longo do Tempo")
        strike_layout = QVBoxLayout()
        self.strike_graph = pg.PlotWidget()
        self.strike_graph.setLabel('left', 'Tipo de Pisada')
        self.strike_graph.setLabel('bottom', 'Tempo (s)')
        self.strike_graph.showGrid(x=True, y=True)
        strike_layout.addWidget(self.strike_graph)
        self.strike_group.setLayout(strike_layout)

        # Grupo para as informações de cadência, velocidade e comprimento da passada
        self.info_group = QGroupBox("Informações da Passada")
        info_layout = QVBoxLayout()
        info_layout.addLayout(input_layout)
        info_layout.addWidget(self.cadence_label)
        info_layout.addWidget(self.stride_length_label)
        info_layout.addWidget(self.speed_label)
        self.info_group.setLayout(info_layout)

        # Estilo para os QGroupBox
        groupbox_style = """
        QGroupBox {
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
            border: 2px solid #2980b9;
            border-radius: 5px;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 3px;
        }
        """

        self.strike_group.setStyleSheet(groupbox_style)
        self.info_group.setStyleSheet(groupbox_style)

        # Adicionar grupos ao layout principal
        self.parent_layout.addWidget(self.strike_group)
        self.parent_layout.addWidget(self.info_group)

    def process_frame(self, annotated_frame, results):
        """Processa cada frame para a análise de passada."""
        if not results.pose_landmarks:
            return

        landmarks = results.pose_landmarks.landmark
        image_height, image_width, _ = annotated_frame.shape

        # Obter coordenadas dos pontos relevantes
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        left_heel = landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL.value]
        right_heel = landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL.value]
        left_foot_index = landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
        right_foot_index = landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]

        # Converter para coordenadas em pixels
        left_hip_x = int(left_hip.x * image_width)
        left_hip_y = int(left_hip.y * image_height)
        right_hip_x = int(right_hip.x * image_width)
        right_hip_y = int(right_hip.y * image_height)

        left_knee_x = int(left_knee.x * image_width)
        left_knee_y = int(left_knee.y * image_height)
        right_knee_x = int(right_knee.x * image_width)
        right_knee_y = int(right_knee.y * image_height)

        left_ankle_x = int(left_ankle.x * image_width)
        left_ankle_y = int(left_ankle.y * image_height)
        right_ankle_x = int(right_ankle.x * image_width)
        right_ankle_y = int(right_ankle.y * image_height)

        left_heel_x = int(left_heel.x * image_width)
        left_heel_y = int(left_heel.y * image_height)
        right_heel_x = int(right_heel.x * image_width)
        right_heel_y = int(right_heel.y * image_height)

        left_foot_x = int(left_foot_index.x * image_width)
        left_foot_y = int(left_foot_index.y * image_height)
        right_foot_x = int(right_foot_index.x * image_width)
        right_foot_y = int(right_foot_index.y * image_height)

        # Marcar os landmarks
        points = [
            (left_hip_x, left_hip_y),
            (left_knee_x, left_knee_y),
            (left_ankle_x, left_ankle_y),
            (left_heel_x, left_heel_y),
            (left_foot_x, left_foot_y),
            (right_hip_x, right_hip_y),
            (right_knee_x, right_knee_y),
            (right_ankle_x, right_ankle_y),
            (right_heel_x, right_heel_y),
            (right_foot_x, right_foot_y)
        ]

        for x, y in points:
            cv2.circle(annotated_frame, (x, y), 5, (0, 0, 255), -1)

        # Desenhar linhas unindo os landmarks para cada perna
        # Perna esquerda
        cv2.line(annotated_frame, (left_hip_x, left_hip_y), (left_knee_x, left_knee_y), (0, 255, 0), 2)
        cv2.line(annotated_frame, (left_knee_x, left_knee_y), (left_ankle_x, left_ankle_y), (0, 255, 0), 2)
        cv2.line(annotated_frame, (left_ankle_x, left_ankle_y), (left_heel_x, left_heel_y), (0, 255, 0), 2)
        cv2.line(annotated_frame, (left_heel_x, left_heel_y), (left_foot_x, left_foot_y), (0, 255, 0), 2)

        # Perna direita
        cv2.line(annotated_frame, (right_hip_x, right_hip_y), (right_knee_x, right_knee_y), (0, 255, 0), 2)
        cv2.line(annotated_frame, (right_knee_x, right_knee_y), (right_ankle_x, right_ankle_y), (0, 255, 0), 2)
        cv2.line(annotated_frame, (right_ankle_x, right_ankle_y), (right_heel_x, right_heel_y), (0, 255, 0), 2)
        cv2.line(annotated_frame, (right_heel_x, right_heel_y), (right_foot_x, right_foot_y), (0, 255, 0), 2)

        # Determinar a linha do solo dinamicamente
        ground_points_y = [
            left_heel_y,
            right_heel_y,
            left_foot_y,
            right_foot_y
        ]
        self.ground_line_y = max(ground_points_y)


        # Desenhar a linha do solo
        cv2.line(annotated_frame, (0, self.ground_line_y), (image_width, self.ground_line_y), (255, 0, 0), 2)

        # Identificar o pé da frente
        if left_foot_x < right_foot_x:
            front_heel_y = left_heel_y
            front_foot_y = left_foot_y
        else:
            front_heel_y = right_heel_y
            front_foot_y = right_foot_y

        # Verificar contato com o solo
        current_time = time.time()
        foot_contact = False

        # Se qualquer parte do pé estiver na mesma altura ou abaixo da linha do solo
        if front_foot_y >= self.ground_line_y or front_heel_y >= self.ground_line_y:
            foot_contact = True

            # Registrar tempo do passo
            if self.last_step_time is not None:
                time_between_steps = current_time - self.last_step_time
                if time_between_steps > 0:
                    self.step_times.append(time_between_steps)
                    self.update_cadence()
            self.last_step_time = current_time

            # Analisar o tipo de pisada
            self.analyze_foot_strike(front_heel_y, front_foot_y, current_time)

        # Atualizar labels e gráficos
        self.cadence_label.setText(f"Cadência: {self.cadence:.1f} passos/min")
        self.speed_label.setText(f"Velocidade Estimada: {self.speed:.2f} km/h")
        self.stride_length_label.setText(f"Comprimento da Passada: {self.stride_length:.1f} cm")
        self.update_strike_graph()

    def analyze_foot_strike(self, heel_y, foot_y, current_time):
        """Determina o tipo de pisada."""
        # Verificar qual parte do pé toca o solo primeiro
        if heel_y > foot_y:
            strike_type = 1  # Calcanhar
        elif foot_y > heel_y:
            strike_type = 3  # Antepé
        else:
            strike_type = 2  # Meio do Pé

        # Armazenar o tipo de pisada e o timestamp
        self.strike_types.append(strike_type)
        self.strike_times.append(current_time - self.initial_frames)

    def update_strike_graph(self):
        """Atualiza o gráfico do tipo de pisada."""
        if len(self.strike_times) > 0:
            times = np.array(self.strike_times)
            times = times - times[0]  # Normalizar o tempo
            types = np.array(self.strike_types)

            self.strike_graph.clear()

            # Mapear tipos de pisada para cores
            color_map = {1: 'b', 2: 'g', 3: 'r'}
            colors = [color_map[t] for t in types]

            # Plotar os pontos
            for t, y, c in zip(times, types, colors):
                self.strike_graph.plot([t], [y], pen=None, symbol='o', symbolBrush=c, symbolSize=10)

            # Configurar o eixo y
            self.strike_graph.getPlotItem().getAxis('left').setTicks([[(1, 'Calcanhar'), (2, 'Meio do Pé'), (3, 'Antepé')]])

            self.strike_graph.repaint()

    def update_cadence(self):
        """Atualiza a cadência e calcula o comprimento da passada."""
        if len(self.step_times) > 0:
            average_step_time = sum(self.step_times) / len(self.step_times)
            self.cadence = (60 / average_step_time)

            # Calcular o comprimento da passada
            try:
                speed = float(self.speed_input.text())
                self.speed = speed  # Atualizar a velocidade

                # Converter velocidade para m/s
                speed_m_per_sec = speed / 3.6
                # Converter cadência para passos por segundo
                cadence_per_sec = self.cadence / 60
                # Calcular comprimento da passada em metros
                stride_length_m = speed_m_per_sec / cadence_per_sec
                # Converter para centímetros
                self.stride_length = stride_length_m * 100

            except ValueError:
                # Se a velocidade não for um número válido
                self.speed = 0
                self.stride_length = 0

    def reset(self):
        """Reseta as variáveis específicas da análise de passada."""
        self.step_times.clear()
        self.last_step_time = None
        self.cadence = 0
        self.speed = 0
        self.stride_length = 0
        self.cadence_label.setText("Cadência: 0 passos/min")
        self.speed_label.setText("Velocidade Estimada: 0 km/h")
        self.stride_length_label.setText("Comprimento da Passada: 0 cm")
        self.speed_input.clear()
        self.strike_types.clear()
        self.strike_times.clear()
        if self.strike_graph:
            self.strike_graph.clear()
