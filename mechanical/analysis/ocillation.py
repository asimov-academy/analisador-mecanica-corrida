from PyQt5.QtWidgets import QVBoxLayout, QGroupBox, QSizePolicy
import pyqtgraph as pg
import cv2


class OscillationAnalysis:
    def __init__(self, parent_layout, mp_pose, max_points=100, initial_frames=30):
        self.parent_layout = parent_layout
        self.mp_pose = mp_pose
        self.max_points = max_points
        self.initial_frames = initial_frames

        # Inicializa variáveis específicas da análise
        self.zero_point_head_x = None
        self.zero_point_head_y = None
        self.zero_point_left_shoulder_x = None
        self.zero_point_right_shoulder_x = None
        self.zero_point_left_hip_x = None
        self.zero_point_right_hip_x = None
        self.displacements_head_x = []
        self.displacements_head_y = []
        self.displacements_left_shoulder = []
        self.displacements_right_shoulder = []
        self.displacements_left_hip = []
        self.displacements_right_hip = []
        self.frames_captured = 0

        # Inicializar PlotDataItems
        self.head_x_curve = None
        self.head_y_curve = None
        self.left_shoulder_curve = None
        self.right_shoulder_curve = None
        self.left_hip_curve = None
        self.right_hip_curve = None

    def setup_ui(self):
        """Configura os componentes da UI para a análise de oscilação."""
        # Gráfico da Cabeça - X
        self.plot_widget_head_x = pg.PlotWidget(title="Deslocamento Lateral da Cabeça (X)")
        self.plot_widget_head_x.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.head_x_curve = self.plot_widget_head_x.plot(pen='r', name='Cabeça X')

        # Gráfico da Cabeça - Y
        self.plot_widget_head_y = pg.PlotWidget(title="Deslocamento Vertical da Cabeça (Y)")
        self.plot_widget_head_y.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.head_y_curve = self.plot_widget_head_y.plot(pen='b', name='Cabeça Y')

        # Grupo de Movimentos da Cabeça
        self.head_movement_group = QGroupBox("Análise dos Movimentos da Cabeça")
        head_movement_layout = QVBoxLayout()
        head_movement_layout.addWidget(self.plot_widget_head_x)
        head_movement_layout.addWidget(self.plot_widget_head_y)
        self.head_movement_group.setLayout(head_movement_layout)

        # Gráfico dos Ombros
        self.plot_widget_shoulders = pg.PlotWidget(title="Deslocamento Lateral dos Ombros (X)")
        self.plot_widget_shoulders.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.left_shoulder_curve = self.plot_widget_shoulders.plot(pen='g', name='Ombro Esquerdo')
        self.right_shoulder_curve = self.plot_widget_shoulders.plot(pen='m', name='Ombro Direito')
        self.plot_widget_shoulders.addLegend()

        # Grupo de Movimentos dos Ombros
        self.shoulders_movement_group = QGroupBox("Análise dos Ombros")
        shoulders_movement_layout = QVBoxLayout()
        shoulders_movement_layout.addWidget(self.plot_widget_shoulders)
        self.shoulders_movement_group.setLayout(shoulders_movement_layout)

        # Gráfico dos Quadris
        self.plot_widget_hips = pg.PlotWidget(title="Deslocamento Lateral dos Quadris (X)")
        self.plot_widget_hips.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.left_hip_curve = self.plot_widget_hips.plot(pen='c', name='Quadril Esquerdo')
        self.right_hip_curve = self.plot_widget_hips.plot(pen='y', name='Quadril Direito')
        self.plot_widget_hips.addLegend()

        # Grupo de Movimentos dos Quadris
        self.hips_movement_group = QGroupBox("Análise dos Quadris")
        hips_movement_layout = QVBoxLayout()
        hips_movement_layout.addWidget(self.plot_widget_hips)
        self.hips_movement_group.setLayout(hips_movement_layout)
    

        # Adicionar grupos ao layout principal
        self.parent_layout.addWidget(self.head_movement_group)
        self.parent_layout.addWidget(self.shoulders_movement_group)
        self.parent_layout.addWidget(self.hips_movement_group)

    def process_frame(self, annotated_frame, results):
        """Processa cada frame para a análise de oscilação corporal."""
        landmark_indices = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
        ]

        pose_connections = [
            (self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.LEFT_SHOULDER),
            (self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP),
        ]

        # Marcar os pontos chave
        for idx in landmark_indices:
            landmark = results.pose_landmarks.landmark[idx]
            x = int(landmark.x * annotated_frame.shape[1])
            y = int(landmark.y * annotated_frame.shape[0])
            cv2.circle(annotated_frame, (x, y), 5, (0, 0, 255), -1)

        # Conectar os pontos com linhas
        for connection in pose_connections:
            start_idx = connection[0]
            end_idx = connection[1]
            start_landmark = results.pose_landmarks.landmark[start_idx]
            end_landmark = results.pose_landmarks.landmark[end_idx]
            x_start = int(start_landmark.x * annotated_frame.shape[1])
            y_start = int(start_landmark.y * annotated_frame.shape[0])
            x_end = int(end_landmark.x * annotated_frame.shape[1])
            y_end = int(end_landmark.y * annotated_frame.shape[0])
            cv2.line(annotated_frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

        # Obter coordenadas
        nose_landmark = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder_landmark = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder_landmark = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip_landmark = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip_landmark = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]

        nose_x = nose_landmark.x * annotated_frame.shape[1]
        nose_y = nose_landmark.y * annotated_frame.shape[0]
        left_shoulder_x = left_shoulder_landmark.x * annotated_frame.shape[1]
        right_shoulder_x = right_shoulder_landmark.x * annotated_frame.shape[1]
        left_hip_x = left_hip_landmark.x * annotated_frame.shape[1]
        right_hip_x = right_hip_landmark.x * annotated_frame.shape[1]

        if self.frames_captured < self.initial_frames:
            if self.zero_point_head_x is None:
                self.zero_point_head_x = nose_x
                self.zero_point_head_y = nose_y
                self.zero_point_left_shoulder_x = left_shoulder_x
                self.zero_point_right_shoulder_x = right_shoulder_x
                self.zero_point_left_hip_x = left_hip_x
                self.zero_point_right_hip_x = right_hip_x
            else:
                self.zero_point_head_x = (self.zero_point_head_x * self.frames_captured + nose_x) / (self.frames_captured + 1)
                self.zero_point_head_y = (self.zero_point_head_y * self.frames_captured + nose_y) / (self.frames_captured + 1)
                self.zero_point_left_shoulder_x = (self.zero_point_left_shoulder_x * self.frames_captured + left_shoulder_x) / (self.frames_captured + 1)
                self.zero_point_right_shoulder_x = (self.zero_point_right_shoulder_x * self.frames_captured + right_shoulder_x) / (self.frames_captured + 1)
                self.zero_point_left_hip_x = (self.zero_point_left_hip_x * self.frames_captured + left_hip_x) / (self.frames_captured + 1)
                self.zero_point_right_hip_x = (self.zero_point_right_hip_x * self.frames_captured + right_hip_x) / (self.frames_captured + 1)
            self.frames_captured += 1
            cv2.putText(annotated_frame, f'Capturando pontos zero... ({self.frames_captured}/{self.initial_frames})', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            delta_head_x = nose_x - self.zero_point_head_x
            delta_head_y = nose_y - self.zero_point_head_y
            delta_left_shoulder_x = left_shoulder_x - self.zero_point_left_shoulder_x
            delta_right_shoulder_x = right_shoulder_x - self.zero_point_right_shoulder_x
            delta_left_hip_x = left_hip_x - self.zero_point_left_hip_x
            delta_right_hip_x = right_hip_x - self.zero_point_right_hip_x

            self.displacements_head_x.append(delta_head_x)
            self.displacements_head_y.append(delta_head_y)
            self.displacements_left_shoulder.append(delta_left_shoulder_x)
            self.displacements_right_shoulder.append(delta_right_shoulder_x)
            self.displacements_left_hip.append(delta_left_hip_x)
            self.displacements_right_hip.append(delta_right_hip_x)

            if len(self.displacements_head_x) > self.max_points:
                self.displacements_head_x = self.displacements_head_x[-self.max_points:]
                self.displacements_head_y = self.displacements_head_y[-self.max_points:]
                self.displacements_left_shoulder = self.displacements_left_shoulder[-self.max_points:]
                self.displacements_right_shoulder = self.displacements_right_shoulder[-self.max_points:]
                self.displacements_left_hip = self.displacements_left_hip[-self.max_points:]
                self.displacements_right_hip = self.displacements_right_hip[-self.max_points:]

            # Atualizar gráficos usando setData
            self.head_x_curve.setData(self.displacements_head_x)
            self.head_y_curve.setData(self.displacements_head_y)
            self.left_shoulder_curve.setData(self.displacements_left_shoulder)
            self.right_shoulder_curve.setData(self.displacements_right_shoulder)
            self.left_hip_curve.setData(self.displacements_left_hip)
            self.right_hip_curve.setData(self.displacements_right_hip)

    def reset(self):
        """Reseta as variáveis específicas da análise de oscilação corporal."""
        self.zero_point_head_x = None
        self.zero_point_head_y = None
        self.zero_point_left_shoulder_x = None
        self.zero_point_right_shoulder_x = None
        self.zero_point_left_hip_x = None
        self.zero_point_right_hip_x = None
        self.displacements_head_x.clear()
        self.displacements_head_y.clear()
        self.displacements_left_shoulder.clear()
        self.displacements_right_shoulder.clear()
        self.displacements_left_hip.clear()
        self.displacements_right_hip.clear()
        self.frames_captured = 0

        # Limpar os dados dos gráficos
        self.head_x_curve.clear()
        self.head_y_curve.clear()
        self.left_shoulder_curve.clear()
        self.right_shoulder_curve.clear()
        self.left_hip_curve.clear()
        self.right_hip_curve.clear()
