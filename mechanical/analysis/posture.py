from PyQt5.QtWidgets import QLabel, QVBoxLayout, QGroupBox
from PyQt5.QtCore import Qt
from utils import calculate_angle
import cv2

class PostureAnalysis:
    def __init__(self, parent_layout, mp_pose, max_points=100, initial_frames=30):
        self.parent_layout = parent_layout
        self.mp_pose = mp_pose
        self.max_points = max_points  # Not used but included for compatibility
        self.initial_frames = initial_frames  # Not used but included for compatibility

        # Labels para os ângulos
        self.head_angle_label = None
        self.shoulder_angle_label = None
        self.hip_angle_label = None
        self.knee_angle_label = None

    def setup_ui(self):
        """Configura os componentes da UI para a análise postural."""
        # Estilo para os labels
        label_style = """
        QLabel {
            background-color: #f0f0f0;
            border: 1px solid #cccccc;
            border-radius: 5px;
            padding: 10px;
            font-size: 24px;
            font-weight: bold;
            color: #333333;
        }
        """

        # Inicialização das labels
        self.head_angle_label = QLabel("Ângulo da Cabeça: 0.0°")
        self.head_angle_label.setAlignment(Qt.AlignCenter)
        self.head_angle_label.setStyleSheet(label_style)

        self.head_group = QGroupBox("Análise da Cabeça")
        head_layout = QVBoxLayout()
        head_layout.addWidget(self.head_angle_label)
        self.head_group.setLayout(head_layout)

        self.shoulder_angle_label = QLabel("Ângulo do Ombro: 0.0°")
        self.shoulder_angle_label.setAlignment(Qt.AlignCenter)
        self.shoulder_angle_label.setStyleSheet(label_style)

        self.shoulder_group = QGroupBox("Análise do Ombro")
        shoulder_layout = QVBoxLayout()
        shoulder_layout.addWidget(self.shoulder_angle_label)
        self.shoulder_group.setLayout(shoulder_layout)

        self.hip_angle_label = QLabel("Ângulo do Quadril: 0.0°")
        self.hip_angle_label.setAlignment(Qt.AlignCenter)
        self.hip_angle_label.setStyleSheet(label_style)

        self.hip_group = QGroupBox("Análise do Quadril")
        hip_layout = QVBoxLayout()
        hip_layout.addWidget(self.hip_angle_label)
        self.hip_group.setLayout(hip_layout)

        self.knee_angle_label = QLabel("Ângulo do Joelho: 0.0°")
        self.knee_angle_label.setAlignment(Qt.AlignCenter)
        self.knee_angle_label.setStyleSheet(label_style)

        self.knee_group = QGroupBox("Análise do Joelho")
        knee_layout = QVBoxLayout()
        knee_layout.addWidget(self.knee_angle_label)
        self.knee_group.setLayout(knee_layout)

        # Estilo para os QGroupBoxes
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

        self.head_group.setStyleSheet(groupbox_style)
        self.shoulder_group.setStyleSheet(groupbox_style)
        self.hip_group.setStyleSheet(groupbox_style)
        self.knee_group.setStyleSheet(groupbox_style)

        # Adicionar grupos ao layout principal
        self.parent_layout.addWidget(self.head_group)
        self.parent_layout.addWidget(self.shoulder_group)
        self.parent_layout.addWidget(self.hip_group)
        self.parent_layout.addWidget(self.knee_group)

    def process_frame(self, annotated_frame, results):
        """Processa cada frame para a análise postural."""
        if not results.pose_landmarks:
            return

        landmarks = results.pose_landmarks.landmark

        image_height, image_width, _ = annotated_frame.shape

        # Obter coordenadas dos pontos
        try:
            # Cabeça
            left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
            right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
            head_x = int((left_ear.x + right_ear.x) / 2 * image_width)
            head_y = int((left_ear.y + right_ear.y) / 2 * image_height)

            # Ombros
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            shoulder_x = int((left_shoulder.x + right_shoulder.x) / 2 * image_width)
            shoulder_y = int((left_shoulder.y + right_shoulder.y) / 2 * image_height)

            # Quadris
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            hip_x = int((left_hip.x + right_hip.x) / 2 * image_width)
            hip_y = int((left_hip.y + right_hip.y) / 2 * image_height)

            # Joelhos
            left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            knee_x = int((left_knee.x + right_knee.x) / 2 * image_width)
            knee_y = int((left_knee.y + right_knee.y) / 2 * image_height)

            # Tornozelos (pés)
            left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            ankle_x = int((left_ankle.x + right_ankle.x) / 2 * image_width)
            ankle_y = int((left_ankle.y + right_ankle.y) / 2 * image_height)
        except IndexError:
            # Caso algum landmark não esteja disponível
            return

        points = [
            (head_x, head_y),
            (shoulder_x, shoulder_y),
            (hip_x, hip_y),
            (knee_x, knee_y)
        ]

        # Definir zero_line_x e reference_y usando o tornozelo (pé)
        zero_line_x = ankle_x
        reference_y = ankle_y

        # Calcular ângulos relativos à linha zero dinâmica
        head_angle = calculate_angle(head_x, head_y, zero_line_x, reference_y)
        shoulder_angle = calculate_angle(shoulder_x, shoulder_y, zero_line_x, reference_y)
        hip_angle = calculate_angle(hip_x, hip_y, zero_line_x, reference_y)
        knee_angle = calculate_angle(knee_x, knee_y, zero_line_x, reference_y)

        # Atualizar labels na interface
        self.head_angle_label.setText(f"Ângulo da Cabeça: {head_angle:.1f}°")
        self.shoulder_angle_label.setText(f"Ângulo do Ombro: {shoulder_angle:.1f}°")
        self.hip_angle_label.setText(f"Ângulo do Quadril: {hip_angle:.1f}°")
        self.knee_angle_label.setText(f"Ângulo do Joelho: {knee_angle:.1f}°")

        # Desenhar a linha zero dinâmica na imagem
        cv2.line(annotated_frame, (zero_line_x, 0), (zero_line_x, image_height), (255, 0, 0), 2)

        # Marcar os pontos no corpo do atleta
        for x, y in points:
            cv2.circle(annotated_frame, (x, y), 5, (0, 0, 255), -1)

        # Conectar os pontos formando uma linha reta na postura do atleta
        cv2.line(annotated_frame, points[0], points[1], (0, 255, 0), 2)  # Cabeça aos Ombros
        cv2.line(annotated_frame, points[1], points[2], (0, 255, 0), 2)  # Ombros aos Quadris
        cv2.line(annotated_frame, points[2], points[3], (0, 255, 0), 2)  # Quadris aos Joelhos

    def reset(self):
        """Reseta as variáveis específicas da análise postural."""
        self.head_angle_label.setText("Ângulo da Cabeça: 0.0°")
        self.shoulder_angle_label.setText("Ângulo do Ombro: 0.0°")
        self.hip_angle_label.setText("Ângulo do Quadril: 0.0°")
        self.knee_angle_label.setText("Ângulo do Joelho: 0.0°")
