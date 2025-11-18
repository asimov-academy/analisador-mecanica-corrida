import sys
import os
import cv2
from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QHBoxLayout,
    QComboBox, QSizePolicy, QFileDialog
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import mediapipe as mp

from analysis.ocillation import OscillationAnalysis
from analysis.posture import PostureAnalysis
from analysis.stride import StrideAnalysis

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms/'


class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Análise da Mecânica de Corrida")

        # Configurar tamanho da janela para tela cheia
        screen = QApplication.primaryScreen()
        size = screen.size()
        screen_width = size.width()
        screen_height = size.height()
        self.resize(screen_width, screen_height)

        # Label para exibir o vídeo
        self.video_label = QLabel()
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)

        # Seletores de análise e câmera
        self.analysis_selector = QComboBox()
        self.analysis_selector.addItems(["Análise de Oscilação Corporal", "Análise Postural Lateral", "Análise de Passada"])
        self.analysis_selector.setCurrentIndex(0)

        self.camera_selector = QComboBox()
        self.available_cameras = self.get_available_cameras()
        self.camera_selector.addItems([f"Câmera {i} (Índice {i})" for i in self.available_cameras])
        self.camera_selector.setCurrentIndex(0)

        # Botões de iniciar, carregar vídeo e parar
        self.start_button = QPushButton("Iniciar")
        self.start_button.clicked.connect(self.start_video)

        self.load_video_button = QPushButton("Carregar Vídeo")
        self.load_video_button.clicked.connect(self.load_video)

        self.stop_button = QPushButton("Parar")
        self.stop_button.clicked.connect(self.stop_video)
        self.stop_button.setEnabled(False)

        # Layout de controle
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.load_video_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(QLabel("Análise:"))
        control_layout.addWidget(self.analysis_selector)
        control_layout.addWidget(QLabel("Câmera:"))
        control_layout.addWidget(self.camera_selector)

        # Layout para as análises
        self.analysis_layout = QVBoxLayout()

        # Layout esquerdo (vídeo e controles)
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.video_label)
        left_layout.addLayout(control_layout)

        # Layout direito (dados das análises)
        self.right_layout = QVBoxLayout()
        self.right_layout.addLayout(self.analysis_layout)

        # Layout principal
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(self.right_layout)
        self.setLayout(main_layout)

        # Inicializar captura de vídeo
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Inicializar Mediapipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Inicializar variáveis de análise
        self.analysis_type = "Análise de Oscilação Corporal"
        self.frames_captured = 0
        self.initial_frames = 30
        self.max_points = 100

        # Instanciar classes de análise
        self.analyses = {
            "Análise de Oscilação Corporal": OscillationAnalysis,
            "Análise Postural Lateral": PostureAnalysis,
            "Análise de Passada": StrideAnalysis 
        }
        self.current_analysis = None

        # Conectar sinal de mudança de análise
        self.analysis_selector.currentIndexChanged.connect(self.on_analysis_change)

    def on_analysis_change(self, index):
        analysis_name = self.analysis_selector.currentText()
        self.setup_analysis(analysis_name)

    def setup_analysis(self, analysis_name):
        """Configura a análise selecionada."""
        # Limpar layout atual
        self.clear_analysis_layout()

        # Instanciar a classe de análise
        analysis_class = self.analyses.get(analysis_name)
        if analysis_class:
            self.current_analysis = analysis_class(self.analysis_layout, self.mp_pose, self.max_points, self.initial_frames)
            self.current_analysis.setup_ui()

    def clear_analysis_layout(self):
        """Limpa o layout de análises."""
        while self.analysis_layout.count():
            child = self.analysis_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def get_available_cameras(self):
        """Retorna uma lista de índices de câmeras disponíveis."""
        available_cameras = []
        max_tested = 10
        for index in range(0, max_tested):
            cap = cv2.VideoCapture(index)
            if cap.read()[0]:
                available_cameras.append(index)
                cap.release()
            else:
                cap.release()
        if not available_cameras:
            available_cameras.append(0)
        return available_cameras

    def start_video(self):
        """Inicia a captura de vídeo da webcam."""
        selected_camera_index = self.camera_selector.currentIndex()
        camera_index = self.available_cameras[selected_camera_index]
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"Erro ao abrir a câmera no índice {camera_index}")
            return

        self.analysis_type = self.analysis_selector.currentText()
        self.setup_analysis(self.analysis_type)

        self.timer.start(30)
        self.start_button.setEnabled(False)
        self.load_video_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.camera_selector.setEnabled(False)
        self.analysis_selector.setEnabled(False)

        self.frames_captured = 0

    def load_video(self):
        """Permite ao usuário selecionar um arquivo de vídeo para análise."""
        video_path, _ = QFileDialog.getOpenFileName(self, "Selecionar Vídeo", "", "Vídeos (*.mp4 *.avi *.mov)")
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                print(f"Erro ao abrir o vídeo {video_path}")
                return

            self.analysis_type = self.analysis_selector.currentText()
            self.setup_analysis(self.analysis_type)

            self.timer.start(30)
            self.start_button.setEnabled(False)
            self.load_video_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.camera_selector.setEnabled(False)
            self.analysis_selector.setEnabled(False)

            self.frames_captured = 0

    def stop_video(self):
        """Para a captura de vídeo."""
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.clear()
        self.start_button.setEnabled(True)
        self.load_video_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.camera_selector.setEnabled(True)
        self.analysis_selector.setEnabled(True)

        if self.current_analysis:
            self.current_analysis.reset()
            self.current_analysis = None

        self.clear_analysis_layout()

    def update_frame(self):
        """Atualiza o frame do vídeo e processa a análise."""
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            annotated_frame = frame.copy()

            if results.pose_landmarks:
                if self.current_analysis:
                    self.current_analysis.process_frame(annotated_frame, results)
            else:
                cv2.putText(annotated_frame, "Aguardando detecção...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            qt_image = self.convert_cv_qt(annotated_frame)
            self.video_label.setPixmap(qt_image)
        else:
            # Se não houver mais frames (fim do vídeo), pare a reprodução
            self.stop_video()

    def convert_cv_qt(self, cv_img):
        """Converte uma imagem OpenCV para QPixmap para exibição no QLabel."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.showMaximized()
    sys.exit(app.exec())
