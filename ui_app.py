import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QDockWidget, QLabel, QPushButton, QFileDialog,
                             QVBoxLayout, QWidget, QScrollArea, QSizePolicy, QGroupBox, QHBoxLayout, 
                             QProgressBar, QSpinBox, QComboBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from filter_image import FilterImage


class FilterThread(QThread):
    update_progress = pyqtSignal(int)
    filter_complete = pyqtSignal(np.ndarray)

    def __init__(self, filter_func, image):
        super().__init__()
        self.filter_func = filter_func
        self.image = image

    def run(self):
        rows, cols = self.image.shape[:2]
        total_pixels = rows * cols
        processed_pixels = 0

        def progress_callback(pixel_count):
            nonlocal processed_pixels
            processed_pixels += pixel_count
            progress = int((processed_pixels / total_pixels) * 100)
            self.update_progress.emit(progress)

        result = self.filter_func(self.image, progress_callback)
        self.filter_complete.emit(result)

class ImageEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.filter_image = FilterImage()

    def initUI(self):
        self.setWindowTitle('Advanced Image Editor')
        self.setGeometry(50, 50, 1000, 700)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidget(self.label)
        self.scrollArea.setWidgetResizable(True)
        
        self.setCentralWidget(self.scrollArea)

        self.sidebar = QDockWidget("Tools", self)
        self.sidebar.setAllowedAreas(Qt.LeftDockWidgetArea)
        
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout()
        
        # Add progress bar
        self.progressBar = QProgressBar(self)
        self.progressBar.setVisible(False)
        sidebar_layout.addWidget(self.progressBar)
        
        # File operations group
        file_group = QGroupBox("File Operations")
        file_layout = QHBoxLayout()
        self.openButton = QPushButton('Open Image', self)
        self.openButton.clicked.connect(self.openImage)
        self.saveButton = QPushButton('Save Image', self)
        self.saveButton.clicked.connect(self.saveImage)
        file_layout.addWidget(self.openButton)
        file_layout.addWidget(self.saveButton)
        file_group.setLayout(file_layout)
        sidebar_layout.addWidget(file_group)

        # Basic filters group
        basic_filters_group = QGroupBox("Basic Filters")
        basic_filters_layout = QVBoxLayout()
        
        self.gaussianButton = QPushButton('Gaussian Blur', self)
        self.gaussianButton.clicked.connect(self.gaussian_filter)
        basic_filters_layout.addWidget(self.gaussianButton)

        self.highPassButton = QPushButton('High-Pass Filter', self)
        self.highPassButton.clicked.connect(self.high_pass_filter)
        basic_filters_layout.addWidget(self.highPassButton)

        self.lowPassButton = QPushButton('Low-Pass Filter', self)
        self.lowPassButton.clicked.connect(self.low_pass_filter)
        basic_filters_layout.addWidget(self.lowPassButton)

        self.meanFilterButton = QPushButton('Mean Filter', self)
        self.meanFilterButton.clicked.connect(self.mean_filter)
        basic_filters_layout.addWidget(self.meanFilterButton)

        basic_filters_group.setLayout(basic_filters_layout)
        sidebar_layout.addWidget(basic_filters_group)

        # Advanced filters group
        advanced_filters_group = QGroupBox("Advanced Filters")
        advanced_filters_layout = QVBoxLayout()

        self.histogramButton = QPushButton('Histogram Equalization', self)
        self.histogramButton.clicked.connect(self.histogramme_filter)
        advanced_filters_layout.addWidget(self.histogramButton)

        self.contourButton = QPushButton('Detect Contours', self)
        self.contourButton.clicked.connect(self.contour_detection)
        advanced_filters_layout.addWidget(self.contourButton)

        self.noiseButton = QPushButton('Add Noise', self)
        self.noiseButton.clicked.connect(self.add_noise)
        advanced_filters_layout.addWidget(self.noiseButton)

        self.minMaxButton = QPushButton('Min-Max Smoothing', self)
        self.minMaxButton.clicked.connect(self.min_max_smoothing)
        advanced_filters_layout.addWidget(self.minMaxButton)

        self.medianFilterButton = QPushButton('Median Filter', self)
        self.medianFilterButton.clicked.connect(self.median_filter)
        advanced_filters_layout.addWidget(self.medianFilterButton)

        self.hybridMedianButton = QPushButton('Hybrid Median Filter', self)
        self.hybridMedianButton.clicked.connect(self.hybrid_median_filter)
        advanced_filters_layout.addWidget(self.hybridMedianButton)

        advanced_filters_group.setLayout(advanced_filters_layout)
        sidebar_layout.addWidget(advanced_filters_group)

       # Add Morphological Operations group
        morph_group = QGroupBox("Morphological Operations")
        morph_layout = QVBoxLayout()

        self.morph_combo = QComboBox()
        self.morph_combo.addItems(['Dilate', 'Erode', 'Open', 'Close'])
        morph_layout.addWidget(self.morph_combo)

        kernel_layout = QHBoxLayout()
        kernel_layout.addWidget(QLabel("Kernel Size:"))
        self.kernel_spin = QSpinBox()
        self.kernel_spin.setRange(3, 15)
        self.kernel_spin.setSingleStep(2)
        self.kernel_spin.setValue(3)
        kernel_layout.addWidget(self.kernel_spin)

        iterations_layout = QHBoxLayout()
        iterations_layout.addWidget(QLabel("Iterations:"))
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 10)
        self.iterations_spin.setValue(1)
        iterations_layout.addWidget(self.iterations_spin)

        morph_layout.addLayout(kernel_layout)
        morph_layout.addLayout(iterations_layout)

        self.morphButton = QPushButton('Apply Morphological Operation', self)
        self.morphButton.clicked.connect(self.apply_morph_operation)
        morph_layout.addWidget(self.morphButton)

        morph_group.setLayout(morph_layout)
        sidebar_layout.addWidget(morph_group)
        sidebar_layout.addStretch(1)  # Add stretch to push everything up
        sidebar_widget.setLayout(sidebar_layout)
        self.sidebar.setWidget(sidebar_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.sidebar)

        self.image = None

    def openImage(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.xpm *.jpg *.jpeg *.bmp)', options=options)
        if fileName:
            self.image = cv2.imread(fileName)
            self.displayImage()

    def saveImage(self):
        if self.image is not None:
            options = QFileDialog.Options()
            fileName, _ = QFileDialog.getSaveFileName(self, 'Save Image', '', 'Images (*.png *.xpm *.jpg *.jpeg *.bmp)', options=options)
            if fileName:
                if not (fileName.endswith('.png') or fileName.endswith('.jpg') or fileName.endswith('.jpeg') or fileName.endswith('.bmp')):
                    fileName += '.png'
                try:
                    cv2.imwrite(fileName, self.image)
                except cv2.error as e:
                    print(f"Error: {e}")
                    print("Unable to save the image. Please ensure the filename has a correct extension and try again.")
    def apply_morph_operation(self):
        if self.image is not None:
            operation = self.morph_combo.currentText().lower()
            kernel_size = self.kernel_spin.value()
            iterations = self.iterations_spin.value()

            self.progressBar.setVisible(True)
            self.progressBar.setValue(0)
            self.thread = FilterThread(
                lambda img, cb: self.filter_image.morph_operation(img, operation, kernel_size, iterations),
                self.image
            )
            self.thread.update_progress.connect(self.updateProgress)
            self.thread.filter_complete.connect(self.filterComplete)
            self.thread.start()

    def hybrid_median_filter(self):
        if self.image is not None:
            self.progressBar.setVisible(True)
            self.progressBar.setValue(0)
            self.thread = FilterThread(self.filter_image.hybrid_median_filter, self.image)
            self.thread.update_progress.connect(self.updateProgress)
            self.thread.filter_complete.connect(self.filterComplete)
            self.thread.start()
    def updateProgress(self, value):
        self.progressBar.setValue(value)

    def filterComplete(self, result):
        self.image = result
        self.displayImage()
        self.progressBar.setVisible(False)
    def gaussian_filter(self):
        if self.image is not None:
            self.image = self.filter_image.gaussian_filter(self.image)
            self.displayImage()

    def high_pass_filter(self):
        if self.image is not None:
            self.image = self.filter_image.high_pass_filter(self.image)
            self.displayImage()

    def histogramme_filter(self):
        if self.image is not None:
            self.image = self.filter_image.histogramme_filter(self.image)
            self.displayImage()

    def contour_detection(self):
        if self.image is not None:
            self.image = self.filter_image.detect_contours(self.image)
            self.displayImage()

    def add_noise(self):
        if self.image is not None:
            self.image = self.filter_image.add_gaussian_noise(self.image)
            self.displayImage()

    def mean_filter(self):
        if self.image is not None:
            self.image = self.filter_image.mean_filter(self.image)
            self.displayImage()

    def low_pass_filter(self):
        if self.image is not None:
            self.image = self.filter_image.low_pass_filter(self.image)
            self.displayImage()

    def min_max_smoothing(self):
        if self.image is not None:
            self.progressBar.setVisible(True)
            self.progressBar.setValue(0)
            self.thread = FilterThread(self.filter_image.min_max_smoothing, self.image)
            self.thread.update_progress.connect(self.updateProgress)
            self.thread.filter_complete.connect(self.filterComplete)
            self.thread.start()

    def median_filter(self):
        if self.image is not None:
            self.image = self.filter_image.median_filter(self.image)
            self.displayImage()

    def hybrid_median_filter(self):
        if self.image is not None:
            self.image = self.filter_image.hybrid_median_filter(self.image)
            self.displayImage()

    def displayImage(self):
        if self.image is not None:
            if len(self.image.shape) == 3:  # Color image
                h, w, _ = self.image.shape
                qformat = QImage.Format_RGB888
            elif len(self.image.shape) == 2:  # Grayscale image
                h, w = self.image.shape
                qformat = QImage.Format_Grayscale8
            else:
                return

            img = QImage(self.image.data, w, h, self.image.strides[0], qformat)
            img = img.rgbSwapped()  # Convert BGR to RGB
            pixmap = QPixmap.fromImage(img)
            
            # Scale the pixmap while maintaining aspect ratio
            self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    editor = ImageEditor()
    editor.show()
    sys.exit(app.exec_())