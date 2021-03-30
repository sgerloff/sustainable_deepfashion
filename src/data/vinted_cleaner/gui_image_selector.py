from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel, QScrollArea, QAction, \
    QMainWindow, QStyle, qApp, QPushButton
from src.data.vinted_cleaner.file_processing import *


class MainWindow(QMainWindow):
    def __init__(self, directory_dir, directory_count=0, parent=None):
        super(MainWindow, self).__init__(parent)

        self.all_directories = get_all_directories_path(directory_dir)
        self.all_directories.sort()
        self.deleted_images = []
        self.count = directory_count

        deleteAct = QAction(qApp.style().standardIcon(QStyle.SP_TrashIcon), 'Delete Selection', self)
        deleteAct.setStatusTip('Will delete the selected images')
        deleteAct.setShortcut('D')
        deleteAct.triggered.connect(self.delete_images)

        self.toolbar = self.addToolBar('Exit')
        self.toolbar.addAction(deleteAct)

        self.backButton = QPushButton('Back', self)
        self.backButton.setShortcut('B')
        self.backButton.clicked.connect(self.back)

        self.nextButton = QPushButton('Next', self)
        self.nextButton.setShortcut('N')
        self.nextButton.clicked.connect(self.next)

        self.toolbar.addWidget(self.backButton)
        self.toolbar.addWidget(self.nextButton)

        self.countLabel = QLabel(self)
        self.countLabel.setText('Directory count: ' + str(self.count))
        self.toolbar.addWidget(self.countLabel)

        self.images = get_all_images_path(self.all_directories[self.count])
        self.window = Window(self.images)
        self.setCentralWidget(self.window)

        self.show()

    def delete_images(self):
        current_selection = self.window.get_selection()
        delete_images(current_selection)
        print(current_selection)

        self.deleted_images += current_selection
        remaining_paths = self.window.paths

        for path in self.deleted_images:
            try:
                remaining_paths.remove(path)
            except ValueError:
                pass

        self.window = Window(remaining_paths)
        self.setCentralWidget(self.window)

    def next(self):
        self.count += 1
        image_paths = get_all_images_path(self.all_directories[self.count])

        self.window = Window(image_paths)
        self.setCentralWidget(self.window)
        self.countLabel.setText('Directory count: ' + str(self.count))

    def back(self):
        self.count -= 1
        image_paths = get_all_images_path(self.all_directories[self.count])

        self.window = Window(image_paths)
        self.setCentralWidget(self.window)
        self.countLabel.setText('Directory count: ' + str(self.count))

class ClickableLabel(QLabel):

    def __init__(self, path):
        super(ClickableLabel, self).__init__()
        self.width = 250
        self.height = 250
        self.isChecked = False

        pixmap = QPixmap(path)
        pixmap = pixmap.scaled(self.width, self.height, QtCore.Qt.KeepAspectRatio)
        self.setPixmap(pixmap)

    def mousePressEvent(self, event):
        self.isChecked = not self.isChecked

        if self.isChecked:
            self.setStyleSheet("border: 5px inset red;")
        else:
            self.setStyleSheet("")


class Window(QScrollArea):
    def __init__(self, paths):
        QScrollArea.__init__(self)

        widget = QWidget()

        self.layout = QGridLayout(widget)
        self.paths = paths
        self.all_labels = []
        self.nb_columns = 4

        self.populate_grid(self.paths)

        self.setWidget(widget)

        self.setWidgetResizable(True)
        self.setMinimumWidth(1000)
        self.setMinimumHeight(600)

    def populate_grid(self, paths):
        row = 0
        column = 0
        for idx, path in enumerate(paths):
            label = ClickableLabel(path)
            self.all_labels.append(label)
            self.layout.addWidget(self.all_labels[idx], row, column)

            column += 1
            if column % self.nb_columns == 0:
                row += 1
                column = 0

    def get_selection(self):
        selection = []
        for idx, path in enumerate(self.paths):
            if self.all_labels[idx].isChecked:
                selection.append(path)

        return selection
