import sys
import asyncio
import threading
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTextEdit, QLineEdit, 
                             QLabel, QFileDialog, QProgressBar, QListWidget)
from PySide6.QtCore import Qt, Signal, QObject, QThread
from rag_engine import RAGEngine
from styles import DARK_THEME

class WorkerSignals(QObject):
    finished = Signal(object)
    progress = Signal(int)
    error = Signal(str)

class RAGWorker(QThread):
    def __init__(self, engine, mode, **kwargs):
        super().__init__()
        self.engine = engine
        self.mode = mode
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            if self.mode == "index":
                success = self.engine.index_document(self.kwargs["file_path"])
                self.signals.finished.emit(success)
            elif self.mode == "query":
                # Create a specific event loop for this thread's async operations
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                context = self.engine.search(self.kwargs["prompt"])
                response = loop.run_until_complete(
                    self.engine.query_groq(self.kwargs["prompt"], context)
                )
                self.signals.finished.emit(response)
                loop.close()
        except Exception as e:
            self.signals.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple RAG Document Summariser")
        self.setMinimumSize(800, 600)
        self.setStyleSheet(DARK_THEME)
        
        self.engine = RAGEngine()
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Header / Status
        self.status_label = QLabel("Ready. Upload a document to start.")
        main_layout.addWidget(self.status_label)

        # Top Bar: File selection
        top_bar = QHBoxLayout()
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(100)
        top_bar.addWidget(self.file_list)

        upload_btn = QPushButton("Upload Document")
        upload_btn.clicked.connect(self.upload_file)
        top_bar.addWidget(upload_btn)
        
        main_layout.addLayout(top_bar)

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        main_layout.addWidget(self.chat_display)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Input box
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask a question about your documents...")
        self.input_field.returnPressed.connect(self.send_query)
        input_layout.addWidget(self.input_field)

        self.send_btn = QPushButton("Ask")
        self.send_btn.clicked.connect(self.send_query)
        input_layout.addWidget(self.send_btn)

        main_layout.addLayout(input_layout)

    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Document", "", "Documents (*.pdf *.docx *.txt)"
        )
        if file_path:
            self.status_label.setText(f"Indexing {file_path}...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0) # Pulsing
            
            self.worker = RAGWorker(self.engine, "index", file_path=file_path)
            self.worker.signals.finished.connect(lambda s: self.on_upload_finished(s, file_path))
            self.worker.signals.error.connect(self.on_error)
            self.worker.start()

    def on_upload_finished(self, success, file_path):
        self.progress_bar.setVisible(False)
        if success:
            self.file_list.addItem(file_path)
            self.status_label.setText(f"Successfully indexed {file_path}")
        else:
            self.status_label.setText("Failed to index document.")

    def send_query(self):
        prompt = self.input_field.text().strip()
        if not prompt:
            return

        self.chat_display.append(f"<b>You:</b> {prompt}")
        self.input_field.clear()
        
        self.status_label.setText("Searching and generating answer...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        self.worker = RAGWorker(self.engine, "query", prompt=prompt)
        self.worker.signals.finished.connect(self.on_query_finished)
        self.worker.signals.error.connect(self.on_error)
        self.worker.start()

    def on_query_finished(self, response):
        self.progress_bar.setVisible(False)
        self.chat_display.append(f"<b>Assistant:</b> {response}")
        self.status_label.setText("Ready.")

    def on_error(self, message):
        self.progress_bar.setVisible(False)
        self.chat_display.append(f"<span style='color: red;'><b>Error:</b> {message}</span>")
        self.status_label.setText("An error occurred.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
