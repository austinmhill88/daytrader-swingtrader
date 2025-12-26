"""
AI Console - Interactive AI assistant panel.
"""
import asyncio
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QLabel, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from loguru import logger


class AIWorker(QThread):
    """Worker thread for AI requests to avoid blocking UI."""
    
    response_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, ai_client, prompt):
        super().__init__()
        self.ai_client = ai_client
        self.prompt = prompt
    
    def run(self):
        """Run AI request in background thread."""
        try:
            # Create event loop for async call
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful trading assistant. Provide clear, "
                              "concise answers about trading strategies, risk management, "
                              "and market analysis."
                },
                {
                    "role": "user",
                    "content": self.prompt
                }
            ]
            
            response = loop.run_until_complete(
                self.ai_client.chat(messages, options={"num_predict": 300})
            )
            
            loop.close()
            
            if response:
                self.response_ready.emit(response)
            else:
                self.error_occurred.emit("No response from AI server")
        except Exception as e:
            self.error_occurred.emit(str(e))


class AIConsole(QWidget):
    """
    AI Assistant console for interactive queries.
    """
    
    def __init__(self, ai_client=None):
        """
        Initialize AI console.
        
        Args:
            ai_client: Optional AI client instance
        """
        super().__init__()
        self.ai_client = ai_client
        self.worker = None
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        
        title = QLabel("🤖 AI Trading Assistant")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Model selector
        header_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "qwen2.5:3b-instruct",
            "llama3.2:3b-instruct"
        ])
        header_layout.addWidget(self.model_combo)
        
        # Status indicator
        self.status_label = QLabel("● Ready")
        self.status_label.setStyleSheet("color: green;")
        header_layout.addWidget(self.status_label)
        
        layout.addLayout(header_layout)
        
        # Chat history display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Consolas", 10))
        layout.addWidget(self.chat_display)
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask a question about trading, strategies, or analysis...")
        self.input_field.returnPressed.connect(self._send_message)
        input_layout.addWidget(self.input_field)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._send_message)
        input_layout.addWidget(self.send_btn)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_chat)
        input_layout.addWidget(self.clear_btn)
        
        layout.addLayout(input_layout)
        
        # Quick actions
        actions_layout = QHBoxLayout()
        actions_layout.addWidget(QLabel("Quick Actions:"))
        
        quick_actions = [
            ("Market Summary", "Provide a brief summary of current market conditions"),
            ("Risk Check", "What should I monitor for risk management today?"),
            ("Strategy Tips", "Give me 3 tips for improving trading performance"),
        ]
        
        for label, prompt in quick_actions:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, p=prompt: self._quick_action(p))
            actions_layout.addWidget(btn)
        
        actions_layout.addStretch()
        layout.addLayout(actions_layout)
        
        # Welcome message
        self._append_message("System", "AI Assistant ready. Ask me anything about trading!")
    
    def set_ai_client(self, ai_client):
        """Set the AI client instance."""
        self.ai_client = ai_client
        if ai_client and ai_client.enabled:
            self._append_message("System", f"Connected to AI server: {ai_client.base_url}")
        else:
            self._append_message("System", "AI server not available. Check configuration.")
    
    def _append_message(self, sender: str, message: str):
        """
        Append a message to the chat display.
        
        Args:
            sender: Message sender (User, AI, System)
            message: Message content
        """
        timestamp = logger._core.now().strftime("%H:%M:%S")
        
        if sender == "User":
            color = "blue"
        elif sender == "AI":
            color = "green"
        else:
            color = "gray"
        
        formatted = f'<span style="color: {color}; font-weight: bold;">[{timestamp}] {sender}:</span> {message}<br>'
        self.chat_display.append(formatted)
    
    def _send_message(self):
        """Send user message to AI."""
        prompt = self.input_field.text().strip()
        if not prompt:
            return
        
        if not self.ai_client or not self.ai_client.enabled:
            self._append_message("System", "AI server not available")
            return
        
        # Display user message
        self._append_message("User", prompt)
        self.input_field.clear()
        
        # Disable input while processing
        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.status_label.setText("● Processing...")
        self.status_label.setStyleSheet("color: orange;")
        
        # Create worker thread
        self.worker = AIWorker(self.ai_client, prompt)
        self.worker.response_ready.connect(self._on_response)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.start()
    
    def _on_response(self, response: str):
        """Handle AI response."""
        self._append_message("AI", response)
        
        # Re-enable input
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.status_label.setText("● Ready")
        self.status_label.setStyleSheet("color: green;")
    
    def _on_error(self, error: str):
        """Handle error."""
        self._append_message("System", f"Error: {error}")
        
        # Re-enable input
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.status_label.setText("● Error")
        self.status_label.setStyleSheet("color: red;")
    
    def _quick_action(self, prompt: str):
        """Execute a quick action."""
        self.input_field.setText(prompt)
        self._send_message()
    
    def _clear_chat(self):
        """Clear chat history."""
        self.chat_display.clear()
        self._append_message("System", "Chat cleared")
