"""Editor console widget (stdout/stderr capture)."""
from __future__ import annotations

import sys
from PySide6 import QtCore, QtGui, QtWidgets


class ConsoleWidget(QtWidgets.QWidget):
    """Console widget for displaying logs, warnings, and errors."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
        # Store original stdout/stderr
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        
        # Redirect stdout/stderr to console
        sys.stdout = self
        sys.stderr = self
        
        # Log levels
        self._log_colors = {
            'DEBUG': '#888888',
            'INFO': '#ffffff',
            'WARNING': '#ffaa00',
            'ERROR': '#ff4444',
            'CRITICAL': '#ff0000',
        }
    
    def _setup_ui(self) -> None:
        """Set up the console UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Toolbar with clear button and filter options
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setSpacing(4)
        
        # Clear button
        clear_btn = QtWidgets.QPushButton("Clear", self)
        clear_btn.setFixedWidth(60)
        clear_btn.clicked.connect(self.clear)
        toolbar.addWidget(clear_btn)
        
        # Filter checkboxes
        self._show_info = QtWidgets.QCheckBox("Info", self)
        self._show_info.setChecked(True)
        self._show_info.stateChanged.connect(self._apply_filter)
        toolbar.addWidget(self._show_info)
        
        self._show_warnings = QtWidgets.QCheckBox("Warnings", self)
        self._show_warnings.setChecked(True)
        self._show_warnings.stateChanged.connect(self._apply_filter)
        toolbar.addWidget(self._show_warnings)
        
        self._show_errors = QtWidgets.QCheckBox("Errors", self)
        self._show_errors.setChecked(True)
        self._show_errors.stateChanged.connect(self._apply_filter)
        toolbar.addWidget(self._show_errors)
        
        toolbar.addStretch(1)
        
        layout.addLayout(toolbar)
        
        # Text edit for log output
        self._text_edit = QtWidgets.QTextEdit(self)
        self._text_edit.setReadOnly(True)
        self._text_edit.setFont(QtGui.QFont("Consolas", 9))
        self._text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #333;
            }
        """)
        layout.addWidget(self._text_edit, 1)
        
        # Store all messages for filtering
        self._all_messages: list[tuple[str, str]] = []  # (level, message)
    
    def write(self, text: str) -> None:
        """Write text to console (used for stdout/stderr redirection)."""
        if not text or text.strip() == '':
            return
        
        # Determine log level from text
        level = 'INFO'
        upper_text = text.upper()
        if 'ERROR' in upper_text or 'CRITICAL' in upper_text or 'EXCEPTION' in upper_text:
            level = 'ERROR'
        elif 'WARNING' in upper_text or 'WARN' in upper_text:
            level = 'WARNING'
        elif 'DEBUG' in upper_text:
            level = 'DEBUG'
        
        # Store message
        self._all_messages.append((level, text))
        
        # Display if passes filter
        self._display_message(level, text)
        
        # Also write to original stdout
        self._original_stdout.write(text)
    
    def _display_message(self, level: str, text: str) -> None:
        """Display a message with appropriate color."""
        # Check filters
        if level == 'INFO' and not self._show_info.isChecked():
            return
        if level == 'WARNING' and not self._show_warnings.isChecked():
            return
        if level == 'ERROR' and not self._show_errors.isChecked():
            return
        
        color = self._log_colors.get(level, '#ffffff')
        
        # Add timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Format: [HH:MM:SS] [LEVEL] message
        formatted = f'<span style="color: #666;">[{timestamp}]</span> <span style="color: {color};">{self._escape_html(text)}</span>'
        
        self._text_edit.append(formatted)
        
        # Auto-scroll to bottom
        scrollbar = self._text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;'))
    
    def _apply_filter(self) -> None:
        """Re-apply filters and refresh display."""
        self._text_edit.clear()
        for level, text in self._all_messages:
            self._display_message(level, text)
    
    def clear(self) -> None:
        """Clear the console."""
        self._text_edit.clear()
        self._all_messages.clear()
    
    def flush(self) -> None:
        """Flush the console (required for file-like interface)."""
        self._original_stdout.flush()
    
    def log(self, message: str, level: str = 'INFO') -> None:
        """Log a message with specified level."""
        self._all_messages.append((level, message + '\n'))
        self._display_message(level, message + '\n')
    
    def restore_stdout(self) -> None:
        """Restore original stdout/stderr."""
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

