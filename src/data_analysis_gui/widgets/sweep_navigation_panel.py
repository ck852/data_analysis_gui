"""
PatchBatch Electrophysiology Data Analysis Tool

Sweep Navigation Panel - A comprehensive sweep browsing widget combining
dropdown selection, arrow buttons, and a slider for quick navigation.

Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)
"""

from typing import Dict, List, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QComboBox, QSlider, QLabel
)
from PySide6.QtCore import Qt, Signal, QTimer

from data_analysis_gui.config.themes import (
    style_button, style_combo_box, style_label, MODERN_COLORS
)
from data_analysis_gui.config.logging import get_logger

logger = get_logger(__name__)


class SweepNavigationPanel(QWidget):
    """
    A comprehensive sweep navigation widget with synchronized controls.
    
    Provides three methods of sweep selection:
    1. Dropdown combo box for direct selection
    2. Arrow buttons for sequential navigation (with click-and-hold support)
    3. Horizontal slider for quick browsing
    
    Features:
    - Synchronized controls prevent signal loops
    - Rate-limited slider dragging for performance
    - Click-and-hold arrow buttons for continuous scrolling
    - Sweep timing display from dataset metadata
    - Consistent theming with application style
    """
    
    # Signal emitted when sweep selection changes
    sweep_changed = Signal(str)  # Emits sweep index as string
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        # State
        self.sweep_times: Dict[str, float] = {}
        self.pending_sweep_index: Optional[str] = None
        self._is_dragging = False
        
        # Rate limiting timer for slider
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(False)
        self.update_timer.setInterval(50)  # 50ms throttle
        self.update_timer.timeout.connect(self._emit_pending_change)
        
        # Click-and-hold timers for arrow buttons
        self.prev_hold_timer = QTimer(self)
        self.prev_hold_timer.setSingleShot(True)
        self.prev_hold_timer.setInterval(500)  # Initial delay
        self.prev_hold_timer.timeout.connect(self._start_prev_repeat)
        
        self.next_hold_timer = QTimer(self)
        self.next_hold_timer.setSingleShot(True)
        self.next_hold_timer.setInterval(500)  # Initial delay
        self.next_hold_timer.timeout.connect(self._start_next_repeat)
        
        self.prev_repeat_timer = QTimer(self)
        self.prev_repeat_timer.setInterval(100)  # Repeat rate
        self.prev_repeat_timer.timeout.connect(self._prev_sweep)
        
        self.next_repeat_timer = QTimer(self)
        self.next_repeat_timer.setInterval(100)  # Repeat rate
        self.next_repeat_timer.timeout.connect(self._next_sweep)
        
        self._init_ui()
        self._connect_signals()
    
    def _init_ui(self):
        """Initialize the widget layout and controls."""
        # Main vertical layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        # ===== Top Row: Labels, Buttons, and Combo =====
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(0)  # No spacing between widgets
        
        # Sweep count label - on the left
        self.count_label = QLabel("1/1")
        self.count_label.setMinimumWidth(35)
        self.count_label.setMaximumWidth(45)
        style_label(self.count_label, "muted")
        top_row.addWidget(self.count_label)
        
        # Previous button
        self.prev_btn = QPushButton("◀")
        self.prev_btn.setMaximumWidth(10)
        self.prev_btn.setEnabled(False)
        style_button(self.prev_btn, "secondary")
        top_row.addWidget(self.prev_btn)
        
        # Sweep combo box - sized for 4 digits
        self.sweep_combo = QComboBox()
        self.sweep_combo.setMinimumWidth(50)
        self.sweep_combo.setMaximumWidth(60)
        self.sweep_combo.setEnabled(False)
        style_combo_box(self.sweep_combo)
        top_row.addWidget(self.sweep_combo)
        
        # Next button
        self.next_btn = QPushButton("▶")
        self.next_btn.setMaximumWidth(10)
        self.next_btn.setEnabled(False)
        style_button(self.next_btn, "secondary")
        top_row.addWidget(self.next_btn)
        
        # Time label - on the right
        self.time_label = QLabel("t = N/A")
        self.time_label.setMinimumWidth(65)
        self.time_label.setMaximumWidth(80)
        style_label(self.time_label, "muted")
        top_row.addWidget(self.time_label)
        
        layout.addLayout(top_row)
        
        # ===== Bottom Row: Slider Only =====
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(0)
        
        # Sweep slider - fixed 420px width (matches top row max width)
        self.sweep_slider = QSlider(Qt.Orientation.Horizontal)
        self.sweep_slider.setValue(0)
        self.sweep_slider.setEnabled(False)
        self.sweep_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.sweep_slider.setSingleStep(1)
        self.sweep_slider.setPageStep(1)
        self.sweep_slider.setMinimumWidth(400)
        self.sweep_slider.setMaximumWidth(400)
        self._style_slider()
        bottom_row.addWidget(self.sweep_slider)
        
        layout.addLayout(bottom_row)
    
    def _style_slider(self):
        """Apply custom styling to the slider using theme colors."""
        self.sweep_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: 1px solid {MODERN_COLORS['border']};
                height: 6px;
                background: {MODERN_COLORS['surface']};
                border-radius: 3px;
            }}
            
            QSlider::groove:horizontal:disabled {{
                background: {MODERN_COLORS['disabled']};
            }}
            
            QSlider::handle:horizontal {{
                background: {MODERN_COLORS['primary']};
                border: 1px solid {MODERN_COLORS['primary']};
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}
            
            QSlider::handle:horizontal:hover {{
                background: #0066CC;
                border: 1px solid #0066CC;
            }}
            
            QSlider::handle:horizontal:pressed {{
                background: #0052A3;
            }}
            
            QSlider::handle:horizontal:disabled {{
                background: {MODERN_COLORS['disabled']};
                border: 1px solid {MODERN_COLORS['border']};
            }}
        """)
    
    def _connect_signals(self):
        """Connect internal signals for synchronized control updates."""
        # Combo box changes -> update slider
        self.sweep_combo.currentIndexChanged.connect(self._on_combo_changed)
        
        # Slider changes -> update combo (with rate limiting)
        self.sweep_slider.valueChanged.connect(self._on_slider_changed)
        self.sweep_slider.sliderPressed.connect(self._on_slider_pressed)
        self.sweep_slider.sliderReleased.connect(self._on_slider_released)
        
        # Arrow buttons -> single click navigation
        self.prev_btn.clicked.connect(self._prev_sweep)
        self.next_btn.clicked.connect(self._next_sweep)
        
        # Arrow buttons -> click-and-hold navigation
        self.prev_btn.pressed.connect(self._on_prev_pressed)
        self.prev_btn.released.connect(self._on_prev_released)
        self.next_btn.pressed.connect(self._on_next_pressed)
        self.next_btn.released.connect(self._on_next_released)
    
    def _on_combo_changed(self, index: int):
        """
        Handle combo box selection change.
        Syncs slider position and emits sweep_changed signal.
        """
        if index < 0:
            return
        
        # Sync slider position (block signals to prevent loop)
        self.sweep_slider.blockSignals(True)
        self.sweep_slider.setValue(index)
        self.sweep_slider.blockSignals(False)
        
        # Update labels
        self._update_labels()
        
        # Emit change signal (unless we're in the middle of dragging)
        if not self._is_dragging:
            sweep_text = self.sweep_combo.currentText()
            if sweep_text:
                self.sweep_changed.emit(sweep_text)
                logger.debug(f"Sweep changed via combo: {sweep_text}")
    
    def _on_slider_changed(self, value: int):
        """
        Handle slider value change.
        Syncs combo box and implements rate limiting during drag.
        """
        if value < 0 or value >= self.sweep_combo.count():
            return
        
        # Sync combo box immediately (visual feedback)
        self.sweep_combo.blockSignals(True)
        self.sweep_combo.setCurrentIndex(value)
        self.sweep_combo.blockSignals(False)
        
        # Update labels immediately
        self._update_labels()
        
        # Store pending sweep for rate-limited emission
        self.pending_sweep_index = self.sweep_combo.itemText(value)
        
        # If dragging, use timer-based rate limiting
        if self._is_dragging:
            if not self.update_timer.isActive():
                self.update_timer.start()
        else:
            # Not dragging (e.g., keyboard control), emit immediately
            self.sweep_changed.emit(self.pending_sweep_index)
            logger.debug(f"Sweep changed via slider (immediate): {self.pending_sweep_index}")
    
    def _on_slider_pressed(self):
        """Mark that slider dragging has started."""
        self._is_dragging = True
        logger.debug("Slider drag started")
    
    def _on_slider_released(self):
        """
        Handle slider release - stop timer and emit final position.
        Ensures the final sweep is always rendered.
        """
        self._is_dragging = False
        self.update_timer.stop()
        
        # Emit the final position immediately
        if self.pending_sweep_index is not None:
            self.sweep_changed.emit(self.pending_sweep_index)
            logger.debug(f"Sweep changed via slider (final): {self.pending_sweep_index}")
            self.pending_sweep_index = None
    
    def _emit_pending_change(self):
        """
        Timer callback - emit the pending sweep change.
        This is the rate-limited emission during dragging.
        """
        if self.pending_sweep_index is not None:
            self.sweep_changed.emit(self.pending_sweep_index)
            logger.debug(f"Sweep changed via slider (throttled): {self.pending_sweep_index}")
    
    def _update_labels(self):
        """Update the sweep count and time labels."""
        current_idx = self.sweep_combo.currentIndex()
        total = self.sweep_combo.count()
        
        # Update count label (1-indexed for user display)
        if total > 0:
            self.count_label.setText(f"{current_idx + 1}/{total}")
        else:
            self.count_label.setText("0/0")
        
        # Update time label with "t =" format
        sweep_text = self.sweep_combo.currentText()
        if sweep_text and sweep_text in self.sweep_times:
            time_value = self.sweep_times[sweep_text]
            self.time_label.setText(f"t = {time_value:.2f}s")
        else:
            self.time_label.setText("t = N/A")
    
    def _prev_sweep(self):
        """Navigate to previous sweep."""
        current_idx = self.sweep_combo.currentIndex()
        if current_idx > 0:
            self.sweep_combo.setCurrentIndex(current_idx - 1)
    
    def _next_sweep(self):
        """Navigate to next sweep."""
        current_idx = self.sweep_combo.currentIndex()
        if current_idx < self.sweep_combo.count() - 1:
            self.sweep_combo.setCurrentIndex(current_idx + 1)
    
    # ========== Click-and-Hold Implementation ==========
    
    def _on_prev_pressed(self):
        """Handle previous button press - start hold timer."""
        self.prev_hold_timer.start()
        logger.debug("Previous button pressed - starting hold timer")
    
    def _on_prev_released(self):
        """Handle previous button release - stop all timers."""
        self.prev_hold_timer.stop()
        self.prev_repeat_timer.stop()
        logger.debug("Previous button released - stopped timers")
    
    def _start_prev_repeat(self):
        """Start continuous previous navigation after initial delay."""
        self.prev_repeat_timer.start()
        logger.debug("Previous button held - starting auto-repeat")
    
    def _on_next_pressed(self):
        """Handle next button press - start hold timer."""
        self.next_hold_timer.start()
        logger.debug("Next button pressed - starting hold timer")
    
    def _on_next_released(self):
        """Handle next button release - stop all timers."""
        self.next_hold_timer.stop()
        self.next_repeat_timer.stop()
        logger.debug("Next button released - stopped timers")
    
    def _start_next_repeat(self):
        """Start continuous next navigation after initial delay."""
        self.next_repeat_timer.start()
        logger.debug("Next button held - starting auto-repeat")
    
    # ========== Public API ==========
    
    def get_current_sweep(self) -> str:
        """
        Get the currently selected sweep index as a string.
        
        Returns:
            str: Current sweep index (e.g., "0", "1", "2"), or empty string if none.
        """
        return self.sweep_combo.currentText()
    
    def set_current_sweep(self, sweep_index: str):
        """
        Programmatically set the current sweep selection.
        
        Args:
            sweep_index: Sweep index as a string (e.g., "0", "1", "2")
        """
        index = self.sweep_combo.findText(sweep_index)
        if index >= 0:
            # Set the index (this will trigger currentIndexChanged if index changes)
            self.sweep_combo.setCurrentIndex(index)
            
            # Force emit sweep_changed even if index didn't change
            # (e.g., when loading a file and first sweep is already selected)
            self.sweep_changed.emit(sweep_index)
            
            logger.debug(f"Set current sweep to: {sweep_index}")
        else:
            logger.warning(f"Sweep index not found: {sweep_index}")
    
    def set_sweep_list(self, sweep_names: List[str]):
        """
        Populate the sweep selection controls with a list of sweep names.
        
        Args:
            sweep_names: List of sweep index strings (e.g., ["0", "1", "2", ...])
        """
        # Block signals during bulk update
        self.sweep_combo.blockSignals(True)
        self.sweep_slider.blockSignals(True)
        
        # Clear and repopulate
        self.sweep_combo.clear()
        self.sweep_combo.addItems(sweep_names)
        
        # Update slider range
        count = len(sweep_names)
        self.sweep_slider.setMaximum(max(0, count - 1))
        self.sweep_slider.setValue(0)
        
        # Re-enable signals
        self.sweep_combo.blockSignals(False)
        self.sweep_slider.blockSignals(False)
        
        # Update labels
        self._update_labels()
        
        logger.debug(f"Loaded {count} sweeps into navigation panel")
    
    def set_sweep_times(self, sweep_times: Dict[str, float]):
        """
        Set the sweep timing data for display.
        
        Args:
            sweep_times: Dictionary mapping sweep index strings to time values in seconds
                        (e.g., {"0": 0.0, "1": 5.5, "2": 11.0})
        """
        self.sweep_times = sweep_times
        self._update_labels()
        logger.debug(f"Set sweep times for {len(sweep_times)} sweeps")
    
    def set_enabled(self, enabled: bool):
        """
        Enable or disable all navigation controls.
        
        Args:
            enabled: True to enable controls, False to disable
        """
        self.prev_btn.setEnabled(enabled)
        self.next_btn.setEnabled(enabled)
        self.sweep_combo.setEnabled(enabled)
        self.sweep_slider.setEnabled(enabled)
        logger.debug(f"Navigation panel enabled: {enabled}")
    
    def clear(self):
        """
        Clear all sweep data and reset the panel to empty state.
        Called when no file is loaded.
        """
        # Stop any active timers
        self.prev_hold_timer.stop()
        self.next_hold_timer.stop()
        self.prev_repeat_timer.stop()
        self.next_repeat_timer.stop()
        
        self.sweep_combo.blockSignals(True)
        self.sweep_slider.blockSignals(True)
        
        self.sweep_combo.clear()
        self.sweep_slider.setMaximum(0)
        self.sweep_slider.setValue(0)
        self.sweep_times.clear()
        
        self.sweep_combo.blockSignals(False)
        self.sweep_slider.blockSignals(False)
        
        self._update_labels()
        self.set_enabled(False)
        
        logger.debug("Navigation panel cleared")