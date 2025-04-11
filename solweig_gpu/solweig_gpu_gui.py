import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QFileDialog, QCheckBox,
    QDateEdit, QSpinBox, QComboBox, QProgressBar, QTextEdit, QGroupBox, QFormLayout,
    QHBoxLayout, QFrame, QToolButton, QStyle, QMessageBox, QScrollArea, QDesktopWidget
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from solweig_gpu import thermal_comfort


class EmittingStream:
    def __init__(self, signal):
        self.signal = signal

    def write(self, text):
        self.signal.emit(str(text))

    def flush(self):
        pass


class SOLWEIGWorker(QThread):
    log_signal = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        sys.stdout = EmittingStream(self.log_signal)
        try:
            thermal_comfort(**self.params)
        except Exception as e:
            self.log_signal.emit(f"\nError: {str(e)}")
        finally:
            self.finished.emit()


class SOLWEIGApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SOLWEIG GPU")

        screen = QDesktopWidget().screenGeometry()
        width = int(screen.width() * 0.85)
        height = int(screen.height() * 0.85)
        self.resize(width, height)

        self.setStyleSheet(self.load_styles())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.main_layout = QVBoxLayout(container)
        scroll.setWidget(container)

        outer_layout = QVBoxLayout(self)
        outer_layout.addWidget(scroll)
        self.setLayout(outer_layout)

        self._build_widgets()


    def select_path(self, target_widget, file=True):
        if file:
            path = QFileDialog.getOpenFileName()[0]
        else:
            path = QFileDialog.getExistingDirectory()
        if path:
            target_widget.setText(path)


    def _build_widgets(self):
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: monospace;")

        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setValue(0)

        self.run_button = QPushButton("Run SOLWEIG")
        self.run_button.setCursor(Qt.PointingHandCursor)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #00c853;
                color: white;
                border-radius: 6px;
                font-size: 16px;
                font-weight: bold;
                padding: 10px 20px;
                margin: 10px auto;
            }
            QPushButton:hover {
                background-color: #00e676;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #999;
            }
        """)
        self.run_button.clicked.connect(self.run_solweig)

        header = QLabel("🌤 SOLWEIG GPU")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px 0; color: #ffffff;")
        self.main_layout.addWidget(header)

        self.main_layout.addWidget(self._make_divider())
        self.main_layout.addWidget(self._create_group("Initial Settings", self._general_layout()))
        self.main_layout.addWidget(self._create_group("Input Files", self._input_files_layout()))
        self.main_layout.addWidget(self._create_group("📊 Meteorological Inputs", self._met_inputs_layout()))
        self.main_layout.addWidget(self._create_group("📤 Output Options", self._output_options_layout()))

        btn_container = QHBoxLayout()
        btn_container.addStretch()
        btn_container.addWidget(self.run_button)
        btn_container.addStretch()
        self.main_layout.addLayout(btn_container)

        self.main_layout.addWidget(self.progress)
        #self.main_layout.addWidget(QLabel("Execution Log:", parent=self).setStyleSheet("color: #ffffff;"))
        log_label = QLabel("", parent=self)
        log_label.setStyleSheet("color: #ffffff;")
        self.main_layout.addWidget(log_label)
        self.main_layout.addWidget(self.log_output)

        self._monitor_required_inputs()
        self.toggle_met_selector("Own")

    def _make_divider(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: #444;")
        return line

    def _create_group(self, title, inner_layout):
        group = QGroupBox(title)
        group.setStyleSheet("color: #ffffff;")
        group.setLayout(inner_layout)
        return group

    def _help_icon(self, tooltip):
        btn = QToolButton()
        btn.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation))
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(lambda: QMessageBox.information(self, "Help", tooltip))
        return btn

    def _label_with_help(self, text, tooltip):
        layout = QHBoxLayout()
        label = QLabel(text)
        label.setStyleSheet("color: #ffffff;")
        layout.addWidget(label)
        layout.addWidget(self._help_icon(tooltip))
        layout.addStretch()
        container = QWidget()
        container.setLayout(layout)
        return container

    def _with_button(self, widget, button):
        layout = QHBoxLayout()
        layout.addWidget(widget)
        layout.addWidget(button)
        container = QWidget()
        container.setLayout(layout)
        return container

    def _with_browse(self, target, file=True):
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self.select_path(target, file))
        return self._with_button(target, btn)

    def _general_layout(self):
        layout = QFormLayout()
        self.base_path_input = QLineEdit()
        self.date_input = QDateEdit()
        self.date_input.setCalendarPopup(True)
        layout.addRow(self._label_with_help("Base Directory", "Directory where output and intermediate files will be stored."), self._with_browse(self.base_path_input, False))
        layout.addRow(self._label_with_help("Simulation Date", "Date for which thermal comfort is computed. MM/DD/YY"), self.date_input)
        return layout

    def _input_files_layout(self):
        layout = QFormLayout()
        self.building_dsm = QLineEdit()
        self.dem = QLineEdit()
        self.trees = QLineEdit()
        self.tile_size_input = QSpinBox()
        self.tile_size_input.setRange(100, 4000)
        self.tile_size_input.setValue(3600)

        layout.addRow(self._label_with_help("Building DSM", "Digital Surface Model representing building heights."), self._with_browse(self.building_dsm))
        layout.addRow(self._label_with_help("DEM", "Digital Elevation Model of the terrain."), self._with_browse(self.dem))
        layout.addRow(self._label_with_help("Trees", "Raster layer representing vegetation height."), self._with_browse(self.trees))
        layout.addRow(self._label_with_help("Tile Size", "Controls resolution of GPU processing (100–4000)."), self.tile_size_input)
        return layout

    def _met_inputs_layout(self):
        layout = QFormLayout()
        self.met_source = QComboBox()
        self.met_source.addItems([ "ERA5 (netcdf)", "WRF (netcdf)", "Metfile (txt)"])
        self.met_source.currentTextChanged.connect(self.toggle_met_selector)

        self.met_path_input = QLineEdit()
        self.met_path_button = QPushButton("Browse", clicked=self.browse_met_source)
        self.start_time = QLineEdit()
        self.start_time.setPlaceholderText("YYYY-MM-DD HH:MM:SS")
        self.end_time = QLineEdit()
        self.end_time.setPlaceholderText("YYYY-MM-DD HH:MM:SS")

        layout.addRow(self._label_with_help("Source", "Choose type of meteorological input (metfile or dynamic data)."), self.met_source)
        layout.addRow(self._label_with_help("Path", "File (metfile) or folder (ERA5/WRF) containing input data."), self._with_button(self.met_path_input, self.met_path_button))
        layout.addRow(self._label_with_help("Start Time", "Start time in format YYYY-MM-DD HH:MM:SS."), self.start_time)
        layout.addRow(self._label_with_help("End Time", "End time in format YYYY-MM-DD HH:MM:SS."), self.end_time)
        return layout

    def _output_options_layout(self):
        layout = QVBoxLayout()
        self.save_tmrt = QCheckBox("TMRT")
        self.save_svf = QCheckBox("SVF")
        self.save_kup = QCheckBox("Short wave↑")
        self.save_kdown = QCheckBox("Short wave↓")
        self.save_lup = QCheckBox("Long wave↑")
        self.save_ldown = QCheckBox("Long wave↓")
        self.save_shadow = QCheckBox("Shadow")

        for cb in [self.save_tmrt, self.save_svf, self.save_kup, self.save_kdown,
                   self.save_lup, self.save_ldown, self.save_shadow]:
            cb.setStyleSheet("color: #ffffff;")
            layout.addWidget(cb)
        return layout

    def toggle_met_selector(self, source):
        is_metfile = "Metfile" in source

        if "Metfile (txt)" in source:
            self.met_path_input.setPlaceholderText("Select the meteorological .txt file")
        elif "WRF (netcdf)" in source:
            self.met_path_input.setPlaceholderText("Select the folder for WRF data")
        else:
            self.met_path_input.setPlaceholderText("Select the folder for ERA5 data")

        # Show/hide start and end time fields
        self.start_time.setVisible(not is_metfile)
        self.end_time.setVisible(not is_metfile)


    def browse_met_source(self):
        source = self.met_source.currentText()
        if "Metfile" in source:
            path = QFileDialog.getOpenFileName(caption="Select Meteorological File", filter="Text files (*.txt)")[0]
            if path and not path.endswith(".txt"):
                QMessageBox.warning(self, "Invalid File", "Please select a valid .txt file.")
                return
        else:
            path = QFileDialog.getExistingDirectory(caption="Select Data Folder")
            if path:
                nc_files = [f for f in os.listdir(path) if f.endswith(".nc")]
                if not nc_files:
                    QMessageBox.warning(self, "No NetCDF Files", "The selected folder does not contain any .nc files.")
                    return
                self.log_output.append(f"📂 Found {len(nc_files)} NetCDF files in {path}:")
                for f in nc_files:
                    self.log_output.append(f"  • {f}")
        if path:
            self.met_path_input.setText(path)

    def _monitor_required_inputs(self):
        fields = [self.base_path_input, self.building_dsm, self.dem, self.trees, self.met_path_input, self.start_time, self.end_time]
        for field in fields:
            field.textChanged.connect(self._check_run_button)
        self.date_input.dateChanged.connect(self._check_run_button)
        self.met_source.currentIndexChanged.connect(self._check_run_button)

    def _check_run_button(self):
        required = [
            self.base_path_input.text(),
            self.building_dsm.text(),
            self.dem.text(),
            self.trees.text(),
            self.met_path_input.text()
        ]

        if self.start_time.isVisible():
            required.append(self.start_time.text())
        if self.end_time.isVisible():
            required.append(self.end_time.text())
        is_complete = all(required) and self.date_input.date().isValid()
        self.run_button.setEnabled(is_complete)
        self.run_button.setToolTip("" if is_complete else "Please complete all required fields to enable Run.")

    def run_solweig(self):
        source = self.met_source.currentText()
        use_own_met = "Metfile" in source
        data_source_type = None if use_own_met else source.split()[0]
        params = dict(
            base_path=self.base_path_input.text(),
            selected_date_str=self.date_input.date().toString("yyyy-MM-dd"),
            building_dsm_filename=self.building_dsm.text(),
            dem_filename=self.dem.text(),
            trees_filename=self.trees.text(),
            tile_size=self.tile_size_input.value(),
            use_own_met=use_own_met,
            own_met_file=self.met_path_input.text() if use_own_met else None,
            data_source_type=data_source_type,
            data_folder=self.met_path_input.text() if not use_own_met else None,
            start_time=self.start_time.text(),
            end_time=self.end_time.text(),
            save_tmrt=self.save_tmrt.isChecked(),
            save_svf=self.save_svf.isChecked(),
            save_kup=self.save_kup.isChecked(),
            save_kdown=self.save_kdown.isChecked(),
            save_lup=self.save_lup.isChecked(),
            save_ldown=self.save_ldown.isChecked(),
            save_shadow=self.save_shadow.isChecked()
        )
        self.log_output.clear()
        self.progress.setValue(0)
        self.run_button.setEnabled(False)
        self.run_button.setText("Running... ⏳")

        import pprint
        pprint.pprint(params)
        self.log_output.append(str(params))

        self.worker = SOLWEIGWorker(params)
        self.worker.log_signal.connect(self.update_log)
        self.worker.finished.connect(self.on_solweig_done)
        self.worker.start()

    def update_log(self, text):
        self.log_output.moveCursor(self.log_output.textCursor().End)
        self.log_output.insertPlainText(text)
        self.log_output.ensureCursorVisible()

        # Initialize tracking
        if not hasattr(self, "_progress_state"):
            self._progress_state = {
                "tile_count": 0,
                "metfiles_saved": 0,
                "total_tiles": None,  # Will be inferred
                "final_steps": 0,
                "steps_done": 0,
                "phase": 0  # 0: initial tiles, 1: metfile, 2: workers, 3: execution, 4: tiles executed
            }

        state = self._progress_state

        # 5% for tile creation phase
        if "Created tile:" in text:
            state["tile_count"] += 1
            self.progress.setValue(5)

        # 10% after all metfiles are saved
        if "All raster extents processed and metfiles saved" in text:
            self.progress.setValue(10)
            state["phase"] = 1

        # 30% after workers launched
        if "Running Solweig" in text:
            self.progress.setValue(30)
            state["phase"] = 2

        # Count number of final tiles for dividing the rest of the progress
        if "Processing 24 time steps for" in text:
            # Count how many final tiles to expect (metfiles = number of tiles)
            state["metfiles_saved"] += 1

        if "Using" in text and "parallel workers" in text:
            # Total tiles = number of metfiles saved
            state["total_tiles"] = state["metfiles_saved"]
            if state["total_tiles"] > 0:
                state["phase"] = 3
                state["steps_done"] = 0
                state["final_steps"] = state["total_tiles"]
            else:
                self.log_output.append("⚠️ No tiles detected for progress tracking.")

        # Final 70%: each executed tile advances progress
        if "Time taken to execute tile" in text:
            state["steps_done"] += 1
            if state["final_steps"]:
                progress_val = 30 + int((70 * state["steps_done"]) / state["final_steps"])
                self.progress.setValue(min(100, progress_val))
    
    def on_solweig_done(self):
        self.run_button.setEnabled(True)
        self.run_button.setText("✅ Completed - Run Again")
        self.progress.setValue(100)

    def load_styles(self):
        return """
        QWidget {
            background-color: #121212;
            color: #dddddd;
            font-family: 'Arial', 'Helvetica', sans-serif;
            font-size: 13px;
        }
        QGroupBox {
            font-size: 16px;
            font-weight: bold;
            margin-top: 10px;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 10px;
        }
        QPushButton {
            background-color: #2979ff;
            color: white;
            border-radius: 4px;
            padding: 6px 12px;
        }
        QPushButton:hover {
            background-color: #1565c0;
        }
        QProgressBar {
            background: #333;
            border: 1px solid #555;
            border-radius: 5px;
        }
        QProgressBar::chunk {
            background: #2979ff;
            border-radius: 5px;
        }
        QLineEdit, QComboBox, QDateEdit {
            padding: 4px;
            background: #1e1e1e;
            border: 1px solid #444;
            color: #ffffff;
            border-radius: 4px;
        }
        """


def main():
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    win = SOLWEIGApp()
    win.show()
    sys.exit(app.exec_())

