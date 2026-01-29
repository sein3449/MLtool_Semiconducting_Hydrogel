import os
import sys
import traceback

from PySide6.QtCore import QObject, Signal, Slot, QThread
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QTextEdit, QMessageBox, QTabWidget, QCheckBox
)

from hydrogel_engine_HG import train_pipeline, predict_pipeline, default_hg_config


class Worker(QObject):
    log = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, fn, kwargs):
        super().__init__()
        self.fn = fn
        self.kwargs = kwargs

    @Slot()
    def run(self):
        try:
            result = self.fn(logger=lambda m: self.log.emit(m), **self.kwargs)
            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hydrogel ML System (HG Schema)")

        self.cfg = default_hg_config()
        self.thread = None
        self.worker = None

        tabs = QTabWidget()
        tabs.addTab(self._train_tab(), "Train")
        tabs.addTab(self._predict_tab(), "Predict")
        self.setCentralWidget(tabs)

    # ---------- UI helpers ----------
    def _row(self, btn, edit):
        h = QHBoxLayout()
        h.addWidget(btn)
        h.addWidget(edit, 1)
        return h

    def _pick_xlsx(self, title):
        path, _ = QFileDialog.getOpenFileName(self, title, "", "Excel Files (*.xlsx)")
        return path

    def _pick_joblib(self, title):
        path, _ = QFileDialog.getOpenFileName(self, title, "", "Joblib Files (*.joblib)")
        return path

    def _pick_outdir(self, title, default):
        return QFileDialog.getExistingDirectory(self, title, default)

    def _set_if(self, edit: QLineEdit, path: str):
        if path:
            edit.setText(path)

    def _run_in_thread(self, fn, kwargs, log_cb, done_cb, err_cb):
        self.thread = QThread()
        self.worker = Worker(fn, kwargs)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(log_cb)
        self.worker.finished.connect(done_cb)
        self.worker.error.connect(err_cb)

        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    # ---------- Train tab ----------
    def _train_tab(self):
        w = QWidget()
        lay = QVBoxLayout()

        self.A_edit = QLineEdit()
        self.B_edit = QLineEdit()
        self.C_edit = QLineEdit()
        self.after_train_edit = QLineEdit()
        self.outdir_edit = QLineEdit(os.path.abspath("outputs_train"))

        btnA = QPushButton("Select A (Monomer) xlsx")
        btnB = QPushButton("Select B (gCP) xlsx")
        btnC = QPushButton("Select C (Solvent) xlsx")
        btnAfter = QPushButton("Select afterHG TRAIN xlsx")
        btnOut = QPushButton("Select output folder")

        self.export_enriched_train = QCheckBox("Overwrite afterHG with enriched version (formulation_id + HSP_dist)")
        self.export_enriched_train.setChecked(True)

        self.btn_train = QPushButton("RUN TRAINING")

        btnA.clicked.connect(lambda: self._set_if(self.A_edit, self._pick_xlsx("Select A (Monomer) xlsx")))
        btnB.clicked.connect(lambda: self._set_if(self.B_edit, self._pick_xlsx("Select B (gCP) xlsx")))
        btnC.clicked.connect(lambda: self._set_if(self.C_edit, self._pick_xlsx("Select C (Solvent) xlsx")))
        btnAfter.clicked.connect(lambda: self._set_if(self.after_train_edit, self._pick_xlsx("Select afterHG TRAIN xlsx")))
        btnOut.clicked.connect(lambda: self._set_if(self.outdir_edit, self._pick_outdir("Select output folder", self.outdir_edit.text())))
        self.btn_train.clicked.connect(self.run_train)

        self.log_train = QTextEdit()
        self.log_train.setReadOnly(True)
        self.log_train.setFont(self.log_train.font() if hasattr(self.log_train, 'font') else None)

        lay.addWidget(QLabel("A table (Input_Schema_beforeHG_monomer.xlsx):"))
        lay.addLayout(self._row(btnA, self.A_edit))
        lay.addWidget(QLabel("B table (Input_Schema_beforeHG_gCPs.xlsx):"))
        lay.addLayout(self._row(btnB, self.B_edit))
        lay.addWidget(QLabel("C table (Input_Schema_beforeHG_solvents.xlsx):"))
        lay.addLayout(self._row(btnC, self.C_edit))

        lay.addWidget(QLabel("AFTER train (Input_Schema_afterHG.xlsx):"))
        lay.addLayout(self._row(btnAfter, self.after_train_edit))
        lay.addWidget(QLabel("Output folder:"))
        lay.addLayout(self._row(btnOut, self.outdir_edit))

        lay.addWidget(self.export_enriched_train)
        lay.addWidget(self.btn_train)
        lay.addWidget(QLabel("Logs:"))
        lay.addWidget(self.log_train)

        w.setLayout(lay)
        return w

    def _train_log(self, msg: str):
        self.log_train.append(msg)

    def run_train(self):
        A = self.A_edit.text().strip()
        B = self.B_edit.text().strip()
        C = self.C_edit.text().strip()
        after = self.after_train_edit.text().strip()
        outdir = self.outdir_edit.text().strip()

        if not all(os.path.isfile(p) for p in [A, B, C, after]):
            QMessageBox.warning(self, "Missing files", "Select valid A/B/C before files and afterHG train file.")
            return

        os.makedirs(outdir, exist_ok=True)
        self.btn_train.setEnabled(False)
        self._train_log("=== TRAINING START ===")

        export_flag = self.export_enriched_train.isChecked()

        self._run_in_thread(
            fn=train_pipeline,
            kwargs=dict(
                A_path=A, B_path=B, C_path=C,
                after_train_path=after,
                outdir=outdir,
                cfg=self.cfg,
                export_enriched_after=export_flag
            ),
            log_cb=self._train_log,
            done_cb=self._on_train_done,
            err_cb=self._on_train_err
        )

    def _on_train_done(self, result: dict):
        self._train_log("=== TRAINING DONE ===")
        self._train_log(f"Output: {result.get('outdir')}")
        self._train_log(f"Best classifier: {result.get('best_classifier')}")
        self._train_log(f"Best regressor : {result.get('best_regressor')}")
        self._train_log(f"Rows (ML/Clf/Reg): {result.get('n_rows_ml')}/{result.get('n_rows_clf')}/{result.get('n_rows_reg')}")
        self._train_log("Saved: cv_*.csv, ml_table_merged.csv, best_*.joblib")
        if self.export_enriched_train.isChecked():
            self._train_log("✓ Enriched afterHG file saved (original file overwritten)")
        self.btn_train.setEnabled(True)

    def _on_train_err(self, tb: str):
        self._train_log("=== TRAINING ERROR ===")
        self._train_log(tb)
        QMessageBox.critical(self, "Training failed", "Check logs.")
        self.btn_train.setEnabled(True)

    # ---------- Predict tab ----------
    def _predict_tab(self):
        w = QWidget()
        lay = QVBoxLayout()

        self.A2_edit = QLineEdit()
        self.B2_edit = QLineEdit()
        self.C2_edit = QLineEdit()
        self.after_cand_edit = QLineEdit()

        self.clf_edit = QLineEdit()
        self.reg_edit = QLineEdit()
        self.out_csv_edit = QLineEdit(os.path.abspath("predictions_candidate.csv"))

        btnA = QPushButton("Select A (Monomer) xlsx")
        btnB = QPushButton("Select B (gCP) xlsx")
        btnC = QPushButton("Select C (Solvent) xlsx")
        btnAfter = QPushButton("Select afterHG CANDIDATE xlsx")

        btnClf = QPushButton("Select classifier.joblib")
        btnReg = QPushButton("Select regressor.joblib")
        btnOut = QPushButton("Select output CSV path")

        self.export_enriched_pred = QCheckBox("Overwrite candidate afterHG with enriched version")
        self.export_enriched_pred.setChecked(True)

        self.btn_pred = QPushButton("RUN PREDICTION")

        btnA.clicked.connect(lambda: self._set_if(self.A2_edit, self._pick_xlsx("Select A (Monomer) xlsx")))
        btnB.clicked.connect(lambda: self._set_if(self.B2_edit, self._pick_xlsx("Select B (gCP) xlsx")))
        btnC.clicked.connect(lambda: self._set_if(self.C2_edit, self._pick_xlsx("Select C (Solvent) xlsx")))
        btnAfter.clicked.connect(lambda: self._set_if(self.after_cand_edit, self._pick_xlsx("Select afterHG candidate xlsx")))

        btnClf.clicked.connect(lambda: self._set_if(self.clf_edit, self._pick_joblib("Select classifier.joblib")))
        btnReg.clicked.connect(lambda: self._set_if(self.reg_edit, self._pick_joblib("Select regressor.joblib")))
        btnOut.clicked.connect(self.pick_out_csv)

        self.btn_pred.clicked.connect(self.run_predict)

        self.log_pred = QTextEdit()
        self.log_pred.setReadOnly(True)

        lay.addWidget(QLabel("A/B/C tables (same versions as training):"))
        lay.addLayout(self._row(btnA, self.A2_edit))
        lay.addLayout(self._row(btnB, self.B2_edit))
        lay.addLayout(self._row(btnC, self.C2_edit))

        lay.addWidget(QLabel("afterHG candidate file (same column schema as training):"))
        lay.addLayout(self._row(btnAfter, self.after_cand_edit))

        lay.addWidget(QLabel("Load trained models (*.joblib):"))
        lay.addLayout(self._row(btnClf, self.clf_edit))
        lay.addLayout(self._row(btnReg, self.reg_edit))

        lay.addWidget(QLabel("Output predictions CSV:"))
        lay.addLayout(self._row(btnOut, self.out_csv_edit))

        lay.addWidget(self.export_enriched_pred)
        lay.addWidget(self.btn_pred)
        lay.addWidget(QLabel("Logs:"))
        lay.addWidget(self.log_pred)

        w.setLayout(lay)
        return w

    def _pred_log(self, msg: str):
        self.log_pred.append(msg)

    def pick_out_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save predictions CSV", self.out_csv_edit.text(), "CSV Files (*.csv)")
        if path:
            if not path.lower().endswith(".csv"):
                path += ".csv"
            self.out_csv_edit.setText(path)

    def run_predict(self):
        A = self.A2_edit.text().strip()
        B = self.B2_edit.text().strip()
        C = self.C2_edit.text().strip()
        after = self.after_cand_edit.text().strip()
        clf = self.clf_edit.text().strip()
        reg = self.reg_edit.text().strip()
        out_csv = self.out_csv_edit.text().strip()

        if not all(os.path.isfile(p) for p in [A, B, C, after, clf, reg]):
            QMessageBox.warning(self, "Missing files", "Select valid A/B/C, candidate afterHG, and model joblib files.")
            return

        self.btn_pred.setEnabled(False)
        self._pred_log("=== PREDICTION START ===")

        export_flag = self.export_enriched_pred.isChecked()

        self._run_in_thread(
            fn=predict_pipeline,
            kwargs=dict(
                A_path=A, B_path=B, C_path=C,
                after_candidate_path=after,
                classifier_joblib=clf,
                regressor_joblib=reg,
                out_csv=out_csv,
                cfg=self.cfg,
                export_enriched_after=export_flag
            ),
            log_cb=self._pred_log,
            done_cb=self._on_pred_done,
            err_cb=self._on_pred_err
        )

    def _on_pred_done(self, result: dict):
        self._pred_log("=== PREDICTION DONE ===")
        self._pred_log(f"Saved: {result.get('out_csv')}")
        self._pred_log(f"Rows: {result.get('n_rows')}")
        if self.export_enriched_pred.isChecked():
            self._pred_log("✓ Enriched candidate afterHG file saved (original file overwritten)")
        self.btn_pred.setEnabled(True)

    def _on_pred_err(self, tb: str):
        self._pred_log("=== PREDICTION ERROR ===")
        self._pred_log(tb)
        QMessageBox.critical(self, "Prediction failed", "Check logs.")
        self.btn_pred.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1100, 800)
    win.show()
    sys.exit(app.exec())
