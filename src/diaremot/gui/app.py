"""Tkinter-based desktop application for the DiaRemot pipeline."""

from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from typing import Any, Dict

import importlib.util

if importlib.util.find_spec("tkinter") is None:  # pragma: no cover - environment guard
    raise RuntimeError(
        "Tkinter is required for the DiaRemot GUI; install the system tk package (e.g. "
        "python3-tk on Debian/Ubuntu)."
    )

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ..cli import (
    BUILTIN_PROFILES,
    core_build_config,
    core_diagnostics,
    core_run_pipeline,
)


class _QueueHandler(logging.Handler):
    """Logging handler that forwards log records to a tkinter-safe queue."""

    def __init__(self, event_queue: "queue.Queue[tuple[str, Any]]") -> None:
        super().__init__()
        self._queue = event_queue

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - GUI side effect
        msg = self.format(record)
        self._queue.put(("log", msg))


class DiaRemotDesktopApp:
    """Main application window for running the DiaRemot pipeline."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("DiaRemot Desktop")
        self.root.geometry("900x600")
        self.root.minsize(760, 520)

        self.event_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self._pipeline_thread: threading.Thread | None = None
        self._running = False

        logging.basicConfig(level=logging.INFO)

        self._build_ui()
        self.root.after(100, self._process_events)

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self.root, padding=16)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Input selection
        input_frame = ttk.LabelFrame(main_frame, text="Input audio")
        input_frame.pack(fill=tk.X, expand=False, pady=(0, 12))

        self.input_var = tk.StringVar()
        input_entry = ttk.Entry(input_frame, textvariable=self.input_var)
        input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 4), pady=8)

        ttk.Button(input_frame, text="Browse…", command=self._choose_input).pack(
            side=tk.LEFT, padx=8, pady=8
        )

        # Output selection
        output_frame = ttk.LabelFrame(main_frame, text="Output directory")
        output_frame.pack(fill=tk.X, expand=False, pady=(0, 12))

        self.output_var = tk.StringVar()
        output_entry = ttk.Entry(output_frame, textvariable=self.output_var)
        output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 4), pady=8)

        ttk.Button(output_frame, text="Browse…", command=self._choose_output).pack(
            side=tk.LEFT, padx=8, pady=8
        )

        # Configuration controls
        config_frame = ttk.LabelFrame(main_frame, text="Configuration")
        config_frame.pack(fill=tk.X, expand=False, pady=(0, 12))

        ttk.Label(config_frame, text="Profile").grid(row=0, column=0, sticky=tk.W, padx=8, pady=4)
        self.profile_var = tk.StringVar(value="default")
        profile_choices = list(BUILTIN_PROFILES.keys())
        self.profile_combo = ttk.Combobox(
            config_frame,
            textvariable=self.profile_var,
            values=profile_choices,
            state="readonly",
        )
        self.profile_combo.grid(row=0, column=1, sticky=tk.EW, padx=8, pady=4)

        ttk.Label(config_frame, text="Whisper model").grid(
            row=1, column=0, sticky=tk.W, padx=8, pady=4
        )
        self.whisper_model_var = tk.StringVar(value="faster-whisper-tiny.en")
        ttk.Entry(config_frame, textvariable=self.whisper_model_var).grid(
            row=1, column=1, sticky=tk.EW, padx=8, pady=4
        )

        ttk.Label(config_frame, text="Compute type").grid(
            row=2, column=0, sticky=tk.W, padx=8, pady=4
        )
        self.compute_type_var = tk.StringVar(value="float32")
        compute_values = ("float32", "float16", "int8", "int8_float16")
        ttk.Combobox(
            config_frame,
            textvariable=self.compute_type_var,
            values=compute_values,
            state="readonly",
        ).grid(row=2, column=1, sticky=tk.EW, padx=8, pady=4)

        ttk.Label(config_frame, text="Beam size").grid(row=0, column=2, sticky=tk.W, padx=8, pady=4)
        self.beam_size_var = tk.IntVar(value=1)
        ttk.Spinbox(config_frame, from_=1, to=10, textvariable=self.beam_size_var, width=5).grid(
            row=0, column=3, sticky=tk.W, padx=8, pady=4
        )

        ttk.Label(config_frame, text="ASR threads").grid(row=1, column=2, sticky=tk.W, padx=8, pady=4)
        self.asr_threads_var = tk.IntVar(value=1)
        ttk.Spinbox(config_frame, from_=1, to=16, textvariable=self.asr_threads_var, width=5).grid(
            row=1, column=3, sticky=tk.W, padx=8, pady=4
        )

        self.disable_affect_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            config_frame,
            text="Disable affect analysis",
            variable=self.disable_affect_var,
        ).grid(row=2, column=2, sticky=tk.W, padx=8, pady=4)

        self.enable_sed_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            config_frame,
            text="Enable sound event tagging",
            variable=self.enable_sed_var,
        ).grid(row=2, column=3, sticky=tk.W, padx=8, pady=4)

        self.clear_cache_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            config_frame,
            text="Clear cache before run",
            variable=self.clear_cache_var,
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=8, pady=4)

        for column in range(4):
            config_frame.columnconfigure(column, weight=1)

        # Action buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, expand=False, pady=(0, 12))

        self.run_button = ttk.Button(action_frame, text="Run pipeline", command=self._on_run)
        self.run_button.pack(side=tk.LEFT, padx=8)

        ttk.Button(action_frame, text="Diagnostics", command=self._show_diagnostics).pack(
            side=tk.LEFT, padx=8
        )

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(action_frame, textvariable=self.status_var).pack(side=tk.RIGHT, padx=8)

        # Log output
        log_frame = ttk.LabelFrame(main_frame, text="Activity log")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=12)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.log_text.configure(state=tk.DISABLED)

        # Manifest summary at bottom
        self.manifest_var = tk.StringVar(value="")
        ttk.Label(main_frame, textvariable=self.manifest_var, anchor=tk.W, justify=tk.LEFT).pack(
            fill=tk.X, expand=False, pady=(8, 0)
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _choose_input(self) -> None:
        path = filedialog.askopenfilename(title="Select audio file")
        if path:
            self.input_var.set(path)

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_var.set(path)

    def _on_run(self) -> None:
        if self._running:
            messagebox.showinfo("DiaRemot", "A pipeline run is already in progress.")
            return

        input_path = Path(self.input_var.get()).expanduser()
        output_dir = Path(self.output_var.get()).expanduser()

        if not input_path.exists():
            messagebox.showerror("DiaRemot", "Input audio path does not exist.")
            return

        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                messagebox.showerror(
                    "DiaRemot", f"Unable to create output directory: {exc}"
                )
                return

        overrides = self._build_overrides()
        clear_cache = self.clear_cache_var.get()

        self._start_pipeline_thread(str(input_path), str(output_dir), overrides, clear_cache)

    def _show_diagnostics(self) -> None:
        try:
            report = core_diagnostics()
        except Exception as exc:  # pragma: no cover - runtime failure
            messagebox.showerror("DiaRemot", f"Diagnostics failed: {exc}")
            return

        pretty = json.dumps(report, indent=2)
        messagebox.showinfo("DiaRemot diagnostics", pretty)

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    def _build_overrides(self) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {}
        profile_name = self.profile_var.get()
        if profile_name in BUILTIN_PROFILES:
            overrides.update(BUILTIN_PROFILES[profile_name])

        overrides.update(
            {
                "whisper_model": self.whisper_model_var.get().strip(),
                "compute_type": self.compute_type_var.get().strip(),
                "beam_size": int(self.beam_size_var.get()),
                "cpu_threads": int(self.asr_threads_var.get()),
                "disable_affect": self.disable_affect_var.get(),
                "enable_sed": self.enable_sed_var.get(),
            }
        )
        return overrides

    def _start_pipeline_thread(
        self, input_path: str, output_dir: str, overrides: Dict[str, Any], clear_cache: bool
    ) -> None:
        self._running = True
        self.run_button.configure(state=tk.DISABLED)
        self.status_var.set("Running pipeline…")
        self.manifest_var.set("")
        self._append_log("Starting pipeline run…")

        def worker() -> None:
            handler = _QueueHandler(self.event_queue)
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            root_logger = logging.getLogger()
            root_logger.addHandler(handler)
            previous_level = root_logger.level
            root_logger.setLevel(logging.INFO)
            try:
                config = core_build_config(overrides)
                manifest = core_run_pipeline(
                    input_path,
                    output_dir,
                    config=config,
                    clear_cache=clear_cache,
                )
            except Exception as exc:  # pragma: no cover - runtime failure
                self.event_queue.put(("error", str(exc)))
            else:
                self.event_queue.put(("result", manifest))
            finally:
                root_logger.removeHandler(handler)
                root_logger.setLevel(previous_level)

        self._pipeline_thread = threading.Thread(target=worker, daemon=True)
        self._pipeline_thread.start()

    # ------------------------------------------------------------------
    # Queue processing / logging helpers
    # ------------------------------------------------------------------
    def _process_events(self) -> None:
        try:
            while True:
                event, payload = self.event_queue.get_nowait()
                if event == "log":
                    self._append_log(payload)
                elif event == "result":
                    self._handle_success(payload)
                elif event == "error":
                    self._handle_failure(payload)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._process_events)

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _handle_success(self, manifest: Dict[str, Any]) -> None:
        self._running = False
        self.run_button.configure(state=tk.NORMAL)
        self.status_var.set("Completed successfully")
        self._append_log("Pipeline finished.")
        outputs = manifest.get("outputs", {})
        summary_path = outputs.get("summary_html") or outputs.get("csv")
        if summary_path:
            self.manifest_var.set(f"Key output: {summary_path}")
        else:
            self.manifest_var.set("Pipeline run completed; see log for details.")
        messagebox.showinfo("DiaRemot", "Pipeline run completed successfully.")

    def _handle_failure(self, error_message: str) -> None:
        self._running = False
        self.run_button.configure(state=tk.NORMAL)
        self.status_var.set("Run failed")
        self._append_log(f"Pipeline failed: {error_message}")
        messagebox.showerror("DiaRemot", f"Pipeline execution failed:\n{error_message}")

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Start the Tkinter main loop."""

        self.root.mainloop()


def main() -> None:
    """Launch the DiaRemot desktop application."""

    app = DiaRemotDesktopApp()
    app.run()


if __name__ == "__main__":  # pragma: no cover
    main()
