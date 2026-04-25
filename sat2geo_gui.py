from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from sat2geo.locator import locate_image


ROOT = Path(__file__).resolve().parent


class Sat2GeoApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Sat2Geo Aarhus")
        self.geometry("780x520")
        self.minsize(700, 440)

        self.query_path = tk.StringVar()
        self.status = tk.StringVar(value="Choose a satellite/map crop from the Aarhus demo coverage.")

        frame = ttk.Frame(self, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(frame, text="Sat2Geo Aarhus", font=("Segoe UI", 18, "bold"))
        title.pack(anchor="w")

        chooser = ttk.Frame(frame)
        chooser.pack(fill=tk.X, pady=(16, 8))
        ttk.Entry(chooser, textvariable=self.query_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(chooser, text="Browse", command=self.choose_image).pack(side=tk.LEFT, padx=(8, 0))
        self.locate_button = ttk.Button(chooser, text="Locate", command=self.locate)
        self.locate_button.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(frame, textvariable=self.status).pack(anchor="w", pady=(0, 8))

        self.output = tk.Text(frame, wrap=tk.WORD, height=18)
        self.output.pack(fill=tk.BOTH, expand=True)

    def choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose query image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp"), ("All files", "*.*")],
        )
        if path:
            self.query_path.set(path)

    def locate(self) -> None:
        query = Path(self.query_path.get())
        if not query.exists():
            messagebox.showerror("Missing image", "Choose an existing image first.")
            return

        self.locate_button.configure(state=tk.DISABLED)
        self.status.set("Loading DINOv2 and searching the Aarhus index...")
        self.output.delete("1.0", tk.END)
        threading.Thread(target=self._locate_worker, args=(query,), daemon=True).start()

    def _locate_worker(self, query: Path) -> None:
        try:
            result = locate_image(
                query_image=query,
                db_path=ROOT / "data" / "index" / "chips.sqlite",
                index_path=ROOT / "data" / "index" / "chips_faiss.index",
                embeddings_path=ROOT / "data" / "embeddings" / "chip_embeddings.npy",
                ids_path=ROOT / "data" / "embeddings" / "chip_ids.npy",
                top_k=5,
                search_k=100,
            )
            lines: list[str] = []
            lines.append(f"Confidence: {result.confidence}")
            lines.append("")
            for match in result.matches:
                lines.append(
                    f"{match.rank:02d}. {match.lat:.7f}, {match.lon:.7f}  "
                    f"score={match.score:.6f}  method={match.method}  chip_id={match.chip_id}"
                )
                lines.append(match.google_maps_url)
            if result.fused_lat is not None and result.fused_lon is not None:
                lines.append("")
                lines.append(f"Fused coordinate: {result.fused_lat:.7f}, {result.fused_lon:.7f}")
                lines.append(f"https://www.google.com/maps?q={result.fused_lat:.7f},{result.fused_lon:.7f}")
            if result.confidence == "low":
                lines.append("")
                lines.append("Low confidence: use a larger crop, remove labels/UI overlays, or build a matching hybrid/labeled index.")
            text = "\n".join(lines) if lines else "No matches found."
            self.after(0, self._show_result, text)
        except Exception as exc:
            self.after(0, self._show_error, str(exc))

    def _show_result(self, text: str) -> None:
        self.output.insert(tk.END, text)
        self.status.set("Done.")
        self.locate_button.configure(state=tk.NORMAL)

    def _show_error(self, text: str) -> None:
        messagebox.showerror("Sat2Geo failed", text)
        self.status.set("Failed.")
        self.locate_button.configure(state=tk.NORMAL)


if __name__ == "__main__":
    Sat2GeoApp().mainloop()
