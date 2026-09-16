import json
from pathlib import Path

nb_path = Path(r"e:/2026/Project/Fruits/Kiwifruit_Allison_Smart_Irrigation_XAI_Updated_Colab (2).ipynb")
out_path = Path("nb_cells.py")

nb = json.loads(nb_path.read_text(encoding="utf-8"))

code_cells = [(i, c) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]

lines = ["# Extracted code cells from the kiwifruit notebook\n"]
for i, cell in code_cells:
    src = "".join(cell.get("source", []))
    if not src.strip():
        continue
    lines.append(f"# === CELL {i} {'=' * 60}")
    lines.append(src.rstrip("\n"))
    lines.append("")

out_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {out_path.resolve()} ({len(lines)} lines)")
print(f"Total code cells: {len(code_cells)}")
