from pathlib import Path

source = Path("app.py").read_text(encoding="utf-8")
marker = 'APP_CSS = """'

if marker not in source:
    raise SystemExit("Could not find the expected section in app.py.")

core = source.split(marker, 1)[0]
core = core.replace("import gradio as gr\n", "")
core = core.replace("gr.Error(", "ValueError(")

Path("weather_morph_core.py").write_text(core, encoding="utf-8")
print("Created weather_morph_core.py")
