import subprocess
import time
import requests
import sys

OLLAMA_URL = "http://127.0.0.1:11434"
FASTAPI_APP = "HyperLung_XR.app.main:app"
DEV_MODE = False   


def ollama_ready():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


print("Starting Autonomous Thoracic Diagnosis System...")

if not ollama_ready():
    print("Starting Ollama server...")

    # IMPORTANT: shell=False on Windows
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Wait until Ollama is actually ready
    for _ in range(30):
        if ollama_ready():
            print("Ollama is ready")
            break
        time.sleep(1)
    else:
        print("Ollama failed to start")
        sys.exit(1)
else:
    print("Ollama already running")

print("Starting FastAPI server...")

cmd = [
    "uvicorn",
    FASTAPI_APP,
    "--host", "127.0.0.1",
    "--port", "8000"
]

if DEV_MODE:
    cmd.append("--reload")

subprocess.run(cmd)
