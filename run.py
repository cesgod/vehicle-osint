"""Start the app and open it in the browser:  python run.py  (run.bat does this for you)."""
import os
import socket
import threading
import webbrowser

import uvicorn


def free_port(preferred: int) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT") or free_port(8000))
    url = f"http://localhost:{port}"
    print(f"\n  Vehicle OSINT Aggregator -> {url}\n  Press Ctrl+C to stop.\n")
    if os.getenv("OPEN_BROWSER", "1") == "1":
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run("app.main:app", host=host, port=port, log_level="warning")
