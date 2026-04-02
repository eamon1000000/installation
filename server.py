import threading
from flask import Flask, render_template, jsonify, send_from_directory

app = Flask(__name__)

# ── Shared state ───────────────────────────────────────────────────────────────
# Written by the pipeline thread, read by Flask route handlers.
_state = {
    "status":      "idle",       # idle | processing | done | error
    "message":     "Press SPACE in the camera window to capture.",
    "detections":  [],           # [{label, confidence, area}, ...]
    "gif_url":     None,
    "capture_url": None,
}
_lock = threading.Lock()


def get_state():
    with _lock:
        return dict(_state)


def update_state(**kwargs):
    with _lock:
        _state.update(kwargs)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    return jsonify(get_state())


@app.route("/static/output/<path:filename>")
def output_file(filename):
    return send_from_directory("static/output", filename)


@app.route("/static/input/<path:filename>")
def input_file(filename):
    return send_from_directory("static/input", filename)


def run_server(host="0.0.0.0", port=5000):
    app.run(host=host, port=port, debug=True, use_reloader=False)
