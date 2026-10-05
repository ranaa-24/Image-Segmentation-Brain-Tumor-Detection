import base64
import io

import numpy as np
import torch
from flask import Flask, jsonify, request, send_file
from PIL import Image

from unet import UNet, preprocess

MIN_PIXELS = 50 
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

net = UNet().to(dev)
net.load_state_dict(torch.load("model.pt", map_location=dev))
net.eval()

app = Flask(__name__)


def to_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/")
def index():
    return send_file("index.html")


@app.post("/predict")
def predict():
    f = request.files.get("image")
    if f is None:
        return jsonify(error="Choose an MRI slice first."), 400
    try:
        img = Image.open(f.stream).convert("RGB").resize((256, 256))
    except Exception:
        return jsonify(error="Could not read that file. Upload a PNG, JPG or TIF slice."), 400

    with torch.no_grad():
        prob = torch.sigmoid(net(preprocess(img)[None].to(dev)))[0, 0].cpu().numpy()
    mask = prob > 0.5
    found = int(mask.sum()) >= MIN_PIXELS

    base = np.asarray(img)
    overlay = base.copy()
    if found:
        overlay[mask] = (0.45 * base[mask] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)

    return jsonify(
        found=found,
        area_pct=round(100 * float(mask.mean()), 2) if found else 0.0,
        confidence=round(float(prob[mask].mean()), 3) if found else None,
        original=to_b64(base),
        overlay=to_b64(overlay),
        mask=to_b64((mask * 255).astype(np.uint8)),
    )


if __name__ == "__main__":
    app.run(port=5000, debug=False)
