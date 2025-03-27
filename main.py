import os
import tempfile
import requests
import json

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import functions_framework
from google.cloud import storage
from flask import jsonify, Request

BUCKET_NAME = "trash_check_ai"
BLOB_PATH = "model/stocker.pth"

# 前処理
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model_from_gcs():
    # GCS からモデルをダウンロード & ロード
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(BLOB_PATH)

    temp_dir = tempfile.gettempdir()
    model_local_path = os.path.join(temp_dir, "stocker.pth")
    blob.download_to_filename(model_local_path)

    # ResNet18 の最終層を2クラスに
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_local_path, map_location="cpu"))
    model.eval()
    return model

# モデルロード
model = load_model_from_gcs()

# クラス名 (学習時と同じ順番)
class_names = ['ストッカー×', 'ストッカー〇']

@functions_framework.http
def predict(request: Request):
    if request.method != "POST":
        return jsonify({"error": "Only POST is allowed"}), 405

    req_json = request.get_json(silent=True)
    if not req_json or "urls" not in req_json:
        return jsonify({"error": "No 'urls' field in JSON"}), 400

    # "urls"フィールドに複数の画像URLが入っている想定
    image_urls = req_json["urls"]
    if not isinstance(image_urls, list):
        return jsonify({"error": "'urls' must be an array"}), 400

    results = []
    for url in image_urls:
        # 1. 画像をダウンロード
        resp = requests.get(url)
        if resp.status_code != 200:
            # エラー時は、結果に "error" として格納しておく
            results.append({"url": url, "error": f"Download failed (status={resp.status_code})"})
            continue

        # 2. PILで開く
        from io import BytesIO
        image = Image.open(BytesIO(resp.content)).convert("RGB")

        # 3. 前処理
        input_tensor = inference_transform(image).unsqueeze(0)

        # 4. 推論
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()

        # 5. 結果を配列に追加
        result_label = class_names[class_idx]
        results.append({
            "url": url,
            "result": result_label
        })

        print("[INFO] === Trash_stocker_id_AI: End ===\n")

    # まとめて JSON で返す
    return jsonify({
        "results": results
    }), 200
