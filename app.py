from flask import Flask, request, jsonify
from deepface import DeepFace
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Konfigurasi folder upload
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Path database wajah
DB_PATH = "dataset/isal"  # Sesuaikan dengan lokasi dataset wajah

@app.route("/recognize", methods=["POST"])
def recognize():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Simpan file sementara
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        # Jalankan Face Recognition menggunakan DeepFace
        result = DeepFace.find(img_path=file_path, db_path=DB_PATH, model_name="Facenet", enforce_detection=False)

        if result and not result[0].empty:
            identity = result[0]["identity"][0]  # Ambil wajah terdekat
            person_name = os.path.basename(identity).split(".")[0]  # Ambil nama dari filename
            return jsonify({"status": "success", "recognized_as": person_name}), 200
        else:
            return jsonify({"status": "failed", "message": "Face not recognized"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Hapus file setelah diproses untuk menghemat storage
        os.remove(file_path)

if __name__ == "__main__":
    app.run(debug=True)
