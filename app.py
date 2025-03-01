from flask import Flask, request, jsonify
from deepface import DeepFace
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)

# Konfigurasi folder upload
UPLOAD_FOLDER = "dataset"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Path database wajah
# DB_PATH = "dataset/isal"  # Sesuaikan dengan lokasi dataset wajah
DB_PATH = "dataset"  # Sesuaikan dengan lokasi dataset wajah

def save_face_image(image_path, name):
    # Baca gambar menggunakan OpenCV
    img = cv2.imread(image_path)

    # Mendeteksi wajah di gambar
    # result = DeepFace.detectFace(img, detector_backend='opencv')
    
    # Jika wajah ditemukan, simpan wajah tersebut
    if img is not None:
        # Nama file berdasarkan timestamp dan nama pengguna
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        face_filename = f"{name}_{timestamp}.jpg"
        face_path = os.path.join(DB_PATH, face_filename)

        # Simpan gambar wajah ke folder dataset
        cv2.imwrite(face_path, img)
        return face_path
    return None

@app.route("/register", methods=["POST"])
def register():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Ambil nama pengguna dari parameter form
    name = request.form.get("name")
    if not name:
        return jsonify({"error": "Name not provided"}), 400

    # Tentukan path folder untuk nama pengguna
    user_folder = os.path.join(DB_PATH, name)

    # Jika folder untuk nama pengguna belum ada, buat folder baru
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    # Simpan file sementara
    filename = secure_filename(file.filename)
    file_path = os.path.join(user_folder, filename)
    file.save(file_path)

    # Simpan wajah yang terdeteksi ke dataset
    try:
        face_path = save_face_image(file_path, name)
        if face_path:
            return jsonify({"status": "success", "message": f"Face registered as {name}", "face_path": face_path}), 200
        else:
            return jsonify({"status": "failed", "message": "No face detected in the image"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Hapus file sementara setelah diproses
        os.remove(file_path)
        
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
    app.run(host='192.168.1.10', port=5000,debug=True)
