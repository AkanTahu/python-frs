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
    temp_file_path = os.path.join("dataset", filename)  # Tempat sementara untuk menyimpan file
    file.save(temp_file_path)

    # Simpan wajah yang terdeteksi ke folder yang sesuai
    try:
        # Baca gambar yang disimpan sementara
        face_path = save_face_image(temp_file_path, name)
        
        if face_path:
            # Pindahkan file yang sudah diproses ke folder yang sesuai berdasarkan nama pengguna
            final_face_path = os.path.join(user_folder, os.path.basename(face_path))
            os.rename(face_path, final_face_path)  # Memindahkan file ke folder yang sesuai

            return jsonify({"status": "success", "message": f"Face registered as {name}", "face_path": final_face_path}), 200
        else:
            return jsonify({"status": "failed", "message": "No face detected in the image"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Hapus file sementara setelah diproses
        os.remove(temp_file_path)

        
@app.route("/recognize", methods=["POST"])
def recognize():
    print("Received request for recognition")
    if "file" not in request.files or "username" not in request.form:
        return jsonify({"error": "File and username are required"}), 400

    file = request.files["file"]
    username = request.form["username"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Define user-specific dataset path
    user_dataset_path = os.path.join(DB_PATH, username)

    # Check if the user's dataset exists
    if not os.path.exists(user_dataset_path):
        return jsonify({"status": "3"}), 404

    # Save the uploaded file temporarily
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    
    try:
        # Ambil semua gambar di dataset pengguna (misalnya 3 gambar pertama)
        dataset_images = [os.path.join(user_dataset_path, img) for img in os.listdir(user_dataset_path)[:3]]  # Mengambil 3 gambar pertama
        print(f"Dataset images: {dataset_images}")

        # Loop untuk memeriksa gambar input dengan setiap gambar di dataset
        for dataset_image in dataset_images:
            print(f"Comparing with: {dataset_image}")
            result = DeepFace.verify(img1_path=file_path, img2_path=dataset_image, model_name="Facenet")
            print(f"Result: {result}")  # Cek hasil dari verify()

            if result["verified"]:  # Jika ada gambar yang cocok
                print("Wajah terdeteksi")
                return jsonify({"status": "1"}), 200  # Wajah terdeteksi, kembalikan dengan dictionary

        # Jika tidak ada gambar yang cocok
        return jsonify({"status": "0"}), 404  # Wajah tidak dikenali

    except Exception as e:
        print(f"Error during face verification: {str(e)}")  # Print error detail
        return jsonify({"error": str(e)}), 500
    finally:
        # Remove the temporary uploaded file to save space
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    # app.run(host='192.168.72.7', port=5000,debug=True)
    app.run(host='192.168.1.8', port=5000,debug=True)
