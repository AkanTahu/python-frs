from flask import Flask, request, jsonify
from deepface import DeepFace
import os
import cv2
import requests
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)

# Konfigurasi folder upload
UPLOAD_FOLDER = "dataset"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
RESULT_FOLDER = os.path.join(os.getcwd(), "result")
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

LARAVEL_API_URL = "http://192.168.1.8:8000/scan-faces"
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

    name = request.form.get("name")
    if not name:
        return jsonify({"error": "Name not provided"}), 400

    user_folder = os.path.join(DB_PATH, name)

    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    filename = secure_filename(file.filename)
    temp_file_path = os.path.join("dataset", filename)
    file.save(temp_file_path)

    try:
        face_path = save_face_image(temp_file_path, name)
        
        if face_path:
            final_face_path = os.path.join(user_folder, os.path.basename(face_path))
            os.rename(face_path, final_face_path)

            return jsonify({"status": "success", "message": f"Face registered as {name}", "face_path": final_face_path}), 200
        else:
            return jsonify({"status": "failed", "message": "No face detected in the image"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_file_path)

        
@app.route("/recognize", methods=["POST"])
def recognize():
    print("Received request for recognition")
    if "file" not in request.files or "username" not in request.form:
        return jsonify({"error": "File and username are required"}), 400

    file = request.files["file"]
    username = request.form["username"]
    user_id  = request.form["id"]
    
    print(f"Received file: {file}, username: {username}, user_id: {user_id}")

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    user_dataset_path = os.path.join(DB_PATH, username)

    if not os.path.exists(user_dataset_path):
        return jsonify({"status": "3"}), 404

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    
    try:
        dataset_images = [os.path.join(user_dataset_path, img) for img in os.listdir(user_dataset_path)[:3]]
        print(f"Dataset images: {dataset_images}")

        for dataset_image in dataset_images:
            print(f"Comparing with: {dataset_image}")
            result = DeepFace.verify(img1_path=file_path, img2_path=dataset_image, model_name="Facenet")
            print(f"Result: {result}")

            if result["verified"]: 
                result_image_path = os.path.join(RESULT_FOLDER, f"{username}_{filename}")
                cv2.imwrite(result_image_path, cv2.imread(file_path))
                
                status = "SUKSES"
                send_data_to_laravel(user_id, result_image_path, status)
                
                return jsonify({"status": "1"}), 200 

        status = "GAGAL"
        send_data_to_laravel(user_id, result_image_path, status)
        return jsonify({"status": "0"}), 404 

    except Exception as e:
        print(f"Error during face verification: {str(e)}") 
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            
def send_data_to_laravel(user_id, result_image_path, status):
    """Fungsi untuk mengirim data ke Laravel"""
    data = {
        "user_id": user_id,
        "image_path": result_image_path if result_image_path else None,
        "status": status
    }
    
    headers = {
        'Accept': 'application/json',
        'X-Requested-With': 'XMLHttpRequest'
    }

    try:
        # Mengirim request POST ke Laravel API
        response = requests.post(LARAVEL_API_URL, data=data, headers=headers)
        print(f"Response from Laravel: {response.text}")
        if response.status_code == 201 :
            print("Data berhasil dikirim ke Laravel")
        else:
            print(f"Failed to send data to Laravel. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error while sending data to Laravel: {str(e)}")

if __name__ == "__main__":
    # app.run(host='192.168.72.7', port=5000,debug=True)
    app.run(host='192.168.1.8', port=5000,debug=True)
