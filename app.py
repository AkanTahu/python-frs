from flask import Flask, request, jsonify
from deepface import DeepFace
import os
import cv2
import requests
import numpy as np
import time
# import openpyxl
import pandas as pd
from werkzeug.utils import secure_filename
from datetime import datetime


app = Flask(__name__)


# Konfigurasi folder dataset wajah dan hasil scan di Laravel
BASE_PYTHON_STORAGE = os.path.abspath("./testing")
BASE_LARAVEL_STORAGE = os.path.abspath("../rekachain-web/storage/app/public")
BASE_SHARED = "/shared-storage"
DB_PATH = os.path.join(BASE_SHARED, "dataset_faces")
RESULT_FOLDER = os.path.join(BASE_SHARED, "result_scan_faces")

# Pastikan folder dataset_faces dan result_scan_faces ada
os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = DB_PATH
LARAVEL_API_URL = "http://192.168.1.7/scan-faces"
# LARAVEL_API_URL = "http://192.168.76.52/scan-faces"

DeepFace.build_model('Facenet')

def save_face_image(image_path, nip):
    # Baca gambar menggunakan OpenCV
    img = cv2.imread(image_path)

    if img is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        face_filename = f"{nip}_{timestamp}.jpg"
        face_path = os.path.join(DB_PATH, nip, face_filename)

        # Pastikan folder user ada
        os.makedirs(os.path.dirname(face_path), exist_ok=True)

        cv2.imwrite(face_path, img)
        return face_path
    return None

@app.route("/frs/register", methods=["POST"])
def register():
    start_time_reg = time.time()
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 401

    nip = request.form.get("nip")
    if not nip:
        return jsonify({"error": "Name not provided"}), 402

    user_folder = os.path.join(DB_PATH, nip)

    user_folder = os.path.join(DB_PATH, nip)
    os.makedirs(user_folder, exist_ok=True)  # Pastikan folder user ada

    filename = secure_filename(file.filename)
    temp_file_path = os.path.join(DB_PATH, filename)
    file.save(temp_file_path)

    try:
        face_path = save_face_image(temp_file_path, nip)
        
        if face_path:
            final_face_path = os.path.join(user_folder, os.path.basename(face_path))
            os.rename(face_path, final_face_path)
            
            return jsonify({
                "status": "success",
                "message": f"Face registered as {nip}",
                "face_path": final_face_path
            }), 200
        else:
            return jsonify({"status": "failed", 
                            "message": "No face detected in the image"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
        end_time_reg = time.time()
        detection_time_reg = end_time_reg - start_time_reg
        log_to_excel_generate(nip, detection_time_reg)    

        
@app.route("/frs/recognize", methods=["POST"])
def recognize():
    start_time_recog = time.time()
    print("Received request for recognition")
    if "file" not in request.files or "nip" not in request.form:
        return jsonify({"error": "File and nip are required"}), 400

    file = request.files["file"]
    nip = request.form["nip"]
    user_id  = request.form["id"]
    panel  = request.form["panel"]
    kpm  = request.form["kpm"]
    
    print(f"Received file: {file}, nip: {nip}, user_id: {user_id}, panel: {panel}, kpm: {kpm}")

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    user_dataset_path = os.path.join(DB_PATH, nip)

    if not os.path.exists(user_dataset_path):
        return jsonify({"status": "2"}), 200

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    
    try:
        dataset_images = [os.path.join(user_dataset_path, img) for img in os.listdir(user_dataset_path)[:3]]
        print(f"Dataset images: {dataset_images}")

        for dataset_image in dataset_images:
            print(f"Comparing with: {dataset_image}")
            result = DeepFace.verify(img1_path=file_path, img2_path=dataset_image, model_name="Facenet", enforce_detection=False)
            print(f"Result: {result}")

            if result["verified"]: 
                result_filename = f"{nip}_{filename}"
                result_image_path = os.path.join(RESULT_FOLDER, result_filename)
                cv2.imwrite(result_image_path, cv2.imread(file_path))
                
                status = "SUKSES"
                send_data_to_laravel(user_id, result_filename, status, panel, kpm)
                
                end_time_recog = time.time()
                detection_time_recog = end_time_recog - start_time_recog
                
                return jsonify({"status": "1"}), 200 
            else:
                result_filename = f"{nip}_{filename}"
                result_image_path = os.path.join(RESULT_FOLDER, result_filename)
                cv2.imwrite(result_image_path, cv2.imread(file_path))
                        
                status = "GAGAL"
                send_data_to_laravel(user_id, result_filename, status, panel, kpm)
                
                return jsonify({"status": "0"}), 200 

    except Exception as e:
        print(f"Error during face verification: {str(e)}") 
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        
        end_time_recog = time.time()
        detection_time_recog = end_time_recog - start_time_recog
        log_to_excel_recognition(nip, detection_time_recog, status)
            
def send_data_to_laravel(user_id, result_image_path, status, panel, kpm):
    """Fungsi untuk mengirim data ke Laravel"""
    data = {
        "user_id": user_id,
        "image_path": result_image_path if result_image_path else None,
        "status": status,
        "panel": panel,
        "kpm": kpm
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

def log_to_excel_generate(nip, detection_time):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = {"nip": nip, "detection_time": detection_time, "created_at": now}
    
    excel_path = os.path.join(BASE_PYTHON_STORAGE, "generate_face_log.xlsx")
    
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        new_df = pd.DataFrame([new_row])  # Convert new_row to DataFrame
        df = pd.concat([df, new_df], ignore_index=True)  # Use concat instead of append
    else:
        df = pd.DataFrame([new_row])

    df.to_excel(excel_path, index=False)

def log_to_excel_recognition(nip, detection_time, status):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = {"nip": nip, "detection_time": detection_time, "created_at": now, "status": status}
    
    excel_path = os.path.join(BASE_PYTHON_STORAGE, "recognition_face_log.xlsx")
    
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        new_df = pd.DataFrame([new_row])  # Convert new_row to DataFrame
        df = pd.concat([df, new_df], ignore_index=True)  # Use concat instead of append
    else:
        df = pd.DataFrame([new_row])

    df.to_excel(excel_path, index=False)
    
if __name__ == "__main__":
    print("Running Flask in development mode")
    app.run(debug=True, host='0.0.0.0', port=5000)
