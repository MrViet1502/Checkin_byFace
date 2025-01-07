


from flask import Flask, render_template, request, jsonify
from sklearn import neighbors
import cv2
import os
import time
import pickle
import face_recognition
import numpy as np

app = Flask(__name__)

# Paths for storing local files
TRAIN_DIR = "knn_folder/train/"
USER_INFO = "user_info.csv"
CHECKIN_LOG = "checkin_log.csv"
MODEL_PATH = "knn_folder/trained_knn_model.clf"

# Create directories and files if not exist
if not os.path.exists(TRAIN_DIR):
    os.makedirs(TRAIN_DIR)

if not os.path.exists(USER_INFO):
    with open(USER_INFO, "w") as f:
        f.write("name,dob,phone,img_path\n")

if not os.path.exists(CHECKIN_LOG):
    with open(CHECKIN_LOG, "w") as f:
        f.write("name,dob,phone,time\n")

# Train KNN model
def train_knn_model():
    X = []
    y = []

    for user in os.listdir(TRAIN_DIR):
        user_dir = os.path.join(TRAIN_DIR, user)
        if not os.path.isdir(user_dir):
            continue

        for img_path in os.listdir(user_dir):
            img_full_path = os.path.join(user_dir, img_path)
            image = face_recognition.load_image_file(img_full_path)
            face_locations = face_recognition.face_locations(image)

            if len(face_locations) == 1:
                X.append(face_recognition.face_encodings(image, known_face_locations=face_locations)[0])
                y.append(user)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=2, weights='distance')
    knn_clf.fit(X, y)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(knn_clf, f)

    return knn_clf

# Load KNN model
def load_knn_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return train_knn_model()

knn_model = load_knn_model()

@app.route('/')
def index():
    return render_template('index.html')

# Register new user
@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    dob = request.form['dob']
    phone = request.form['phone']
    file = request.files['file']

    if not file:
        return jsonify({"status": "error", "message": "No file uploaded"})

    # Save the image and user information
    user_dir = os.path.join(TRAIN_DIR, name)
    os.makedirs(user_dir, exist_ok=True)
    img_path = os.path.join(user_dir, f"{int(time.time())}.jpg")
    file.save(img_path)

    with open(USER_INFO, "a") as f:
        f.write(f"{name},{dob},{phone},{img_path}\n")

    # Retrain the model
    global knn_model
    knn_model = train_knn_model()

    return jsonify({"status": "success", "message": f"User {name} registered successfully."})

# Check-in using camera
@app.route('/check_in', methods=['GET'])
def check_in():
    cap = cv2.VideoCapture(0)  # Mở camera
    if not cap.isOpened():
        return jsonify({"status": "error", "message": "Could not access the camera."})

    while True:
        ret, frame = cap.read()  # Đọc khung hình từ camera
        if not ret:
            break

        # Resize và chuyển đổi màu sắc cho nhận diện
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Xác định khuôn mặt và trích xuất đặc trưng
        face_locations = face_recognition.face_locations(rgb_frame)
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for encoding, location in zip(face_encodings, face_locations):
                closest_distances = knn_model.kneighbors([encoding], n_neighbors=1)
                is_match = closest_distances[0][0][0] <= 0.4

                if is_match:
                    name = knn_model.predict([encoding])[0]

                    # Lấy thông tin người dùng
                    dob, phone = None, None
                    with open(USER_INFO, "r") as f:
                        for line in f.readlines()[1:]:
                            user_data = line.strip().split(",")
                            if user_data[0] == name:
                                dob, phone = user_data[1], user_data[2]
                                break

                    # Nếu không tìm thấy thông tin người dùng
                    if dob is None or phone is None:
                        cap.release()
                        cv2.destroyAllWindows()
                        return jsonify({"status": "error", "message": f"User {name} not found in user info."})

                    # Ghi thông tin check-in
                    with open(CHECKIN_LOG, "a") as log_file:
                        log_file.write(f"{name},{dob},{phone},{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

                    # Vẽ khung và thông báo thành công
                    top, right, bottom, left = location
                    top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, f"Checked In: {name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.imshow("Camera", frame)
                    cv2.waitKey(10000)  # Hiển thị thông báo trong 2 giây
                    cap.release()
                    cv2.destroyAllWindows()
                    return jsonify({"status": "success", "message": f"{name} checked in successfully."})

        # Hiển thị khung hình và chờ người dùng nhấn 'q' để thoát
        cv2.imshow("Camera", frame)
        if cv2.waitKey(10000) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"status": "error", "message": "No face detected. Please try again."})

# Get check-in list
@app.route('/check_in_list', methods=['GET'])
def check_in_list():
    with open(CHECKIN_LOG, "r") as log_file:
        data = log_file.readlines()
    check_in_data = [{"name": line.split(",")[0],
                      "dob": line.split(",")[1],
                      "phone": line.split(",")[2],
                      "time": line.split(",")[3].strip()} for line in data[1:]]
    return jsonify({"status": "success", "data": check_in_data})

# Delete user
@app.route('/delete', methods=['POST'])
def delete_user():
    name = request.json['name']
    user_dir = os.path.join(TRAIN_DIR, name)

    if os.path.exists(user_dir):
        for file in os.listdir(user_dir):
            os.remove(os.path.join(user_dir, file))
        os.rmdir(user_dir)

    # Remove user information from CSV
    with open(USER_INFO, "r") as f:
        lines = f.readlines()
    with open(USER_INFO, "w") as f:
        for line in lines:
            if not line.startswith(name + ","):
                f.write(line)

    # Retrain the model
    global knn_model
    knn_model = train_knn_model()

    return jsonify({"status": "success", "message": f"User {name} deleted successfully."})

if __name__ == '__main__':
    app.run(debug=True)
