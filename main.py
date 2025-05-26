# from fastapi import FastAPI, File, UploadFile, HTTPException
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import io
# import sqlite3
# import datetime

# app = FastAPI()

# model = tf.keras.models.load_model('backend2/resnet_backend.h5')
# class_names = ['Bjorn_Necondition', 'Bjorn_Nonstandard', 'Bjorn_Standard', 'Valigora_Necondition', 'Valigora_Nonstandard', 'Valigora_Standard', 'Meva_Necondition', 'Meva_Nonstandard', 'Meva_Standard', 'Bjorn_Util', 'Valigora_Util', 'Meva_Util', 'Hevioso_Util', 'Hevioso_Nonstandard', 'Hevioso_Standard', 'Hevioso_Necondition', 'Sanfrendo_Standard', 'Sanfrendo_Nonstandart', 'Sanfrendo_Necondition', 'Sanfrendo_Util']

# # Функция предсказания
# def predict_image(image: Image.Image):
#     image = image.resize((224, 224))  # Изменение размера под модель
#     image_array = np.array(image) # /255
#     image_array = np.expand_dims(image_array, axis=0)  # Добавляем batch dimension
#     predictions = model.predict(image_array)
#     predicted_class = class_names[np.argmax(predictions)]
#     return image_array.flatten(), predicted_class

# # Логирование в SQLite
# def log_to_db(image_array, predicted_class, com):
#     try:
#         conn = sqlite3.connect("logs.db")
#         cursor = conn.cursor()
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS logs (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 timestamp TEXT,
#                 predicted_class TEXT,
#                 image_array TEXT,
#                 comment TEXT
#             )
#         """)
#         cursor.execute("INSERT INTO logs (timestamp, predicted_class, image_array, comment) VALUES (?, ?, ?, ?)",
#                     (datetime.datetime.now().isoformat(), predicted_class, str(image_array.tolist()), com[0]))
#         conn.commit()
#         conn.close()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Ошибка логирования в БД: {str(e)}")


# @app.post("/predict")
# async def predict(com: str = '', file: UploadFile = File(...)):
#     image = Image.open(io.BytesIO(await file.read()))
#     image_array, predicted_class = predict_image(image)
#     log_to_db(image_array, predicted_class, com)
#     return {"predicted_class": predicted_class}




# # @app.post("/file/upload-file")
# # def upload_file(file: UploadFile):
# #   return file

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

from pydantic import BaseModel
import bcrypt
from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import sqlite3
import datetime
from ultralytics import YOLO
import requests
import cv2

api_url = "http://kb.ai-hippocrates.ru:8886/editor-solver/solver/key-value-request/72"
api_headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}

app = FastAPI()

# Модель для пользователя
class User(BaseModel):
    username: str
    password: str

# Функция для проверки пользователя в БД
def check_user_in_db(user: User):
    try:
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        # Проверяем, есть ли пользователь с таким логином
        cursor.execute("SELECT password, role FROM users WHERE username = ?", (user.username,))
        result = cursor.fetchone()
        
        conn.close()

        # Если пользователь не найден, возвращаем False
        if result is None:
            return False, None

        # Сравниваем хешированный пароль с введенным
        stored_password, role = result
        if bcrypt.checkpw(user.password.encode('utf-8'), stored_password.encode('utf-8')):
            return True, role  # Пароль совпал, возвращаем True и роль
        else:
            return False, None  # Неверный пароль

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка работы с БД: {str(e)}")

# Метод для регистрации пользователя с хешированием пароля
def register_user(user: User):
    try:
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        # Хешируем пароль перед добавлением в базу данных
        hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Вставляем нового пользователя в базу данных
        cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                       (user.username, hashed_password, 'worker'))  # Роль по умолчанию - worker
        conn.commit()
        conn.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка регистрации пользователя: {str(e)}")

@app.post("/login")
async def login(user: User):
    # Проверяем наличие пользователя в базе данных
    is_valid, role = check_user_in_db(user)
    if is_valid:
        return {"message": "Login successful", "role": role}
    else:
        raise HTTPException(status_code=400, detail="Invalid username or password")

@app.post("/register")
async def register(user: User):
    # Регистрируем нового пользователя
    register_user(user)
    return {"message": "User registered successfully"}


# --------------------


yolo = YOLO('backend2/best.pt')

model_pimples = tf.keras.models.load_model('backend2/resnet_pimples.h5')
class_names_pimples = [0, 2, 1]

model_shape = tf.keras.models.load_model('backend2/resnet_shape.h5')
class_names_shape = ['Кривой', 'Ровный', 'Немного_кривой']

model_spots = tf.keras.models.load_model('backend2/resnet_spots.h5')
class_names_spots = ['отсутствуют', 'светлые']

model_tom_color = tf.keras.models.load_model('backend2/resnet_tom_color.h5')
class_names_tom_color = ['red', 'green']

model_tom_spots = tf.keras.models.load_model('backend2/resnet_tom_spots.h5')
class_names_tom_spots = ['темные', 'светлые', 'отсутствуют']

# функция обрезки
def crop(image: Image.Image) -> list[np.ndarray] | None:
    if image is None:
        print("Ошибка: не удалось загрузить изображение.")
        return None

    results = yolo(image)[0]
    image_np = np.array(image)
    cropped_images = []

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        crop_np = image_np[y1:y2, x1:x2]
        resized = cv2.resize(crop_np, (224, 224))
        batched = np.expand_dims(resized, axis=0)
        cropped_images.append(batched)
    
    # cropped = cropped.resize((224, 224))  # Изменение размера под модель
    # image_array = np.array(image_np) # /255
    # image_array = np.expand_dims(image_array, axis=0)  # Добавляем batch dimension

    return cropped_images[0]

# Функция предсказания
def predict_image(image: Image.Image, length: int = 0):

    # томат 1, огурец 0
    results = yolo(image, conf=0.5)
    result = results[0]

    predicted_classes = result.boxes.cls.cpu().tolist()

    if not predicted_classes:
        return "Плод не обнаружен"
    # томат 1, огурец 0
    elif predicted_classes[0] == 0:
        image_array = crop(image)
        
        predictions_pimples = int(np.argmax(model_pimples.predict(image_array)))
        predictions_shape = int(np.argmax(model_shape.predict(image_array)))
        predictions_spots = int(np.argmax(model_spots.predict(image_array)))

        data_sort = {
            "vegetable": "cucumber",
            "opt_length": length,
            "cucumber_texture": class_names_pimples[predictions_pimples]
        }

        data_q = {
            "vegetable": "cucumber",
            "spots": class_names_spots[predictions_spots],
            "Кривизна": class_names_shape[predictions_shape]
        }
    else:
        image_array = crop(image)

        predictions_tom_color = int(np.argmax(model_tom_color.predict(image_array)))
        predictions_tom_spots = int(np.argmax(model_tom_spots.predict(image_array)))

        data_sort = {
            "vegetable": "tomato",
            "diameter_of_tomatoes": length,
        }

        data_q = {
            "vegetable": "tomato",
            "spots": class_names_tom_spots[predictions_tom_spots],
            "color": class_names_tom_color[predictions_tom_color]
        }

    # print('#####################################################')
    # print(data_sort)
    # print('#####################################################')
    # print('#####################################################')
    # print(data_q)
    # print('#####################################################')


    try:
        response_sort = requests.post(api_url, headers=api_headers, json=data_sort, timeout=10)
        response_sort.raise_for_status()  # вызывает исключение при коде 4xx/5xx
        response_q = requests.post(api_url, headers=api_headers, json=data_q, timeout=10)
        response_q.raise_for_status()  # вызывает исключение при коде 4xx/5xx
        return image_array.flatten(), response_sort.json(), response_q.json()
    except requests.exceptions.HTTPError as e:
        print("HTTP error:", e.response.status_code, e.response.text)
    except requests.exceptions.Timeout:
        print("Request timed out")
    except requests.exceptions.RequestException as e:
        print("Request failed:", str(e))
    return None


    


# Логирование в SQLite
def log_to_db(image_array, predicted_sort, predicted_quality, com):
    try:
        conn = sqlite3.connect("logs.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                predicted_sort TEXT,
                predicted_quality TEXT,
                image_array TEXT,
                comment TEXT
            )
        """)
        cursor.execute("INSERT INTO logs (timestamp, predicted_class, image_array, comment) VALUES (?, ?, ?, ?)",
                    (datetime.datetime.now().isoformat(), predicted_sort, predicted_quality, str(image_array.tolist()), com[0]))
        conn.commit()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка логирования в БД: {str(e)}")


@app.post("/predict")
async def predict(file: UploadFile = File(...), length: int = 0):
    image = Image.open(io.BytesIO(await file.read()))
    image_array, predicted_sort, predicted_quality = predict_image(image, length)
    return {"predicted_sort": predicted_sort, "predicted_quality": predicted_quality, "image_array":str(image_array.tolist())}

@app.post("/log_comment")
async def log_comment(com: str = '', predicted_sort: str = '', predicted_quality: str = '', image_array: list = None):
    log_to_db(image_array, predicted_sort, predicted_quality, com)
    return {"predicted_sort": predicted_sort, "predicted_quality": predicted_quality, "image_array":str(image_array.tolist())}

# http://192.168.0.232:8000/docs#/default/predict_predict_post


# {
#   "vegetable": "tomato",
#   "diameter_of_tomatoes":6,
#   "spots":"темные",
#   "color":"red"
# }