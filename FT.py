import cv2
import numpy as np
from deepface import DeepFace
import sqlite3

# データベース接続とテーブル作成
def init_db():
    conn = sqlite3.connect('face_recognition.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        embedding BLOB NOT NULL
    )
    ''')
    conn.commit()
    return conn

# ユーザー登録関数
def register_user(conn, name, embedding):
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (name, embedding) VALUES (?, ?)', (name, embedding.tobytes()))
    conn.commit()

# 登録済みユーザー取得関数
def get_registered_users(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT name, embedding FROM users')
    users = {}
    for row in cursor.fetchall():
        name, embedding_bytes = row
        users[name] = np.frombuffer(embedding_bytes, dtype=np.float64)
    return users

# メイン処理
def main():
    conn = init_db()
    cap = cv2.VideoCapture(0)
    registered_users = get_registered_users(conn)

    print("リアルタイム認識開始： 'r' キーで顔を登録、 'q' キーで終了")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = DeepFace.extract_faces(img_path=frame, enforce_detection=False)

        for face_data in faces:
            face_img = face_data["face"]
            area = face_data["facial_area"]

            rep = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)
            if not rep:
                continue
            embedding = np.array(rep[0]["embedding"])

            name_found = "Unknown"
            best_distance = np.inf
            for name, reg_embedding in registered_users.items():
                distance = np.linalg.norm(embedding - reg_embedding)
                if distance < best_distance:
                    best_distance = distance
                    if distance < 0.7:  # 閾値の調整が必要かもしれません
                        name_found = name

            x, y, w, h = area["x"], area["y"], area["w"], area["h"]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name_found, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Real-time Face Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            if faces:
                largest_face = max(faces, key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"])
                rep = DeepFace.represent(largest_face["face"], model_name="Facenet", enforce_detection=False)
                if rep:
                    embedding = np.array(rep[0]["embedding"])
                    name = input("登録する名前を入力してください: ")
                    register_user(conn, name, embedding)
                    registered_users[name] = embedding
                    print(f"{name} の顔を登録しました。")
            else:
                print("顔が検出されなかったため、登録できませんでした。")

    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    main()
