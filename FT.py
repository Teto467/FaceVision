import cv2
import numpy as np
from deepface import DeepFace

# 登録済みの顔データ（名前と特徴量の辞書）
registered_faces = {}

# 類似度判定の閾値（必要に応じて調整）
threshold = 0.7

# システムに接続されたカメラを起動
cap = cv2.VideoCapture(0)
print("リアルタイム認識開始： 'r' キーで顔を登録、 'q' キーで終了")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # カメラフレームから複数の顔領域を検出（enforce_detection=Falseで顔未検出でもエラー回避）
    faces = DeepFace.extract_faces(img_path=frame, enforce_detection=False)

    # 各検出顔について
    for face_data in faces:
        face_img = face_data["face"]
        # 顔領域の位置情報（x,y,w,h）
        area = face_data["facial_area"]

        # DeepFace.representで顔画像から特徴量（embedding）を算出
        rep = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)
        if not rep:
            continue
        embedding = rep[0]["embedding"]

        # 登録済みの顔データと比較して、最も近い距離を求める
        name_found = "Unknown"
        best_distance = np.inf
        for name, reg_embedding in registered_faces.items():
            distance = np.linalg.norm(np.array(embedding) - np.array(reg_embedding))
            if distance < best_distance:
                best_distance = distance
                if distance < threshold:
                    name_found = name

        # 顔領域に矩形と名前を描画する
        x, y, w, h = area["x"], area["y"], area["w"], area["h"]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name_found, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # フレーム表示
    cv2.imshow("Real-time Face Recognition", frame)
    
    # キー入力処理
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        # 「r」キーで顔の登録を実施
        if faces:
            # 複数顔が検出される場合は、面積が最大の顔を登録対象とする
            largest_face = max(faces, key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"])
            rep = DeepFace.represent(largest_face["face"], model_name="Facenet", enforce_detection=False)
            if rep:
                embedding = rep[0]["embedding"]
                name = input("登録する名前を入力してください: ")
                registered_faces[name] = embedding
                print(f"{name} の顔を登録しました。")
        else:
            print("顔が検出されなかったため、登録できませんでした。")

cap.release()
cv2.destroyAllWindows()
