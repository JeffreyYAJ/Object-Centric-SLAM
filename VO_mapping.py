import cv2
import numpy as np
import matplotlib.pyplot as plt

cx = 480.0

map_cars_x = []
map_cars_z = []

print("Demarrage du SLAM")

vo = VisualOdometry(K, "poses_00.txt")

frame_id = 0 
max_frames = 150

while cap.isOpened() and frame_id < max_frames:
  ret, frame = cap.read()
  if not ret: 
    break

  gray_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  vo.new_frame = gray_frame
  vo.process_frame(frame_id)
  vo.last_frame = gray_frame

  if frame_id % 10 == 0:
    results = model(frame, classes = [2,5,7], conf = 0.5,verbose = False)[0]

    for box in results.boxes:
      x1, y1, x2, y2 = map(int, box.xyxy[0])
      pixel_height = y2 - y1
      u = (x1 + x2) / 2.0
            
      z_local = (focal_length * REAL_CAR_HEIGHT) / pixel_height
      x_local = ((u - cx) * z_local) / focal_length
            
      pos_local = np.array([[x_local], [0], [z_local]])
            
      pos_global = vo.cur_t + vo.cur_R.dot(pos_local)
            
            # On sauvegarde pour l'affichage
      map_cars_x.append(pos_global[0, 0])
      map_cars_z.append(pos_global[2, 0])

    frame_id += 1

cap.release()
print("Traitement terminé ! Génération de la carte sémantique...")

# --- 3. AFFICHAGE DE LA CARTE ---
plt.figure(figsize=(10, 8))

# Affichage de notre trajectoire
plt.plot(vo.traj_x, vo.traj_z, color='blue', linewidth=2, label="Notre Trajectoire")
plt.scatter(vo.traj_x[0], vo.traj_z[0], color='green', marker='o', s=100, label="Départ", zorder=5)

# Affichage des voitures détectées sur la carte
plt.scatter(map_cars_x, map_cars_z, color='orange', marker='s', s=50, label="Voitures détectées (YOLO)", alpha=0.7)

plt.title("Carte Sémantique (Object-Centric SLAM)")
plt.xlabel("Axe X (Latéral en mètres)")
plt.ylabel("Axe Z (Profondeur en mètres)")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
