# test.py içeriği:

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt  # Plotlama için
from visual_odometry import VisualOdometry, rotation_matrix_to_euler_angles
from pathlib import Path  # Path nesnesi için


dataset_path = Path("MH_01_easy/mav0/")


def preprocess_imu_data(dataset_path, cam_to_imu_timeshift=5.63799926987e-05):
    try:
        # Verileri okuma
        imu_df = pd.read_csv(dataset_path / 'imu0/data.csv')
        groundtruth_df = pd.read_csv(dataset_path / 'state_groundtruth_estimate0/data.csv')
        
        # Nanosaniye cinsinden timeshift hesaplama
        timeshift_ns = int(cam_to_imu_timeshift * 1e9)
        
        # Groundtruth verilerindeki timestamp'leri düzeltme
        # t_imu = t_cam + shift formülüne göre
        groundtruth_df['#timestamp'] = groundtruth_df['#timestamp'].apply(
            lambda x: x + timeshift_ns
        )
        
        groundtruth_df.set_index('#timestamp', inplace=True)
        groundtruth_df.sort_index(inplace=True)
        
        # Tüm sayısal sütunları seç
        numeric_cols = groundtruth_df.select_dtypes(include=[np.number]).columns.tolist()
        # Timestamp sütununu çıkart (eğer varsa)
        numeric_cols = [col for col in numeric_cols if col != '#timestamp']
        
        # Zaman kayması düzeltilmiş timestamp'leri kullanarak interpolasyon
        for col in numeric_cols:
            if col in groundtruth_df.columns:
                imu_df[col] = np.interp(
                    imu_df['#timestamp [ns]'],
                    groundtruth_df.index.values,
                    groundtruth_df[col].values
                )
        
        output_file = dataset_path / 'imu0/imu_with_interpolated_groundtruth.csv'
        imu_df.to_csv(output_file, index=False)
        print(f"Preprocessed IMU data saved to {output_file}")
        
        # Düzeltme miktarını kontrol etmek için bilgi yazdırma
        print(f"Applied timeshift correction: {timeshift_ns} ns ({cam_to_imu_timeshift} s)")
        
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        raise


class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy,
                 k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.d = [k1, k2, p1, p2, k3]
        self.distortion = any(abs(p) > 1e-7 for p in self.d)

    def undistort_image(self, img):
        if not self.distortion:
            return img
        camera_matrix = np.array([[self.fx, 0, self.cx],
                                  [0, self.fy, self.cy],
                                  [0,       0,       1]], dtype=np.float32)
        dist_coeffs = np.array(self.d, dtype=np.float32)
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)
        return undistorted

# Euroc MAV kamera parametreleri
cam = PinholeCamera(
    width=752, height=480,
    fx=458.654, fy=457.296,
    cx=367.215, cy=248.375,
    k1=-0.28340811, k2=0.07395907, p1=0.00019359, p2=1.76187114e-05
)

# IMU verilerini ön işleme tabi tut
preprocess_imu_data(dataset_path)

# GT ve kamera verilerini okuyun
gt_data = pd.read_csv(dataset_path / "imu0/imu_with_interpolated_groundtruth.csv")
cam_data = pd.read_csv(dataset_path / "cam0/data.csv")

vo = VisualOdometry(cam, gt_data, cam_data)

# Traj çizimleri için üç ayrı görüntü oluştur
traj_xy = np.zeros((800, 800, 3), dtype=np.uint8)
traj_xz = np.zeros((800, 800, 3), dtype=np.uint8)
traj_yz = np.zeros((800, 800, 3), dtype=np.uint8)

# Plotlama için listeler
predicted_euler_list = []  # VO tahmin Euler (roll,pitch,yaw)
gt_euler_list = []         # Ground truth Euler
predicted_vel_list = []
gt_vel_list = []

# Başlangıç koordinatlarını merkezlemek için
initial_position_set = False
initial_x, initial_y, initial_z = 0, 0, 0
center = 400  # Traj çizim görüntüsü merkezi
scale = 50  # Ölçek faktörü

for idx, row in cam_data.iterrows():
    timestamp = row['#timestamp [ns]']
    img_path = dataset_path / f"cam0/data/{timestamp}.png"
    img = cv2.imread(str(img_path), 0)  # Path nesnesini stringe çevirin
    if img is None:
        print(f"Error: Could not read image at path {img_path}")
        continue

    undistorted_img = cam.undistort_image(img)
    vo.update(undistorted_img, idx)

    # İlk iki frame'i atla (VO henüz stabilize olmamış olabilir)
    if idx <= 2:
        continue

    cur_t = vo.cur_t
    x, y, z = cur_t[0][0], cur_t[1][0], cur_t[2][0]

    # İlk geçerli pozisyonu belirleyin ve merkezleyin
    if not initial_position_set:
        initial_x, initial_y, initial_z = x, y, z
        initial_position_set = True

    # -- Tahmin Euler açısı & hız --
    vo_euler = rotation_matrix_to_euler_angles(vo.cur_R)  # clamp uygulanmış R
    vo_vel = vo.cur_vel.reshape(-1)  # [vx, vy, vz]
    predicted_euler_list.append(vo_euler)
    predicted_vel_list.append(vo_vel)

    # -- Ground Truth Euler açısı & hız --
    curr_gt = gt_data[gt_data['#timestamp [ns]'] == timestamp]
    if len(curr_gt) > 0:
        curr_gt = curr_gt.iloc[0]
        # GT quaternion -> Euler (hiç clamp yok!)
        qw, qx = curr_gt[' q_RS_w []'], curr_gt[' q_RS_x []']
        qy, qz = curr_gt[' q_RS_y []'], curr_gt[' q_RS_z []']
        R_gt = vo.quaternion_to_rotation_matrix(qw, qx, qy, qz)
        gt_euler = rotation_matrix_to_euler_angles(R_gt)

        gt_vel = np.array([
            curr_gt[' v_RS_R_x [m s^-1]'],
            curr_gt[' v_RS_R_y [m s^-1]'],
            curr_gt[' v_RS_R_z [m s^-1]']
        ], dtype=np.float32)
    else:
        # timestamp eşleşmezse 0 koy
        gt_euler = np.array([0,0,0], dtype=np.float32)
        gt_vel = np.array([0,0,0], dtype=np.float32)

    gt_euler_list.append(gt_euler)
    gt_vel_list.append(gt_vel)

    # 2D traj çizimleri (merkezlenmiş ve ölçeklenmiş)
    # XY Projeksiyonu
    draw_xy_x = int((x - initial_x) * scale) + center
    draw_xy_y = int((y - initial_y) * scale) + center

    # XZ Projeksiyonu
    draw_xz_x = int((x - initial_x) * scale) + center
    draw_xz_y = int((z - initial_z) * scale) + center

    # YZ Projeksiyonu
    draw_yz_x = int((y - initial_y) * scale) + center
    draw_yz_y = int((z - initial_z) * scale) + center

    # Ground Truth Pozisyonları
    true_xy_x = int((vo.trueX - initial_x) * scale) + center
    true_xy_y = int((vo.trueY - initial_y) * scale) + center

    true_xz_x = int((vo.trueX - initial_x) * scale) + center
    true_xz_y = int((vo.trueZ - initial_z) * scale) + center

    true_yz_x = int((vo.trueY - initial_y) * scale) + center
    true_yz_y = int((vo.trueZ - initial_z) * scale) + center

    # Çizim sınırlarını kontrol edin ve gerektiğinde kaydırın
    if 0 <= draw_xy_x < traj_xy.shape[1] and 0 <= draw_xy_y < traj_xy.shape[0]:
        cv2.circle(traj_xy, (draw_xy_x, draw_xy_y), 1,
                   (int(idx * 255 / len(cam_data)), 255 - int(idx * 255 / len(cam_data)), 0), 1)
    if 0 <= true_xy_x < traj_xy.shape[1] and 0 <= true_xy_y < traj_xy.shape[0]:
        cv2.circle(traj_xy, (true_xy_x, true_xy_y), 1, (0, 0, 255), 2)

    if 0 <= draw_xz_x < traj_xz.shape[1] and 0 <= draw_xz_y < traj_xz.shape[0]:
        cv2.circle(traj_xz, (draw_xz_x, draw_xz_y), 1,
                   (int(idx * 255 / len(cam_data)), 255 - int(idx * 255 / len(cam_data)), 0), 1)
    if 0 <= true_xz_x < traj_xz.shape[1] and 0 <= true_xz_y < traj_xz.shape[0]:
        cv2.circle(traj_xz, (true_xz_x, true_xz_y), 1, (0, 0, 255), 2)

    if 0 <= draw_yz_x < traj_yz.shape[1] and 0 <= draw_yz_y < traj_yz.shape[0]:
        cv2.circle(traj_yz, (draw_yz_x, draw_yz_y), 1,
                   (int(idx * 255 / len(cam_data)), 255 - int(idx * 255 / len(cam_data)), 0), 1)
    if 0 <= true_yz_x < traj_yz.shape[1] and 0 <= true_yz_y < traj_yz.shape[0]:
        cv2.circle(traj_yz, (true_yz_x, true_yz_y), 1, (0, 0, 255), 2)

    # Koordinat bilgilerini görüntüleyin (XY Projeksiyonu için)
    cv2.rectangle(traj_xy, (10, 20), (750, 60), (0, 0, 0), -1)
    text_xy = f"XY Coord: x={x:.2f} y={y:.2f}"
    cv2.putText(traj_xy, text_xy, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

    # Koordinat bilgilerini görüntüleyin (XZ Projeksiyonu için)
    cv2.rectangle(traj_xz, (10, 20), (750, 60), (0, 0, 0), -1)
    text_xz = f"XZ Coord: x={x:.2f} z={z:.2f}"
    cv2.putText(traj_xz, text_xz, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

    # Koordinat bilgilerini görüntüleyin (YZ Projeksiyonu için)
    cv2.rectangle(traj_yz, (10, 20), (750, 60), (0, 0, 0), -1)
    text_yz = f"YZ Coord: y={y:.2f} z={z:.2f}"
    cv2.putText(traj_yz, text_yz, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

    # Pencereleri güncelleyin
    cv2.imshow('Trajectory XY', traj_xy)
    cv2.imshow('Trajectory XZ', traj_xz)
    cv2.imshow('Trajectory YZ', traj_yz)
    cv2.imshow('Camera', undistorted_img)
    cv2.waitKey(1)

cv2.imwrite('map_xy.png', traj_xy)
cv2.imwrite('map_xz.png', traj_xz)
cv2.imwrite('map_yz.png', traj_yz)
cv2.destroyAllWindows()

# --- Plotlama ---
pred_euler = np.array(predicted_euler_list)  # shape (N, 3), radyan
gt_euler = np.array(gt_euler_list)           # shape (N, 3), radyan
pred_vel = np.array(predicted_vel_list)      # shape (N, 3)
gt_vel = np.array(gt_vel_list)               # shape (N, 3)

# Açıların -180..180 bandında anlık atlama yapmaması için "unwrap" (sadece plot için)
# Her ekseni ayrı ayrı unwrap et (radyan cinsinden).
pred_euler_unwrapped = pred_euler.copy()
gt_euler_unwrapped = gt_euler.copy()

for i in range(3):
    pred_euler_unwrapped[:, i] = np.unwrap(pred_euler_unwrapped[:, i])
    gt_euler_unwrapped[:, i] = np.unwrap(gt_euler_unwrapped[:, i])

# Radyan -> derece
pred_euler_deg = np.rad2deg(pred_euler_unwrapped)
gt_euler_deg = np.rad2deg(gt_euler_unwrapped)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("VO vs Ground Truth: Euler Açıları ve Hız Karşılaştırması", fontsize=16)

# Euler açıları (roll, pitch, yaw)
titles_euler = ['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']
for i in range(3):
    axs[0, i].plot(pred_euler_deg[:, i], label='Tahmin', color='blue')
    axs[0, i].plot(gt_euler_deg[:, i], label='Gerçek', color='orange')
    axs[0, i].set_title(titles_euler[i])
    axs[0, i].legend()
    axs[0, i].grid(True)

# Hızlar (vx, vy, vz)
titles_vel = ['Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)']
for i in range(3):
    axs[1, i].plot(pred_vel[:, i], label='Tahmin', color='green')
    axs[1, i].plot(gt_vel[:, i], label='Gerçek', color='red')
    axs[1, i].set_title(titles_vel[i])
    axs[1, i].legend()
    axs[1, i].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
