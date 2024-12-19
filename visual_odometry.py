import numpy as np
import cv2

# Aşamalar
STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 2500

# LK parametreleri
lk_params = dict(winSize=(5, 5),
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# Body-sensor dönüşüm matrisi (Euroc MAV için T_BS)
T_BS = np.array([
    [ 0.01517066, -0.99983694,  0.00979558, -0.01638528],
    [ 0.99965712,  0.01537559,  0.02119505, -0.06812726],
    [-0.02134221,  0.00947067,  0.99972737,  0.00395795],
    [ 0.0,          0.0,         0.0,         1.0        ]
])

def featureTracking(image_ref, image_cur, px_ref):
    """Önceki karedeki özellik noktalarını yeni karede takip eder."""
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)
    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]
    return kp1, kp2

def rotation_matrix_to_euler_angles(R):
    """Rotasyon matrisinden ZYX (roll, pitch, yaw) [rad] döndürür."""
    sy = np.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2,1], R[2,2])  # Rx
        pitch = np.arctan2(-R[2,0], sy)    # Ry
        yaw = np.arctan2(R[1,0], R[0,0])   # Rz
    else:
        roll = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = 0

    return np.array([roll, pitch, yaw], dtype=np.float64)

def euler_angles_to_rotation_matrix(euler):
    """Euler açılarını (roll, pitch, yaw) -> rotasyon matrisine dönüştürür (ZYX sırası)."""
    roll, pitch, yaw = euler
    Rz = np.array([
        [ np.cos(yaw), -np.sin(yaw), 0],
        [ np.sin(yaw),  np.cos(yaw), 0],
        [          0,           0,   1]
    ], dtype=np.float64)

    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [             0, 1,             0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ], dtype=np.float64)

    Rx = np.array([
        [1,           0,            0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ], dtype=np.float64)

    return Rz @ Ry @ Rx

def clamp_euler_angles(euler_old, euler_new, max_deg=5.0):
    """
    Tahmin Euler açı sürekliliği için clamp.
    Ground Truth açılarına KESİNLİKLE uygulanmaz!
    """
    max_rad = np.deg2rad(max_deg)
    diff = euler_new - euler_old

    # -pi..pi normalizasyonu
    diff = (diff + np.pi) % (2*np.pi) - np.pi

    clamped = euler_old.copy()
    for i in range(3):
        if abs(diff[i]) <= max_rad:
            clamped[i] = euler_new[i]
        else:
            if diff[i] > 0:
                clamped[i] = euler_old[i] + max_rad
            else:
                clamped[i] = euler_old[i] - max_rad
    return clamped

class VisualOdometry:
    def __init__(self, cam, gt_data, cam_data):
        self.frame_stage = STAGE_FIRST_FRAME
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        
        first_gt = gt_data.iloc[0]
        self.cur_t = np.array([
            first_gt[' p_RS_R_x [m]'],
            first_gt[' p_RS_R_y [m]'],
            first_gt[' p_RS_R_z [m]']
        ]).reshape(3,1)

        # Quaternion -> rotasyon matrisi (IMU frame)
        q_w = first_gt[' q_RS_w []']
        q_x = first_gt[' q_RS_x []']
        q_y = first_gt[' q_RS_y []']
        q_z = first_gt[' q_RS_z []']
        R_imu = self.quaternion_to_rotation_matrix(q_w, q_x, q_y, q_z)

        self.cur_R = R_imu.copy()
        self.prev_euler = rotation_matrix_to_euler_angles(self.cur_R)

        self.cur_vel = np.array([
            first_gt[' v_RS_R_x [m s^-1]'],
            first_gt[' v_RS_R_y [m s^-1]'],
            first_gt[' v_RS_R_z [m s^-1]']
        ]).reshape(3, 1)

        self.px_ref = None
        self.px_cur = None
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)

        self.trueX, self.trueY, self.trueZ = 0, 0, 0

        self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        self.gt_data = gt_data
        self.cam_data = cam_data

    def quaternion_to_rotation_matrix(self, w, x, y, z):
        R = np.zeros((3,3))
        R[0,0] = 1 - 2*(y*y + z*z)
        R[0,1] = 2*(x*y - z*w)
        R[0,2] = 2*(x*z + y*w)

        R[1,0] = 2*(x*y + z*w)
        R[1,1] = 1 - 2*(x*x + z*z)
        R[1,2] = 2*(y*z - x*w)

        R[2,0] = 2*(x*z - y*w)
        R[2,1] = 2*(y*z + x*w)
        R[2,2] = 1 - 2*(x*x + y*y)
        return R

    def getAbsoluteScale(self, frame_id):
        if frame_id < 1:
            return 0
        curr_timestamp = self.cam_data.iloc[frame_id]['#timestamp [ns]']
        prev_timestamp = self.cam_data.iloc[frame_id - 1]['#timestamp [ns]']
        
        curr_gt = self.gt_data[self.gt_data['#timestamp [ns]'] == curr_timestamp]
        prev_gt = self.gt_data[self.gt_data['#timestamp [ns]'] == prev_timestamp]

        if len(curr_gt) == 0 or len(prev_gt) == 0:
            return 0
        
        curr_gt = curr_gt.iloc[0]
        prev_gt = prev_gt.iloc[0]

        x_prev = prev_gt[' p_RS_R_x [m]']
        y_prev = prev_gt[' p_RS_R_y [m]']
        z_prev = prev_gt[' p_RS_R_z [m]']
        
        x = curr_gt[' p_RS_R_x [m]']
        y = curr_gt[' p_RS_R_y [m]']
        z = curr_gt[' p_RS_R_z [m]']

        self.trueX, self.trueY, self.trueZ = x, y, z
        scale = np.sqrt((x - x_prev)**2 + (y - y_prev)**2 + (z - z_prev)**2)
        return scale

    def processFirstFrame(self):
        self.px_ref = self.detector.detect(self.new_frame)
        self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        self.frame_stage = STAGE_SECOND_FRAME

    def processSecondFrame(self):
        self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, 
                                       focal=self.focal, pp=self.pp,
                                       method=cv2.RANSAC, prob=0.999, threshold=0.1)
        _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref,
                                        focal=self.focal, pp=self.pp)
        
        # scale=1 varsayımı
        T_cam = np.eye(4)
        T_cam[:3,:3] = R
        T_cam[:3, 3] = t.reshape(3)

        T_BS_inv = np.linalg.inv(T_BS)
        T_cam_corrected = T_BS @ T_cam @ T_BS_inv
        R_cam = T_cam_corrected[:3, :3]

        new_R = self.cur_R @ R_cam

        # Tahmin Euler clamp
        new_euler = rotation_matrix_to_euler_angles(new_R)
        clamped_euler = clamp_euler_angles(self.prev_euler, new_euler, max_deg=5.0)
        final_R = euler_angles_to_rotation_matrix(clamped_euler)

        self.cur_R = final_R
        self.prev_euler = clamped_euler

        self.frame_stage = STAGE_DEFAULT_FRAME
        self.px_ref = self.px_cur

    def processFrame(self, frame_id):
        self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref,
                                       focal=self.focal, pp=self.pp,
                                       method=cv2.RANSAC, prob=0.999, threshold=0.7)
        _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref,
                                        focal=self.focal, pp=self.pp)

        absolute_scale = self.getAbsoluteScale(frame_id)

        if absolute_scale > 0.001:
            T_cam = np.eye(4)
            T_cam[:3, :3] = R
            T_cam[:3, 3] = (absolute_scale * t).reshape(3)

            T_BS_inv = np.linalg.inv(T_BS)
            T_cam_corrected = T_BS @ T_cam @ T_BS_inv

            R_cam = T_cam_corrected[:3, :3]
            t_cam = T_cam_corrected[:3, 3].reshape(3,1)

            new_t = self.cur_t + np.dot(self.cur_R, t_cam)
            new_R = self.cur_R @ R_cam

            # Tahmin Euler clamp
            new_euler = rotation_matrix_to_euler_angles(new_R)
            clamped_euler = clamp_euler_angles(self.prev_euler, new_euler, max_deg=5.0)
            final_R = euler_angles_to_rotation_matrix(clamped_euler)

            self.cur_R = final_R
            self.cur_t = new_t
            self.prev_euler = clamped_euler

            # IMU hız güncelleme
            curr_timestamp = self.cam_data.iloc[frame_id]['#timestamp [ns]']
            curr_gt = self.gt_data[self.gt_data['#timestamp [ns]'] == curr_timestamp]
            if len(curr_gt) > 0:
                curr_gt = curr_gt.iloc[0]
                self.cur_vel = np.array([
                    curr_gt[' v_RS_R_x [m s^-1]'],
                    curr_gt[' v_RS_R_y [m s^-1]'],
                    curr_gt[' v_RS_R_z [m s^-1]']
                ]).reshape(3, 1)

        # Feature sayısı düşükse yeniden tespit
        if self.px_ref.shape[0] < kMinNumFeature:
            self.px_cur = self.detector.detect(self.new_frame)
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)

        self.px_ref = self.px_cur

    def update(self, img, frame_id):
        assert img.ndim == 2 and img.shape[0] == self.cam.height and img.shape[1] == self.cam.width, \
            "Frame ebatları uyuşmuyor veya grayscale değil."
        self.new_frame = img

        if self.frame_stage == STAGE_DEFAULT_FRAME:
            self.processFrame(frame_id)
        elif self.frame_stage == STAGE_SECOND_FRAME:
            self.processSecondFrame()
        elif self.frame_stage == STAGE_FIRST_FRAME:
            self.processFirstFrame()

        self.last_frame = self.new_frame
