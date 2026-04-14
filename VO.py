import numpy as np
import cv2
import matplotlib.pyplot as plt

class VisualOdometry:
    def __init__(self, camera_matrix, file_path_poses):
        """
        Initialise le système avec la matrice intrinsèque de la caméra 
        et le fichier contenant les vraies positions (Ground Truth).
        """
        self.K = camera_matrix
        self.focal = self.K[0, 0]
        self.pp = (self.K[0, 2], self.K[1, 2])
        
        with open(file_path_poses) as f:
            self.true_poses = f.readlines()
            
        self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        
        self.lk_params = dict(winSize=(21, 21), 
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        self.frame_stage = 0 # 0: Initialisation, 1: Processus continu
        self.new_frame = None
        self.last_frame = None
        self.px_ref = None # Points de l'image précédente
        self.px_cur = None # Points de l'image courante
        
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        
        self.traj_x, self.traj_z = [], []

    def get_absolute_scale(self, frame_id):
        """
        Calcule la distance réelle parcourue entre la frame précédente et l'actuelle
        en lisant les données de la vérité terrain (Dataset KITTI).
        """
        ss = self.true_poses[frame_id-1].strip().split()
        x_prev, y_prev, z_prev = float(ss[3]), float(ss[7]), float(ss[11])
        
        ss = self.true_poses[frame_id].strip().split()
        x, y, z = float(ss[3]), float(ss[7]), float(ss[11])
        
        scale = np.sqrt((x - x_prev)**2 + (y - y_prev)**2 + (z - z_prev)**2)
        return scale

    def process_first_frame(self):
        """Initialise les premiers points à suivre."""
        self.px_ref = self.detector.detect(self.new_frame)
        self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        self.frame_stage = 1

    def process_frame(self, frame_id):
        """Le cœur du moteur d'odométrie."""
        if self.frame_stage == 0:
            self.process_first_frame()
            self.traj_x.append(self.cur_t[0, 0])
            self.traj_z.append(self.cur_t[2, 0])
            return

        self.px_cur, st, err = cv2.calcOpticalFlowPyrLK(self.last_frame, self.new_frame, self.px_ref, None, **self.lk_params)
        
        good_old = self.px_ref[st == 1]
        good_new = self.px_cur[st == 1]

        E, _ = cv2.findEssentialMat(good_new, good_old, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        _, R, t, mask = cv2.recoverPose(E, good_new, good_old, focal=self.focal, pp=self.pp)

        absolute_scale = self.get_absolute_scale(frame_id)
        if absolute_scale > 0.1: # On ignore les micro-mouvements (bruit)
            self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
            self.cur_R = R.dot(self.cur_R)

        self.traj_x.append(self.cur_t[0, 0])
        self.traj_z.append(self.cur_t[2, 0])

        if good_old.shape[0] < 1500:
            new_keypoints = self.detector.detect(self.new_frame)
            self.px_cur = np.array([x.pt for x in new_keypoints], dtype=np.float32)

        self.px_ref = self.px_cur
