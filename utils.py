import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_features(img):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img, None)
    return kp1, des1

def match_features(kp1, des1, kp2, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    pts1 = []
    pts2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            pts2.append(list(kp2[m.trainIdx].pt))
            pts1.append(list(kp1[m.queryIdx].pt))
    return pts1, pts2


def normalize(points):
    points_ = points.copy()
    mean = np.mean(points_, axis=0)
    centroid = points_ - mean
    mean_dist = np.mean(np.sqrt(np.sum(centroid**2, axis=1)))
    if mean_dist > 0:
        scale = np.sqrt(2)/mean_dist
    else:
        scale = 1
    T = np.array([[scale,0,-scale*mean[0]],[0,scale,-scale*mean[1]],[0,0,1]])
    normalized_point = np.dot(T, points_.T).T
    return normalized_point, T

def do_svd(pts1, pts2):
    A = []
    pts1_norm, Ta = normalize(pts1)
    pts2_norm, Tb = normalize(pts2)
    for i in range(len(pts1_norm)):
        x1, y1, _ = pts1_norm[i]
        x2, y2, _ = pts2_norm[i]
        A.append([x1*x2 , x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1 ])
    A = np.array(A)
    U , S , Vt = np.linalg.svd(A)
    F = Vt[-1,:]
    F = (np.reshape(F, (3, 3)))
    U , S, Vt = np.linalg.svd(F)
    S = np.array([[1,0,0],[0,1,0],[0,0,0]])
    F_norm = np.dot(np.dot(U,S),Vt)
    F_unnorm = np.dot(Tb.T, np.dot(F_norm, Ta))
    F_unnorm = F_unnorm / np.linalg.norm(F_unnorm)
    return F_unnorm


def do_ransac(pts1,pts2):
    pts1_ = np.hstack((pts1, np.ones((len(pts1), 1))))
    pts2_ = np.hstack((pts2, np.ones((len(pts2), 1))))
    count = 0
    thresh = 0.1
    np.random.seed(0)
    max_score = 0
    while count < 10000:
        count = count + 1
        score = 0
        rand_idx = np.random.randint(len(pts1_), size=8)
        rand_pts1, rand_pts2, inlier_idxs = [], [] ,[]
        for i in range(len(rand_idx)):
            rand_pts1.append(pts1_[rand_idx[i]])
            rand_pts2.append(pts2_[rand_idx[i]])

        rand_pts1 = np.array(rand_pts1)
        rand_pts2 = np.array(rand_pts2)
        F = do_svd(rand_pts1, rand_pts2)
        for i in range(len(pts1_)):
            pt1 = np.array([pts1_[i][0], pts1_[i][1], 1])
            pt2 = np.array([pts2_[i][0], pts2_[i][1], 1])
            constraint = abs(np.dot(pt2,np.dot(F,pt1.T)) )
            if constraint < thresh:
                score+=1
                inlier_idxs.append(i)

        if score > max_score:
            final_idxs = inlier_idxs
            max_score = score
    return final_idxs, pts1_, pts2_

def compute_fundamental_matrix(pts1,pts2):

    inlier_idxs, pts1, pts2 = do_ransac(pts1, pts2)
    inlier_pts1, inlier_pts2 = [], []
    for i in inlier_idxs:
        inlier_pts1.append(pts1[i])
        inlier_pts2.append(pts2[i])
    inlier_pts1 = np.array(inlier_pts1)
    inlier_pts2 = np.array(inlier_pts2)
    F = do_svd(inlier_pts1, inlier_pts2)
    F = F/ F[-1][-1]
    return F, inlier_pts1[:, :2], inlier_pts2[:, :2]

def get_proj(dir_name):
    if dir_name == 'curule':
        K1 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15],  [0, 0, 1]])
        K2 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
        baseline, focal_len = 88.39, 1758.23
    elif dir_name == 'octagon':
        K1 = np.array([[1742.11, 0, 804.90], [0 ,1742.11, 541.22], [0, 0, 1]])
        K2 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
        baseline, focal_len = 221.76, 1742.11
    elif dir_name == 'pendulum':
        K1 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
        K2 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
        baseline, focal_len = 537.75, 1729.05
    return K1, K2, baseline, focal_len

def compute_essential_matrix(K1, K2, F):
    E = np.dot(K2.T, np.dot(F, K1))
    U, S, Vt = np.linalg.svd(E)
    S = np.array([[1,0,0], [0,1,0], [0,0,0]])
    E = np.dot(U, np.dot(S, Vt))
    return E

def decompose_essential(E, K, pts1, pts2):
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

    U, S, Vt = np.linalg.svd(E)
    R = []
    C = []
    C.append(U[:, 2])
    R.append(np.dot(U, np.dot(W, Vt)))
    C.append(-U[:, 2])
    R.append(np.dot(U, np.dot(W, Vt)))
    C.append(U[:, 2])
    R.append(np.dot(U, np.dot(W.T, Vt)))
    C.append(-U[:, 2])
    R.append(np.dot(U, np.dot(W.T, Vt)))
    for i in range(4):
        if (np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            C[i] = -C[i]

    ThreeD_pts = []
    for i in range(4):
        ThreeD_pts.append(triangulation(K, C[i], R[i], pts1, pts2))
    R_final, t_final = get_Rt(C, R, ThreeD_pts)
    return R_final, t_final

def triangulation(K, C, R, pts1, pts2):
    C = C.T
    P1 = np.dot(K, np.dot(np.identity(3), np.hstack((np.identity(3), -1*np.array([[0,0,0]]).T))))
    P2 = np.dot(K, np.dot(R, np.hstack((np.identity(3), -C))))

    X1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    X2 = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    X = []

    for i in range(pts1.shape[0]):
        x1, y1, z1 = X1[i,:]
        x2, y2, z2 = X2[i, :]
        a1 = np.array([[0, -z1, y1], [z1, 0, x1], [y1, x1, 0]])
        a2 = np.array([[0, -z2, y2], [z2, 0, x2], [y2, x2, 0]])
        A = np.vstack((np.dot(a1, P1), np.dot(a2, P2)))
        U, S, Vt = np.linalg.svd(A)
        x = Vt[-1]/Vt[-1, -1]
        x = np.reshape(x, (len(x), -1))
        X.append(x[0:3].T)
    X = np.array(X)
    return X

def get_Rt(t, R, X):
    max_score = 0
    for i in range(4):
        score = 0
        for j in range(X[i].shape[0]):
            if ((np.dot(R[i][2, :], (X[i][j, :] - t[i])) > 0) and X[i][j, 2] >= 0):
                score += 1
        if score > max_score:
            t_final,R_final = t[i], R[i]
            max_score = score
    return R_final, t_final

def draw_epilines(img1, lines, pts1, pts2):
    r, c = img1.shape
    img1_three_channel = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1_three_channel = cv2.line(img1_three_channel, (x0, y0), (x1, y1), color, 1)
        img1_three_channel = cv2.circle(img1_three_channel, tuple(np.int32(pt1)), 5, color, -1)
    return img1_three_channel

def search(idx, j, img, block1, win_size):
    h, w = img.shape[:2]
    min_score = float("inf")
    search_size = 56
    xmin = max(0, j - search_size)
    xmax = min(w-win_size, j+search_size)
    for i in range(xmin, xmax):
        block2 = img[idx:idx+win_size, i:i+win_size]
        ssd = np.sum(abs(block1-block2))
        if ssd < min_score:
            min_score = ssd
            best_idx = i

    return best_idx

def get_disparity(img1, img2):
    disparity = np.zeros_like(img1, dtype=np.uint8)
    win_size=30
    h, w = img1.shape[:2]
    count=0
    for i in range(0, h-win_size):
        for j in range(0, w-win_size):
            block = img1[i:i+win_size, j:j+win_size]
            count+=1
            print(count, end="\r")
            best_x = search(i, j, img2, block, win_size)
            disparity[i][j] = int(abs(best_x-j))

    return np.array(disparity, dtype=np.uint8)

def get_depth(disparity, focal_len, baseline):
    depth = np.zeros_like(disparity, dtype = np.uint8)
    for i in range(0, disparity.shape[0]):
        for j in range(0, disparity.shape[1]):
            depth[i][j] = int(baseline * focal_len / (disparity[i][j]+0.01))
    return np.array(depth,dtype=np.uint8)
