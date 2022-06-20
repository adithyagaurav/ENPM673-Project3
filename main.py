import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import utils

if __name__=='__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--dir", type=str, required=True, help="Directory for stereo pair")
    # args = parser.parse_args()
    for dir in ["curule", "octagon", "pendulum"]:
        print("PROCESSING {}".format(dir))
        args_dir = "data/{}".format(dir)

        img1_path = os.path.join(args_dir, "im0.png")
        img2_path = os.path.join(args_dir, "im1.png")
        calib_path = os.path.join(args_dir, "calib.txt")

        img1 = cv2.imread(img1_path)
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.imread(img2_path)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = utils.get_features(img1_gray)
        kp2, des2 = utils.get_features(img2_gray)


        pts1, pts2 = utils.match_features(kp1, des1, kp2, des2)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        F, inlier_pts1, inlier_pts2 = utils.compute_fundamental_matrix(pts1, pts2)
        print("Fundamental Matrix: \n", F)
        print("*********************")
        K1, K2, baseline, focal_len = utils.get_proj(args_dir.split("/")[-1])
        E = utils.compute_essential_matrix(K1, K2, F)
        print("Essential Matrix: \n", E)
        print("*********************")
        R, t = utils.decompose_essential(E, K1, pts1, pts2)
        print("Camera Pose : \n", R, "\n\n", t)
        print("*********************\n")
        lines1 = cv2.computeCorrespondEpilines(
            np.array(inlier_pts2).reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)
        imgL = utils.draw_epilines(img1_gray, lines1, np.array(inlier_pts1), np.array(inlier_pts2))

        lines2 = cv2.computeCorrespondEpilines(
            np.array(inlier_pts1).reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)
        imgR = utils.draw_epilines(img2_gray, lines2, np.array(inlier_pts2), np.array(inlier_pts2))
        plt.subplot(121)
        plt.imshow(imgL)
        plt.subplot(122)
        plt.imshow(imgR)
        plt.suptitle("Epilines in both images")
        plt.savefig("output/{}/epipolar.png".format(dir))
        plt.close()
        h1, w1 = img1_gray.shape
        h2, w2 = img2_gray.shape
        _, H1, H2 = cv2.stereoRectifyUncalibrated(
            np.float32(inlier_pts1), np.float32(inlier_pts2), F, imgSize=(w1, h1)
        )
        img1_rectified = cv2.warpPerspective(img1_gray, H1, (w1, h1))
        img2_rectified = cv2.warpPerspective(img2_gray, H2, (w2, h2))

        disparity = utils.get_disparity(img1_rectified, img2_rectified)
        depth = utils.get_depth(disparity, focal_len, baseline)
        disparity = (disparity/disparity.max())*255.0
        cv2.imwrite("output/{}/disparity_gray.jpg".format(dir), disparity)
        plt.imshow(disparity, cmap='viridis')
        plt.savefig("output/{}/disparity_color.png".format(dir))
        plt.close()
        # cv2.imwrite("output/{}/disparity_color.jpg".format(dir), heatmap_disparity)
        cv2.imwrite("output/{}/depth_gray.jpg".format(dir), depth)
        plt.imshow(depth, cmap='viridis')
        plt.savefig("output/{}/depth_color.png".format(dir))
        plt.close()
