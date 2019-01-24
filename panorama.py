import cv2
import math
import numpy as np
from numpy.linalg import solve
############### Cyber Paras#######
cols = 400
rows = 600
bias = int(0.2 * rows)
f = 35
# Ratio
p = 0.6
# Ransac
k = 500
r = 50
###################################
def cylin_projection(src_img, f):
    dst_img = np.zeros_like(src_img)
    rows = src_img.shape[0]
    cols = src_img.shape[1]
    center_x = int(cols / 2)
    center_y = int(rows / 2)

    for dst_x in range(cols):
        theta = (dst_x - center_x) / f
        src_x = int(f * math.tan(theta) + center_x)
        if src_x >= 0 and src_x < cols:
            for dst_y in range(rows):
                src_y = int((dst_y - center_y) / math.cos(theta) + center_y)
                if src_y >= 0 and src_y < rows:
                    dst_img[dst_y, dst_x, :] = src_img[src_y, src_x, :]
    return dst_img
###反圆柱投影
def cylin_projection_inverse(src_img, f):
    dst_img = np.zeros_like(src_img)
    rows = src_img.shape[0]
    cols = src_img.shape[1]
    center_x = int(cols / 2)
    center_y = int(rows / 2)

    for dst_x in range(cols):
        theta = math.atan((dst_x - center_x) / f)
        src_x = int((f * theta) + center_x)
        if src_x >= 0 and src_x < cols:
            for dst_y in range(rows):
                src_y = int((dst_y - center_y) * math.cos(theta) + center_y)
                if src_y >= 0 and src_y < rows:
                    dst_img[dst_y, dst_x, :] = src_img[src_y, src_x, :]
    return dst_img

def draw_keypoint_lines(img1, kp1, img2, kp2, matches):
    img_out = np.concatenate([img1, img2], axis=1)
    for i in range(len(matches)):
        # img_out = cv2.drawMatches(img1, kp1, img2, kp2, matches[i],None, flags=2)
        # cv2.imshow('out', img_out)
        # cv2.waitKey(0)
        img1_idx = matches[i][0].queryIdx
        img2_idx = matches[i][0].trainIdx

        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        a = np.random.randint(0, 256)
        b = np.random.randint(0, 256)
        c = np.random.randint(0, 256)
        cv2.line(img_out, (int(np.round(x1)), int(np.round(y1))), (int(np.round(x2) + cols), int(np.round(y2))),
                 (a, b, c), 1, shift=0)
        # for j in range(3):
        #     print(matches[i][j].distance)
        #     print(matches[i][j].queryIdx)
        #     print(matches[i][j].trainIdx)
    cv2.imshow('match', img_out)
    cv2.waitKey(0)
    return img_out

def find_good_matches_ratio(kp1, kp2, matches, p):
    matches_good = []
    for m in matches:
        if p * m[1].distance > m[0].distance:
            matches_good.append(m)
    return matches_good

def find_good_matches_ransac(kp1, kp2, matches, k, r):
    matches_good = []
    x = np.zeros(4, int)
    y = np.zeros(4, int)
    x_ = np.zeros(4, int)
    y_ = np.zeros(4, int)
    # x = [719, 606, 474, 781]
    # y = [400, 34, 396, 495]
    # x_ = [739, 622, 494, 801]
    # y_ = [370, 4, 366, 465]
    good_point_num = np.zeros(k)
    max_points_num = 0
    for i in range(k):
        test_points = [np.random.randint(0, len(matches)) for i in range(4)]
        for j in range(4):
            x[j], y[j] = kp1[matches[test_points[j]][0].queryIdx].pt
            x_[j], y_[j] = kp2[matches[test_points[j]][0].trainIdx].pt
        # print(i, test_points)
        #         # print(x, y, x_, y_)
        P = np.array([[x[0], y[0], 1, 0, 0, 0, -1 * x_[0] * x[0], -1 * x_[0]* y[0]],
             [ 0, 0, 0, x[0], y[0], 1,-1 * y_[0] * x[0], -1 * y_[0] * y[0]],
             [x[1], y[1], 1, 0, 0, 0, -1 * x_[1] * x[1], -1 * x_[1] * y[1]],
             [ 0, 0, 0, x[1], y[1], 1,-1 * y_[1] * x[1], -1 * y_[1] * y[1]],
             [x[2], y[2], 1, 0, 0, 0, -1 * x_[2] * x[2], -1 * x_[2] * y[2]],
             [0, 0, 0, x[0], y[0], 1, -1 * y_[2] * x[2], -1 * y_[2] * y[2]],
             [x[3], y[3], 1, 0, 0, 0, -1 * x_[3] * x[3], -1 * x_[3] * y[3]],
             [0, 0, 0, x[3], y[3], 1, -1 * y_[3] * x[3], -1 * y_[3] * y[3]]
             ])
        b = np.array([x_[0], y_[0], x_[1], y_[1], x_[2], y_[2], x_[3], y_[3]]).T
        if np.linalg.matrix_rank(P) < 8:
            continue
        h = solve(P, b)
        # print(P)
        # print(h, h.shape)
        h = np.concatenate([h,[1]]).reshape((3, 3))
        # print(h)
        # [x__, y__, z] = np.dot(h, [x[1], y[1], 1])
        # print([x__, y__, z])
        matches_good_temp =[]
        for m in matches:
            x_src, y_src = kp1[m[0].queryIdx].pt
            x_dst, y_dst = kp2[m[0].trainIdx].pt
            x_cacu, y_cacu, z= np.dot(h, [x_src, y_src, 1])
            d = (x_cacu - x_dst) * (x_cacu - x_dst) + (y_cacu - y_dst) * (y_cacu - y_dst)
            # print(m[0].queryIdx, x_src, y_src, x_dst, y_dst, x_cacu, y_cacu)
            if d < r:
                good_point_num[i] = good_point_num[i] + 1
                matches_good_temp.append(m)
        # print(max_points_num, good_point_num[i])
        if max_points_num < good_point_num[i]:
            max_points_num = good_point_num[i]
            matches_good = matches_good_temp[:]
    # print(len(matches_good))
    # print(good_point_num)
    return matches_good

def find_image_distance_ratio(kp1, kp2, matches_good):
    sum = np.array([0, 0])
    for m in matches_good:
        x_src, y_src = kp1[m[0].queryIdx].pt
        x_dst, y_dst = kp2[m[0].trainIdx].pt
        sum =sum + [x_src - x_dst, y_src - y_dst]
    d = np.divide(sum, len(matches_good))
    return d

def image_splicing_two(img1, img2, d):
    alpha = 1
    x = int(cols + cols)
    y = int(rows + bia * 2)
    if d[0] < 0:
        t = img1
        img1 = img2
        img2 = t
        d = -d
    # img_spiced[i][j] = np.concatenate([img1, np.zeros_like(img2)], axis=1)
    print(d)
    img_spiced = np.zeros([y, x, 3], np.uint8)
    img_spiced[bia:rows + bia, 0:cols, :] = img1
    for i in range(rows):
        for j in range(cols):
            if sum(img2[i][j][:]) != 0:
                [j_, i_] = np.add([0, 60], [j, i])
                [jj, ii] =np.add([j_, i_], d)
                ii = int(ii)
                jj = int(jj)
                # print(ii, jj)
                if sum(img_spiced[ii][jj][:]) != 0:
                    img_spiced[ii][jj][:] = (1 - alpha) * img_spiced[ii][jj][:] + alpha * img2[i][j][:]
                else:
                    img_spiced[ii][jj][:] = img2[i][j][:]
    cv2.imshow('spiced image', img_spiced)
    cv2.waitKey(0)
    return img_spiced

def image_splicing_all(img_panorama, img2_cylined, d_all, d, n):
    move = int(1 * cols)
    img_panorama = np.concatenate([np.zeros([int(rows + bias + bias), move, 3], np.uint8), img_panorama], axis=1)
    # alpha = 0.5
    # x = int(np.shape(img1)[0] + np.shape(img2)[0])
    # y = int(rows + bia * 2)
    # if d[0] < 0:
    #     t = img1
    #     img1 = img2
    #     img2 = t
    #     d = -d
    # # img_spiced[i][j] = np.concatenate([img1, np.zeros_like(img2)], axis=1)
    # print(d)
    # img_spiced = np.zeros([y, x, 3], np.uint8)
    # img_spiced[int(0.1 * rows):rows, 0:cols, :] = img1
    [j_l, i_l] = np.add(np.add([move * (n - 1), 0], np.add([0, bias], [0, 0])), d_all - d)
    [j_r, i_r] = np.add(np.add([move * (n - 1), 0], np.add([0, bias], [cols, 0])), d_all)
    j_l = int(j_l)
    j_r = int(j_r)
    # print(j_l, j_r)
    for j in range(cols):
        for i in range(rows):
            if sum(img2_cylined[i][j][:]) != 0:
                [jj, ii] = np.add(np.add([move * (n-1), 0] , np.add([0, bias], [j, i])), d_all)
                ii = int(ii)
                jj = int(jj)
                # print(i_, j_, '  ', ii, jj)
                if sum(img_panorama[ii][jj][:]) != 0:
                    alpha = (j_r - jj) / (j_r - j_l)
                    img_panorama[ii][jj][:] = (1 - alpha) * img_panorama[ii][jj][:] + alpha * img2_cylined[i][j][:]
                else:
                    img_panorama[ii][jj][:] = img2_cylined[i][j][:]
    # cv2.imshow('spiced image', img_panorama)
    # cv2.waitKey(0)
    return img_panorama

def image_straight(img_panorama):
    i = 0
    while sum(sum(img_panorama[:, i, :])) == 0:
        i = i + 1
    img_panorama = img_panorama[:, i:, :]
    i = 0
    while sum(img_panorama[i, int(0.5 * cols), :]) == 0:
        i = i + 1
    img_panorama_straight = np.zeros_like(img_panorama)
    for j in range(np.shape(img_panorama)[1]):
        b = int((np.shape(img_panorama)[1] - j ) * (bias - i)/np.shape(img_panorama)[1])
        img_panorama_straight[bias : rows + bias, j, :] = img_panorama[bias - b : rows + bias - b, j, :]
    img_panorama_straight = img_panorama_straight[bias + 10 : rows + bias - 25, : , :]
    return img_panorama_straight

def main():
    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher()

    d_all = 0
    i = 1
    img1 = cv2.resize(cv2.imread('images\srcImage (' + str(i) + ').jpg'), (cols, rows))
    img1_cylined = cylin_projection(img1, f)
    kp1, des1 = sift.detectAndCompute(img1_cylined, None)
    img_panorama = np.zeros([int(rows + bias + bias), int(cols), 3], np.uint8)
    img_panorama[bias : rows + bias, 0: cols,:] = img1_cylined
    print('image', i, 'done')
    for i in range(2, 20):
        img2 = cv2.resize(cv2.imread('images\srcImage (' + str(i) + ').jpg'), (cols, rows))
        img2_cylined = cylin_projection(img2, f)
        kp2, des2 = sift.detectAndCompute(img2_cylined, None)
        matches = bf.knnMatch(des1, des2, k=3)
        matches_good = find_good_matches_ratio(kp1, kp2, matches, p)
        # draw_keypoint_lines(img1_cylined, kp1, img2_cylined, kp2, matches_good)
        matches_good = find_good_matches_ransac(kp1, kp2, matches_good, k, r)
        # draw_keypoint_lines(img1_cylined, kp1, img2_cylined, kp2, matches_good)
        d = find_image_distance_ratio(kp1, kp2, matches_good)
        d_all = d_all + d
        # print(d_all)
        # img_panorama = image_splicing_two(img1_cylined, img2_cylined, d)
        # img1 = img_spiced
        img_panorama = image_splicing_all(img_panorama, img2_cylined, d_all, d, i)
        print('image', i, 'done')

        img1_cylined = img2_cylined
        kp1, des1 = kp2, des2

    img_panorama = image_straight(img_panorama)
    cv2.imshow('panorama.png', img_panorama)
    cv2.waitKey(0)
    cv2.imwrite('results\panorama.png', img_panorama)

if __name__ == '__main__':
    main()

