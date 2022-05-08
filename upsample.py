import ffmpeg
import cv2
import numpy as np
import os
import numpy.linalg as la
import utils
from matplotlib import pyplot as plt

from random import random
import time
import scipy
import scipy.ndimage as nd
import seaborn as sns
import pandas as pd
import scipy.sparse.linalg
from scipy import optimize
import pwlf
import argparse


def compute_gradient_xdir(img_y):
    f2 = np.array([[0, 0, 0],
                   [-1, 0, 1],
                   [0, 0, 0]])
    # f2 = np.array([[-1,-2,-1], [0,0,0], [1,2,1]]).T
    horz_gradient = cv2.filter2D(img_y, -1, f2)
    return horz_gradient


def compute_gradient_ydir(img_y):
    f1 = np.array([[0, -1, 0],
                   [0, 0, 0],
                   [0, 1, 0]])
    # f1 = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
    vert_gradient = cv2.filter2D(img_y, -1, f1)
    return vert_gradient


def calculate_phi(gradient_x, gradient_y):
    grad_norm_x = cv2.normalize(gradient_x, None, alpha=-235, beta=235, norm_type=cv2.NORM_MINMAX)
    grad_norm_y = cv2.normalize(gradient_y, None, alpha=-235, beta=235, norm_type=cv2.NORM_MINMAX)
    grad_norm_x = grad_norm_x.astype(np.int16)
    grad_norm_y = grad_norm_y.astype(np.int16)
    flattened_grad_x = grad_norm_x.flatten()
    flattened_grad = np.append(flattened_grad_x, grad_norm_y.flatten())
    df = pd.Series(flattened_grad).value_counts().reset_index()
    df.columns = ['val', 'count']
    df = df[df['val'] >= 0]
    val = np.array(df['val'])
    log_count = np.log(df['count'] / df['count'].max())
    my_pwlf = pwlf.PiecewiseLinFit(val, log_count)
    breaks = my_pwlf.fit(2)
    l_t = breaks[1]

    def piecewise(x, k, a, b):
        return np.piecewise(x, [x <= l_t, x > l_t], [lambda x: -k * abs(x), lambda x: -1 * (a * (x ** 2) + b)])

    p, e = optimize.curve_fit(piecewise, val, log_count)
    k, a, b = p[0], p[1], p[2]
    return l_t, k, a, b


def generate_full_size_filter(orig_fil, shape):
    full_size_fil = np.zeros(shape)
    x_start = int((full_size_fil.shape[0] - orig_fil.shape[0]) / 2)
    y_start = int((full_size_fil.shape[1] - orig_fil.shape[1]) / 2)
    full_size_fil[x_start:x_start + orig_fil.shape[0], y_start:y_start + orig_fil.shape[1]] = orig_fil
    return full_size_fil


def mu_optimize(l_t, k, a, b, lambda_1, lambda_2, H_x_gradient, H_y_gradient):
    mu_x = np.zeros(H_x_gradient.shape)
    mu_y = np.zeros(H_y_gradient.shape)
    height, width = H_x_gradient.shape[:2]

    # since phi is calculated between 0 and 235
    k = k * 235
    a = a * (235 ** 2)
    for i in range(height):
        for j in range(width):
            curr_x_grad = H_x_gradient[i][j]
            curr_y_grad = H_y_gradient[i][j]

            if abs(curr_x_grad) < l_t:
                if curr_x_grad > 0:
                    mu_x[i][j] = ((2 * lambda_2 * curr_x_grad) - lambda_1 * k) / (2 * lambda_2)
                else:
                    mu_x[i][j] = ((2 * lambda_2 * curr_x_grad) + lambda_1 * k) / (2 * lambda_2)
            else:
                mu_x[i][j] = (2 * lambda_2 * curr_x_grad) / (2 * lambda_1 * a + 2 * lambda_2)

            if abs(curr_y_grad) < l_t:
                if curr_y_grad > 0:
                    mu_y[i][j] = ((2 * lambda_2 * curr_y_grad) - lambda_1 * k) / (2 * lambda_2)
                else:
                    mu_y[i][j] = ((2 * lambda_2 * curr_y_grad) + lambda_1 * k) / (2 * lambda_2)
            else:
                mu_y[i][j] = (2 * lambda_2 * curr_y_grad) / (2 * lambda_1 * a + 2 * lambda_2)

    return mu_x, mu_y


def H_optimize(f_fft, H_prime_fft, lambda_2, H_x_grad_fft, H_y_grad_fft, mu_x, mu_y):
    mu_x_fft = np.fft.fft2(mu_x)
    mu_y_fft = np.fft.fft2(mu_y)

    upper_adding = lambda_2 * np.conj(H_x_grad_fft) * mu_x_fft + lambda_2 * np.conj(H_y_grad_fft) * mu_y_fft
    lower_adding = lambda_2 * np.conj(H_x_grad_fft) * H_x_grad_fft + lambda_2 * np.conj(H_y_grad_fft) * H_y_grad_fft
    #     clip_upper = np.clip(upper_adding, -clip_range, clip_range)
    #     clip_lower = np.clip(lower_adding, -clip_range, clip_range)
    clip_upper = upper_adding
    clip_lower = lower_adding

    H_star_fft = ((np.conj(f_fft) * H_prime_fft) + clip_upper) / ((np.conj(f_fft) * f_fft) + clip_lower)
    H_star = np.fft.ifft2(H_star_fft)
    H_star = np.clip(np.real(H_star), 0, 1)

    shift_result = np.fft.fftshift(H_star, (0, 1))
    return shift_result


def optimize_loop(H_prime_y, H, l_t, k, a, b, kernel_sigma=1.5, lamda_one=0.3, lamda_two=20):
    f1 = np.array([[0, -1, 0],
                   [0, 0, 0],
                   [0, 1, 0]])
    f2 = np.array([[0, 0, 0],
                   [-1, 0, 1],
                   [0, 0, 0]])

    ksize = int(np.ceil(kernel_sigma) * 6 + 1)
    fil = cv2.getGaussianKernel(ksize, kernel_sigma)  # 1D kernel
    fil = fil * np.transpose(fil)

    fil = generate_full_size_filter(fil, H_prime_y.shape)

    f_fft = np.fft.fft2(fil)
    H_prime_fft = np.fft.fft2(H_prime_y)
    lambda_1 = lamda_one
    lambda_2 = lamda_two

    full_f1 = generate_full_size_filter(f1, H_prime_y.shape)
    full_f2 = generate_full_size_filter(f2, H_prime_y.shape)
    H_x_gradient = cv2.filter2D(H, -1, full_f2)
    H_y_gradient = cv2.filter2D(H, -1, full_f1)
    H_x_grad_fft = np.fft.fft2(full_f2)
    H_y_grad_fft = np.fft.fft2(full_f1)

    H_star_arr = np.zeros((100, H_prime_y.shape[0], H_prime_y.shape[1]), dtype=np.float64)
    
    mu_stop = False
    prev_mu_x = np.zeros(H_x_gradient.shape)
    prev_mu_y = np.zeros(H_y_gradient.shape)
    
    for i in range(100):
        if not mu_stop:
            mu_x, mu_y = mu_optimize(l_t, k, a, b, lambda_1, lambda_2, H_x_gradient, H_y_gradient)
            x_diff = (mu_x - prev_mu_x).max()
            y_diff = (mu_y - prev_mu_y).max()
            if x_diff < 0.001 and y_diff < 0.001:
                mu_stop = True
            prev_mu_x = mu_x
            prev_mu_y = mu_y
        H_star = H_optimize(f_fft, H_prime_fft, lambda_2, H_x_grad_fft, H_y_grad_fft, mu_x, mu_y)
        H_star_arr[i] = 1 - H_star
        lambda_2 *= 3
        if i >= 1:
            diff_norm = la.norm(H_star_arr[i] - H_star_arr[i - 1], 2)
            if diff_norm < 0.01:
                break
    H_star = (1 - H_star)
    return H_star, H_star_arr


def pixel_substitute(H_star, orig_im_y, upsample_ratio, kernel_sigma, diff_range=1):
    ksize = int(np.ceil(kernel_sigma) * 6 + 1)
    fil = cv2.getGaussianKernel(ksize, kernel_sigma)
    fil = fil * np.transpose(fil)
    H_prime_refil = cv2.filter2D(H_star, -1, fil)

    n = upsample_ratio
    for i in range(orig_im_y.shape[0]):
        for j in range(orig_im_y.shape[1]):
            curr = abs(H_prime_refil[n * i][n * j] - orig_im_y[i][j])
            if curr < diff_range:
                H_prime_refil[n * i][n * j] = orig_im_y[i][j]

    return H_prime_refil


def upsample_one_image(input_img_location, upsample_ratio=4.0, feedback_loop=2, kernel_sigma=1.5,
                       lamda_one=0.3, lamda_two=20, substitute_diff=0.1):
    im_YUV = cv2.cvtColor(cv2.imread(input_img_location), cv2.COLOR_BGR2YUV)
    im_yuv = im_YUV.astype('double') / 235.0
    im_y, im_u, im_v = cv2.split(im_yuv)

    gradient_xdir = compute_gradient_xdir(im_y)
    gradient_ydir = compute_gradient_ydir(im_y)

    l_t, k, a, b = calculate_phi(gradient_xdir, gradient_ydir)

    bicubic_im_yuv = cv2.resize(im_YUV, None, fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    if bicubic_im_yuv.shape[0] % 2 == 0:
        bicubic_im_yuv = bicubic_im_yuv[:-1, :, :]
    if bicubic_im_yuv.shape[1] % 2 == 0:
        bicubic_im_yuv = bicubic_im_yuv[:, :-1, :]

    H_prime_y, H_prime_u, H_prime_v = cv2.split(bicubic_im_yuv)

    H_prime_y = H_prime_y.astype('double') / 235.0
    H = H_prime_y.copy()

    for i in range(feedback_loop):
        H_star, H_star_arr = optimize_loop(H_prime_y, H, l_t, k, a, b, kernel_sigma, lamda_one, lamda_two)
        H = H_star.copy()
        H_prime_y = pixel_substitute(H_star, im_y, int(upsample_ratio), kernel_sigma, substitute_diff)
        print("one iteration finished")

    H_star_norm = cv2.normalize(H_star, None, alpha=0, beta=235, norm_type=cv2.NORM_MINMAX)
    H_star_norm = H_star_norm.astype(H_prime_u.dtype)
    merged_YUV = cv2.merge([H_star_norm, H_prime_u, H_prime_v])
    merged_rgb = cv2.cvtColor(merged_YUV, cv2.COLOR_YUV2BGR)

    return merged_rgb, H_star


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help="path to low res image", required=True)
    parser.add_argument('-l', '--loop', type=int, help="how many times to run the feedback loop", default=1)
    args = parser.parse_args()

    input_img_location = args.path
    loop = args.loop
    folder, im_name = os.path.split(input_img_location)
    suffixes = (".png", ".jpg")
    if str(im_name).endswith(suffixes):
        print("Upsampling {} using {} feedback loops...".format(im_name, loop))
        first_iteration_result, _ = upsample_one_image(input_img_location, feedback_loop=loop, substitute_diff=0.3)
        cv2.imwrite(os.path.join(folder, "upsampled_" + str(loop) + "_loop_" + im_name), first_iteration_result)
        print("Upsampled image is saved at: ", os.path.join(folder, "upsampled_" + str(loop) + "_loop_" + im_name))
    else:
        print("File must be jpg or png.")
