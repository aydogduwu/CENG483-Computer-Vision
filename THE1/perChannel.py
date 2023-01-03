import numpy as np
import cv2 as cv


def l1_normalization(histogram):
    norm = np.linalg.norm(histogram, 1)
    return histogram / norm + 10 ** -6


def kl_divergence(q, s):
    return (q * (np.log(q / s))).sum()


def js_divergence(q, s):

    return (0.5 * kl_divergence(q, (q + s) / 2)) + (0.5 * kl_divergence(s, (q + s) / 2))


def interval_creator(bin_count):
    interval_size = int(256 / bin_count)
    interval_left = 0
    interval_right = int(256 / bin_count)
    intervals = []
    for i in range(0, bin_count):
        intervals.append([interval_left, interval_right])
        interval_left += interval_size
        interval_right += interval_size

    return intervals


def per_channel_histogram(image, bin_count):
    intervals = interval_creator(bin_count)
    interval_size = int(256 / bin_count)
    interval_count_array = np.zeros([3, len(intervals)], int)
    image_height = image.shape[0]
    image_reshaped = image.reshape(image_height ** 2, 3)

    unique1, counts1 = np.unique(image_reshaped[:, 0], return_counts=True)
    dict1 = dict(zip(unique1, counts1))

    unique2, counts2 = np.unique(image_reshaped[:, 1], return_counts=True)
    dict2 = dict(zip(unique2, counts2))

    unique3, counts3 = np.unique(image_reshaped[:, 2], return_counts=True)
    dict3 = dict(zip(unique3, counts3))

    for j in dict1.keys():
        interval_count_array[0][j // interval_size] += dict1[j]

    for j in dict2.keys():
        interval_count_array[1][j // interval_size] += dict2[j]

    for j in dict3.keys():
        interval_count_array[2][j // interval_size] += dict3[j]

    return interval_count_array


def ddd_histogram(image, bin_count):
    intervals = interval_creator(bin_count)
    interval_size = int(256 / bin_count)
    interval_count_array = np.zeros((bin_count, bin_count, bin_count), int)
    image_height = image.shape[0]
    image_reshaped = image.reshape(image_height ** 2, 3)

    for i in image_reshaped:
        blue_value = i[0]
        blue_index = int(blue_value // interval_size)

        green_value = i[1]
        green_index = int(green_value // interval_size)

        red_value = i[2]
        red_index = int(red_value // interval_size)

        interval_count_array[blue_index][green_index][red_index] += 1

    return interval_count_array


def main(histogram_type, bin_count, grid_size, query_number):
    instance_names = open("InstanceNames.txt").read().splitlines()

    if grid_size == 1:  # No grid operations, store support dataset's values not to calculate again and again
        support_results_blue = []
        support_results_green = []
        support_results_red = []
        support_result_3d = []
        for i in range(0, len(instance_names)):
            image = cv.imread("support_96/{}".format(instance_names[i]))

            if histogram_type == 1:  # per_channel
                support_result = per_channel_histogram(image, bin_count)
                support_results_blue.append(l1_normalization(support_result[0]))
                support_results_green.append(l1_normalization(support_result[1]))
                support_results_red.append(l1_normalization(support_result[2]))

            elif histogram_type == 2:  # 3D
                support_result = ddd_histogram(image, bin_count)
                support_result_3d.append(l1_normalization(support_result.ravel()))

        if histogram_type == 1:  # if per-channel, no grid.
            correctly_predicted = 0
            for i in instance_names:
                image = cv.imread("query_{}/{}".format(query_number, i))  # change X in query_X to calculate for different query sets
                min_js_divergence = 10 ** 5
                index = -1
                query_result = per_channel_histogram(image, bin_count)
                query_result_blue = l1_normalization(query_result[0])
                query_result_green = l1_normalization(query_result[1])
                query_result_red = l1_normalization(query_result[2])
                for j in range(0, len(instance_names)):
                    jsB = js_divergence(query_result_blue, support_results_blue[j])
                    jsG = js_divergence(query_result_green, support_results_green[j])
                    jsR = js_divergence(query_result_red, support_results_red[j])
                    average = (jsB + jsG + jsR) / 3
                    if average < min_js_divergence:
                        min_js_divergence = average
                        index = j

                if i == instance_names[index]:
                    correctly_predicted += 1
            print("Accuracy: ", correctly_predicted / 2, "%")  # prints accuracy.

        elif histogram_type == 2:  # 3d histogram.
            correctly_predicted = 0
            for i in instance_names:
                image = cv.imread("query_{}/{}".format(query_number, i))  # change X in query_X to calculate for different query sets
                min_js_divergence = 10 ** 5
                index = -1
                query_result = ddd_histogram(image, bin_count)
                query_result_norm = l1_normalization(query_result.ravel())
                for j in range(0, len(instance_names)):
                    js = js_divergence(query_result_norm, support_result_3d[j])
                    if js < min_js_divergence:
                        min_js_divergence = js
                        index = j

                if i == instance_names[index]:
                    correctly_predicted += 1
            print("Accuracy: ", correctly_predicted / 2, "%")

    else:  # grid calculations here.
        grid_support_results_blues = []
        grid_support_results_greens = []
        grid_support_results_reds = []
        grid_support_results_3d = []
        if histogram_type == 1:  # store per-channel support results with grids.
            index = 0
            for i in range(0, len(instance_names)):
                image = cv.imread("support_96/{}".format(instance_names[i]))
                size = 96 // grid_size
                grid_support_results_blues.append([])
                grid_support_results_greens.append([])
                grid_support_results_reds.append([])
                for j in range(grid_size):
                    for k in range(grid_size):
                        grid = image[j * size:(j + 1) * size, k * size:(k + 1) * size]
                        support_result = per_channel_histogram(grid, bin_count)
                        grid_support_results_blues[index].append(l1_normalization(support_result[0]))
                        grid_support_results_greens[index].append(l1_normalization(support_result[1]))
                        grid_support_results_reds[index].append(l1_normalization(support_result[2]))
                index += 1
        else:  # 3d support results with grids.
            index = 0
            for i in range(0, len(instance_names)):
                image = cv.imread("support_96/{}".format(instance_names[i]))
                size = 96 // grid_size
                grid_support_results_3d.append([])
                for j in range(grid_size):
                    for k in range(grid_size):
                        grid = image[j * size:(j + 1) * size, k * size:(k + 1) * size]
                        support_result_3d = ddd_histogram(grid, bin_count)
                        grid_support_results_3d[index].append(l1_normalization(support_result_3d.ravel()))
                index += 1

        if histogram_type == 1:  # per-channel calculations with grids.
            correctly_predicted = 0

            for i in instance_names:
                image = cv.imread("query_{}/{}".format(query_number, i))  # change X in query_X to calculate for different query sets
                min_js_divergence = 10 ** 5
                grid_query_results_blues = []
                grid_query_results_greens = []
                grid_query_results_reds = []
                for j in range(0, grid_size):
                    for k in range(0, grid_size):
                        grid = image[j * size:(j + 1) * size, k * size:(k + 1) * size]
                        query_result = per_channel_histogram(grid, bin_count)
                        grid_query_results_blues.append(l1_normalization(query_result[0]))
                        grid_query_results_greens.append(l1_normalization(query_result[1]))
                        grid_query_results_reds.append(l1_normalization(query_result[2]))
                summ = 0
                instance_index = -1
                for j in range(0, len(instance_names)):
                    for k in range(0, grid_size ** 2):
                        jsB = js_divergence(grid_query_results_blues[k], grid_support_results_blues[j][k])
                        jsG = js_divergence(grid_query_results_greens[k], grid_support_results_greens[j][k])
                        jsR = js_divergence(grid_query_results_reds[k], grid_support_results_reds[j][k])
                        summ += (jsB + jsG + jsR)
                    average = summ / ((grid_size ** 2) * 3)
                    summ = 0
                    if average < min_js_divergence:
                        min_js_divergence = average
                        instance_index = j

                if i == instance_names[instance_index]:
                    correctly_predicted += 1
            print("Accuracy: ", correctly_predicted / 2, "%")

        else:  # 3d histogram calculations with grids.

            correctly_predicted = 0
            for i in instance_names:
                image = cv.imread("query_{}/{}".format(query_number, i))  # change X in query_X to calculate for different query sets
                grid_3d_query_results = []
                for j in range(0, grid_size):
                    for k in range(0, grid_size):
                        grid = image[j * size:(j + 1) * size, k * size:(k + 1) * size]
                        query_result = ddd_histogram(grid, bin_count)
                        grid_3d_query_results.append(l1_normalization(query_result.ravel()))
                min_js_divergence = 10 ** 5
                index = -1
                summ = 0
                for j in range(0, len(instance_names)):
                    for k in range(0, len(grid_3d_query_results)):
                        summ += js_divergence(grid_3d_query_results[k], grid_support_results_3d[j][k])
                    average = summ / len(grid_3d_query_results)
                    summ = 0
                    if average < min_js_divergence:
                        min_js_divergence = average
                        index = j

                if i == instance_names[index]:
                    correctly_predicted += 1
            print("Accuracy: ", correctly_predicted / 2, "%")


# First parameter: histogram_type: 1 for perChannel, histogram_type:2 for 3D.
# Second parameter: Bin count (2 or 4 or 8 ...)
# Third parameter: grid number in grid X grid. 1 for no grid.  (2 for 2x2, 4 for 4x4 ...)
# Fourth parameter: query number (1 or 2 or 3)
if __name__ == '__main__':
    main(1, 32, 6, 3)
