import cv2
import numpy as np
import random
import time
import math

poly_list = []
color_list = []
tracking_poly = False
canvas = np.zeros((600, 600, 3)).astype(np.uint8)


def build_poly(event, x, y, flags, image):
    global poly_list, color_list, tracking_poly, canvas

    if event == cv2.EVENT_LBUTTONDOWN:
        print("hello")
        if not tracking_poly:
            tracking_poly = True
            poly_list.append([])
            color_list.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        poly_list[-1].append((x, y))
        canvas = np.zeros(np.shape(image)).astype(np.uint8)
        for i in range(len(poly_list)):
            poly = poly_list[i]
            for point in poly:
                print("drawing point " + str(point))
                cv2.circle(canvas, point, 5, color_list[i], -1)
            print(poly)
            if len(poly) > 2:
                canvas = cv2.fillPoly(canvas, np.int32([poly]), color = color_list[i])
                image_mask = cv2.fillPoly(canvas, np.int32([poly]), (255, 255, 255))
                image_mask = cv2.bitwise_and(image, image_mask)
                cv2.namedWindow('and', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('and', 600, 600)
                cv2.imshow('and', image_mask)
                cv2.waitKey(0)
                result = find_dominant_color(image_mask)
                result = cv2.fillPoly(image_mask, np.int32([poly]), color = result.tolist())
                cv2.imshow('and', result)
                cv2.waitKey(0)


def find_dominant_color(image):
    flat = np.reshape(image, [-1, 3])
    non_zero_indices = np.nonzero(np.any(flat != 0, axis=1))[0]
    non_zero_colors = flat[non_zero_indices]
    # print(non_zero_colors.size)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(np.float32(non_zero_colors), 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    _, counts = np.unique(label, return_counts=True)
    popular_color = center[np.argmax(counts)].tolist()
    return popular_color


def go(file, debug=False):
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('test', 600, 600)

    img = cv2.imread(file)
    global canvas, tracking_poly
    canvas = np.zeros(np.shape(img)).astype(np.uint8)
    cv2.setMouseCallback('test', build_poly, img)

    while True:
        overlay = cv2.addWeighted(img, 0.5, canvas, 0.5, 50)
        cv2.imshow('test', overlay)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("x"):
            break
        elif key == ord("s"):
            tracking_poly = False
            print('starting new polygon')


def squares(file):
    cv2.namedWindow('squares', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('squares', 1000, 750)
    cv2.namedWindow('tmp', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('tmp', 1000, 750)

    img = cv2.imread(file)
    cv2.imshow('squares', img)
    [height, width, _] = np.shape(img)
    num_width_steps = 200
    num_height_steps = 200
    acc = np.zeros(np.shape(img)).astype(np.uint8)
    for j in range(num_height_steps):
        for i in range(num_width_steps):
            width_length = int(width/num_width_steps)
            height_length = int(height/num_height_steps)
            mask = np.zeros(np.shape(img)).astype(np.uint8)
            cv2.rectangle(mask, (i * width_length, j * height_length), ((i+1) * width_length - 1, (j+1) * height_length - 1), (255, 255, 255), -1)
            cv2.bitwise_and(img, mask, mask)
            dominant_color = find_dominant_color(mask)
            cv2.rectangle(mask, (i * width_length, j * height_length), ((i+1) * width_length - 1, (j+1) * height_length - 1), dominant_color, -1)
            cv2.bitwise_or(acc, mask, acc)
            cv2.waitKey(1)
            cv2.imshow('tmp', mask)
    cv2.imshow('tmp', acc)
    cv2.waitKey(0)
    cv2.imwrite('squares.jpg', acc)


def get_triangle_lattice(row, col, hexagon_size):
    x = col * hexagon_size
    if row % 2 == 1:
        x += int(hexagon_size/2)
    height_step = hexagon_size/2 * math.sqrt(3)
    y = int(height_step * row)
    return x, y


def get_hexagon_points(hexagon_size, row, col):
    points = []
    if row % 2 == 0:
        points.append(get_triangle_lattice(row, col, hexagon_size))
        points.append(get_triangle_lattice(row+1, col-1, hexagon_size))
        points.append(get_triangle_lattice(row+2, col, hexagon_size))
        points.append(get_triangle_lattice(row+2, col+1, hexagon_size))
        points.append(get_triangle_lattice(row+1, col+1, hexagon_size))
        points.append(get_triangle_lattice(row, col+1, hexagon_size))
    else:
        points.append(get_triangle_lattice(row, col, hexagon_size))
        points.append(get_triangle_lattice(row+1, col, hexagon_size))
        points.append(get_triangle_lattice(row+2, col, hexagon_size))
        points.append(get_triangle_lattice(row+2, col+1, hexagon_size))
        points.append(get_triangle_lattice(row+1, col+2, hexagon_size))
        points.append(get_triangle_lattice(row, col+1, hexagon_size))
    return points


def hexagons(file):
    cv2.namedWindow('hexagons', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('hexagons', 750, 1000)
    cv2.namedWindow('tmp', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('tmp', 750, 1000)

    img = cv2.imread(file)
    cv2.imshow('hexagons', img)
    [height, width, _] = np.shape(img)
    estimated_num_width_steps = 200
    hexagon_size = math.ceil(width / estimated_num_width_steps)
    num_width_steps = math.ceil(width / hexagon_size)
    height_step_length = hexagon_size / 2 * math.sqrt(3)
    num_height_steps = math.ceil(height / height_step_length)
    acc = np.zeros(np.shape(img)).astype(np.uint8)
    for row in range(-1, num_height_steps):
        for col in range((row % 2), num_width_steps, 3):
            triangle_point = get_triangle_lattice(row, col, hexagon_size)
            print("(" + str(row) + ", " + str(col) + ") -> (" + str(triangle_point[0]) + ", " + str(triangle_point[1]) + ")")
            mask = np.zeros(np.shape(img)).astype(np.uint8)
            hexagon = get_hexagon_points(hexagon_size, row, col)
            cv2.fillPoly(mask, np.int32([hexagon]), (255, 255, 255), cv2.LINE_4)
            cv2.bitwise_and(img, mask, mask)
            dominant_color = find_dominant_color(mask)
            cv2.fillPoly(mask, np.int32([hexagon]), dominant_color, cv2.LINE_4)
            cv2.bitwise_or(acc, mask, acc)
            cv2.imshow('tmp', mask)
            cv2.waitKey(1)
    cv2.imwrite('hexagons_high_res.jpg', acc)
    cv2.imshow('tmp', acc)
    cv2.waitKey(0)



    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # if (debug):
    #     cv2.imshow('test', img)
    #     cv2.waitKey(0)
    #
    # img = cv2.medianBlur(img, 3)
    # if debug:
    #     cv2.imshow('test', img)
    #     cv2.waitKey(0)
    #
    # ret, img = cv2.threshold(img, 100, 0, cv2.THRESH_TOZERO)
    # if debug:
    #     cv2.imshow('test', img)
    #     cv2.waitKey(0)
    #
    # img = cv2.Canny(img, 0, 255)
    # if debug:
    #     cv2.imshow('test', img)
    #     cv2.waitKey(0)

    # lpkernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    # cv2.filter2D(src=img, dst=img, ddepth=-1, kernel=cv2.getGaussianKernel(ksize=3, sigma=-1, ktype=cv2.CV_32F))
    # cv2.filter2D(src=img, dst=img, ddepth=-1, kernel=lpkernel)
    # ret, img = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    #
    # # img = cv2.morphologyEx(img, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    # # img = cv2.morphologyEx(img, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    # # img = cv2.morphologyEx(img, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    # # img = cv2.morphologyEx(img, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    # # img = cv2.morphologyEx(img, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    #
    # if debug:
    #     cv2.imshow('test', img)
    #     cv2.waitKey(0)



    # contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # for i in range(len(contours)):
    #     draw_contours = cv2.drawContours(np.zeros(np.shape(img)).astype(np.uint8), contours, i, (255, 255, 255), 2, cv2.LINE_8, hierarchy, 2)
    #     cv2.imshow('test', draw_contours)
    #     cv2.waitKey(0)




    #
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # if (debug):
    #     cv2.imshow('test', img)
    #     cv2.waitKey(0)
    #
    # ret, img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
    # if (debug):
    #     cv2.imshow('test', img)
    #     cv2.waitKey(0)
    #
    # img = img[:,:,2]
    # if (debug):
    #     cv2.imshow('test', img)
    #     cv2.waitKey(0)
    #
    # # Find contours of the tree
    # contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    # contours = cv2.drawContours(np.zeros(np.shape(img)).astype(np.uint8), contours, -1, (255, 255, 255), 2)
    # if debug:
    #     cv2.imshow('test', contours)
    #     cv2.waitKey(0)
    #
    #
    # img = cv2.medianBlur(img, 7)
    # if (debug):
    #     cv2.imshow('test', img)
    #     cv2.waitKey(0)
    #
    # kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    # edges = cv2.filter2D(img, cv2.CV_32F, kernel)
    # edges = cv2.convertScaleAbs(edges)
    # if (debug):
    #     cv2.imshow('test', edges)
    #     cv2.waitKey(0)
    #
    # edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7)))
    # edges = cv2.morphologyEx(edges, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)))
    # if (debug):
    #     cv2.imshow('test', edges)
    #     cv2.waitKey(0)
    #
    #
    #
    # img = cv2.morphologyEx(img, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)))
    # if (debug):
    #     cv2.imshow('test', img)
    #     cv2.waitKey(0)
    # img = cv2.morphologyEx(img, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # if (debug):
    #     cv2.imshow('test', img)
    #     cv2.waitKey(0)
    # img = cv2.morphologyEx(img, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # if (debug):
    #     cv2.imshow('test', img)
    #     cv2.waitKey(0)
    # img = cv2.morphologyEx(img, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
    # if (debug):
    #     cv2.imshow('test', img)
    #     cv2.waitKey(0)
    # img = cv2.morphologyEx(img, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # if (debug):
    #     cv2.imshow('test', img)
    #     cv2.waitKey(0)
    # img = cv2.morphologyEx(img, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    # if (debug):
    #     cv2.imshow('test', img)
    #     cv2.waitKey(0)
    # img = cv2.morphologyEx(img, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
    # if (debug):
    #     cv2.imshow('test', img)
    #     cv2.waitKey(0)
    # img = cv2.morphologyEx(img, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # if (debug):
    #     cv2.imshow('test', img)
    #     cv2.waitKey(0)
    # img = cv2.morphologyEx(img, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # if (debug):
    #     cv2.imshow('test', img)
    #     cv2.waitKey(0)
    # img = cv2.morphologyEx(img, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # if (debug):
    #     cv2.imshow('test', img)
    #     cv2.waitKey(0)
    #
    # #
    # img = cv2.subtract(img, contours)
    # cv2.imshow('test', img)
    # cv2.waitKey(0)
    #
    #
    # num_comp, ltype, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    # img2 = np.empty((np.shape(img)[0], np.shape(img)[1], 3), dtype=np.uint8)
    # img2[:,:,2] = img2[:,:,1] = img2[:,:,0] = img
    # for i in range(1, num_comp):
    #     print(stats[i, cv2.CC_STAT_AREA])
    #     if(stats[i, cv2.CC_STAT_AREA] > 500000 or stats[i, cv2.CC_STAT_AREA] < 50):
    #         continue;
    #     (x1, y1) = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP])
    #     (x2, y2) = (stats[i, cv2.CC_STAT_WIDTH], stats[i,cv2.CC_STAT_HEIGHT])
    #     (x2, y2) = (x1 + x2, y1 + y2)
    #     cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 0, 255), 10)
    #     mask = np.zeros(np.shape(img), dtype=np.uint8)
    #     mask[y1:y1+stats[i, cv2.CC_STAT_HEIGHT], x1:x1+stats[i, cv2.CC_STAT_WIDTH]] = img[y1:y1+stats[i, cv2.CC_STAT_HEIGHT], x1:x1+stats[i, cv2.CC_STAT_WIDTH]]
    #     cv2.floodFill(mask, np.zeros((img.shape[0] +  2, img.shape[1] + 2), dtype=np.uint8), seedPoint=(0,0), newVal=255)
    #     mask = cv2.bitwise_not(mask)
    #     num_pips, mask_ltype, mask_stats, mask_centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    #     print(num_pips - 1)
    #     cv2.imshow('test', mask)
    #     cv2.waitKey(0)
    #
    # cv2.imshow('test', img2)
    # cv2.waitKey(0)

def main():
    random.seed(time.time())
    hexagons('C:\\Users\\andre\\OneDrive\\Desktop\\shinso-ji_temple.jpg')


if __name__ == '__main__':
    main()