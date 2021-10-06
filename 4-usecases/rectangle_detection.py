
import numpy as np
import cv2

from image import normalize_size

def convert_homography_points(homography_points):
    new_homography_points = []
    for points in homography_points:
        new_points = []
        for point in points:
            new_points.append(float(point))
        new_homography_points.append(new_points)
    return new_homography_points

def wrap(box, img):
    input_pts, max_height, max_width, output_pts = reorder_points(box)
    homography_points = cv2.getPerspectiveTransform(input_pts, output_pts)
    img_out = cv2.warpPerspective(img, homography_points, (max_width, max_height), flags=cv2.INTER_LINEAR)
    return img_out, convert_homography_points(homography_points)


def reorder_points(box):
    (pt_A, pt_B, pt_C, pt_D) = box
    points_x = [pt_A[0], pt_B[0], pt_C[0], pt_D[0]]
    points = [pt_A, pt_B, pt_C, pt_D]
    min_x1 = min(points_x)
    points_x.remove(min_x1)
    min_x2 = min(points_x)
    points_x.remove(min_x2)
    p1 = None
    p2 = None
    other_points = []
    for point in points:
        if p1 is None and point[0] == min_x1:
            p1 = point
        elif p2 is None and point[0] == min_x2:
            p2 = point
        else:
            other_points.append(point)
    if p1[1] > p2[1]:
        pt_B = p1
        pt_A = p2
    else:
        pt_B = p2
        pt_A = p1
    points_y = [other_points[0][1], other_points[1][1]]
    min_y3 = min(points_y)
    points_y.remove(min_y3)
    min_y4 = min(points_y)
    p3 = None
    p4 = None
    for point in other_points:
        if p3 is None and point[1] == min_y3:
            p3 = point
        if p4 is None and point[1] == min_y4:
            p4 = point
    pt_D = p3
    pt_C = p4
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    max_width = max(int(width_AD), int(width_BC))
    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    max_height = max(int(height_AB), int(height_CD))
    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                             [0, max_height - 1],
                             [max_width - 1, max_height - 1],
                             [max_width - 1, 0]])
    return input_pts, max_height, max_width, output_pts

def _find_contour(threshold, destination_path="", threshold_min_are=0.48, contour_mode=cv2.RETR_EXTERNAL):
    (h, w) = threshold.shape[:2]
    origin_area = h * w
    percentage_area_min = int(origin_area * threshold_min_are)
    contours, _ = cv2.findContours(threshold, contour_mode,
                                   cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    contours_area = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        contours_area.append({"area": area, "contour": cnt})
        areas.append(area)

    if len(areas) == 0:
        return None

    max_area = max(areas)
    if max_area < percentage_area_min:
        return None
    # Searching through every region selected to
    # find the required polygon.
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Shortlisting the regions based on there area.
        if area == max_area:
            return cnt

    return None


def compute_outer_crop_margin_ratio(img_width, img_heigth, coordinates, margin_ratio):
    if margin_ratio <= 0:
        return coordinates
    xmin, ymin, xmax, ymax = coordinates
    margin_ratio = int(max(img_width, img_heigth) * margin_ratio / 100)
    xmin = xmin - margin_ratio
    ymin = ymin - margin_ratio
    xmax = xmax + margin_ratio
    ymax = ymax + margin_ratio
    if xmin < 0:
        xmin=0
    if ymin < 0:
        ymin=0
    if xmax > img_width:
        xmax=img_width
    if ymax > img_heigth:
        ymax=img_heigth
    return xmin, ymin, xmax, ymax

class BackgroundExtractor:
    def __init__(self, target_img, target_img_small, environment_mode="production"):
        self.target_img = target_img
        self.target_img_small = target_img_small
        self.bgdModel = np.zeros((1, 65), np.float64)
        self.fgdModel = np.zeros((1, 65), np.float64)
        self.mask = np.zeros(target_img_small.shape[:2], np.uint8)
        self.environment_mode = environment_mode

    def extract_background(self, rectangle, nb_iterations=1, destination_path=""):
        target_img = self.target_img
        img2 = self.target_img_small
        mask = self.mask
        if nb_iterations <= 0:
            return target_img

        cv2.grabCut(img2, mask, rectangle, self.bgdModel, self.fgdModel, nb_iterations, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img2 = img2 * mask2[:, :, np.newaxis]
        if self.environment_mode == "development":
            cv2.imwrite(destination_path + "3_contour_origin.png", img2)
        (h, w) = target_img.shape[:2]
        mask3 = cv2.resize(mask2, (w, h), interpolation=cv2.INTER_AREA)
        target_img = target_img * mask3[:, :, np.newaxis]
        return target_img

def add_border(src, color=[0, 0, 0], border_type=cv2.BORDER_CONSTANT):
    top = int(0.05 * src.shape[0])  # shape[0] = rows
    bottom = top
    left = int(0.05 * src.shape[1])  # shape[1] = cols
    right = left
    border_type = border_type
    value = color
    dst = cv2.copyMakeBorder(src, top, bottom, left, right, border_type, None, value)
    return dst, top, left

def add_border_cleaned(thresh):
    (H, W) = thresh.shape[:2]
    thresh, border_size_top, border_size_left = add_border(thresh)
    top = int(0.02 * thresh.shape[0])
    left = int(0.02 * thresh.shape[1])
    margin = int(top + left / 2)
    cv2.rectangle(thresh, (border_size_left, border_size_top), (W + border_size_left, H + border_size_top), (0, 0, 0),
                  margin)
    return thresh

def extract_rectangle(target_img, destination_path="", environment_mode="production"):
    gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    thresh = add_border_cleaned(thresh)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    dilate_iteration = 16
    img_tmp_size = 360
    thresh = cv2.dilate(thresh, kernel, iterations=dilate_iteration)
    if environment_mode == "development":
        cv2.imwrite(destination_path + "1_threshold.png", thresh)
    contour = _find_contour(thresh, destination_path=destination_path)
    target_img, border_size_top, border_size_left = add_border(target_img, border_type=cv2.BORDER_REPLICATE)
    if environment_mode == "development":
        cv2.imwrite(destination_path + "2_margin.png", target_img)
    img2 = target_img.copy()
    img2, ratio = normalize_size(img2, img_tmp_size)
    color = (0, 0, 255)
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        coordinates = (int(x * ratio), int(y * ratio), int(w * ratio), int(h * ratio))
        rectangle = coordinates
    else:
        x, y, w, h = (border_size_left, border_size_top, target_img.shape[1] - border_size_left * 2,
                      target_img.shape[0] - border_size_top * 2)
        rectangle = (int(x * ratio), int(y * ratio), int(w * ratio), int(h * ratio))
        color = (0, 255, 0)
    background_extrator = BackgroundExtractor(target_img.copy(), img2, environment_mode)
    img_foreground = background_extrator.extract_background(rectangle, 7, destination_path=destination_path)
    if environment_mode == "development":
        cv2.imwrite(destination_path + "4_img_foreground.png", img_foreground)

    contour2 = _find_contour(cv2.cvtColor(img_foreground, cv2.COLOR_BGR2GRAY), destination_path=destination_path,
                             contour_mode=cv2.RETR_EXTERNAL)
    if contour2 is not None:
        contour = contour2
        color = (255, 0, 0)

    if contour is not None:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if environment_mode == "development":
            cv2.drawContours(img_foreground, [box], 0, color, 20)
            cv2.imwrite(destination_path + "5_img_foreground.png", img_foreground)
        return wrap(box, target_img)
    else:
        destination_img = target_img
    return destination_img, None