# Don't name this file 'cv2.py' or 'opencv.py' (it can shadow the real cv2 package)
import cv2
import numpy as np

min_contour_width = 40  
min_contour_height = 40  
offset = 10  
line_height = 550  
matches = []
vehicles = 0

def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

cap = cv2.VideoCapture('Video.mp4')
cap.set(3, 1920)
cap.set(4, 1080)

if cap.isOpened():
    ret, frame1 = cap.read()
else:
    ret = False

ret, frame1 = cap.read()
ret, frame2 = cap.read()

# Resize factor
resize_factor = 0.5

while ret:
    # Absolute difference between frames
    d = cv2.absdiff(frame1, frame2)
    cv2.imshow("Absolute Difference", cv2.resize(d, None, fx=resize_factor, fy=resize_factor))

    # Convert to grayscale
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale", cv2.resize(grey, None, fx=resize_factor, fy=resize_factor))

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    cv2.imshow("Gaussian Blur", cv2.resize(blur, None, fx=resize_factor, fy=resize_factor))

    # Thresholding
    ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    cv2.imshow("Threshold", cv2.resize(th, None, fx=resize_factor, fy=resize_factor))

    # Dilation
    dilated = cv2.dilate(th, np.ones((3, 3)))
    cv2.imshow("Dilated", cv2.resize(dilated, None, fx=resize_factor, fy=resize_factor))

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Closing", cv2.resize(closing, None, fx=resize_factor, fy=resize_factor))

    # Find contours
    contours, h = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    frame_contours = frame1.copy()
    cv2.drawContours(frame_contours, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Contours", cv2.resize(frame_contours, None, fx=resize_factor, fy=resize_factor))

    for(i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        contour_valid = (w >= min_contour_width) and (h >= min_contour_height)

        if not contour_valid:
            continue

        cv2.rectangle(frame1, (x-10, y-10), (x+w+10, y+h+10), (255, 0, 0), 2)
        cv2.line(frame1, (0, line_height), (1200, line_height), (0, 255, 0), 2)
        centroid = get_centroid(x, y, w, h)
        matches.append(centroid)
        cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)
        cv2.imshow("Bounding Boxes", cv2.resize(frame1, None, fx=resize_factor, fy=resize_factor))

        cx, cy = get_centroid(x, y, w, h)
        for (x, y) in matches:
            if y < (line_height+offset) and y > (line_height-offset):
                vehicles += 1
                matches.remove((x, y))

    cv2.putText(frame1, "Total Vehicle Detected: " + str(vehicles), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 170, 0), 2)

    cv2.imshow("Vehicle Detection", cv2.resize(frame1, None, fx=resize_factor, fy=resize_factor))
    if cv2.waitKey(1) == 27:
        break
    frame1 = frame2
    ret, frame2 = cap.read()

cv2.destroyAllWindows()
cap.release()