import cv2
import numpy as np
#import matplotlib.pyplot as plt
def mk_coord(im, l_para):
    slope, intercept = l_para
    print(im.shape)
    y1 = im.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1,y1,x2,y2])
def avg_slope_intercept(im, ln):
    l_fit = []
    r_fit = []
    for l in ln:
        x1,y1,x2,y2 = l.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2), 1)
        #print(parameters)
        slope = parameters[0]
        intercept = parameters[1]
        if slope<0:
            l_fit.append((slope,intercept))
        else:
            r_fit.append((slope,intercept))
    #print(l_fit,r_fit)
    l_fit_avg = np.average(l_fit, axis=0)
    r_fit_avg = np.average(r_fit, axis=0)
    #print(l_fit_avg,r_fit_avg)
    l_line = mk_coord(im, l_fit_avg)
    r_line = mk_coord(im, r_fit_avg)
    return np.array([l_line, r_line])

def canny(im):
    l_im = np.copy(im)
    gr = cv2.cvtColor(l_im, cv2.COLOR_RGB2GRAY) #convert to grayscale
    blur = cv2.GaussianBlur(gr, (5, 5), 0) #smoothening
    canny = cv2.Canny(blur, 50, 150)
    return canny
def display_lines(im, ln):
    line_image = np.zeros_like(im)
    if ln is not None:
        for x1,y1,x2,y2 in ln:
            #x1,y1,x2,y2 = l#.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (0,255,0), 10)
    return line_image
def roi(im):
    h = im.shape[0]
    tri = np.array([[(280,h),(1100,h),(550,250)]])
    mask = np.zeros_like(im)
    cv2.fillPoly(mask, tri, 255) #cv2.fillPoly(mask, pts = [tri], color =(255,255,255))
    masked_image = cv2.bitwise_and(im, mask)
    return masked_image


# image = cv2.imread('test_image.jpg')
# lane_image = np.copy(image)
# canny_image = canny(lane_image)
# cropped_image = roi(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# avg_lines = avg_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, avg_lines)
# combo_image = cv2.addWeighted(lane_image, 1, line_image, 0.5, 1)
# cv2.imshow('res', combo_image)
# cv2.waitKey(0)

c = cv2.VideoCapture(0)
while(c.isOpened()):
    _, frame = c.read()
    canny_image = canny(frame)
    cropped_image = roi(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    avg_lines = avg_slope_intercept(frame, lines)
    line_image = display_lines(frame, avg_lines)
    combo_image = cv2.addWeighted(frame, 1, line_image, 1, 1)
    cv2.imshow('res', combo_image)
    if cv2.waitKey(2) == ord('s'):
        break
c.release()
cv2.destroyAllWindows()