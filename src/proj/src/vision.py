#!/home/toshan/miniconda3/envs/tf/bin/python
import cv2
import numpy as np
from PIL import Image
import rospy
from std_msgs.msg import String
import math

yellow = [0, 255, 255]
h = 480
w = 640
y_origin = h//2
x_origin = w//2
x_origin_real = 0
y_origin_real = 0
source = cv2.VideoCapture(2)
source.set(cv2.CAP_PROP_FRAME_HEIGHT,h)
source.set(cv2.CAP_PROP_FRAME_WIDTH,w)
#camera_matrix = np.load('../../../calibrationFiles/cameraMatrix.npy')
#dist_coefs = np.load('../../../calibrationFiles/cameraDistortion.npy')

'''def calibrate(image, camera_matrix, dist_coefs):
    K_undistort = camera_matrix

    img_und = cv2.undistort(image, camera_matrix, dist_coefs, newCameraMatrix=K_undistort)

    return(img_und)
'''
def get_limits(color):
    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]

    if hue >= 165:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit

def Publisher():

    error_lists = [[], []]
    sum_error = [0, 0]   #x, y
    diff_error = [0, 0]
    kp = [0, 0]
    ki = [0, 0]
    kd = [0, 0]

    pub = rospy.Publisher('coordinates', String, queue_size=10)
    rospy.init_node('xy_pub_node', anonymous=False)
    rate = rospy.Rate(50)

    while not rospy.is_shutdown():
        ret, frame = source.read() 

        frame = cv2.resize(frame, (w, h))

        frame = cv2.flip(frame, 1)
               
        #frame = calibrate(frame, camera_matrix, dist_coefs)
        hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lowerLimit, upperLimit = get_limits(color=yellow)

        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
        #y = cv2.bitwise_and(frame, frame, mask=mask)
        mask_ = Image.fromarray(mask)

        bbox = mask_.getbbox()

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            #frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

            x_centre_ball = (x1 + x2)//2
            y_centre_ball = (y1 + y2)//2

            x_centre_real = -(x_origin - x_centre_ball) / 37.795
            y_centre_real = -(y_origin - y_centre_ball) / 37.795

            #frame = cv2.circle(frame, (x_origin,y_origin), radius=0, color=(0, 0, 255), thickness=5)
            #frame = cv2.circle(frame, (x_centre_ball,y_centre_ball), radius=0, color=(0, 0, 255), thickness=5)

            x_error = x_origin_real - x_centre_real
            y_error = y_origin_real - y_centre_real

            error_lists[0].append(x_error)
            error_lists[1].append(y_error)

            nx = -(kp[0]*x_error + ki[0]*sum_error[0] + kd[0]*diff_error[0])
            ny = -(kp[1]*y_error + ki[1]*sum_error[1] + kd[1]*diff_error[1])

            nmag = math.sqrt(math.pow(nx, 2) + math.pow(ny, 2) + 1)
            nx /= nmag
            ny /= nmag
            nz = 1/nmag

            message = f'x_direction : {nx} y_direction : {ny}'

            #frame = cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                
        else:
            message = 'None'

        sum_error[0] += x_error
        sum_error[1] += y_error

        if(len(error_lists[0][0]) == 2):
            diff_error[0] = (error_lists[0][1] - error_lists[0][0])/0.02
            diff_error[1] = (error_lists[1][1] - error_lists[1][0])/0.02

        pub.publish(message)
        rate.sleep()

    source.release() 
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try :
        Publisher()
    except rospy.ROSInterruptException:
        pass