from Matcher import Matcher
from ROI import ROI
import numpy as np
import cv2
import argparse
'''
Author: Zhaorui Chen 2017

Code based on OpenCV doc - python tutorial.
'''

# The intrinsic parameters for macbook-pro early 2015
# generated using AR toolbox
k1 = 0.0814318880
k2 = -0.0729859173
p1 = 0.0077049430
p2 = -0.0046416679
fx = 1019.955688
fy = 1019.131287
x0 = 643.416321
y0 = 381.590668
s = 0.957716


# Set up the intrinsix matrices
disCoeff = np.float32([k1, k2, p1, p2])
cameraMatrix = np.float32([[fx, 0.0, x0], [0.0, fy, y0], [0.0, 0.0, 1.0]])


class App:
    def __init__(self, alg, mode):
        self.referencePoints = []
        self.cropping = False
        self.roi = None  # the marker object
        self.alg = alg  # the algorithm to choose: sift/orb
        self.mode = mode  # the way of choosing marker: static/screen capture

    # Code borrowed from:
    # http://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
    def click_and_crop(self, event, x, y, flags, param):

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.referencePoints = [(x, y)]
            self.cropping = True

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            self.referencePoints.append((x, y))
            self.cropping = False

    def draw(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
        return img

    def main(self):
        ''' Screencapture from the webcam frame to get the marker pattern '''

        cap = cv2.VideoCapture(0)

        if self.mode == 'capture':
            while True:

                # Capture frame-by-frame
                ret, frame = cap.read()

                currentFrame = frame.copy()
                cv2.namedWindow("choose marker")
                cv2.setMouseCallback("choose marker", self.click_and_crop)

                # Display the resulting frame
                cv2.imshow('choose marker', frame)

                # if there are two reference points, then crop the region of interest
                # from teh image and display it
                if len(self.referencePoints) == 2:
                    cropImage \
                        = currentFrame[self.referencePoints[0][1]:self.referencePoints[1][1],
                                       self.referencePoints[0][0]:self.referencePoints[1][0]]
                    cv2.rectangle(
                        currentFrame,
                        self.referencePoints[0],
                        self.referencePoints[1], (0, 255, 0), 2)
                    # initialize a marker object for the marker
                    self.roi = ROI(cropImage, self.alg)

                    cv2.imshow('choose marker', currentFrame)
                    cv2.waitKey(1000)
                    cv2.destroyWindow('choose marker')
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break
        else:
            roi = cv2.imread('roi_rec.jpg')
            print(roi.shape)
            roi = cv2.resize(roi, (200, 200))
            self.roi = ROI(roi, self.alg)

        cv2.waitKey(100)
        matcher = Matcher(self.roi, self.alg, disCoeff, cameraMatrix)

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            currentFrame = frame.copy()
            mirrorFrame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)

            cv2.namedWindow('webcam')
            cv2.imshow('webcam', currentFrame)

            matcher.setFrame(currentFrame)

            result = matcher.getCorrespondence()
            if result:
                # get the corners
                (src, dst, corners) = result
            else:
                # Not enough matching points found
                print('Not enough points')
                cv2.waitKey(1)
                continue

            (retvalCorner, rvecCorner, tvecCorner) = matcher.computePose(
                self.roi.getPoints3d(), corners)

            # if(rvecCorner[0] < 0):
            #     rvecCorner = -rvecCorner

            print(rvecCorner)

            if retvalCorner:
                # Set up where to draw the axises of the the cube in the frame
                # axis = np.float32(
                #         [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                #          [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]])
                axis = np.float32(
                        [[1, 0, -1], [1, 1, -1], [0, 1, -1], [0, 0, -1],
                         [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])


                # project 3d points to 2d coordinates in the frame coordination
                imgpts, jac = cv2.projectPoints(
                    axis, rvecCorner, tvecCorner, cameraMatrix, disCoeff)

                # re-draw the frame
                currentFrame = cv2.polylines(
                    currentFrame,
                    [np.int32(corners)], True, 255, 3, cv2.LINE_AA)
                # currentFrame = self.drawlines(currentFrame, imgpts)
                cv2.imshow('webcam', currentFrame)
                cv2.waitKey(1)

            else:
                cv2.waitKey(1)
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

    def drawlines(self, img, imgpts):
        imgpts = np.int32(imgpts).reshape(-1, 2)
        # draw floor in grey
        # img = cv2.drawContours(img, [imgpts[:4]], -1, (175, 175, 175), -3)
        # draw pillars in blue color
        # for i, j in zip(range(4), range(4, 8)):
        # for i, j in zip(range(4), range(4, 8)):
        #     img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

        # img = cv2.line(img,
        #                tuple((imgpts[0] + imgpts[1] + imgpts[2] + imgpts[3]) // 4),
        #                tuple((imgpts[0] + imgpts[1]) // 2),
        #                (0, 255, 0), 3)
        # img = cv2.line(img,
        #                tuple((imgpts[0] + imgpts[1] + imgpts[2] + imgpts[3]) // 4),
        #                tuple((imgpts[1] + imgpts[2]) // 2),
        #                (0, 0, 255), 3)
        img = cv2.line(img,
                       tuple((imgpts[4] + imgpts[5] + imgpts[6] + imgpts[7]) // 4),
                       tuple((imgpts[4] + imgpts[5]) // 2),
                       (0, 255, 0), 3)
        img = cv2.line(img,
                       tuple((imgpts[4] + imgpts[5] + imgpts[6] + imgpts[7]) // 4),
                       tuple((imgpts[5] + imgpts[6]) // 2),
                       (0, 0, 255), 3)
        img = cv2.line(img,
                       tuple((imgpts[0] + imgpts[1] + imgpts[2] + imgpts[3]) // 4),
                       tuple((imgpts[4] + imgpts[5] + imgpts[6] + imgpts[7]) // 4),
                       (255, 0, 0), 3)

        return img


if __name__ == '__main__':
    app = App("orb", "static")
    app.main()
