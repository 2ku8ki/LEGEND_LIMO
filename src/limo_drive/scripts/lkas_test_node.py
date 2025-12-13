#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from math import atan2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


class LKAS:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node("LKAS_node")

        # Publishers
        self.ctrl_pub = rospy.Publisher("/cmd_vel_lkas", Twist, queue_size=1)
        self.debug_pub = rospy.Publisher("/sliding_windows/compressed", CompressedImage, queue_size=1)

        # Subscribers
        rospy.Subscriber("/camera/rgb/image_raw/compressed", CompressedImage, self.img_CB, queue_size=1)
        rospy.Subscriber("/lkas_enable", Bool, self.enable_cb, queue_size=1)

        # State
        self.enabled = True
        self.cmd_vel_msg = Twist()
        self.last_pub_time = rospy.get_time()

        # Camera / Warp
        self.img_x = 0
        self.img_y = 0
        self.offset_x = 80

        # Control params
        self.speed = 0.12
        self.turn_gain = 0.0038
        self.lookahead_y_ratio = 0.62

        # Sliding window
        self.nwindows = 10
        self.margin = 80
        self.minpix = 35

    def enable_cb(self, msg: Bool):
        self.enabled = msg.data

    def img_CB(self, data: CompressedImage):
        if not self.enabled:
            return

        now = rospy.get_time()
        if now - self.last_pub_time < 0.1:
            return

        img = self.bridge.compressed_imgmsg_to_cv2(data)
        self.img_y, self.img_x = img.shape[0], img.shape[1]

        # Warp image
        warp = self.img_warp(img)
        binary = self.make_lane_binary(warp)

        # Sliding window for lane detection
        debug_img, lane_ok, center_x, departure = self.lane_center_from_windows(binary)

        # Debug image publish
        self.publish_debug(debug_img)

        # If lane detection failed, stop the vehicle
        if not lane_ok:
            self.publish_cmd(0.08, 0.0)
            self.last_pub_time = now
            return

        # Calculate steering based on lane center offset
        img_center = binary.shape[1] * 0.5
        err_px = (center_x - img_center)

        # Low-pass filter to smooth the center of lane
        if self.prev_center_x is None:
            self.prev_center_x = center_x
        else:
            self.prev_center_x = 0.7 * self.prev_center_x + 0.3 * center_x

        steer = -err_px * self.turn_gain

        # Publish control command
        self.publish_cmd(self.speed, steer)
        self.last_pub_time = now

    def img_warp(self, img):
        h, w = img.shape[0], img.shape[1]

        # src: Trapezoidal points (can be adjusted for your environment)
        src_center_offset = [200, 315]
        src = np.array(
            [
                [0, h - 1],
                [src_center_offset[0], src_center_offset[1]],
                [w - src_center_offset[0], src_center_offset[1]],
                [w - 1, h - 1],
            ],
            dtype=np.float32,
        )

        dst = np.array(
            [
                [self.offset_x, h],
                [self.offset_x, 0],
                [w - self.offset_x, 0],
                [w - self.offset_x, h],
            ],
            dtype=np.float32,
        )

        M = cv2.getPerspectiveTransform(src, dst)
        warp_img = cv2.warpPerspective(img, M, (w, h))
        return warp_img

    def make_lane_binary(self, warp_bgr):
        """
        목표:
        - 차선만 인식하고, 숫자/문자/기타 잡음을 무시
        """
        hsv = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2HSV)

        # 차선 색(흰색, 노란색 차선) 범위
        white = cv2.inRange(hsv, (0, 0, 205), (179, 60, 255))
        yellow = cv2.inRange(hsv, (15, 80, 60), (45, 255, 255))
        mask = cv2.bitwise_or(white, yellow)

        # 작은 노이즈 제거
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

        # 숫자/중앙 억제: 중앙 영역을 제외하고 차선만 남기기
        h, w = mask.shape
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        keep = np.zeros((h, w), dtype=np.uint8)

        left_band_end = int(w * 0.46)
        right_band_start = int(w * 0.54)
        bottom_band = int(h * 0.35)
        bottom_touch = h - bottom_band

        strict_center_exclude = int(w * 0.10)
        cx0 = (w // 2) - strict_center_exclude
        cx1 = (w // 2) + strict_center_exclude

        min_area = int(h * w * 0.00018)

        for i in range(1, num):
            x, y, ww, hh, area = stats[i]
            if area < min_area:
                continue

            cy = centroids[i][1]
            cx = centroids[i][0]
            bottom = y + hh

            if bottom < bottom_touch:
                continue

            in_left = (cx < left_band_end)
            in_right = (cx > right_band_start)

            if (not in_left) and (not in_right):
                if cx0 <= cx <= cx1:
                    continue
                if cy < h * 0.55:
                    continue

            keep[labels == i] = 255

        # Slightly dilate to make lane more visible
        keep = cv2.dilate(keep, k, iterations=1)

        binary = (keep > 0).astype(np.uint8)  # Convert to 0/1 binary
        return binary

    def lane_center_from_windows(self, binary):
        h, w = binary.shape

        # Bottom half histogram
        hist = np.sum(binary[h // 2:, :], axis=0)

        midpoint = w // 2
        leftx_base = int(np.argmax(hist[:midpoint]))
        rightx_base = int(np.argmax(hist[midpoint:]) + midpoint)

        # Initial check for valid lane positions
        lane_ok = True
        if leftx_base < 5 or rightx_base > (w - 5):
            lane_ok = False

        # Sliding window parameters
        nwindows = self.nwindows
        window_height = h // nwindows
        margin = self.margin

        nonzero = binary.nonzero()
        nonzeroy = np.array(nonzero[0], dtype=np.int32)
        nonzerox = np.array(nonzero[1], dtype=np.int32)

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        out_img = (np.dstack((binary, binary, binary)) * 255).astype(np.uint8)

        minpix = max(self.minpix, int((2 * margin * window_height) * 0.0025))

        for window in range(nwindows):
            win_y_low = h - (window + 1) * window_height
            win_y_high = h - window * window_height

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 0, 255), 2)

            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)

            if len(good_left) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left]))
            if len(good_right) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right]))

        left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) else np.array([], dtype=int)
        right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) else np.array([], dtype=int)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if (len(leftx) < 30) and (len(rightx) < 30):
            return out_img, False, None, True

        y_look = int(h * self.lookahead_y_ratio)
        if len(leftx) >= 30 and len(rightx) >= 30:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            lx = left_fit[0] * y_look**2 + left_fit[1] * y_look + left_fit[2]
            rx = right_fit[0] * y_look**2 + right_fit[1] * y_look + right_fit[2]
            center_x = 0.5 * (lx + rx)
        else:
            if len(leftx) >= 30:
                left_fit = np.polyfit(lefty, leftx, 2)
                lx = left_fit[0] * y_look**2 + left_fit[1] * y_look + left_fit[2]
                center_x = lx + 0.5 * self.lane_width_px
            else:
                right_fit = np.polyfit(righty, rightx, 2)
                rx = right_fit[0] * y_look**2 + right_fit[1] * y_look + right_fit[2]
                center_x = rx - 0.5 * self.lane_width_px

            lane_ok = False

        img_center = w * 0.5
        departure = (abs(center_x - img_center) > (w * 0.42))

        cv2.circle(out_img, (int(center_x), y_look), 6, (255, 255, 0), -1)

        return out_img, lane_ok, float(center_x), departure

    def publish_cmd(self, v, steer):
        msg = Twist()
        msg.linear.x = float(np.clip(v, 0.0, 0.25))
        msg.angular.z = float(np.clip(steer, -1.6, 1.6))
        self.ctrl_pub.publish(msg)

if __name__ == "__main__":
    lkas = LKAS()
    rospy.spin()