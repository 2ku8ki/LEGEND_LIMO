#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import rospy
import numpy as np
import cv2

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from std_msgs.msg import Bool


def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


class LKAS(object):
    def __init__(self):
        rospy.init_node("lkas_node")

        # -----------------------------
        # (FSM 호환) 토픽 파라미터
        # -----------------------------
        self.image_topic  = rospy.get_param("~image_topic",  "/camera/rgb/image_raw/compressed")
        self.cmd_topic    = rospy.get_param("~cmd_topic",    "/cmd_vel_lkas")
        self.enable_topic = rospy.get_param("~enable_topic", "/lkas_enable")
        self.use_fsm      = rospy.get_param("~use_fsm", True)

        self.bridge = CvBridge()

        # Publishers/Subscribers
        self.ctrl_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        rospy.Subscriber(self.image_topic, CompressedImage, self.img_CB, queue_size=1)
        rospy.Subscriber(self.enable_topic, Bool, self.enable_cb, queue_size=1)

        # -----------------------------
        # 주행 파라미터
        # -----------------------------
        self.speed      = float(rospy.get_param("~speed", 0.18))
        self.turn_mult  = float(rospy.get_param("~turn_mult", 0.14))
        self.max_yaw    = float(rospy.get_param("~max_yaw", 0.8))   # 조향 과도 방지

        # 프레임 스킵
        self.frame_skip = int(rospy.get_param("~frame_skip", 3))
        self._frame_count = 0

        # warp 관련
        self.img_x = 0
        self.img_y = 0
        self.offset_x = int(rospy.get_param("~offset_x", 40))

        # 슬라이딩 윈도우
        self.nwindows = int(rospy.get_param("~nwindows", 10))
        self.window_height = 0
        self.nothing_flag = False

        # publish rate limit
        self.last_pub_time = rospy.get_time()
        self.pub_period = float(rospy.get_param("~pub_period", 0.1))
        self.have_published_cmd = False

        # enable 상태
        if self.use_fsm:
            self.enabled = False
        else:
            self.enabled = True
        self.prev_enabled = self.enabled

        # -----------------------------
        # 디버그(선택)
        # -----------------------------
        self.debug_view  = bool(rospy.get_param("~debug_view", False))
        self.debug_wait  = int(rospy.get_param("~debug_wait", 1))
        self.debug_every = int(rospy.get_param("~debug_every", 1))
        if self.debug_view and not os.environ.get("DISPLAY"):
            rospy.logwarn("[LKAS] DISPLAY not set. Disable imshow to avoid crash.")
            self.debug_view = False

        self.debug_throttle = float(rospy.get_param("~debug_throttle", 1.0))  # log throttle sec

        # -----------------------------
        # (핵심) “차선 사이 동적 제거 + 폭 벌어지면 한쪽만 추종” 파라미터
        # -----------------------------
        self.roi_y_start_ratio = float(rospy.get_param("~roi_y_start_ratio", 0.45))  # 하단 ROI 시작
        self.roi_y_end_ratio   = float(rospy.get_param("~roi_y_end_ratio",   1.00))  # 하단 ROI 끝
        self.between_margin_px = int(rospy.get_param("~between_margin_px", 12))      # 차선 안쪽 여유

        self.hist_left_max_ratio  = float(rospy.get_param("~hist_left_max_ratio", 0.45))
        self.hist_right_min_ratio = float(rospy.get_param("~hist_right_min_ratio", 0.55))
        self.min_lane_sep_ratio   = float(rospy.get_param("~min_lane_sep_ratio", 0.20))

        # lane width EMA
        self.lane_width_ema = None
        self.lane_width_alpha = float(rospy.get_param("~lane_width_alpha", 0.20))
        self.widen_factor = float(rospy.get_param("~widen_factor", 1.35))
        self.nominal_lane_width_ratio = float(rospy.get_param("~nominal_lane_width_ratio", 0.52))

        # fallback 중앙밴드(동적 제거 실패 시)
        self.fallback_center_band_half_ratio = float(
            rospy.get_param("~fallback_center_band_half_ratio", 0.23)
        )

        rospy.loginfo("[LKAS] image_topic=%s cmd_topic=%s enable_topic=%s use_fsm=%s",
                      self.image_topic, self.cmd_topic, self.enable_topic, str(self.use_fsm))

    def enable_cb(self, msg):
        self.enabled = bool(msg.data)

        # enable True -> False 순간에 stop 1회 publish (잔류 cmd 방지)
        if (self.prev_enabled is True) and (self.enabled is False) and self.have_published_cmd:
            stop = Twist()
            stop.linear.x = 0.0
            stop.angular.z = 0.0
            self.ctrl_pub.publish(stop)
        self.prev_enabled = self.enabled

    # -----------------------------
    # 색 기반 차선 검출
    # -----------------------------
    def detect_color(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        yellow_lower = np.array([15, 80, 0], dtype=np.uint8)
        yellow_upper = np.array([45, 255, 255], dtype=np.uint8)

        white_lower = np.array([0, 0, 230], dtype=np.uint8)
        white_upper = np.array([179, 40, 255], dtype=np.uint8)

        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        white_mask  = cv2.inRange(hsv, white_lower,  white_upper)

        blend_mask = yellow_mask | white_mask
        return cv2.bitwise_and(img, img, mask=blend_mask)

    # -----------------------------
    # BEV warp
    # -----------------------------
    def img_warp(self, img):
        self.img_x, self.img_y = img.shape[1], img.shape[0]

        src_center_offset = [100, 158]
        src = np.array(
            [
                [0, self.img_y - 1],
                [src_center_offset[0], src_center_offset[1]],
                [self.img_x - src_center_offset[0], src_center_offset[1]],
                [self.img_x - 1, self.img_y - 1],
            ],
            dtype=np.float32,
        )

        dst = np.array(
            [
                [self.offset_x, self.img_y],
                [self.offset_x, 0],
                [self.img_x - self.offset_x, 0],
                [self.img_x - self.offset_x, self.img_y],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, matrix, (self.img_x, self.img_y))

    # -----------------------------
    # 동적 “차선 사이 제거”
    # -----------------------------
    def remove_between_lanes(self, binary_line, left_base, right_base, y0, y1):
        if left_base is None or right_base is None:
            return binary_line
        if right_base <= left_base:
            return binary_line

        x0 = int(left_base + self.between_margin_px)
        x1 = int(right_base - self.between_margin_px)
        x0 = max(0, x0)
        x1 = min(binary_line.shape[1], x1)
        if x1 <= x0:
            return binary_line

        binary_line[y0:y1, x0:x1] = 0
        return binary_line

    # -----------------------------
    # 이진화 + “차선 사이” 동적 제거
    # -----------------------------
    def img_binary(self, blend_line):
        gray = cv2.cvtColor(blend_line, cv2.COLOR_BGR2GRAY)
        _, binary_line = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY)

        h, w = binary_line.shape
        y0 = int(h * self.roi_y_start_ratio)
        y1 = int(h * self.roi_y_end_ratio)
        y0 = clamp(y0, 0, h)
        y1 = clamp(y1, 0, h)
        if y1 <= y0:
            y0 = int(h * 0.45)
            y1 = h

        roi = binary_line[y0:y1, :]
        histogram = np.sum(roi, axis=0)

        left_end = int(w * self.hist_left_max_ratio)
        right_start = int(w * self.hist_right_min_ratio)
        left_end = clamp(left_end, 1, w - 1)
        right_start = clamp(right_start, 0, w - 2)

        left_base = None
        right_base = None

        # 좌측 피크
        if left_end > 10:
            lidx = int(np.argmax(histogram[:left_end]))
            if histogram[lidx] > 0:
                left_base = lidx

        # 우측 피크
        if right_start < w - 10:
            ridx = int(np.argmax(histogram[right_start:])) + right_start
            if histogram[ridx] > 0:
                right_base = ridx

        # 좌우 간격이 너무 좁으면(중앙 잡음/숫자) 무효
        if left_base is not None and right_base is not None:
            if (right_base - left_base) < int(w * self.min_lane_sep_ratio):
                left_base, right_base = None, None

        # 한쪽만 잡히면 폭 추정으로 반대쪽 보정
        width_est = self.lane_width_ema if self.lane_width_ema is not None else (w * self.nominal_lane_width_ratio)
        if left_base is not None and right_base is None:
            right_base = int(left_base + width_est)
        if right_base is not None and left_base is None:
            left_base = int(right_base - width_est)

        if left_base is not None:
            left_base = clamp(left_base, 0, w - 1)
        if right_base is not None:
            right_base = clamp(right_base, 0, w - 1)

        if left_base is not None and right_base is not None and right_base > left_base:
            binary_line = self.remove_between_lanes(binary_line, left_base, right_base, y0, y1)
        else:
            # fallback: 중앙 밴드 제거(동적 실패시)
            cx = w // 2
            half = int(w * self.fallback_center_band_half_ratio)
            x0b = clamp(cx - half, 0, w)
            x1b = clamp(cx + half, 0, w)
            binary_line[y0:y1, x0b:x1b] = 0

        return binary_line

    # -----------------------------
    # nothing fallback
    # -----------------------------
    def detect_nothing(self):
        offset = int(self.img_x * 0.140625)
        self.nothing_left_x_base = offset
        self.nothing_right_x_base = self.img_x - offset

        self.nothing_pixel_left_x = np.full(self.nwindows, self.nothing_left_x_base, dtype=np.int32)
        self.nothing_pixel_right_x = np.full(self.nwindows, self.nothing_right_x_base, dtype=np.int32)

        base_y = int(self.window_height / 2)
        self.nothing_pixel_y = np.arange(0, self.nwindows * base_y, base_y, dtype=np.int32)

    # -----------------------------
    # sliding window
    # -----------------------------
    def window_search(self, binary_line):
        h, w = binary_line.shape

        bottom_half = binary_line[h // 2:, :]
        histogram = np.sum(bottom_half, axis=0)

        midpoint = w // 2
        left_x_base = int(np.argmax(histogram[:midpoint]))
        right_x_base = int(np.argmax(histogram[midpoint:]) + midpoint)

        left_x_current = left_x_base if left_x_base != 0 else self.nothing_left_x_base
        right_x_current = right_x_base if right_x_base != midpoint else self.nothing_right_x_base

        out_img = (np.dstack((binary_line, binary_line, binary_line)).astype(np.uint8) * 255)

        margin = 80
        min_pix = int((margin * 2 * self.window_height) * 0.005)

        lane_y, lane_x = binary_line.nonzero()
        lane_y = lane_y.astype(np.int32)
        lane_x = lane_x.astype(np.int32)

        left_lane_idx_list = []
        right_lane_idx_list = []

        for window in range(self.nwindows):
            win_y_low  = h - (window + 1) * self.window_height
            win_y_high = h - window * self.window_height

            left_low  = left_x_current - margin
            left_high = left_x_current + margin
            right_low  = right_x_current - margin
            right_high = right_x_current + margin

            in_window = (lane_y >= win_y_low) & (lane_y < win_y_high)

            good_left_idx = np.where(in_window & (lane_x >= left_low) & (lane_x < left_high))[0]
            good_right_idx = np.where(in_window & (lane_x >= right_low) & (lane_x < right_high))[0]

            left_lane_idx_list.append(good_left_idx)
            right_lane_idx_list.append(good_right_idx)

            if len(good_left_idx) > min_pix:
                left_x_current = int(np.mean(lane_x[good_left_idx]))
            if len(good_right_idx) > min_pix:
                right_x_current = int(np.mean(lane_x[good_right_idx]))

        left_lane_idx = np.concatenate(left_lane_idx_list) if left_lane_idx_list else np.array([], dtype=int)
        right_lane_idx = np.concatenate(right_lane_idx_list) if right_lane_idx_list else np.array([], dtype=int)

        left_x = lane_x[left_lane_idx]
        left_y = lane_y[left_lane_idx]
        right_x = lane_x[right_lane_idx]
        right_y = lane_y[right_lane_idx]

        left_count = int(len(left_x))
        right_count = int(len(right_x))

        # fallback
        if left_count == 0 and right_count == 0:
            left_x = self.nothing_pixel_left_x
            left_y = self.nothing_pixel_y
            right_x = self.nothing_pixel_right_x
            right_y = self.nothing_pixel_y
            left_count = 0
            right_count = 0
        else:
            if left_count == 0:
                left_x = right_x - self.img_x // 2
                left_y = right_y
                left_count = 0
            elif right_count == 0:
                right_x = left_x + self.img_x // 2
                right_y = left_y
                right_count = 0

        # draw
        try:
            left_fit = np.polyfit(left_y, left_x, 2)
            right_fit = np.polyfit(right_y, right_x, 2)

            plot_y = np.linspace(0, h - 1, 5)
            left_fit_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
            right_fit_x = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]
            center_fit_x = (right_fit_x + left_fit_x) / 2.0

            left_pts = np.int32(np.column_stack((left_fit_x, plot_y)))
            right_pts = np.int32(np.column_stack((right_fit_x, plot_y)))
            center_pts = np.int32(np.column_stack((center_fit_x, plot_y)))

            cv2.polylines(out_img, [left_pts], False, (0, 0, 255), 5)
            cv2.polylines(out_img, [right_pts], False, (0, 255, 0), 5)
            cv2.polylines(out_img, [center_pts], False, (255, 0, 0), 3)
        except Exception:
            pass

        return out_img, left_x, left_y, right_x, right_y, left_count, right_count

    # -----------------------------
    # 폭 벌어지면 “한쪽 차선” 기반으로 center 계산
    # -----------------------------
    def compute_lane_center(self, img, left_x, left_y, right_x, right_y, left_count, right_count):
        h = img.shape[0]
        bottom_y = h - 1
        y2 = bottom_y * bottom_y

        have_left = (left_count >= 20) and (len(left_y) >= 3)
        have_right = (right_count >= 20) and (len(right_y) >= 3)

        bottom_x_left = None
        bottom_x_right = None

        if have_left:
            lf = np.polyfit(left_y, left_x, 2)
            bottom_x_left = lf[0] * y2 + lf[1] * bottom_y + lf[2]
        if have_right:
            rf = np.polyfit(right_y, right_x, 2)
            bottom_x_right = rf[0] * y2 + rf[1] * bottom_y + rf[2]

        # width 측정(둘 다 있을 때)
        width = None
        prev_ema = self.lane_width_ema
        if bottom_x_left is not None and bottom_x_right is not None:
            width = float(bottom_x_right - bottom_x_left)
            if width > 10:
                if self.lane_width_ema is None:
                    self.lane_width_ema = width
                else:
                    self.lane_width_ema = (1.0 - self.lane_width_alpha) * self.lane_width_ema + self.lane_width_alpha * width

        nominal = self.lane_width_ema
        if nominal is None:
            nominal = img.shape[1] * self.nominal_lane_width_ratio

        # (핵심) 폭이 “기존 폭(EMA)” 대비 갑자기 커진 경우: 한쪽 기준으로 center 결정
        # 비교는 prev_ema(업데이트 전)를 우선 사용
        ref = prev_ema if prev_ema is not None else self.lane_width_ema
        if (width is not None) and (ref is not None) and (width > self.widen_factor * ref):
            if left_count >= right_count and bottom_x_left is not None:
                return float(bottom_x_left + nominal / 2.0)
            if bottom_x_right is not None:
                return float(bottom_x_right - nominal / 2.0)

        # 정상: 둘 다 있으면 중간
        if bottom_x_left is not None and bottom_x_right is not None:
            return float((bottom_x_left + bottom_x_right) / 2.0)

        # 한쪽만 있으면 nominal로 보정
        if bottom_x_left is not None:
            return float(bottom_x_left + nominal / 2.0)
        if bottom_x_right is not None:
            return float(bottom_x_right - nominal / 2.0)

        return float(img.shape[1] / 2.0)

    # -----------------------------
    # 제어
    # -----------------------------
    def ctrl_cmd(self, img_center_x, target_center_x):
        w = float(max(1, self.img_x))
        norm_err = (img_center_x - target_center_x) / w  # 대략 -0.5 ~ +0.5

        yaw = -norm_err * self.turn_mult
        yaw = clamp(yaw, -self.max_yaw, self.max_yaw)

        msg = Twist()
        msg.linear.x = self.speed
        msg.angular.z = yaw
        return msg

    # -----------------------------
    # callback
    # -----------------------------
    def img_CB(self, data):
        if not self.enabled:
            return

        self._frame_count += 1
        if self.frame_skip > 1 and (self._frame_count % self.frame_skip != 0):
            return

        try:
            # cv_bridge 호환 처리 (환경에 따라 desired_encoding 인자 지원이 다를 수 있음)
            try:
                img = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
            except TypeError:
                img = self.bridge.compressed_imgmsg_to_cv2(data)

            if img is None:
                return

            # downscale
            h, w = img.shape[:2]
            if h > 2 and w > 2:
                img = cv2.resize(img, (w // 2, h // 2))

            # window params
            self.window_height = img.shape[0] // self.nwindows

            warp_img = self.img_warp(img)
            blend_img = self.detect_color(warp_img)
            binary_img = self.img_binary(blend_img)

            if not self.nothing_flag:
                self.detect_nothing()
                self.nothing_flag = True

            sliding_window_img, left_x, left_y, right_x, right_y, left_cnt, right_cnt = self.window_search(binary_img)

            target_center_x = self.compute_lane_center(
                sliding_window_img, left_x, left_y, right_x, right_y, left_cnt, right_cnt
            )
            img_center_x = sliding_window_img.shape[1] / 2.0

            ctrl_cmd_msg = self.ctrl_cmd(img_center_x, target_center_x)

            # imshow (optional)
            if self.debug_view and (self._frame_count % self.debug_every == 0):
                binary_vis = (binary_img * 255).astype(np.uint8)
                cv2.imshow("01_raw_resized", img)
                cv2.imshow("02_warp", warp_img)
                cv2.imshow("03_color_blend", blend_img)
                cv2.imshow("04_binary", binary_vis)
                cv2.imshow("05_sliding_window", sliding_window_img)
                key = cv2.waitKey(self.debug_wait) & 0xFF
                if key == ord("q") or key == 27:
                    rospy.signal_shutdown("User requested shutdown via OpenCV window")
                    return

            now = rospy.get_time()
            if now - self.last_pub_time >= self.pub_period:
                self.ctrl_pub.publish(ctrl_cmd_msg)
                self.have_published_cmd = True
                self.last_pub_time = now

                rospy.loginfo_throttle(
                    self.debug_throttle,
                    "[LKAS] pub cmd_vel_lkas v=%.3f yaw=%.3f (left_cnt=%d right_cnt=%d, enabled=%s)",
                    ctrl_cmd_msg.linear.x, ctrl_cmd_msg.angular.z, left_cnt, right_cnt, str(self.enabled)
                )

        except Exception as e:
            rospy.logerr("[LKAS] exception in img_CB: %s", str(e))
            # 안전 정지 (단, publish 자체는 해서 fsm_mux have_lkas_cmd가 영원히 False가 되는 상황을 막음)
            stop = Twist()
            self.ctrl_pub.publish(stop)
            self.have_published_cmd = True


if __name__ == "__main__":
    node = LKAS()
    try:
        rospy.spin()
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass