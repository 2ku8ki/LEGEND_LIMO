#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import threading


def clamp_int(v, lo, hi):
    return int(max(lo, min(hi, v)))


class BlackROI_ScanSteer:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node("blackroi_scansteer", anonymous=False)

        # topics
        self.image_topic = rospy.get_param("~image_topic", "/camera/rgb/image_raw/compressed")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel_lkas")  # ✅ default 변경
        self.enable_topic = rospy.get_param("~enable_topic", "/lkas_enable")

        # enable gate
        self.enabled = bool(rospy.get_param("~enabled_default", True))
        rospy.Subscriber(self.enable_topic, Bool, self._enable_cb, queue_size=1)

        # rates
        self.cmd_hz = float(rospy.get_param("~cmd_hz", 10.0))

        # scale
        self.scale = float(rospy.get_param("~scale", 0.5))

        # --- unified params ---
        self.black_thr = int(rospy.get_param("~black_thr", 180))
        self.top_cut_ratio = float(rospy.get_param("~top_cut_ratio", 0.30))
        self.bottom_cut_ratio = float(rospy.get_param("~bottom_cut_ratio", 0.00))

        self.use_morph = bool(rospy.get_param("~use_morph", True))
        self.open_ksize = int(rospy.get_param("~open_ksize", 8))
        self.close_ksize = int(rospy.get_param("~close_ksize", 2))

        self.min_area = int(rospy.get_param("~min_area", 800))
        self.min_height = int(rospy.get_param("~min_height", 50))
        self.min_width = int(rospy.get_param("~min_width", 10))

        # scanline params
        self.y_start = int(rospy.get_param("~y_start", 190))
        self.y_gap = int(rospy.get_param("~y_gap", 10))
        self.num_points = int(rospy.get_param("~num_points", 11))

        # =========================
        # (A) 기존 "엣지 마진" 조건
        # =========================
        self.edge_margin_px = int(rospy.get_param("~edge_margin_px",
                               rospy.get_param("~space_margin_px", 5)))
        self.require_margin_all_points = bool(rospy.get_param("~require_margin_all_points", True))
        self.min_span_px = int(rospy.get_param("~min_span_px", 30))

        self.use_red_points = bool(rospy.get_param("~use_red_points", True))
        self.edge_red_weight = float(rospy.get_param("~edge_red_weight",
                                   rospy.get_param("~red_weight", 0.1)))

        # =========================
        # (B) 새 "미들(센터) 조건" (왼쪽 점 xl만!)
        # =========================
        self.use_middle_points = bool(rospy.get_param("~use_middle_points", True))
        self.middle_margin_px = int(rospy.get_param("~middle_margin_px", 10))
        self.middle_weight = float(rospy.get_param("~middle_weight", self.edge_red_weight))

        # steering params
        self.publish_cmd = bool(rospy.get_param("~publish_cmd", True))
        self.speed = float(rospy.get_param("~speed", 0.5))
        self.steer_gain = float(rospy.get_param("~steer_gain", 0.018))
        self.max_steer = float(rospy.get_param("~max_steer", 2))

        # follow y
        self.y_follow = int(rospy.get_param("~y_follow", self.y_start))

        # polyfit
        self.poly_degree = int(rospy.get_param("~poly_degree", 2))

        # hold
        self.hold_last_when_fail = bool(rospy.get_param("~hold_last_when_fail", True))
        self.hold_decay = float(rospy.get_param("~hold_decay", 0.90))
        self._last_steer = 0.0

        # buffers
        self._lock = threading.Lock()
        self._latest_bgr = None

        rospy.Subscriber(self.image_topic, CompressedImage, self._img_cb,
                         queue_size=1, buff_size=2**24)
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)

        self._last_cmd_time = rospy.get_time()

        rospy.loginfo(
            "[scansteer] enable_topic=%s cmd_topic=%s scale=%.2f thr=%d y_start=%d gap=%d n=%d y_follow=%d "
            "edge_margin=%d edge_w=%.2f use_middle=%d middle_margin=%d middle_w=%.2f",
            self.enable_topic, self.cmd_topic,
            self.scale, self.black_thr, self.y_start, self.y_gap, self.num_points,
            self.y_follow, self.edge_margin_px, self.edge_red_weight,
            int(self.use_middle_points), self.middle_margin_px, self.middle_weight
        )

    def _enable_cb(self, msg: Bool):
        self.enabled = msg.data

    def _img_cb(self, msg: CompressedImage):
        try:
            bgr0 = self.bridge.compressed_imgmsg_to_cv2(msg)
        except Exception:
            return

        if self.scale != 1.0:
            h0, w0 = bgr0.shape[:2]
            nw = max(1, int(w0 * self.scale))
            nh = max(1, int(h0 * self.scale))
            bgr = cv2.resize(bgr0, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            bgr = bgr0

        with self._lock:
            self._latest_bgr = bgr

    def black_mask(self, bgr):
        b, g, r = cv2.split(bgr)
        mask = ((b <= self.black_thr) & (g <= self.black_thr) & (r <= self.black_thr)).astype(np.uint8) * 255

        h, w = mask.shape
        top_cut = clamp_int(h * self.top_cut_ratio, 0, h)
        bot_cut = clamp_int(h * self.bottom_cut_ratio, 0, h)

        if top_cut > 0:
            mask[:top_cut, :] = 0
        if bot_cut > 0:
            mask[h - bot_cut:h, :] = 0

        if self.use_morph:
            ok = max(1, self.open_ksize)
            ck = max(1, self.close_ksize)
            k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (ok, ok))
            k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (ck, ck))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)

        return mask

    def pick_component(self, mask255):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask255, connectivity=4)
        if n <= 1:
            return None

        best_idx = None
        best_area = -1
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < self.min_area:
                continue
            if h < self.min_height or w < self.min_width:
                continue
            if area > best_area:
                best_area = area
                best_idx = i

        if best_idx is None:
            return None

        return (labels == best_idx).astype(np.uint8)  # 0/1

    def scan_row_lr_center(self, bin01, y, w):
        """
        우편향 스캔:
        - xr: 해당 y줄에서 가장 오른쪽(최대 x)의 1
        - xl: xr에서 왼쪽으로 이동하면서 '연속 1 구간'이 끝나는 지점 다음(=오른쪽 덩어리의 시작)

        반환:
        xl, xr, xc,
        left_ok, right_ok, has_space,
        middle_bad (왼쪽 점 xl만 중앙/근처에 걸리면 True),
        middle_thr_x
        """
        row = bin01[y, :]
        xs = np.flatnonzero(row)
        if xs.size == 0:
            return None

        xr = int(xs[-1])

        x = xr
        while x >= 0 and row[x] != 0:
            x -= 1
        xl = x + 1

        span = xr - xl
        if span < self.min_span_px:
            return None

        left_margin = xl
        right_margin = (w - 1) - xr

        left_ok = (left_margin >= self.edge_margin_px)
        right_ok = (right_margin >= self.edge_margin_px)
        has_space = (left_ok or right_ok)

        img_cx = 0.5 * (w - 1)
        middle_thr_x = img_cx - float(max(0, self.middle_margin_px))
        middle_bad = (xl >= middle_thr_x)

        xc = 0.5 * (xl + xr)
        return xl, xr, xc, left_ok, right_ok, has_space, middle_bad, middle_thr_x

    def compute_scan_points(self, bin01):
        h, w = bin01.shape[:2]

        ys = []
        for i in range(max(1, self.num_points)):
            y = self.y_start - i * self.y_gap
            if 0 <= y < h:
                ys.append(y)

        points = []
        weights = []
        lr_list = []

        # pt_src: 0=미사용, 1=strong(1.0), 2=edge_red, 3=middle
        for idx, y in enumerate(ys):
            out = self.scan_row_lr_center(bin01, y, w)
            if out is None:
                lr_list.append((None, None, y, False, False, False, False, False, None, 0, None))
                continue

            xl, xr, xc, left_ok, right_ok, has_space, middle_bad, middle_thr_x = out

            if idx == 0:
                valid = has_space
            else:
                valid = (has_space if self.require_margin_all_points else True)

            pt_weight = None
            pt_src = 0

            # 우선순위:
            # (1) middle_bad면 middle_weight
            # (2) 아니면 valid면 1.0
            # (3) 아니면 edge_red_weight
            if self.use_middle_points and middle_bad:
                points.append((float(xc), float(y)))
                weights.append(float(self.middle_weight))
                pt_weight = float(self.middle_weight)
                pt_src = 3
            else:
                if valid:
                    points.append((float(xc), float(y)))
                    weights.append(1.0)
                    pt_weight = 1.0
                    pt_src = 1
                else:
                    if self.use_red_points:
                        points.append((float(xc), float(y)))
                        weights.append(float(self.edge_red_weight))
                        pt_weight = float(self.edge_red_weight)
                        pt_src = 2
                    else:
                        pt_weight = None
                        pt_src = 0

            lr_list.append((xl, xr, y, left_ok, right_ok, has_space, valid, middle_bad, pt_weight, pt_src, middle_thr_x))

        return points, weights, lr_list, (h, w)

    def fit_curve_x_of_y(self, points, weights):
        if len(points) < 2:
            return None

        ys = np.array([p[1] for p in points], dtype=np.float32)
        xs = np.array([p[0] for p in points], dtype=np.float32)
        ws = np.array(weights, dtype=np.float32)

        deg = min(self.poly_degree, len(points) - 1)
        if deg < 1:
            return None

        coef = np.polyfit(ys, xs, deg=deg, w=ws)
        return np.poly1d(coef)

    def compute_steer_follow_curve(self, poly, w, h):
        if poly is None:
            if self.hold_last_when_fail:
                self._last_steer *= self.hold_decay
                return float(self._last_steer), False, None
            return 0.0, False, None

        img_cx = 0.5 * (w - 1)
        y_eval = float(clamp_int(self.y_follow, 0, h - 1))
        x_target = float(poly(y_eval))
        x_target = float(max(0.0, min(float(w - 1), x_target)))

        offset_px = x_target - img_cx
        steer = float(-offset_px * self.steer_gain)
        steer = max(-self.max_steer, min(self.max_steer, steer))

        self._last_steer = steer
        return steer, True, (x_target, y_eval)

    def spin(self):
        rate = rospy.Rate(max(self.cmd_hz, 1.0))

        while not rospy.is_shutdown():
            if not self.enabled:
                rate.sleep()
                continue

            with self._lock:
                bgr = None if self._latest_bgr is None else self._latest_bgr.copy()

            if bgr is None:
                rate.sleep()
                continue

            h, w = bgr.shape[:2]

            mask255 = self.black_mask(bgr)

            bin01 = self.pick_component(mask255)
            if bin01 is None:
                bin01 = np.zeros((h, w), dtype=np.uint8)

            points, weights, _, _ = self.compute_scan_points(bin01)
            poly = self.fit_curve_x_of_y(points, weights)
            steer, _, _ = self.compute_steer_follow_curve(poly, w, h)

            now = rospy.get_time()
            if self.publish_cmd and (now - self._last_cmd_time) >= (1.0 / max(self.cmd_hz, 1e-6)):
                cmd = Twist()
                cmd.linear.x = self.speed
                cmd.angular.z = steer
                self.cmd_pub.publish(cmd)
                self._last_cmd_time = now

            rate.sleep()


if __name__ == "__main__":
    node = BlackROI_ScanSteer()
    node.spin()
