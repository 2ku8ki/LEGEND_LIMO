#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import cv2
import threading


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class LKAS:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node("LKAS_node", anonymous=False)

        rospy.Subscriber("/camera/rgb/image_raw/compressed",
                         CompressedImage, self.img_CB, queue_size=1)

        # ✅ /cmd_vel 로 바로 publish
        self.ctrl_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # ---- control ----
        self.speed = 0.25
        self.trun_mutip = 0.1

        # 안전 클램프
        self.max_ang = float(rospy.get_param("~max_ang", 1.2))   # rad/s 제한
        self.max_spd = float(rospy.get_param("~max_spd", self.speed))

        # ---- timing ----
        self.start_time = rospy.get_time()
        self.frame_skip = 3
        self._frame_count = 0

        # ---- sliding window params ----
        self.nwindows = 10
        self.window_height = 0
        self.nothing_flag = False

        # ---- warp base ----
        self.img_x = 0
        self.img_y = 0

        # ✅ 항상 enable
        self.enabled = True

        # ✅ 유효영역 컷 (패딩/차체 제거)  -> 이 값만큼 "아래를 예측으로 채움"
        self.cut_bot = int(rospy.get_param("~cut_bot", 80))

        # ✅ warp src(320x240 기준)
        self.warp_top_left  = (20, 118)
        self.warp_top_right = (300, 118)

        # ✅ robust mask: HSV + Canny OR
        self.use_edge_or = bool(rospy.get_param("~use_edge_or", True))
        self.canny_low = int(rospy.get_param("~canny_low", 60))
        self.canny_high = int(rospy.get_param("~canny_high", 160))
        self.blur_ksize = int(rospy.get_param("~blur_ksize", 5))

        # ✅ binary ROI
        self.top_cut_ratio = float(rospy.get_param("~top_cut_ratio", 0.25))
        self.center_left_ratio = float(rospy.get_param("~center_left_ratio", 0.35))
        self.center_right_ratio = float(rospy.get_param("~center_right_ratio", 0.65))
        self.keep_bottom = int(rospy.get_param("~keep_bottom", 60))

        # ---- debug imshow (콜백 저장, 메인 imshow) ----
        self.debug_imshow = bool(rospy.get_param("~debug_imshow", True))
        self._lock = threading.Lock()
        self._dbg = {
            "raw": None,
            "cropped": None,
            "src_dbg": None,
            "warp": None,
            "blend": None,
            "mask": None,
            "bin": None,
            "slide": None,
        }

        if self.debug_imshow:
            cv2.namedWindow("01_raw(resized)", cv2.WINDOW_NORMAL)
            cv2.namedWindow("01b_cropped(valid)", cv2.WINDOW_NORMAL)
            cv2.namedWindow("02_warp_src_debug", cv2.WINDOW_NORMAL)
            cv2.namedWindow("03_warp(BEV)", cv2.WINDOW_NORMAL)
            cv2.namedWindow("04_blend(HSV)", cv2.WINDOW_NORMAL)
            cv2.namedWindow("05_mask(HSV|EDGE)", cv2.WINDOW_NORMAL)
            cv2.namedWindow("06_binary(0/1)", cv2.WINDOW_NORMAL)
            cv2.namedWindow("07_sliding(+pred)", cv2.WINDOW_NORMAL)

        rospy.on_shutdown(self._on_shutdown)

        self.cmd_vel_msg = Twist()

    # -------------------------
    # HSV 색 기반
    # -------------------------
    def detect_color(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        yellow_lower = np.array([15, 80, 0], dtype=np.uint8)
        yellow_upper = np.array([45, 255, 255], dtype=np.uint8)

        white_lower = np.array([0, 0, 230], dtype=np.uint8)
        white_upper = np.array([179, 40, 255], dtype=np.uint8)

        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        white_mask = cv2.inRange(hsv, white_lower, white_upper)

        blend_mask = yellow_mask | white_mask
        return cv2.bitwise_and(img, img, mask=blend_mask)

    # -------------------------
    # 하단 유효영역 컷
    # -------------------------
    def crop_bottom_invalid(self, img):
        if self.cut_bot <= 0:
            return img
        h = img.shape[0]
        cut = min(self.cut_bot, h - 2)
        return img[:h - cut, :]

    # -------------------------
    # BEV warp (4점 + 검정깨짐 최소화)
    # -------------------------
    def img_warp(self, img, w_ref=320.0, h_ref=240.0, h_before_crop=240):
        h, w = img.shape[:2]
        self.img_x, self.img_y = w, h

        sx = w / w_ref
        sy = h_before_crop / h_ref

        tl = (int(self.warp_top_left[0] * sx),  int(self.warp_top_left[1] * sy))
        tr = (int(self.warp_top_right[0] * sx), int(self.warp_top_right[1] * sy))
        bl = (0, h - 1)
        br = (w - 1, h - 1)

        def clamp_pt(x, y):
            return (max(0, min(w - 1, x)), max(0, min(h - 1, y)))

        tl = clamp_pt(*tl)
        tr = clamp_pt(*tr)

        src = np.array([bl, tl, tr, br], dtype=np.float32)
        dst = np.array([(0, h - 1), (0, 0), (w - 1, 0), (w - 1, h - 1)], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)

        warp = cv2.warpPerspective(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        src_dbg = img.copy()
        pts = np.int32([bl, tl, tr, br])
        cv2.polylines(src_dbg, [pts], True, (0, 255, 0), 2)
        for (x, y) in pts:
            cv2.circle(src_dbg, (x, y), 4, (0, 0, 255), -1)

        return warp, src_dbg

    # -------------------------
    # Canny edge mask
    # -------------------------
    def edge_mask(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        k = self.blur_ksize
        if k >= 3 and (k % 2 == 1):
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        return edges  # 0/255

    # -------------------------
    # binary (중앙 제거: 아래 keep_bottom 살림)
    # -------------------------
    def img_binary(self, img_or_mask):
        if img_or_mask.ndim == 3:
            gray = cv2.cvtColor(img_or_mask, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_or_mask

        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        h, w = bw.shape

        top_cut = int(h * self.top_cut_ratio)
        bw[:top_cut, :] = 0

        left_end = int(w * self.center_left_ratio)
        right_start = int(w * self.center_right_ratio)
        y_limit = max(0, h - self.keep_bottom)
        bw[:y_limit, left_end:right_start] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

        return (bw > 0).astype(np.uint8)

    # -------------------------
    # nothing base
    # -------------------------
    def detect_nothing(self):
        offset = int(self.img_x * 0.140625)
        self.nothing_left_x_base = offset
        self.nothing_right_x_base = self.img_x - offset

        self.nothing_pixel_left_x = np.full(self.nwindows, self.nothing_left_x_base, dtype=np.int32)
        self.nothing_pixel_right_x = np.full(self.nwindows, self.nothing_right_x_base, dtype=np.int32)

        base_y = int(self.window_height / 2)
        self.nothing_pixel_y = np.arange(0, self.nwindows * base_y, base_y, dtype=np.int32)

    # -------------------------
    # sliding window + polyfit + "아래 cut_bot 예측선" 디버그 캔버스 생성
    # -------------------------
    def window_search(self, binary_line):
        h, w = binary_line.shape

        bottom_half = binary_line[h // 2:, :]
        histogram = np.sum(bottom_half, axis=0)

        midpoint = w // 2
        left_x_base = np.argmax(histogram[:midpoint])
        right_x_base = np.argmax(histogram[midpoint:]) + midpoint

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
            win_y_low = h - (window + 1) * self.window_height
            win_y_high = h - window * self.window_height

            left_low = left_x_current - margin
            left_high = left_x_current + margin
            right_low = right_x_current - margin
            right_high = right_x_current + margin

            if left_x_current != 0:
                cv2.rectangle(out_img, (left_low, win_y_low), (left_high, win_y_high), (0, 255, 0), 2)
            if right_x_current != midpoint:
                cv2.rectangle(out_img, (right_low, win_y_low), (right_high, win_y_high), (0, 0, 255), 2)

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

        if len(left_x) == 0 and len(right_x) == 0:
            left_x = self.nothing_pixel_left_x
            left_y = self.nothing_pixel_y
            right_x = self.nothing_pixel_right_x
            right_y = self.nothing_pixel_y
        else:
            if len(left_x) == 0:
                left_x = right_x - self.img_x // 2
                left_y = right_y
            elif len(right_x) == 0:
                right_x = left_x + self.img_x // 2
                right_y = left_y

        # polyfit (안전)
        try:
            left_fit = np.polyfit(left_y, left_x, 2)
            right_fit = np.polyfit(right_y, right_x, 2)
        except Exception:
            left_fit = np.array([0.0, 0.0, float(w * 0.25)], dtype=np.float32)
            right_fit = np.array([0.0, 0.0, float(w * 0.75)], dtype=np.float32)

        # ----- 디버그용: 원본 높이(h) + 아래 예측(cut_bot) 캔버스 만들기 -----
        ext = max(0, int(self.cut_bot))
        out_tall = np.zeros((h + ext, w, 3), dtype=np.uint8)
        out_tall[:h, :, :] = out_img

        # 경계선(여기부터는 "예측" 영역)
        if ext > 0:
            cv2.line(out_tall, (0, h - 1), (w - 1, h - 1), (255, 255, 255), 2)

        # 전체(원본+예측) y 범위에서 라인 그리기
        plot_y_all = np.linspace(0, (h + ext) - 1, 40)
        left_fit_x_all = left_fit[0] * plot_y_all**2 + left_fit[1] * plot_y_all + left_fit[2]
        right_fit_x_all = right_fit[0] * plot_y_all**2 + right_fit[1] * plot_y_all + right_fit[2]
        center_fit_x_all = (left_fit_x_all + right_fit_x_all) / 2.0

        # 화면 밖 방지
        left_fit_x_all = np.clip(left_fit_x_all, 0, w - 1)
        right_fit_x_all = np.clip(right_fit_x_all, 0, w - 1)
        center_fit_x_all = np.clip(center_fit_x_all, 0, w - 1)

        left_pts_all = np.int32(np.column_stack((left_fit_x_all, plot_y_all)))
        right_pts_all = np.int32(np.column_stack((right_fit_x_all, plot_y_all)))
        center_pts_all = np.int32(np.column_stack((center_fit_x_all, plot_y_all)))

        # 원본영역(0~h-1)은 진하게, 예측영역(h~h+ext)은 다른 색
        # 그냥 한 번에 그리되, "예측 영역" 색이 보이게 한번 더 덧그림
        cv2.polylines(out_tall, [left_pts_all], False, (0, 0, 255), 4)
        cv2.polylines(out_tall, [right_pts_all], False, (0, 255, 0), 4)
        cv2.polylines(out_tall, [center_pts_all], False, (255, 0, 0), 3)

        if ext > 0:
            # 예측구간만 따로 강조(노랑/주황 느낌)
            y_pred = np.linspace(h - 1, (h + ext) - 1, 20)
            lx = left_fit[0] * y_pred**2 + left_fit[1] * y_pred + left_fit[2]
            rx = right_fit[0] * y_pred**2 + right_fit[1] * y_pred + right_fit[2]
            cx = (lx + rx) / 2.0
            lx = np.clip(lx, 0, w - 1); rx = np.clip(rx, 0, w - 1); cx = np.clip(cx, 0, w - 1)

            lp = np.int32(np.column_stack((lx, y_pred)))
            rp = np.int32(np.column_stack((rx, y_pred)))
            cp = np.int32(np.column_stack((cx, y_pred)))

            cv2.polylines(out_tall, [lp], False, (0, 255, 255), 3)
            cv2.polylines(out_tall, [rp], False, (0, 255, 255), 3)
            cv2.polylines(out_tall, [cp], False, (255, 255, 0), 2)

        return out_tall, left_x, left_y, right_x, right_y, left_fit, right_fit

    # -------------------------
    # meter_per_pixel (원본 그대로)
    # -------------------------
    def meter_per_pixel(self, virtual_extra_y=0):
        # 원래 코드 스타일 유지 (x,y 스케일 계산)
        world_warp = np.array([[97, 1610], [109, 1610], [109, 1606], [97, 1606]], dtype=np.float32)

        dx_x = world_warp[0, 0] - world_warp[3, 0]
        dy_x = world_warp[0, 1] - world_warp[3, 1]
        meter_x = dx_x * dx_x + dy_x * dy_x

        dx_y = world_warp[0, 0] - world_warp[1, 0]
        dy_y = world_warp[0, 1] - world_warp[1, 1]
        meter_y = dx_y * dx_y + dy_y * dy_y

        # ✅ y는 "가상으로 늘린 높이" 기준으로 잡아줌 (예측영역 반영)
        img_y_virtual = float(max(self.img_y + int(virtual_extra_y), 1))
        img_x_virtual = float(max(self.img_x, 1))

        meter_per_pix_x = meter_x / img_x_virtual
        meter_per_pix_y = meter_y / img_y_virtual
        return meter_per_pix_x, meter_per_pix_y

    # -------------------------
    # ✅ 곡률 계산: 예측 포함한 y_eval에서 곡률반경 계산
    #    (픽셀 poly 계수를 meters로 변환해서 빠르게 계산)
    # -------------------------
    def calc_curvature_radius(self, left_fit, right_fit, y_eval_pix, virtual_extra_y):
        mx, my = self.meter_per_pixel(virtual_extra_y=virtual_extra_y)

        # x_pix = a y^2 + b y + c
        # y_m = my*y, x_m = mx*x
        # => x_m = (mx*a/my^2) y_m^2 + (mx*b/my) y_m + mx*c
        def to_meter_coeff(f):
            a, b, c = f
            A = (mx * a) / (my * my + 1e-9)
            B = (mx * b) / (my + 1e-9)
            C = (mx * c)
            return A, B, C

        Al, Bl, _ = to_meter_coeff(left_fit)
        Ar, Br, _ = to_meter_coeff(right_fit)

        y_eval_m = (my * float(y_eval_pix))

        # R = (1 + (2A y + B)^2)^(3/2) / |2A|
        def radius(A, B, y):
            denom = abs(2.0 * A)
            if denom < 1e-9:
                return 1e9
            return ((1.0 + (2.0 * A * y + B) ** 2) ** 1.5) / denom

        Rl = radius(Al, Bl, y_eval_m)
        Rr = radius(Ar, Br, y_eval_m)
        return float(Rl), float(Rr), float(0.5 * (Rl + Rr))

    # -------------------------
    # ✅ 예측 포함 offset 계산 (아래 잘린 cut_bot을 y로 "가상 확장"해서 center 평가)
    # -------------------------
    def calc_vehicle_offset_from_fit(self, img_w, img_h, left_fit, right_fit, virtual_extra_y):
        # y_eval: (현재 이미지 바닥) + (잘린 아래 만큼)
        y_eval = float((img_h - 1) + int(virtual_extra_y))

        a_l, b_l, c_l = left_fit
        a_r, b_r, c_r = right_fit

        x_left = a_l * (y_eval ** 2) + b_l * y_eval + c_l
        x_right = a_r * (y_eval ** 2) + b_r * y_eval + c_r

        lane_center_x = 0.5 * (x_left + x_right)
        img_center_x = img_w / 2.0
        pixel_offset = img_center_x - lane_center_x

        mx, _ = self.meter_per_pixel(virtual_extra_y=virtual_extra_y)
        vehicle_offset = pixel_offset * (2 * mx)
        return float(vehicle_offset), float(y_eval)

    # -------------------------
    # ctrl cmd
    # -------------------------
    def ctrl_cmd(self, vehicle_offset):
        self.cmd_vel_msg.linear.x = clamp(self.speed, 0.0, self.max_spd)
        ang = -vehicle_offset * self.trun_mutip
        self.cmd_vel_msg.angular.z = clamp(ang, -self.max_ang, self.max_ang)
        return self.cmd_vel_msg

    # -------------------------
    # callback: 처리 + publish(0.1s) + 디버그 저장
    # -------------------------
    def img_CB(self, data):
        now = rospy.get_time()
        if not self.enabled:
            return

        self._frame_count += 1
        if self._frame_count % self.frame_skip != 0:
            return

        img = self.bridge.compressed_imgmsg_to_cv2(data)

        # resize to half
        h0, w0 = img.shape[:2]
        img = cv2.resize(img, (w0 // 2, h0 // 2))
        h_before_crop = img.shape[0]

        # ✅ 유효영역만 남기기
        cropped = self.crop_bottom_invalid(img)

        # sliding window 파라미터
        self.window_height = cropped.shape[0] // self.nwindows

        # ✅ warp
        warp_img, src_dbg = self.img_warp(cropped, h_before_crop=h_before_crop)

        # HSV blend
        blend_img = self.detect_color(warp_img)

        # ✅ robust mask
        if self.use_edge_or:
            mask_hsv = cv2.cvtColor(blend_img, cv2.COLOR_BGR2GRAY)
            edges = self.edge_mask(warp_img)
            mask = cv2.bitwise_or(mask_hsv, edges)
        else:
            mask = cv2.cvtColor(blend_img, cv2.COLOR_BGR2GRAY)

        # binary 0/1
        binary_img = self.img_binary(mask)

        # nothing 초기값
        if not self.nothing_flag:
            self.detect_nothing()
            self.nothing_flag = True

        # sliding window + fit + (예측선 포함 디버그 이미지)
        sliding_pred_img, left_x, left_y, right_x, right_y, left_fit, right_fit = self.window_search(binary_img)

        # ✅ "예측 포함" 오프셋 계산 (아래 cut_bot을 가상으로 붙인 y에서 평가)
        h_used, w_used = binary_img.shape
        vehicle_offset, y_eval = self.calc_vehicle_offset_from_fit(
            img_w=w_used, img_h=h_used,
            left_fit=left_fit, right_fit=right_fit,
            virtual_extra_y=self.cut_bot
        )

        # ✅ "예측 포함" 곡률 계산 (같은 y_eval에서)
        Rl, Rr, Ravg = self.calc_curvature_radius(
            left_fit, right_fit,
            y_eval_pix=y_eval,
            virtual_extra_y=self.cut_bot
        )

        # cmd
        ctrl_cmd_msg = self.ctrl_cmd(vehicle_offset)

        # publish 0.1s
        if now - self.start_time >= 0.1:
            self.ctrl_pub.publish(ctrl_cmd_msg)
            self.start_time = now

        # debug 저장
        if self.debug_imshow:
            # 슬라이딩 디버그에 곡률/평가 y 표기
            disp = sliding_pred_img.copy()
            txt1 = f"y_eval={y_eval:.1f}  cut_bot={self.cut_bot}"
            txt2 = f"Ravg={Ravg:.1f} (Rl={Rl:.1f}, Rr={Rr:.1f})"
            cv2.putText(disp, txt1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(disp, txt2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            with self._lock:
                self._dbg["raw"] = img
                self._dbg["cropped"] = cropped
                self._dbg["src_dbg"] = src_dbg
                self._dbg["warp"] = warp_img
                self._dbg["blend"] = blend_img
                self._dbg["mask"] = mask
                self._dbg["bin"] = (binary_img * 255).astype(np.uint8)
                self._dbg["slide"] = disp

    # -------------------------
    # main loop: imshow only
    # -------------------------
    def spin(self):
        rate = rospy.Rate(60)
        while not rospy.is_shutdown():
            if self.debug_imshow:
                with self._lock:
                    raw = self._dbg["raw"]
                    cropped = self._dbg["cropped"]
                    src_dbg = self._dbg["src_dbg"]
                    warp = self._dbg["warp"]
                    blend = self._dbg["blend"]
                    mask = self._dbg["mask"]
                    binv = self._dbg["bin"]
                    slide = self._dbg["slide"]

                if raw is not None:
                    cv2.imshow("01_raw(resized)", raw)
                if cropped is not None:
                    cv2.imshow("01b_cropped(valid)", cropped)
                if src_dbg is not None:
                    cv2.imshow("02_warp_src_debug", src_dbg)
                if warp is not None:
                    cv2.imshow("03_warp(BEV)", warp)
                if blend is not None:
                    cv2.imshow("04_blend(HSV)", blend)
                if mask is not None:
                    cv2.imshow("05_mask(HSV|EDGE)", mask)
                if binv is not None:
                    cv2.imshow("06_binary(0/1)", binv)
                if slide is not None:
                    cv2.imshow("07_sliding(+pred)", slide)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    rospy.signal_shutdown("q pressed")

            rate.sleep()

    def _on_shutdown(self):
        try:
            self.ctrl_pub.publish(Twist())
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    node = LKAS()
    node.spin()
