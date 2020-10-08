#!/usr/bin/env python3
#
# Copyright (C) 2020 Herbert Poetzl, Sebastian Pichelhofer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys
import cv2
import numpy as np
import json
import csv
import pprint
import getopt
import serial
import errno
import posix
import struct
from time import sleep, time
from random import randint
import transforms3d as t3d
from threading import Thread, Condition, Lock
import wget
import os
import socket

# from paramiko import SSHClient, AutoAddPolicy

# fifo = posix.open(sys.argv[1], posix.O_WRONLY | posix.O_NONBLOCK)

tty = '/dev/ttyUSB0'  # sys.argv[1]
baud = 1000000  # int(sys.argv[2])
enable = True
active = True
exit = False
halt = False
quit = False
skip_homing = False
home = False
homing = False
move = False
moveZ = False
moving = False
move_abs = False
moveZ_abs = False
measuring_run = False
save_image = False
move_stepsize_xy = 2.0
move_stepsize_z = 2.0
focus_height_z = 6.0  # 6mm to protect into crashing PCB
pcb_height_z = 5.0
probing_height = 2  # 2.1  # 4.0  # set above the pcb to safety for now, set to 2.3 at this height the probe needle slightly touches the PCB
csv_file = "pcb.csv"
webcamid = 0
analyze_filter_id = 6
pi_zero_ip = "192.168.0.1"
pi_zero_port = 2000
measurement_count = 16

# current position
ender_X = 0.0
ender_Y = 0.0
ender_Z = 0.0

# fiducials
data = {}


def main(argv):
    global csv_file, webcamid, skip_homing, baud, tty, pi_zero_ip
    try:
        opts, args = getopt.getopt(argv, "hi:c:b:u:p:",
                                   ["help", "ifile=", "cam", "skip-homing", "baud-rate", "usb-device", "pi-ip"])
    except getopt.GetoptError:
        print('probe.py -i <CSV file> -c <webcam ID> -b <baud rate> -u <Marlin serial USB device> -p <raspberry pi IP>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('probe.py -i <CSV file> -c <webcam ID> -b <baud rate> -u <Marlin serial USB device>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            csv_file = arg.strip()
        elif opt in ("-c", "--cam"):
            webcamid = int(arg.replace(" ", ""))
        elif opt in ("-p", "--pi-ip"):
            pi_zero_ip = arg.strip()
        elif opt in ("--skip-homing"):
            skip_homing = True
        elif opt in ("-b", "--baud-rate"):
            baud = int(arg.replace(" ", ""))
        elif opt in ("-u", "--usb-device"):
            tty = arg

    print('Loading CSV file: ', csv_file)
    print('Using Webcam ID: ', webcamid)
    print('Marlin Serial USB Device: ', tty)
    print('Baudrate: ', baud)
    print('Raspberry Pi IP: ', pi_zero_ip)


if __name__ == "__main__":
    main(sys.argv[1:])

ser = serial.Serial(
    port=tty,
    baudrate=baud,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    interCharTimeout=0.5,
    timeout=5.0,
    xonxoff=False,
    rtscts=False,
    dsrdtr=False);


def safetofile():
    with open('prober.json', 'w') as f:
        json.dump(data, f)


f = open('prober.json')
data = json.load(f)

fid_hightlight_index = 0;
pad_hightlight_index = 0;

camera_to_probe_offset_x = -27.56
camera_to_probe_offset_y = -0.73

# camera_pixels_per_mm = 155  # measured at slightly above work height of 6mm
# camera_pixels_per_mm = 230  # measured at slightly above work height of 1mm
camera_pixels_per_mm = 270  # measured at slightly above work height of 6mm

frame = None
frame_cnt = 0
frame_cnt_lock = Lock()
frame_cnt_cond = Condition(frame_cnt_lock)

frame_time = time()
frame_delta = 1
frame_fps = 1

analysis = None
analysis_cnt = 0

analysis_time = time()
analysis_delta = 1

settle_time = time()
dwell_time = 0.5

ana_size = (720, 720)
ana_roi = (int((1920 - ana_size[0]) / 2),
           int((1080 - ana_size[1]) / 2),
           int((1920 - ana_size[0]) / 2) + ana_size[0],
           int((1080 - ana_size[1]) / 2) + ana_size[1])

ana_obj = [0] * 4
ana_pos = [0] * 2
ana_pas = [0] * 4
ana_idx = -1
ana_seq = [0] * 5

thr_val = [70, 231, 150, 205, 115]  # pcb fiducials
# thr_val = [-1, 231, 130, 100, 170]  # testpattern points
# thr_val2 = [-26, 231, 212, 131, 221]  # testpattern points blue

cap = cv2.VideoCapture(webcamid)

ini_con = cap.get(cv2.CAP_PROP_CONTRAST)
ini_bri = cap.get(cv2.CAP_PROP_BRIGHTNESS)
ini_sat = cap.get(cv2.CAP_PROP_SATURATION)

cap.set(cv2.CAP_PROP_FPS, 25)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cap.set(cv2.CAP_PROP_CONTRAST, 0.10)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.95)
cap.set(cv2.CAP_PROP_SATURATION, 0.15)

col = [(0, 0, 255), (0, 200, 255), (255, 50, 50), (0, 200, 0), (255, 255, 255), (0, 0, 0)]


def floatcompare(float1, float2, decimals):
    # print (float2 - float1)
    # print(10**-decimals)
    if (abs(float2 - float1) <= 10 ** -decimals):
        return True
    else:
        return False


# load test points from CSV file

def loadCSV(filename):
    fid1_detected = False
    fid2_detected = False
    fid3_detected = False
    fid4_detected = False
    with open(filename, newline='') as csvfile:
        linereader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in linereader:
            if len(row) > 0:
                if (row[0] == "INDEX" or row[0] == ""):
                    continue  # skip header or empty rows
                testpads[int(row[0])] = {}
                testpads[int(row[0])]['x'] = float(row[5])
                testpads[int(row[0])]['y'] = float(row[6])
                testpads[int(row[0])]['partname'] = row[1]
                if (row[1] == "FID1"):
                    fid1_detected = True
                if (row[1] == "FID2"):
                    fid2_detected = True
                if (row[1] == "FID3"):
                    fid3_detected = True
                if (row[1] == "FID4"):
                    fid4_detected = True
                testpads[int(row[0])]['net'] = row[10]
                testpads[int(row[0])]['trans-x'] = 0.0
                testpads[int(row[0])]['trans-y'] = 0.0

    # print (testpads) # debug
    if not fid1_detected or not fid2_detected or not fid3_detected or not fid4_detected:
        print("CSV: 4 fiducial identification failed")
    else:
        print("CSV: 4 fiducials identified successfully")


testpads = {}
loadCSV(csv_file)

for i in range(measurement_count):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((pi_zero_ip, pi_zero_port))
        s.sendall(b'A')
        measurement_data = s.recv(1024)
        # print('Received', repr(measurement_data))
        f = open("testresults.csv", "a")
        # f.write("PARTNAME(NETNAME);TRANSFORMED-X;TRANSFORMED-Y;Measurement-Result" # HEADER
        f.write("AIR;;;" + repr(measurement_data) + "\r\n")
        f.close()
        print("Measurement: " + repr(measurement_data))


# pprint.pprint(testpads)
# print (testpads) # debug

# beta powerboard
# 6;FID1;;0;0;-49.53;-27.305;0;0;0;FID1
# 7;FID2;;0;0;49.53;-27.305;0;0;0;FID2
# 8;FID3;;0;0;-49.53;27.305;0;0;0;FID3
# 9;FID4;;0;0;49.53;27.305;0;0;0;FID4

def findkey(dict, key, search):
    for item in dict.items():
        # print(item[1])
        if item[1][key] == search:
            return item


def transformpoints():
    # print(findkey(testpads, 'partname', 'FID1')) # debug

    # transformation explanation
    #
    # P are the fiducial positions from CSV (PCB)
    # Q are the fiducial positions in prober space
    # p are the testpad positions from CSV (PCB)
    # resulting: q are the testpad positions in prober space

    P = np.array([[float(findkey(testpads, 'partname', 'FID1')[1]['x']),
                   float(findkey(testpads, 'partname', 'FID1')[1]['y']), pcb_height_z],
                  [float(findkey(testpads, 'partname', 'FID2')[1]['x']),
                   float(findkey(testpads, 'partname', 'FID2')[1]['y']), pcb_height_z],
                  [float(findkey(testpads, 'partname', 'FID3')[1]['x']),
                   float(findkey(testpads, 'partname', 'FID3')[1]['y']), pcb_height_z],
                  [float(findkey(testpads, 'partname', 'FID4')[1]['x']),
                   float(findkey(testpads, 'partname', 'FID4')[1]['y']), pcb_height_z]])
    # print ("P") # debug
    # pprint.pprint(P) # debug

    T = [0.5, 0.6, 0.7]
    R = t3d.euler.euler2mat(0.1, 0.2, 0.3, 'sxyz')
    Z = [0.5, 0.4, 0.3]

    # transformation matrix
    A = t3d.affines.compose(T, R, Z)

    # transformed points
    Q = np.array([[data['fiducial'][0]['x'], data['fiducial'][0]['y'], pcb_height_z],
                  [data['fiducial'][1]['x'], data['fiducial'][1]['y'], pcb_height_z],
                  [data['fiducial'][2]['x'], data['fiducial'][2]['y'], pcb_height_z],
                  [data['fiducial'][3]['x'], data['fiducial'][3]['y'], pcb_height_z]])

    # print ("Q") # debug
    # pprint.pprint(Q) # debug

    # calculate matrix from points
    n = P.shape[0]
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:, :-1]

    X = pad(P)
    Y = pad(Q)

    B, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)

    trans = lambda x: unpad(np.dot(pad(x), B))

    for key, value in testpads.items():
        # if value['partname'] != "FID1" or value['partname'] != "FID2" or value['partname'] != "FID3" or value[
        # 'partname'] != "FID4":
        p = np.array([[value['x'], value['y'], pcb_height_z]])

        # print("p")  # debug
        # pprint.pprint(p)  # debug

        q = trans(p)
        # print(q) #debug
        value['trans-x'] = round(q[0][0], 4)
        value['trans-y'] = round(q[0][1], 4)

    # print("testpads") # debug
    # pprint.pprint(testpads) # debug


transformpoints()


def ovtext(img, txt="test", pos=(0, 0), col=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1

    # cv2.putText(img, txt, pos,
    #    font, font_scale, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(img, txt, pos,
                font, font_scale, col, 2, cv2.LINE_AA)


def overlay(img):
    ox, oy, ow, oh = 4, 4, 1920 - 8, 32

    sub = img[oy:oy + oh, ox:ox + ow]
    img[oy:oy + oh, ox:ox + ow] = sub >> 1  # dark background

    rx0, ry0, rx1, ry1 = ana_roi
    cx, cy, r = int((rx0 + rx1) / 2), int((ry0 + ry1) / 2), 64

    # crosshair and center window
    cv2.rectangle(img, (rx0, ry0), (rx1, ry1), (0, 0, 255), 1)
    cv2.line(img, (cx, cy - r), (cx, cy + r), (0, 0, 0), 3)
    cv2.line(img, (cx - r, cy), (cx + r, cy), (0, 0, 0), 3)
    cv2.line(img, (cx, cy - r), (cx, cy + r), (0, 255, 0), 1)
    cv2.line(img, (cx - r, cy), (cx + r, cy), (0, 255, 0), 1)

    # ovtext(img, "FPS %3.1f" % (frame_fps), (10, 30))
    # ovtext(img, "*%08d" % (prev_frame_cnt), (240, 30))
    ovtext(img, "X:%3.2f Y:%3.2f Z:%3.2f" % (ender_X, ender_Y, ender_Z), (500, 30))
    ovtext(img, "Stepsize(XY): %3.2fmm Stepsize(Z): %3.2fmm" % (move_stepsize_xy, move_stepsize_z), (1120, 30))

    # fiducials
    ox1, oy1, ow1, oh1 = 0, 90, 760, 240
    sub2 = img[oy1:oy1 + oh1, ox1:ox1 + ow1]
    img[oy1:oy1 + oh1, ox1:ox1 + ow1] = sub2 >> 1  # dark background
    ovtext(img, "Fid 1 (v): %3.4f, %3.4f" % (data['fiducial'][0]['x'], data['fiducial'][0]['y']), (10, 120))
    ovtext(img, "Fid 2 (b): %3.4f, %3.4f" % (data['fiducial'][1]['x'], data['fiducial'][1]['y']), (10, 160))
    ovtext(img, "Fid 3 (n): %3.4f, %3.4f" % (data['fiducial'][2]['x'], data['fiducial'][2]['y']), (10, 200))
    ovtext(img, "Fid 4 (m): %3.4f, %3.4f" % (data['fiducial'][3]['x'], data['fiducial'][3]['y']), (10, 240))
    cv2.rectangle(img, (0, fid_hightlight_index * 40 + 125), (8, fid_hightlight_index * 40 + 95), (0, 98, 255), -1)

    ovtext(img, "Pad (%d/%d): %s (%s) X: %3.4f Y:%3.4f" % (
        pad_hightlight_index, len(testpads) - 1, testpads[pad_hightlight_index]['partname'],
        testpads[pad_hightlight_index]['net'],
        testpads[pad_hightlight_index]['trans-x'], testpads[pad_hightlight_index]['trans-y']), (10, 320))

    if homing:
        ovtext(img, "HOMING", (10, 64))
    elif moving:
        ovtext(img, "MOVING", (10, 64))
    if measuring_run:
        ovtext(img, "PROBING RUN", (200, 64))
    # elif halt:
    #    ovtext(img, "HALTING", (10, 64))
    # elif quit:
    #    ovtext(img, "EXITING", (10, 64))

    # if enable:
    #    ovtext(img, "ENABLED", (480, 30))
    # else:
    #    ovtext(img, "ACTIVE", (10, 64))


def overana(img):
    ox, oy, ow, oh = 4, 4, ana_roi[3] - 8, 66

    sub = img[oy:oy + oh, ox:ox + ow]
    img[oy:oy + oh, ox:ox + ow] = sub >> 1

    cx, cy, r = int(ana_size[0] / 2), int(ana_size[1] / 2), 64

    cv2.line(img, (cx, cy - r), (cx, cy + r), (0, 0, 0), 3)
    cv2.line(img, (cx - r, cy), (cx + r, cy), (0, 0, 0), 3)
    cv2.line(img, (cx, cy - r), (cx, cy + r), (0, 255, 0), 1)
    cv2.line(img, (cx - r, cy), (cx + r, cy), (0, 255, 0), 1)

    # rx0, ry0, rx1, ry1 = caproi()

    # cv2.rectangle(img, (0, 0), (640, ry0), (0,0,0), -1)

    # cv2.line(img, (rx0, ry0), (rx0, ry1), (255,255,255), 1)

    lag = frame_cnt - analysis_cnt
    ovtext(img, "TIME %3.1fms" % (analysis_delta), (10, 30))
    ovtext(img, "*%08d" % (prev_analysis_cnt), (240, 30))
    ovtext(img, "LAG %d" % (lag), (480, 30))

    # ovtext(img, "STATE %s" % (state_str(state)), (10, 64))
    # ovtext(img, "+%3.1fs" % (sdelta()), (300, 64))

    ovtext(img, "%3d %3d %3d %3d %3d" % tuple(thr_val), (10, 64))

    if analyze_filter_id == 0:
        ovtext(img, "Hue Ch", (480, 64))
    elif analyze_filter_id == 1:
        ovtext(img, "Sat Ch", (480, 64))
    elif analyze_filter_id == 2:
        ovtext(img, "Val Ch", (480, 64))
    elif analyze_filter_id == 3:
        ovtext(img, "Hue Filter", (480, 64))
    elif analyze_filter_id == 4:
        ovtext(img, "Sat Filter", (480, 64))
    elif analyze_filter_id == 5:
        ovtext(img, "Val Filter", (480, 64))
    elif analyze_filter_id == 6:
        ovtext(img, "Mixed Filter", (480, 64))


def choice(img):
    pass
    # rx0, ry0, rx1, ry1 = caproi()

    # yp = ry0 + int((idx + 0.5)*row_size)
    # cv2.circle(img, (selvis, yp), 10, col[idx], -1)
    # cv2.circle(img, (selvis, yp), 15, col[4], 5)


def capture():
    global frame, frame_cnt, frame_cnt_cond
    global frame_time, frame_delta, frame_fps
    global cap, exit, home, homing, move, moving

    frame_time = time()
    fps = [0] * 25

    while cap.isOpened():
        ret, new = cap.read()
        if ret:
            with frame_cnt_cond:
                frame = new
                frame_cnt += 1
                frame_cnt_cond.notify_all()

                prev = frame_time
                frame_time = time()
                frame_delta = frame_time - prev

                fps = [1 / frame_delta] + fps[:-1]

                if frame_cnt % 10 == 0:
                    frame_fps = round(sum(fps)) / 25;

        if exit:
            break


def analyze():
    global frame, frame_cnt, frame_cnt_cond
    global analysis, analysis_cnt
    global analysis_time, analysis_delta
    global align, finish, exit
    global ana_obj, ana_pos, ana_pas
    global ana_idx, ana_seq

    global thr_val

    this = None
    this_cnt = 0

    rx0, ry0, rx1, ry1 = ana_roi
    # hsv_grid = [np.array([0,0,30]), np.array([200,50,160])]
    # hsv_high = [np.array([0,40,0]), np.array([72,255,255])]
    kern_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    par = cv2.SimpleBlobDetector_Params()
    par.minThreshold = 50
    par.maxThreshold = 400

    par.filterByArea = True
    par.minArea = 1000
    par.maxArea = 70000

    par.filterByColor = False

    par.filterByCircularity = False
    par.minCircularity = 0.2

    par.filterByConvexity = False
    par.minConvexity = 0.8

    par.filterByInertia = False
    par.minInertiaRatio = 0.5

    detector = cv2.SimpleBlobDetector_create(par)

    while not exit:
        with frame_cnt_cond:
            while frame_cnt == this_cnt:
                frame_cnt_cond.wait()
            this = frame.copy()
            this_cnt = frame_cnt

        mark = time()

        roi = this[ry0:ry1, rx0:rx1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # msk_grid = cv2.inRange(hsv, hsv_grid[0], hsv_grid[1])
        # msk_high = cv2.inRange(hsv, hsv_high[0], hsv_high[1])
        # msk = cv2.bitwise_not(cv2.bitwise_or(msk_grid, msk_high))
        # res = cv2.bitwise_and(roi, roi, mask=msk)
        # h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        h, s, v = cv2.split(hsv)

        ret, ht = cv2.threshold(h, thr_val[0], 255, cv2.THRESH_BINARY)
        # ret, ht = cv2.threshold(ht, thr_val[1], 0, cv2.THRESH_BINARY_INV)
        ret, st = cv2.threshold(s, thr_val[2], 255, cv2.THRESH_BINARY_INV)
        ret, vt = cv2.threshold(v, thr_val[3], 255, cv2.THRESH_BINARY_INV)

        # thr = cv2.merge((ht, st, vt))
        thr = cv2.merge((ht, st, vt))
        # hsv1 = cv2.cvtColor(thr, cv2.COLOR_BGR2HSV)
        # h1, s1, v1 = cv2.split(hsv1)
        # thr = cv2.threshold(v1, 200, 255, cv2.THRESH_BINARY)
        # thr = cv2.cvtColor(v1, cv2.COLOR_GRAY2BGR)
        # cv2.bitwise_and(thr, thr, ht, mask = ht)
        # cv2.bitwise_and(thr, thr, st, mask = st)

        thr = cv2.bitwise_and(thr, cv2.cvtColor(vt, cv2.COLOR_GRAY2BGR))
        thr = cv2.bitwise_and(thr, cv2.cvtColor(st, cv2.COLOR_GRAY2BGR))
        thr = cv2.bitwise_and(thr, cv2.cvtColor(ht, cv2.COLOR_GRAY2BGR))

        # cv2.bitwise_and(, thr, mask = st)
        # cv2.bitwise_and(thr, thr, mask = ht)

        # ret, thr = cv2.threshold(s, 80, 250, cv2.THRESH_TOZERO)
        # gry = cv2.cvtColor(thr, cv2.COLOR_BGR2GRAY)

        mor = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kern_open)
        img = cv2.morphologyEx(mor, cv2.MORPH_CLOSE, kern_close)
        # img = cv2.cvtColor(mor, cv2.COLOR_GRAY2RGB)
        # img = thr

        kpt = detector.detect(img)
        img = cv2.drawKeypoints(img, kpt, np.array([]), \
                                (255, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # filter channel display options
        if analyze_filter_id == 0:
            img = h
        elif analyze_filter_id == 1:
            img = s
        elif analyze_filter_id == 2:
            img = v
        elif analyze_filter_id == 3:
            img = ht
        elif analyze_filter_id == 4:
            img = st
        elif analyze_filter_id == 5:
            img = vt

        # split = align - rx0
        # limit = finish - rx0
        # purge = ignore - rx0

        pos = [0] * len(kpt)

        for idx, kp in enumerate(kpt):
            xf, yf = kp.pt
            x, y = int(round(xf)), int(round(yf))
            # interesting = False

            pos[idx] = (x, y)

            # print(pos[idx])
            radius = 20
            if kp.size < thr_val[4]:
                ovtext(img, "%3d" % kp.size, (x - 30, y - 15), (255, 0, 0))
                cv2.line(img, (x - radius, y), (x + radius, y), (255, 0, 0), 2)
                cv2.line(img, (x, y - radius), (x, y + radius), (255, 0, 0), 2)
            else:
                if 150 < pos[idx][0] < 570 and 150 < pos[idx][1] < 570:
                    ovtext(img, "%3d" % kp.size, (x - 30, y - 15), (0, 0, 255))
                    # crosshair
                    cv2.line(img, (x - radius, y), (x + radius, y), (0, 0, 255), 7)
                    cv2.line(img, (x, y - radius), (x, y + radius), (0, 0, 255), 7)
                    ana_pos = pos[idx]
                else:
                    ovtext(img, "%3d" % kp.size, (x - 30, y - 15), (255, 255, 255))

        analysis = img
        analysis_cnt += 1

        prev = analysis_time
        analysis_time = time()
        analysis_delta = (analysis_time - mark) * 1000


def gcode(ser, cmd):
    ser.write(cmd + b'\n')


def ender():
    global ser, exit, home, homing, move, moving, move_abs, moveZ, moveZ_abs, ender_X, ender_Y, ender_Z, skip_homing, settle_time
    init = True
    ender_ready = False

    while not exit:
        if init:
            while ser.in_waiting > 0:
                res = ser.readline()
                init = False
                ender_ready = True
                print("SER:", res)

            if ender_ready:
                if not skip_homing:
                    print('Ender: homing.')
                    gcode(ser, b'G28')
                    homing = True

                    print('Ender: Move to Start Position.')
                    gcode(ser, b'G0 F400')  # set the feedrate (mm/m)
                    gcode(ser, b'G0 X0 Y0 Z' + str(focus_height_z).encode())  # move to safe z height
                    gcode(ser, b'M114')  # report current position
                    moving = True

        elif ser.in_waiting > 0:
            res = ser.readline()
            # line =
            if (res.find("X:0.00 Y:0.00 Z:0.00 E:0.00 Count X:0 Y:0 Z:0".encode()) >= 0):
                homing = False
                moving = False

            if (res.find("E:0.00 Count".encode()) >= 0):
                A = [_.split(b':') for _ in res.rstrip().split(b' ')]
                ender_X = float(A[0][1])
                ender_Y = float(A[1][1])
                ender_Z = float(A[2][1])
                settle_time = time()
                moving = False

            print("N:", res)

        else:
            if home:
                print('Ender: homing.')
                gcode(ser, b'G28')
                gcode(ser, b'G0 X0 Y0 Z' + str(focus_height_z).encode())  # move to safe z height
                gcode(ser, b'M114')
                home = False
                homing = True

            if move:

                # safety: never move while the probe is down
                if (ender_Z < focus_height_z):
                    print('Ender: Move Z to safe distance first')
                    gcode(ser, b'G90')  # set absolute position mode
                    gcode(ser, b'G0 Z' + str(focus_height_z).encode())  # move to safe z height
                    gcode(ser, b'M114')

                print('Ender: Move.')
                gcode(ser, b'G0 F3000')  # set the feedrate
                gcode(ser, b'G91')  # set relative position mode

                parts = [_.split(' ') for _ in move.rstrip().split(' ')]
                if len(parts) == 1:
                    if parts[0][0][:1] == 'X':
                        x = float(parts[0][0][1:])
                        moveparts = 'X' + str(round(x, 3) + 0.5) + ' Y0.5'
                    elif parts[0][0][:1] == 'Y':
                        y = float(parts[0][0][1:])
                        moveparts = 'X0.5 Y' + str(round(y, 3) + 0.5)
                elif len(parts) == 2:
                    x = float(parts[0][0][1:])
                    y = float(parts[1][0][1:])
                    moveparts = 'X' + str(round(x, 3) + 0.5) + ' Y' + str(round(y, 3) + 0.5)

                gcode(ser, b'G0' + moveparts.encode())
                print(b'G0 ' + moveparts.encode())  # debug

                gcode(ser, b'G0 X-0.5 Y-0.5')  # backlash compensation: always approach each point from same side
                # gcode(ser, b'G0 X+0.5 Y+0.5')  # backlash compensation: always approach each point from same side
                # print(b'G0 X-0.5 Y-0.5')  # debug

                gcode(ser, b'M114')
                move = False
                moving = True

            if move_abs:
                print('Ender: Move to Position: %s' % move_abs)
                gcode(ser, b'G0 F3000')  # set the feedrate
                gcode(ser, b'G90')  # set absolute position mode

                # safety: never move while the probe is down
                if (ender_Z < focus_height_z):
                    print('Ender: Move Z to safe distance first')
                    gcode(ser, b'G90')  # set absolute position mode
                    gcode(ser, b'G0 Z' + str(focus_height_z).encode())  # move to safe z height
                    gcode(ser, b'M114')

                parts = [_.split(' ') for _ in move_abs.rstrip().split(' ')]
                x = float(parts[0][0][1:])
                y = float(parts[1][0][1:])
                # movepart1 = 'X' + str(x + 0.5) + ' Y' + str(y + 0.5)  # backlash compensation
                movepart1 = 'X' + str(x - 0.5) + ' Y' + str(y - 0.5)  # backlash compensation
                movepart2 = 'X' + str(x) + ' Y' + str(y)

                # print(b'G0 ' + movepart1.encode())  # backlash compensation: always approach each point from same side
                # print(b'G0 ' + movepart2.encode())
                gcode(ser,
                      b'G0 ' + movepart1.encode())  # backlash compensation: always approach each point from same side
                gcode(ser, b'G0 ' + movepart2.encode())

                # gcode(ser, b'G0 ' + move_abs.encode())
                # print (b'G0 ' + move_abs.encode()) # debug
                gcode(ser, b'M114')
                move_abs = False
                moving = True

            if moveZ_abs:
                print('Ender: Move to Position: %s' % moveZ_abs)
                gcode(ser, b'G0 F500')  # set the feedrate
                gcode(ser, b'G90')  # set absolute position mode
                gcode(ser, b'G0 ' + moveZ_abs.encode())
                gcode(ser, b'M114')
                moveZ_abs = False
                moving = True

            if moveZ:
                print('Ender: Move Z.')
                gcode(ser, b'G0 F300')  # set the feedrate to 1600
                gcode(ser, b'G91')  # set relative position mode
                gcode(ser, b'G0 ' + moveZ.encode())
                print(b'G0 ' + moveZ.encode())  # debug
                gcode(ser, b'M114')
                moveZ = False
                moving = True

            # print("X")

        sleep(0.05)


def engine():
    global timer, exit

    while not exit:
        sleep(0.05)


win_flags = cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL

cv2.namedWindow('Capture', win_flags)
# cv2.resize(
cv2.resizeWindow('Capture', 800, 450)
cv2.moveWindow('Capture', 0, 0)

# cv2.setMouseCallback("Capture", mouse_event)

capture_thread = Thread(target=capture)
capture_thread.start()

cv2.namedWindow('Analyze', win_flags)
cv2.resizeWindow('Analyze', int(ana_size[0] / 2), int(ana_size[1] / 2))
cv2.moveWindow('Analyze', 961, 0)

analyze_thread = Thread(target=analyze)
analyze_thread.start()

ender_thread = Thread(target=ender)
ender_thread.start()

engine_thread = Thread(target=engine)
engine_thread.start()

try:
    prev_frame_cnt = 0
    prev_analysis_cnt = 0

    while not exit:
        if prev_frame_cnt != frame_cnt:
            img = frame.copy()
            prev_frame_cnt = frame_cnt

            overlay(img)

            # cseq = [str(_) for _ in ana_seq]
            # ovtext(img, " " + ".".join(cseq), (240, 64))
            # cobj = [str(min(9,_)) for _ in ana_obj]
            # ovtext(img, ".".join(cobj), (480, 64))

            scale_percent = 50  # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)

            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            cv2.imshow('Capture', resized)

        if prev_analysis_cnt != analysis_cnt:
            img_ana = analysis
            prev_analysis_cnt = analysis_cnt

            overana(img_ana)
            # choice(img)

            scale_percent = 70  # percent of original size
            width = int(img_ana.shape[1] * scale_percent / 100)
            height = int(img_ana.shape[0] * scale_percent / 100)
            dim = (width, height)

            resized = cv2.resize(img_ana, dim, interpolation=cv2.INTER_AREA)

            cv2.imshow('Analyze', resized)

            # print(ana_obj, ana_pos, ana_pas)

        if measuring_run:
            if not moving:
                arrived_x = floatcompare(ender_X, testpads[pad_hightlight_index]['trans-x'] + camera_to_probe_offset_x,
                                         1)
                arrived_y = floatcompare(ender_Y, testpads[pad_hightlight_index]['trans-y'] + camera_to_probe_offset_y,
                                         1)
                if (arrived_x and arrived_y):
                    # print("Arrived at Measurement Location") #debug
                    arrived_z = floatcompare(ender_Z, probing_height, 1)
                    if (arrived_z):
                        # print("Probing") #debug
                        if (time() > (settle_time + dwell_time)):
                            settle_time = time()
                            print("Dwell Time passed at Probing Height")

                            url = 'http://' + pi_zero_ip + ':8080/stream/snapshot.jpeg?delay_s=0'
                            filename = wget.download(url, bar=None)
                            os.rename(filename,
                                      testpads[pad_hightlight_index]['partname'] + "-" + testpads[pad_hightlight_index][
                                          'net'] + '.jpg')
                            print("Captured Picture: ",
                                  testpads[pad_hightlight_index]['partname'] + "-" + testpads[pad_hightlight_index][
                                      'net'] + '.jpg')

                            print("Starting Measurement:")

                            for i in range(measurement_count):
                                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                                    s.connect((pi_zero_ip, pi_zero_port))
                                    s.sendall(b'A')
                                    measurement_data = s.recv(1024)
                                    # print('Received', repr(measurement_data))
                                    f = open("testresults.csv", "a")
                                    # f.write("PARTNAME(NETNAME);TRANSFORMED-X;TRANSFORMED-Y;Measurement-Result" # HEADER
                                    f.write(
                                        testpads[pad_hightlight_index]['partname'] + " (" +
                                        testpads[pad_hightlight_index][
                                            'net'] + ");" + str(
                                            testpads[pad_hightlight_index]['trans-x']) + ";" +
                                        str(testpads[pad_hightlight_index]['trans-y']) + ";" + repr(
                                            measurement_data) + "\r\n")
                                    f.close()
                                    print("Measurement: " + repr(measurement_data))

                            pad_hightlight_index += 1
                            if pad_hightlight_index >= len(testpads):
                                measuring_run = False
                            else:
                                moveZ_abs = "Z" + str(focus_height_z)
                                move_abs = "X" + str(round(testpads[pad_hightlight_index]['trans-x'],
                                                           2) + camera_to_probe_offset_x) + " Y" + str(
                                    round(testpads[pad_hightlight_index]['trans-y'], 2) + camera_to_probe_offset_y)
                    else:
                        moveZ_abs = "Z" + str(probing_height)

        # if save_image:
        # url = 'http://' + pi_zero_ip + ':8080/stream/snapshot.jpeg?delay_s=0'
        # filename = wget.download(url, bar=None)
        # os.rename(filename,
        #           testpads[pad_hightlight_index]['partname'] + "-" + testpads[pad_hightlight_index][
        #              'net'] + '.jpg')
        # print ("Captured Picture: ", testpads[pad_hightlight_index]['partname'] + "-" + testpads[pad_hightlight_index][
        #              'net'] + '.jpg')

        # print("Starting Measurement:")

        # this only works if running minicom -c on -b 1000000 -D /dev/ttyS0 -w S0 first to configure the signals

        # for i in range(4):
        #     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        #         s.connect((pi_zero_ip, pi_zero_port))
        #         s.sendall(b'A')
        #         data = s.recv(1024)
        #         #print('Received', repr(data))
        #         f = open("testresults.csv", "a")
        #         # f.write("PARTNAME(NETNAME);TRANSFORMED-X;TRANSFORMED-Y;Measurement-Result" # HEADER
        #         f.write(
        #             testpads[pad_hightlight_index]['partname'] + " (" + testpads[pad_hightlight_index][
        #                 'net'] + ");" + str(
        #                 testpads[pad_hightlight_index]['trans-x']) + ";" +
        #             str(testpads[pad_hightlight_index]['trans-y']) + ";" + repr(data) + "\r\n")
        #         f.close()
        #         print("Measurement: " + repr(data))
        #
        #    cv2.imwrite(save_image + "-cam.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        #    print('Writing Image: ', save_image + "-cam.jpg", [cv2.IMWRITE_JPEG_QUALITY, 80])
        #    cv2.imwrite(save_image + "-analysis.jpg", img_ana, [cv2.IMWRITE_JPEG_QUALITY, 80])
        #    print('Writing Image: ', save_image + "-analysis.jpg", [cv2.IMWRITE_JPEG_QUALITY, 80])

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # escape
            safetofile()
            exit = True

        # elif key == ord('c'):  # continue
        #    halt = False
        # elif key == ord('e'):  # enable
        #    enable = True
        # elif key == ord('h'):   # halt
        # halt = True

        elif key == ord('q'):  # cycle through image analysing filters
            analyze_filter_id += 1
            if (analyze_filter_id > 6):
                analyze_filter_id = 0

        elif key == ord('h'):  # home
            home = True

        # Movements
        elif key == 85:  # move higher
            moveZ = "Z" + str(move_stepsize_z)
        elif key == 86:  # move lower
            if (ender_Z >= move_stepsize_z + focus_height_z):  # prevent crashing into PCB
                moveZ = "Z-" + str(move_stepsize_z)
        elif key == 83:  # move right
            move = "X" + str(move_stepsize_xy)
        elif key == 81:  # move left
            move = "X-" + str(move_stepsize_xy)
        elif key == 82:  # move up
            move = "Y" + str(move_stepsize_xy)
        elif key == 84:  # move down
            move = "Y-" + str(move_stepsize_xy)

        elif key == ord('1'):  # thr[0]--
            thr_val[0] -= 1
        elif key == ord('2'):  # thr[0]++
            thr_val[0] += 1
        elif key == ord('3'):  # thr[1]--
            thr_val[1] -= 1
        elif key == ord('4'):  # thr[1]++
            thr_val[1] += 1
        elif key == ord('5'):  # thr[2]--
            thr_val[2] -= 1
        elif key == ord('6'):  # thr[2]++
            thr_val[2] += 1
        elif key == ord('7'):  # thr[3]--
            thr_val[3] -= 1
        elif key == ord('8'):  # thr[3]++
            thr_val[3] += 1
        elif key == ord('9'):  # thr[4]--
            thr_val[4] -= 1
        elif key == ord('0'):  # thr[4]++
            thr_val[4] += 1

        # step sizes
        elif key == ord('u'):
            if move_stepsize_xy >= 6:
                move_stepsize_xy -= 1
            elif move_stepsize_xy <= 5.9 and move_stepsize_xy > 0.1:
                move_stepsize_xy -= 0.1
            elif move_stepsize_xy <= 0.1:
                move_stepsize_xy -= 0.01

            if move_stepsize_xy < 0.01:
                move_stepsize_xy = 0.01

        elif key == ord('i'):
            if move_stepsize_xy >= 5:
                move_stepsize_xy += 1
            elif move_stepsize_xy <= 5 and move_stepsize_xy > 0.09:
                move_stepsize_xy += 0.1
            elif move_stepsize_xy <= 0.09:
                move_stepsize_xy += 0.01

        elif key == ord('o'):
            move_stepsize_z -= 0.1
            if move_stepsize_z < 0.1:
                move_stepsize_z = 0.1
        elif key == ord('p'):
            move_stepsize_z += 0.1
            if move_stepsize_z > 2.0:
                move_stepsize_z = 2.0

        # set fiducial locations
        elif key == ord('v'):  # fid1
            data['fiducial'][0]['x'] = ender_X
            data['fiducial'][0]['y'] = ender_Y
            transformpoints()
        elif key == ord('b'):  # fid2
            data['fiducial'][1]['x'] = ender_X
            data['fiducial'][1]['y'] = ender_Y
            transformpoints()
        elif key == ord('n'):  # fid3
            data['fiducial'][2]['x'] = ender_X
            data['fiducial'][2]['y'] = ender_Y
            transformpoints()
        elif key == ord('m'):  # fid4
            data['fiducial'][3]['x'] = ender_X
            data['fiducial'][3]['y'] = ender_Y
            transformpoints()

        elif key == 9:  # TAB select next fid
            fid_hightlight_index += 1
            if fid_hightlight_index > 3:
                fid_hightlight_index = 0

        elif key == 10:  # ENTER: move to selected fiducial
            move_abs = "X" + str(data['fiducial'][fid_hightlight_index]['x']) + " Y" + str(
                data['fiducial'][fid_hightlight_index]['y'])

        elif key == 32:  # Space: center to closest detected circle
            if (ana_pos[0] - 360 < 100 and ana_pos[1] - 360 < 100):
                print("correction: X:" + str((ana_pos[0] - 360) / camera_pixels_per_mm) + " Y:" + str(
                    (ana_pos[1] - 360) / -camera_pixels_per_mm))
                move = "X" + str(round((ana_pos[0] - 360) / camera_pixels_per_mm, 2)) + " Y" + str(
                    round((ana_pos[1] - 360) / -camera_pixels_per_mm, 2))
                print(move)

        # elif key == 106:  # J key: test needle offset
        # move_abs = "X" + str(data['fiducial'][fid_hightlight_index]['x'] + camera_to_probe_offset_x) + " Y" + str(
        #  data['fiducial'][fid_hightlight_index]['y'] + camera_to_probe_offset_y)

        elif key == ord('a'):  # A - cycle through testpads
            pad_hightlight_index += 1
            if pad_hightlight_index >= len(testpads):
                pad_hightlight_index = 0

        elif key == ord('c'):  # C - reset testpad index
            pad_hightlight_index = 0

        elif key == ord('s'):  # S - move camera to selected testpad
            move_abs = "X" + str(round(testpads[pad_hightlight_index]['trans-x'], 2)) + " Y" + str(
                round(testpads[pad_hightlight_index]['trans-y'], 2))

        elif key == ord('d'):  # D - move probe to selected testpad
            move_abs = "X" + str(
                round(testpads[pad_hightlight_index]['trans-x'], 2) + camera_to_probe_offset_x) + " Y" + str(
                testpads[pad_hightlight_index]['trans-y'] + camera_to_probe_offset_y)
            moveZ_abs = "Z" + str(probing_height)
            # print (test)
            # print(test2)
            # move_abs = "X" + str(round(testpads[pad_hightlight_index]['trans-x'], 2)) + " Y" + str(
            #     round(testpads[pad_hightlight_index]['trans-y'],2 ))

        elif key == ord('f'):  # F - move to safe z height
            moveZ_abs = "Z" + str(focus_height_z)

        elif key == ord('x'):  # measure
            if not measuring_run:
                measuring_run = True
                print("Starting Measurement Run")
                move_abs = "X" + str(
                    round(testpads[pad_hightlight_index]['trans-x'], 2) + camera_to_probe_offset_x) + " Y" + str(
                    testpads[pad_hightlight_index]['trans-y'] + camera_to_probe_offset_y)
                moveZ_abs = "Z" + str(probing_height)
            else:
                measuring_run = False
                print("Stopping Measurement Run")

        elif key == ord('j'):  # J - save fiducial locations to file
            safetofile()
            print("prober.json saved")

        elif key == 255:  # nokey
            pass

        else:
            print("unknown key %d" % key)

        """
        elif key == ord('0'):   # reset
            cap.set(cv2.CAP_PROP_CONTRAST, ini_con)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, ini_bri)
            cap.set(cv2.CAP_PROP_SATURATION, ini_sat)

        elif key == ord('1'):   # dec contrast
            val = cap.get(cv2.CAP_PROP_CONTRAST)
            cap.set(cv2.CAP_PROP_CONTRAST, val - 0.01)
        elif key == ord('2'):   # inc contrast
            val = cap.get(cv2.CAP_PROP_CONTRAST)
            cap.set(cv2.CAP_PROP_CONTRAST, val + 0.01)

        elif key == ord('3'):   # dec brightness
            val = cap.get(cv2.CAP_PROP_BRIGHTNESS)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, val - 0.01)
        elif key == ord('4'):   # inc brightness
            val = cap.get(cv2.CAP_PROP_BRIGHTNESS)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, val + 0.01)

        elif key == ord('5'):   # dec saturation
            val = cap.get(cv2.CAP_PROP_SATURATION)
            cap.set(cv2.CAP_PROP_SATURATION, val - 0.01)
        elif key == ord('6'):   # inc saturation
            val = cap.get(cv2.CAP_PROP_SATURATION)
            cap.set(cv2.CAP_PROP_SATURATION, val + 0.01)
        """

except KeyboardInterrupt:
    exit = True

capture_thread.join()
analyze_thread.join()
ender_thread.join()
engine_thread.join()

cap.release()
cv2.destroyAllWindows()
