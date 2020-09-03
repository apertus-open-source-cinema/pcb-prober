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

import errno
import posix
import struct

import serial

from time import sleep, time
from random import randint

import transforms3d as t3d

from threading import Thread, Condition, Lock


tty = sys.argv[1]
baud = int(sys.argv[2])

ser = serial.Serial(
    port = tty,
    baudrate = baud,
    bytesize = serial.EIGHTBITS,
    parity = serial.PARITY_NONE,
    stopbits = serial.STOPBITS_ONE,
    interCharTimeout = 0.5,
    timeout = 5.0,
    xonxoff = False,
    rtscts = False,
    dsrdtr = False);

# fifo = posix.open(sys.argv[1], posix.O_WRONLY | posix.O_NONBLOCK)

enable = True
active = True
exit = False
halt = False
quit = False
home = False
homing = False
move = False
moveZ = False
moving = False
move_abs = False
move_stepsize_xy = 2.0
move_stepsize_z = 2.0
focus_height_z = 6.0 # 6mm to protect into crashing PCB
pcb_height_z = 5.0
probing_height = 2.3 # at this height the probe needle slightly touches the PCB


#current position
ender_X = 0.0
ender_Y = 0.0
ender_Z = 0.0


#fiducials
data = {}

def safetofile():
    with open('prober.json', 'w') as f:
        json.dump(data, f)

f = open('prober.json')
data = json.load(f)
fid_hightlight_index = 0;

camera_to_probe_offset_x = 27.56
camera_to_probe_offset_y = 0.73
camera_pixels_per_mm = 270 # measured at slightly above work height of 6mm

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

ana_size = (720, 720)
ana_roi = (int((1920 - ana_size[0])/2),
           int((1080 - ana_size[1])/2),
           int((1920 - ana_size[0])/2) + ana_size[0],
           int((1080 - ana_size[1])/2) + ana_size[1])

ana_obj = [0]*4
ana_pos = [0]*2
ana_pas = [0]*4
ana_idx = -1
ana_seq = [0]*5

thr_val = [56, 231, 68, 209, 125]


cap = cv2.VideoCapture(int(sys.argv[3]))

ini_con = cap.get(cv2.CAP_PROP_CONTRAST)
ini_bri = cap.get(cv2.CAP_PROP_BRIGHTNESS)
ini_sat = cap.get(cv2.CAP_PROP_SATURATION)

cap.set(cv2.CAP_PROP_FPS, 25)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cap.set(cv2.CAP_PROP_CONTRAST, 0.10)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.95)
cap.set(cv2.CAP_PROP_SATURATION, 0.15)


col = [(0,0,255), (0,200,255), (255,50,50), (0,200,0), (255,255,255), (0,0,0)]

# load test points from file
# TODO

# fiducials

P = np.array([[data['fiducial'][0]['x'], data['fiducial'][0]['y'], pcb_height_z],
              [data['fiducial'][1]['x'], data['fiducial'][1]['y'], pcb_height_z],
              [data['fiducial'][2]['x'], data['fiducial'][2]['y'], pcb_height_z],
              [data['fiducial'][3]['x'], data['fiducial'][3]['y'], pcb_height_z]])

# transformation

T = [0.5, 0.6, 0.7]
R = t3d.euler.euler2mat(0.1, 0.2, 0.3, 'sxyz')
Z = [0.5, 0.4, 0.3]

# transformation matrix

A = t3d.affines.compose(T,R,Z)

# transformed points

Q = np.dot(P, A[0:3,0:3]) + A[0:3,3]



# calculate matrix from points

n = P.shape[0]
pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
unpad = lambda x: x[:,:-1]

X = pad(P)
Y = pad(Q)

B, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)

trans = lambda x: unpad(np.dot(pad(x), B))



# create some new points

p = np.array([(P[_]+P[(_+1)%4])/2 for _ in range(4)])

# transform those points

q = trans(p)


def ovtext(img, txt="test", pos=(0,0), col=(255,255,255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1

    # cv2.putText(img, txt, pos,
    #    font, font_scale, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(img, txt, pos,
        font, font_scale, col, 2, cv2.LINE_AA)


def overlay(img):
    ox, oy, ow, oh = 4, 4, 1920-8, 32


    sub = img[oy:oy+oh, ox:ox+ow]
    img[oy:oy+oh, ox:ox+ow] = sub >> 1    #dark background

    rx0, ry0, rx1, ry1 = ana_roi
    cx, cy, r = int((rx0+rx1)/2), int((ry0+ry1)/2), 64
    
    cv2.rectangle(img, (rx0, ry0), (rx1, ry1), (0,0,255), 1)
    cv2.line(img, (cx, cy-r), (cx, cy+r), (0,0,0), 3)
    cv2.line(img, (cx-r, cy), (cx+r, cy), (0,0,0), 3)
    cv2.line(img, (cx, cy-r), (cx, cy+r), (0,255,0), 1)
    cv2.line(img, (cx-r, cy), (cx+r, cy), (0,255,0), 1)

    ovtext(img, "FPS %3.1f" % (frame_fps), (10, 30))
    ovtext(img, "*%08d" % (prev_frame_cnt), (240, 30))
    ovtext(img, "X:%3.2f Y:%3.2f Z:%3.2f" % (ender_X, ender_Y, ender_Z), (720, 30))
    ovtext(img, "Stepsize(XY): %3.2fmm Stepsize(Z): %3.2fmm" % (move_stepsize_xy, move_stepsize_z), (1120, 30))

    #fiducials
    ox1, oy1, ow1, oh1 = 0, 90, 450, 160
    sub2 = img[oy1:oy1+oh1, ox1:ox1+ow1]
    img[oy1:oy1+oh1, ox1:ox1+ow1] = sub2 >> 1    #dark background
    ovtext(img, "Fid 1 (v): %3.2f, %3.2f" % (data['fiducial'][0]['x'], data['fiducial'][0]['y']), (10, 120))
    ovtext(img, "Fid 2 (b): %3.2f, %3.2f" % (data['fiducial'][1]['x'], data['fiducial'][1]['y']), (10, 160))
    ovtext(img, "Fid 3 (n): %3.2f, %3.2f" % (data['fiducial'][2]['x'], data['fiducial'][2]['y']), (10, 200))
    ovtext(img, "Fid 4 (m): %3.2f, %3.2f" % (data['fiducial'][3]['x'], data['fiducial'][3]['y']), (10, 240))
    cv2.rectangle(img, (0, fid_hightlight_index * 40 + 125),(8 ,fid_hightlight_index * 40 + 95), (0, 98, 255), -1)

    if homing:
        ovtext(img, "HOMING", (10, 64))
    elif moving:
        ovtext(img, "MOVING", (10, 64))
    elif halt:
        ovtext(img, "HALTING", (10, 64))
    elif quit:
        ovtext(img, "EXITING", (10, 64))

    if enable:
        ovtext(img, "ENABLED", (480, 30))
    else:
        ovtext(img, "ACTIVE", (10, 64))

def overana(img):
    ox, oy, ow, oh = 4, 4, ana_roi[3]-8, 66

    sub = img[oy:oy+oh, ox:ox+ow]
    img[oy:oy+oh, ox:ox+ow] = sub >> 1

    cx, cy, r = int(ana_size[0]/2), int(ana_size[1]/2), 64

    cv2.line(img, (cx, cy-r), (cx, cy+r), (0,0,0), 3)
    cv2.line(img, (cx-r, cy), (cx+r, cy), (0,0,0), 3)
    cv2.line(img, (cx, cy-r), (cx, cy+r), (0,255,0), 1)
    cv2.line(img, (cx-r, cy), (cx+r, cy), (0,255,0), 1)

    # rx0, ry0, rx1, ry1 = caproi()
    
    # cv2.rectangle(img, (0, 0), (640, ry0), (0,0,0), -1)

    # cv2.line(img, (rx0, ry0), (rx0, ry1), (255,255,255), 1)

    lag = frame_cnt - analysis_cnt
    ovtext(img, "TIME %3.1fms" % (analysis_delta), (10, 30))
    ovtext(img, "*%08d" % (prev_analysis_cnt), (240, 30))
    ovtext(img, "LAG %d" % (lag), (480, 30))

    #ovtext(img, "STATE %s" % (state_str(state)), (10, 64))
    #ovtext(img, "+%3.1fs" % (sdelta()), (300, 64))

    ovtext(img, "%3d %3d %3d %3d %3d" % tuple(thr_val), (10, 64))

    if active:
        ovtext(img, "ACTIVE", (480, 64))

def choice(img):
    pass
    # rx0, ry0, rx1, ry1 = caproi()
    
    #    yp = ry0 + int((idx + 0.5)*row_size)
    #    cv2.circle(img, (selvis, yp), 10, col[idx], -1)
    #    cv2.circle(img, (selvis, yp), 15, col[4], 5)


def capture():
    global frame, frame_cnt, frame_cnt_cond
    global frame_time, frame_delta, frame_fps
    global cap, exit, home, homing, move, moving

    frame_time = time()
    fps = [0]*25

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

                fps = [1/frame_delta] + fps[:-1]

                if frame_cnt % 10 == 0:
                    frame_fps = round(sum(fps))/25;
            
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
    #hsv_grid = [np.array([0,0,30]), np.array([200,50,160])]
    #hsv_high = [np.array([0,40,0]), np.array([72,255,255])]
    kern_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    par = cv2.SimpleBlobDetector_Params()
    par.minThreshold = 50
    par.maxThreshold = 260

    par.filterByArea = True
    par.minArea = 8000
    par.maxArea = 20000

    par.filterByColor = False

    par.filterByCircularity = False
    par.minCircularity = 0.2

    par.filterByConvexity = True
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
        #msk_grid = cv2.inRange(hsv, hsv_grid[0], hsv_grid[1]) 
        #msk_high = cv2.inRange(hsv, hsv_high[0], hsv_high[1]) 
        #msk = cv2.bitwise_not(cv2.bitwise_or(msk_grid, msk_high)) 
        #res = cv2.bitwise_and(roi, roi, mask=msk)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        ret, ht = cv2.threshold(h, thr_val[0], 0, cv2.THRESH_TOZERO)
        ret, ht = cv2.threshold(ht, thr_val[1], 0, cv2.THRESH_TOZERO_INV)
        ret, st = cv2.threshold(s, thr_val[2], 255, cv2.THRESH_BINARY)
        ret, vt = cv2.threshold(v, thr_val[3], 255, cv2.THRESH_BINARY)
        thr = cv2.merge((ht,st,vt))
        #ret, thr = cv2.threshold(s, 80, 250, cv2.THRESH_TOZERO)
        # gry = cv2.cvtColor(thr, cv2.COLOR_BGR2GRAY)
        mor = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kern_open)
        img = cv2.morphologyEx(mor, cv2.MORPH_CLOSE, kern_close)
        #img = cv2.cvtColor(mor, cv2.COLOR_GRAY2RGB)
        #img = thr

        kpt = detector.detect(img)
        img = cv2.drawKeypoints(img, kpt, np.array([]), \
            (255,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #split = align - rx0
        #limit = finish - rx0
        #purge = ignore - rx0

        pos = [0]*len(kpt)

        for idx, kp in enumerate(kpt):
            xf, yf = kp.pt
            x, y = int(round(xf)), int(round(yf))
            #interesting = False

            pos[idx] = (x, y)

            #print(pos[idx])

            if kp.size < thr_val[4]:
                ovtext(img, "%3d" % kp.size, (x-30, y-8), (0,0,0))
            else:
                if 300 < pos[idx][0] < 420 and 300 < pos[idx][1] < 420:
                    ovtext(img, "%3d" % kp.size, (x - 30, y - 8), (0, 0, 255))
                    # interesting = True
                    ana_pos = pos[idx]
                else:
                    ovtext(img, "%3d" % kp.size, (x - 30, y - 8), (255, 255, 255))

        analysis = img
        analysis_cnt += 1

        prev = analysis_time
        analysis_time = time()
        analysis_delta = (analysis_time - mark)*1000


def gcode(ser, cmd):
    ser.write(cmd + b'\n')

def ender():
    global ser, exit, home, homing, move, moving, move_abs, moveZ, ender_X, ender_Y, ender_Z
    init = True

    while not exit:
        if init:
            while ser.in_waiting > 0:
                res = ser.readline()
                init = False
                print("SER:", res)

            print('Ender: homing.')
            gcode(ser, b'G28')
            homing = True

            print('Ender: Move to Start Position.')
            gcode(ser, b'G0 F400') # set the feedrate to 400
            gcode(ser, b'G0 X0 Y0 Z' + str(focus_height_z).encode()) # move to safe z height
            gcode(ser, b'M114')
            moving = True
    
        elif ser.in_waiting > 0:
            res = ser.readline()
            #line = 
            if (res.find("X:0.00 Y:0.00 Z:0.00 E:0.00 Count X:0 Y:0 Z:0".encode()) >= 0):
                homing = False

            if (res.find("E:0.00 Count".encode()) >= 0):
                moving = False
                A = [_.split(b':') for _ in res.rstrip().split(b' ')]
                ender_X = float(A[0][1])
                ender_Y = float(A[1][1])
                ender_Z = float(A[2][1])

            print("N:", res)


        else:
            if home:
                print('Ender: homing.')
                gcode(ser, b'G28')
                gcode(ser, b'G0 X0 Y0 Z' + str(focus_height_z).encode()) # move to safe z height
                gcode(ser, b'M114')
                home = False
                homing = True

            if moveZ:
                print('Ender: Move Z.')
                gcode(ser, b'G0 F300')  # set the feedrate to 1600
                gcode(ser, b'G91')  # set relative position mode
                gcode(ser, b'G0 ' + moveZ.encode())
                print(b'G0 ' + moveZ.encode())  # debug
                gcode(ser, b'M114')
                moveZ = False
                moving = True

            if move:
                print('Ender: Move.')
                gcode(ser, b'G0 F3000') # set the feedrate to 1600
                gcode(ser, b'G91') # set relative position mode

                parts = [_.split(' ') for _ in move.rstrip().split(' ')]
                if len(parts) == 1:
                    if parts[0][0][:1] == 'X':
                        x = float(parts[0][0][1:])
                        moveparts = 'X' + str(round(x,3) + 0.5) + ' Y0.5'
                    elif parts[0][0][:1] == 'Y':
                        y = float(parts[0][0][1:])
                        moveparts = 'X0.5 Y' + str(round(y,3) + 0.5)
                elif len(parts) == 2:
                    x = float(parts[0][0][1:])
                    y = float(parts[1][0][1:])
                    moveparts = 'X' + str(round(x,3) + 0.5) + ' Y' + str(round(y,3) + 0.5)

                gcode(ser, b'G0' + moveparts.encode())
                print (b'G0 ' + moveparts.encode()) # debug

                gcode(ser, b'G0 X-0.5 Y-0.5') #backlash compensation: always approach each point from same side
                print(b'G0 X-0.5 Y-0.5')  # debug

                gcode(ser, b'M114')
                move = False
                moving = True

            if move_abs:
                print('Ender: Move to Position: %s' % move_abs)
                gcode(ser, b'G0 F3000')  # set the feedrate to 800
                gcode(ser, b'G90')  # set absolute position mode

                parts = [_.split(' ') for _ in move_abs.rstrip().split(' ')]
                x = float(parts[0][0][1:])
                y = float(parts[1][0][1:])
                movepart1 = 'X' + str(x + 0.5) + ' Y' + str(y + 0.5)
                movepart2 = 'X' + str(x) + ' Y' + str(y)

                print(b'G0 ' + movepart1.encode()) #backlash compensation: always approach each point from same side
                print(b'G0 ' + movepart2.encode())
                gcode(ser, b'G0 ' + movepart1.encode())  # backlash compensation: always approach each point from same side
                gcode(ser, b'G0 ' + movepart2.encode())

                #gcode(ser, b'G0 ' + move_abs.encode())
                #print (b'G0 ' + move_abs.encode()) # debug
                gcode(ser, b'M114')
                move_abs = False
                moving = True

            #print("X")

        sleep(0.05)


def engine():
    global timer, exit

    while not exit:
        sleep(0.05)




win_flags = cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL

cv2.namedWindow('Capture', win_flags)
#cv2.resize(
cv2.resizeWindow('Capture', 800, 450)
cv2.moveWindow('Capture', 0, 0) 

# cv2.setMouseCallback("Capture", mouse_event)

capture_thread = Thread(target=capture)
capture_thread.start()

cv2.namedWindow('Analyze', win_flags)
cv2.resizeWindow('Analyze', int(ana_size[0]/2), int(ana_size[1]/2))
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


            #cseq = [str(_) for _ in ana_seq]
            #ovtext(img, " " + ".".join(cseq), (240, 64))
            #cobj = [str(min(9,_)) for _ in ana_obj]
            #ovtext(img, ".".join(cobj), (480, 64))

            scale_percent = 50 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height) 

            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 

            cv2.imshow('Capture', resized)

        if prev_analysis_cnt != analysis_cnt:
            img = analysis
            prev_analysis_cnt = analysis_cnt

            overana(img)
            #choice(img)

            scale_percent = 70 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height) 

            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 


            cv2.imshow('Analyze', resized)

            # print(ana_obj, ana_pos, ana_pas)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:           # escape
            safetofile()
            exit = True

        elif key == ord('c'):   # continue
            halt = False
        elif key == ord('d'):   # disable
            enable = False
        elif key == ord('e'):   # enable
            enable = True
        #elif key == ord('h'):   # halt
            #halt = True
        elif key == ord('q'):   # quit
            quit = True
        elif key == ord('h'):   # home
            home = True

        #Movements
        elif key == 85:   # move higher
            moveZ = "Z" + str(move_stepsize_z)
        elif key == 86:   # move lower
            if (ender_Z >= move_stepsize_z + focus_height_z): #prevent crashing into PCB
                moveZ = "Z-" + str(move_stepsize_z)
        elif key == 83:   # move right
            move = "X" + str(move_stepsize_xy) 
        elif key == 81:   # move left
            move = "X-" + str(move_stepsize_xy) 
        elif key == 82:   # move up
            move = "Y" + str(move_stepsize_xy) 
        elif key == 84:   # move down
            move = "Y-" + str(move_stepsize_xy)

        elif key == ord('1'):   # thr[0]--
            thr_val[0] -= 1
        elif key == ord('2'):   # thr[0]++
            thr_val[0] += 1
        elif key == ord('3'):   # thr[1]--
            thr_val[1] -= 1
        elif key == ord('4'):   # thr[1]++
            thr_val[1] += 1
        elif key == ord('5'):   # thr[2]--
            thr_val[2] -= 1
        elif key == ord('6'):   # thr[2]++
            thr_val[2] += 1
        elif key == ord('7'):   # thr[3]--
            thr_val[3] -= 1
        elif key == ord('8'):   # thr[3]++
            thr_val[3] += 1

        #step sizes
        elif key == ord('9'):
            if move_stepsize_xy >= 6:
                move_stepsize_xy -= 1
            elif move_stepsize_xy <= 5.9 and move_stepsize_xy > 0.1:
                move_stepsize_xy -= 0.1
            elif move_stepsize_xy <= 0.1:
                move_stepsize_xy -= 0.01

            if move_stepsize_xy < 0.01:
                move_stepsize_xy = 0.01

        elif key == ord('0'):
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

        #fiducials
        elif key == ord('v'): # fid1
            data['fiducial'][0]['x'] = ender_X
            data['fiducial'][0]['y'] = ender_Y
        elif key == ord('b'):  # fid2
            data['fiducial'][1]['x'] = ender_X
            data['fiducial'][1]['y'] = ender_Y
        elif key == ord('n'):  # fid3
            data['fiducial'][2]['x'] = ender_X
            data['fiducial'][2]['y'] = ender_Y
        elif key == ord('m'):  # fid4
            data['fiducial'][3]['x'] = ender_X
            data['fiducial'][3]['y'] = ender_Y
        elif key == ord('x'):  # move to next fid
            print("x key")
        elif key == 9:  # select next fid
            fid_hightlight_index+= 1
            if fid_hightlight_index > 3:
                fid_hightlight_index = 0
        elif key == 10: # ENTER: move to selected fiducial
            move_abs = "X" + str(data['fiducial'][fid_hightlight_index]['x']) + " Y" + str(data['fiducial'][fid_hightlight_index]['y'])
        elif key == 32:  # Space: center to closest detected circle
            if (ana_pos[0]-360 <100 and ana_pos[1]-360 < 100):
                print("correction: X:" + str((ana_pos[0]-360)/camera_pixels_per_mm) + " Y:" + str((ana_pos[1]-360)/-camera_pixels_per_mm))
                move = "X" + str(round((ana_pos[0]-360)/camera_pixels_per_mm, 2)) + " Y" + str(round((ana_pos[1]-360)/-camera_pixels_per_mm, 2))
                print(move)
        elif key == 255:        # nokey
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

