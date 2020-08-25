#!/usr/bin/env python3
#
# Copyright (C) 2020 Herbert Poetzl

import sys
import keyboard
import serial

from time import sleep

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


def gcode(ser, cmd):
    ser.write(cmd + b'\n')

    while True:
        res = ser.readline()
        print(res)
        if res == b'ok\n':
            break


sleep(2.0)
while ser.in_waiting > 0:
    res = ser.readline()
    print(res)

print('... init skipped.')
while True:
    if ser.in_waiting > 0:
        res = ser.readline()
        print(res)

#gcode(ser, b'G28')



