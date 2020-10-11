# pcb-prober
An open hardware/free software low tech flying-probe tester based on available 
technology (3D printer mechanics/controller, raspberry pi, openCV, etc.)



# Software Requirements
Python3, NumPy, keyboard, opencv-python, serial, wget

```
pip install numpy, keyboard, serial
```
opencv and serial made trouble to install via pip so:
```
sudo apt install python3-opencv, python3-serial
```
did the trick.

To allow temporary non-root to access the serial port:
```
sudo chmod a+rw /dev/ttyUSB0
```
to make it permanent edit the udev rules (filling in your own idVendor and idProduct):
```
ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6010", MODE="0660", GROUP="dialout"
```
# Usage
```
./probe.py -i <CSV file> -c <webcam ID> -b <baud rate> -u <Marlin serial USB device> -p <Raspberry Pi IP>>

./probe.py -u /dev/ttyUSB0 -b 1000000 -c 0 -i probe.csv -p 192.168.0.1
```
Note that backlash compensation is active and results in each point 
being approached from the same side. This means that even small distance movements result 
in the machine moving at least 0.5mm on each axis.

Fiducial locations are saved into prober.json files when exiting the application
with the ESCAPE button and loaded on startup.

# Keyboard Shortcuts
Arrow Up = move Stepsize(XY) upwards

Arrow Down = move Stepsize(XY) downwards

Arrow Left = move Stepsize(XY) left

Arrow Right = move Stepsize(XY) right

Page Up = move Stepsize(Z) higher

Page Down = move Stepsize(Z) lower - limit: 6mm - to prevent crashing needle into board

u = decrease Stepsize(XY) - limit: 0.01mm

i = increase Stepsize(XY)

o = decrease Stepsize(Z) - limit: 0.1mm

p = increase Stepsize(Z) - limit: 2mm

h = manually home machine - normally performed at startup

j = save fiducial locations to file (prober.json)

TAB = cycle selection through the 4 fiducials

ENTER = move camera to selected fiducial

SPACE = optically center crosshair to centered fiducial - requires to be within the fiducial circle already

v = set current camera location as fiducial 1 location

b = set current camera location as fiducial 2 location

n = set current camera location as fiducial 3 location

m = set current camera location as fiducial 4 location

a = cycle selection through the testpads

c = reset to first testpad (index = 0)

x = start/pause the automatic probing process

s = move camera to selected testpads

d = probe here (camera crosshair)

f = raise probe to safe height

1 = decrease hue range filter min

2 = increase hue range filter min

3 = decrease hue range filter max

4 = increase hue range filter max

5 = decrease saturation range filter

6 = increase saturation range filter

7 = decrease value range filter

8 = increase value range filter

9 = decrease size filter

0 = increase filter filter

q = cycle through filter channel display

e = increase camera pixel per mm value by 1

r = decrease camera pixel per mm value by 1

ESCAPE = quit

# Pi Zero
Prepare ports, set outputs:
```
gpio -g mode  12 out
gpio -g mode  13 out
```

Turn on Lights:
```
gpio -g write  12 1
gpio -g write  13 1
```

# OpenCV Image Analysis
The image is converted to HSV color space.

The 5 filter parameters editable with number keys 1 to 8 do:
```
[0] and [1] define the hue (H) range (displayed in red color)
[2] saturation (S) cut off (displayed in green color)
[3] value (V) cut off (displayed in blue color)
[4] size limiter - not editable currently
```

# Measurement Interface
On the Pi Zero:
```
minicom -c on -b 1000000 -D /dev/ttyS0 -w S0
```
sending letter 'A' over ttyS0 should trigger a measurement, results are returned in the format:
```
A       1C2 2BA 1EC 1E7 0F9 067
```