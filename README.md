# pcb-prober
An open hardware/free software low tech flying-probe tester based on available 
technology (3D printer mechanics/controller, raspberry pi, openCV, etc.)



# Software Requirements
Python3, NumPy, keyboard, opencv-python, serial

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
./probe.py -i <CSV file> -c <webcam ID> -b <baud rate> -u <Marlin serial USB device>

./probe.py -u /dev/ttyUSB0 -b 1000000 -c 0 -i probe.csv
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

9 = decrease Stepsize(XY) - limit: 0.01mm

0 = increase Stepsize(XY)

o = decrease Stepsize(Z) - limit: 0.1mm

p = increase Stepsize(Z) - limit: 2mm

h = manually home machine - normally performed at startup

TAB = cycle selection through the 4 fiducials

ENTER = move camera to selected fiducial

SPACE = optically center crosshair to centered fiducial - requires to be within the fiducial circle already

v = set current camera location as fiducial 1 location

b = set current camera location as fiducial 2 location

n = set current camera location as fiducial 3 location

m = set current camera location as fiducial 4 location

a = cycle selection through the testpads

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

q = cycle through filter channel display

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