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
./probe.py <serial device> <serial device baud rate> <webcam id>

./probe.py /dev/ttyUSB0 1000000 0
```

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

v = set current camera location as fiducial 1 location

b = set current camera location as fiducial 2 location

n = set current camera location as fiducial 3 location

m = set current camera location as fiducial 4 location



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