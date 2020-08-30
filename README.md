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

To allow non-root to access the serial port:
```
sudo chmod a+rw /dev/ttyUSB0
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

h = home machine - performed at startup

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
