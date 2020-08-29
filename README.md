# pcb-prober
An open hardware/free software low tech flying-probe tester based on available technology (3d printer mechanics/controller, raspberry pie, openCV, etc.)



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
