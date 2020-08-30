#!/bin/bash

gpio -g mode  12 out
gpio -g mode  13 out
gpio -g write  12 1
gpio -g write  13 1

