#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:01:46 2024

@author: ash
"""

from ultralytics import YOLO
model = YOLO('best.pt')
model.predict(source="/home/ash/Downloads/OneDrive_2_2-29-2024/camera1_Oct12-120506.mp4", show=True)