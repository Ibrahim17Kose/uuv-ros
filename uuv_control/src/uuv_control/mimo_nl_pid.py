#!/usr/bin/env python3

import rospy
import numpy as np


class MimoNonlinearPid:
    def __init__(self, cfg: dict):
        self.cfg = cfg