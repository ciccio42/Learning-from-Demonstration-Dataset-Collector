#!/usr/bin/env python3
import rospy
import actionlib
import actionlib_tutorials.msg


class PrimitiveActionServer():
    # create messages that are used to publish feedback/result
    _feedback = ()
    _result = ()

    def __init__(self) -> None:
        pass

    def execute_cb(self, goal):
        pass
