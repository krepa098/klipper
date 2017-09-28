# Z-Probe support
#
# Copyright (C) 2017  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import pins, homing
from numpy import std


class ProbeResults:
    def __init__(self):
        self.count = 0
        self.points = []  # The probe points in cartesian coords
        self.heights = []  # The heights for the given probe points
        self.std = 0.  # The standard deviation of the results

class PrinterProbe:
    def __init__(self, printer, config):
        self.printer = printer
        self.offset = [config.getfloat('offset_x', 0.0), config.getfloat('offset_y', 0.0), config.getfloat('offset_z', 0.0)]
        self.speed = config.getfloat('speed', 5.0)
        self.z_distance = config.getfloat('max_distance', 20.0)
        self.mcu_probe = pins.setup_pin(printer, 'endstop', config.get('pin'))
        self.toolhead = printer.objects['toolhead']
        z_steppers = self.toolhead.get_z_steppers()
        for s in z_steppers:
            self.mcu_probe.add_stepper(s.mcu_stepper)
        self.min_step_dist = min(s.step_dist for s in z_steppers)
    # External commands
    def probe_height(self):
        # Start homing and issue move
        pos = self.toolhead.get_position()
        pos[2] -= self.z_distance
        print_time = self.toolhead.get_last_move_time()
        self.mcu_probe.home_start(print_time, self.min_step_dist / self.speed)
        self.toolhead.move(pos, self.speed)
        move_end_print_time = self.toolhead.get_last_move_time()
        self.toolhead.reset_print_time()
        self.mcu_probe.home_finalize(move_end_print_time)
        # Wait for probe to trigger
        try:
            self.mcu_probe.home_wait()
        except self.mcu_probe.error as e:
            raise homing.EndstopError("Failed to probe: %s" % (str(e),))
        # Update with new position
        self.toolhead.reset_position() 
        return self.toolhead.get_position()[2] + self.offset[2]
         
    def probe_points(self, points):
        results = ProbeResults()
        results.points = points
        results.count = len(points)
        for p in points:
            # Move toolhead above probe position
            pos = [p[0], p[1], (self.z_distance * 0.5) - self.offset[2], self.toolhead.get_position()[3]]
            self.toolhead.move(pos, self.speed * 5.0)
            # Move down to probe
            height = self.probe_height()
            results.heights.append(height)
            # Move back up
            self.toolhead.move(pos, self.speed * 5.0)

        results.std = std(results.heights)
        return results

def add_printer_objects(printer, config):
    if config.has_section('probe'):
        printer.add_object('probe', PrinterProbe(
            printer, config.getsection('probe')))
