# Code for handling the kinematics of linear delta robots
#
# Copyright (C) 2016,2017  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import math, logging
import stepper, homing
import numpy as np
from lmfit import minimize, Parameters

StepList = (0, 1, 2)

# Slow moves once the ratio of tower to XY movement exceeds SLOW_RATIO
SLOW_RATIO = 3.

class DeltaKinematics:
    def __init__(self, toolhead, printer, config):      
        self.steppers = [stepper.PrinterHomingStepper(
            printer, config.getsection('stepper_' + n), n)
                         for n in ['a', 'b', 'c']]
        self.need_motor_enable = self.need_home = True
        self.delta_probe_radius = config.getfloat('delta_probe_radius', 0.)
        self.endstop_corrections = [config.getsection('stepper_' + n).getfloat('correction_endstop', 0.)
                                    for n in ['a', 'b', 'c']]
        self.angle_corrections = [config.getsection('stepper_' + n).getfloat('correction_angle', 0.)
                                  for n in ['a', 'b', 'c']]
        self.pending_corrections = None
        self.radius = config.getfloat('delta_radius', above=0.)
        self.arm_length = config.getfloat('delta_arm_length', above=self.radius)
        self.arm_length2 = self.arm_length**2
        self.limit_xy2 = -1.
        self.max_z = min([s.position_endstop for s in self.steppers])

        tower_height_at_zeros = math.sqrt(self.arm_length2 - self.radius ** 2)
        self.limit_z = self.max_z - (self.arm_length - tower_height_at_zeros)
        logging.info(
            "Delta max build height %.2fmm (radius tapered above %.2fmm)" % (
                self.max_z, self.limit_z))

        # Setup stepper max halt velocity
        self.max_velocity, self.max_accel = toolhead.get_max_velocity()
        self.max_z_velocity = config.getfloat(
            'max_z_velocity', self.max_velocity,
            above=0., maxval=self.max_velocity)
        max_xy_halt_velocity = toolhead.get_max_axis_halt(self.max_accel)
        for s in self.steppers:
            s.set_max_jerk(max_xy_halt_velocity, self.max_accel)

        self.angles = [config.getsection('stepper_a').getfloat('angle', 210.),
                       config.getsection('stepper_b').getfloat('angle', 330.),
                       config.getsection('stepper_c').getfloat('angle', 90.)]

        self.towers = []
        self.slow_xy2 = 0.0
        self.very_slow_xy2 = 0.0
        self.max_xy2 = 0.0

        self.calculate_params()
        self.set_position([0., 0., 0.])

    def calculate_params(self):
        logging.info("Updating delta kinematics")

        # Apply endstop corrections to the steppers
        for stepper, corr in zip(self.steppers, self.endstop_corrections):
            stepper.position_endstop += corr
            logging.info("Adj. stepper endstop %.2f" % (stepper.position_endstop))

        tower_height_at_zeros = math.sqrt(self.arm_length2 - self.radius ** 2)

        self.max_z = min([s.position_endstop for s in self.steppers])
        self.limit_z = self.max_z - (self.arm_length - tower_height_at_zeros)

        # Determine tower locations in cartesian space
        self.towers = [(math.cos(math.radians(angle + correction)) * self.radius,
                        math.sin(math.radians(angle + correction)) * self.radius)
                        for angle, correction in zip(self.angles, self.angle_corrections)]

        # Find the point where an XY move could result in excessive
        # tower movement
        half_min_step_dist = min([s.step_dist for s in self.steppers]) * .5

        def ratio_to_dist(ratio):
            return (ratio * math.sqrt(self.arm_length2 / (ratio ** 2 + 1.)
                                      - half_min_step_dist ** 2)
                    + half_min_step_dist)

        self.slow_xy2 = (ratio_to_dist(SLOW_RATIO) - self.radius) ** 2
        self.very_slow_xy2 = (ratio_to_dist(2. * SLOW_RATIO) - self.radius) ** 2
        self.max_xy2 = min(self.radius, self.arm_length - self.radius,
                           ratio_to_dist(4. * SLOW_RATIO) - self.radius) ** 2

    def _cartesian_to_actuator(self, coord):
        return [math.sqrt(self.arm_length2
                          - (self.towers[i][0] - coord[0])**2
                          - (self.towers[i][1] - coord[1])**2) + coord[2]
                for i in StepList]

    def _actuator_to_cartesian(self, pos):
        # Based on code from Smoothieware
        tower1 = list(self.towers[0]) + [pos[0]]
        tower2 = list(self.towers[1]) + [pos[1]]
        tower3 = list(self.towers[2]) + [pos[2]]

        s12 = matrix_sub(tower1, tower2)
        s23 = matrix_sub(tower2, tower3)
        s13 = matrix_sub(tower1, tower3)

        normal = matrix_cross(s12, s23)

        magsq_s12 = matrix_magsq(s12)
        magsq_s23 = matrix_magsq(s23)
        magsq_s13 = matrix_magsq(s13)

        inv_nmag_sq = 1.0 / matrix_magsq(normal)
        q = 0.5 * inv_nmag_sq

        a = q * magsq_s23 * matrix_dot(s12, s13)
        b = -q * magsq_s13 * matrix_dot(s12, s23) # negate because we use s12 instead of s21
        c = q * magsq_s12 * matrix_dot(s13, s23)

        circumcenter = [tower1[0] * a + tower2[0] * b + tower3[0] * c,
                        tower1[1] * a + tower2[1] * b + tower3[1] * c,
                        tower1[2] * a + tower2[2] * b + tower3[2] * c]

        r_sq = 0.5 * q * magsq_s12 * magsq_s23 * magsq_s13
        dist = math.sqrt(inv_nmag_sq * (self.arm_length2 - r_sq))

        return matrix_sub(circumcenter, matrix_mul(normal, dist))

    def get_position(self):
        spos = [s.mcu_stepper.get_commanded_position() for s in self.steppers]
        return self._actuator_to_cartesian(spos)

    def set_position(self, newpos):
        pos = self._cartesian_to_actuator(newpos)
        for i in StepList:
            self.steppers[i].mcu_stepper.set_position(pos[i])
        self.limit_xy2 = -1.

    def home(self, homing_state):
        # Check for pending delta corrections
        if self.pending_corrections:
            self.angle_corrections = self.pending_corrections['angle_corrections']
            self.endstop_corrections = self.pending_corrections['endstop_corrections']
            self.radius = self.pending_corrections['radius']
            self.pending_corrections = None
            self.calculate_params()

        # All axes are homed simultaneously
        homing_state.set_axes([0, 1, 2])
        s = self.steppers[0] # Assume homing speed same for all steppers
        self.need_home = False
        # Initial homing
        homing_speed = s.get_homing_speed()
        homepos = [0., 0., self.max_z, None]
        coord = list(homepos)
        coord[2] = -1.5 * math.sqrt(self.arm_length2-self.max_xy2)
        homing_state.home(list(coord), homepos, self.steppers, homing_speed)
        # Retract
        coord[2] = homepos[2] - s.homing_retract_dist
        homing_state.retract(list(coord), homing_speed)
        # Home again
        coord[2] -= s.homing_retract_dist
        homing_state.home(list(coord), homepos, self.steppers
                          , homing_speed/2.0, second_home=True)

        # Set final homed position
        spos = self._cartesian_to_actuator(homepos)
        spos = [spos[i] + self.steppers[i].position_endstop - self.max_z
                + self.steppers[i].get_homed_offset()
                for i in StepList]
        homing_state.set_homed_position(self._actuator_to_cartesian(spos))

    def query_endstops(self, print_time):
        return homing.query_endstops(print_time, self.steppers)

    def get_z_steppers(self):
        return self.steppers

    def motor_off(self, print_time):
        self.limit_xy2 = -1.
        for stepper in self.steppers:
            stepper.motor_enable(print_time, 0)
        self.need_motor_enable = self.need_home = True

    def _check_motor_enable(self, print_time):
        for i in StepList:
            self.steppers[i].motor_enable(print_time, 1)
        self.need_motor_enable = False

    def check_move(self, move):
        end_pos = move.end_pos
        xy2 = end_pos[0]**2 + end_pos[1]**2
        if xy2 <= self.limit_xy2 and not move.axes_d[2]:
            # Normal XY move
            return
        if self.need_home:
            raise homing.EndstopMoveError(end_pos, "Must home first")
        limit_xy2 = self.max_xy2
        if end_pos[2] > self.limit_z:
            limit_xy2 = min(limit_xy2, (self.max_z - end_pos[2])**2)
        if xy2 > limit_xy2 or end_pos[2] < 0. or end_pos[2] > self.max_z:
            raise homing.EndstopMoveError(end_pos)
        if move.axes_d[2]:
            move.limit_speed(self.max_z_velocity, move.accel)
            limit_xy2 = -1.
        # Limit the speed/accel of this move if is is at the extreme
        # end of the build envelope
        extreme_xy2 = max(xy2, move.start_pos[0]**2 + move.start_pos[1]**2)
        if extreme_xy2 > self.slow_xy2:
            r = 0.5
            if extreme_xy2 > self.very_slow_xy2:
                r = 0.25
            max_velocity = self.max_velocity
            if move.axes_d[2]:
                max_velocity = self.max_z_velocity
            move.limit_speed(max_velocity * r, self.max_accel * r)
            limit_xy2 = -1.
        self.limit_xy2 = min(limit_xy2, self.slow_xy2)

    def move(self, print_time, move):
        if self.need_motor_enable:
            self._check_motor_enable(print_time)
        axes_d = move.axes_d
        move_d = move.move_d
        movexy_r = 1.
        movez_r = 0.
        inv_movexy_d = 1. / move_d
        if not axes_d[0] and not axes_d[1]:
            # Z only move
            movez_r = axes_d[2] * inv_movexy_d
            movexy_r = inv_movexy_d = 0.
        elif axes_d[2]:
            # XY+Z move
            movexy_d = math.sqrt(axes_d[0]**2 + axes_d[1]**2)
            movexy_r = movexy_d * inv_movexy_d
            movez_r = axes_d[2] * inv_movexy_d
            inv_movexy_d = 1. / movexy_d

        origx, origy, origz = move.start_pos[:3]

        accel = move.accel
        cruise_v = move.cruise_v
        accel_d = move.accel_r * move_d
        cruise_d = move.cruise_r * move_d
        decel_d = move.decel_r * move_d

        for i in StepList:
            # Calculate a virtual tower along the line of movement at
            # the point closest to this stepper's tower.
            towerx_d = self.towers[i][0] - origx
            towery_d = self.towers[i][1] - origy
            vt_startxy_d = (towerx_d*axes_d[0] + towery_d*axes_d[1])*inv_movexy_d
            tangentxy_d2 = towerx_d**2 + towery_d**2 - vt_startxy_d**2
            vt_arm_d = math.sqrt(self.arm_length2 - tangentxy_d2)
            vt_startz = origz

            # Generate steps
            mcu_stepper = self.steppers[i].mcu_stepper
            move_time = print_time
            if accel_d:
                mcu_stepper.step_delta(
                    move_time, accel_d, move.start_v, accel,
                    vt_startz, vt_startxy_d, vt_arm_d, movez_r)
                vt_startz += accel_d * movez_r
                vt_startxy_d -= accel_d * movexy_r
                move_time += move.accel_t
            if cruise_d:
                mcu_stepper.step_delta(
                    move_time, cruise_d, cruise_v, 0.,
                    vt_startz, vt_startxy_d, vt_arm_d, movez_r)
                vt_startz += cruise_d * movez_r
                vt_startxy_d -= cruise_d * movexy_r
                move_time += move.cruise_t
            if decel_d:
                mcu_stepper.step_delta(
                    move_time, decel_d, cruise_v, -accel,
                    vt_startz, vt_startxy_d, vt_arm_d, movez_r)
                    
    def suggest_probe_points(self, calibration_point_count, probe_offset):
        """
        Suggests a number of probe points taking the probe's offset into account.
        Used to perform calibration tasks.
        :param calibration_point_count: The number of probe points to request
        :param probe_offset: The offset of the probe from the toolhead
        :return: A list of probe points (x,y)
        """

        probe_dist_from_center = math.hypot(probe_offset[0], probe_offset[1])

        # Always probe the center
        calibration_points = [[0, 0]]
        # Generate a list of probe points [x,y] in a circle defined by delta_probe_radius
        for i in range(calibration_point_count):
            calibration_points.append([ math.sin(2*math.pi/calibration_point_count*i) * max(0, self.delta_probe_radius - probe_dist_from_center),
                                        math.cos(2*math.pi/calibration_point_count*i) * max(0, self.delta_probe_radius - probe_dist_from_center)])
        return calibration_points
        
    def calculate_calibration_params(self, probe_results):
        """
        Calculates an optimized set of delta parameters.
        :param probe_results: The probe results (see probe.ProbeResults)
        :return: Dictionary containing the delta parameters or None if the optimization fails
        """

        # convert to actuator positions
        actuator_pos = [self._cartesian_to_actuator([p[0], p[1], h]) for p, h in zip(probe_results.points, probe_results.heights)]

        def residual(params, x, data, actuator_to_cartesian, eps_data):
            angle_corrections = [params['angle_a'], params['angle_b'], params['angle_c']]
            endstop_corrections = [params['endstop_a'], params['endstop_b'], params['endstop_c']]
            radius = params['radius']

            # The model returns the z value for the given design parameters using the actuator positions of the current
            # probe positions
            model = [actuator_to_cartesian(xx, self.angles, self.arm_length2, radius, angle_corrections, endstop_corrections)[2] for xx in x]

            # Return the error i.e. the new distance between the toolhead and the bed
            return (np.array(data) - np.array(model))/eps_data
        
        params = Parameters()
        params.add('angle_a', value=0.0, min=-3.0, max=3.0)
        params.add('angle_b', value=0.0, min=-3.0, max=3.0)
        params.add('angle_c', value=0.0, min=-3.0, max=3.0, vary=False)
        params.add('endstop_a', value=0.0)
        params.add('endstop_b', value=0.0)
        params.add('endstop_c', value=0.0)
        params.add('radius', value=self.radius)

        x = np.array(actuator_pos)
        data = [0.] * probe_results.count  # The expected result is zero in all points
        eps_data = 1.0

        # Perform the optimization using 6 parameters i.e. we need a least 6 test points
        out = minimize(residual, params, args=(x, data, actuator_to_cartesian, eps_data))

        # The optimization may fail under certain circumstances
        if not out.success:
            return None

        # Extract results
        angle_corrections = [out.params['angle_a'].value, out.params['angle_b'].value, out.params['angle_c'].value]
        endstop_corrections = [out.params['endstop_a'].value, out.params['endstop_b'].value, out.params['endstop_c'].value]
        radius = out.params['radius'].value

        # Normalize endstops (currently not needed)
        #endstop_corrections = np.array(endstop_corrections) - min(endstop_corrections)

        # Calculate the std after calibration
        corr = [actuator_to_cartesian(pos, self.angles, self.arm_length2, radius, angle_corrections, endstop_corrections) for pos in actuator_pos]
        z_corr = np.array([p[2] for p in corr])

        return {'angle_corrections': angle_corrections,
                'endstop_corrections': endstop_corrections,
                'radius': radius,
                'std': np.std(z_corr)}

    def set_pending_corrections(self, corrections):
        """
        Sets the correction parameters to be applied on the next call to home.
        :param corrections: Parameters obtained by calling calculate_calibration_params
        """
        self.pending_corrections = corrections
        self.need_home = True

######################################################################
# Calibration helper functions
######################################################################

def actuator_to_cartesian(pos, angles, arm_length2, radius, angle_corrections, endstop_corrections):
    """
    Used to modelize a Delta printer using the parameters below.
    :param pos: The position in actuator coords
    :param angles: The tower angles
    :param arm_length2: The arm length squared
    :param radius: The delta radius
    :param angle_corrections: The tower angle corrections
    :param endstop_corrections: The endstop corrections
    :return: The cartesian coords corresponding to the actuator positions
    """

    towers = [(math.cos(math.radians(angle + correction)) * radius,
               math.sin(math.radians(angle + correction)) * radius)
              for angle, correction in zip(angles, angle_corrections)]

    # Based on code from Smoothieware
    tower1 = list(towers[0]) + [pos[0] + endstop_corrections[0]]
    tower2 = list(towers[1]) + [pos[1] + endstop_corrections[1]]
    tower3 = list(towers[2]) + [pos[2] + endstop_corrections[2]]

    s12 = matrix_sub(tower1, tower2)
    s23 = matrix_sub(tower2, tower3)
    s13 = matrix_sub(tower1, tower3)

    normal = matrix_cross(s12, s23)

    magsq_s12 = matrix_magsq(s12)
    magsq_s23 = matrix_magsq(s23)
    magsq_s13 = matrix_magsq(s13)

    inv_nmag_sq = 1.0 / matrix_magsq(normal)
    q = 0.5 * inv_nmag_sq

    a = q * magsq_s23 * matrix_dot(s12, s13)
    b = -q * magsq_s13 * matrix_dot(s12, s23)  # negate because we use s12 instead of s21
    c = q * magsq_s12 * matrix_dot(s13, s23)

    circumcenter = [tower1[0] * a + tower2[0] * b + tower3[0] * c,
                    tower1[1] * a + tower2[1] * b + tower3[1] * c,
                    tower1[2] * a + tower2[2] * b + tower3[2] * c]

    r_sq = 0.5 * q * magsq_s12 * magsq_s23 * magsq_s13
    dist = math.sqrt(inv_nmag_sq * (arm_length2 - r_sq))

    return matrix_sub(circumcenter, matrix_mul(normal, dist))

######################################################################
# Matrix helper functions for 3x1 matrices
######################################################################

def matrix_cross(m1, m2):
    return [m1[1] * m2[2] - m1[2] * m2[1],
            m1[2] * m2[0] - m1[0] * m2[2],
            m1[0] * m2[1] - m1[1] * m2[0]]

def matrix_dot(m1, m2):
    return m1[0] * m2[0] + m1[1] * m2[1] + m1[2] * m2[2]

def matrix_magsq(m1):
    return m1[0]**2 + m1[1]**2 + m1[2]**2

def matrix_sub(m1, m2):
    return [m1[0] - m2[0], m1[1] - m2[1], m1[2] - m2[2]]

def matrix_mul(m1, s):
    return [m1[0]*s, m1[1]*s, m1[2]*s]
