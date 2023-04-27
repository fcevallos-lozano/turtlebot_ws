import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
import numpy as np
import tensorflow as tf
from pyquaternion import Quaternion
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class NMPCController(Node):
    def __init__(self):
        super().__init__('nmpc_controller')

        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.turtlebot_state = None
        num_points = 40
        x_values = np.linspace(0, 20, num_points)
        y_values = 10 * np.sin(np.pi * x_values / 5)

        waypoints_sine_wave = [Point(x=x_values[i], y=y_values[i], z=0.0) for i in range(num_points)]

        self.waypoints = waypoints_sine_wave
        self.current_waypoint_index = 0

        # Initialize the RPROP optimizer
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)

    def odom_callback(self, msg: Odometry):
        self.turtlebot_state = msg

        if self.turtlebot_state is not None and self.current_waypoint_index < len(self.waypoints):
            control_signal = self.calculate_control_signal()
            twist_msg = Twist()
            twist_msg.linear.x = control_signal[0]
            twist_msg.angular.z = control_signal[1]
            self.cmd_vel_publisher.publish(twist_msg)
        elif self.current_waypoint_index >= len(self.waypoints):
            self.get_logger().info("Finished following waypoints.")
            rclpy.shutdown()

    def calculate_control_signal(self):
        # Get the current TurtleBot3 state
        x, y, theta = self.get_turtlebot_position()

        # Get the reference state (next waypoint)
        reference_state = self.get_reference_state()

        # Implement your NMPC logic here
        control_signal = self.nmpc_optimizer(np.array([x, y, theta]), reference_state)

        return control_signal

    def get_turtlebot_position(self):
        position = self.turtlebot_state.pose.pose.position
        orientation = self.turtlebot_state.pose.pose.orientation
        q = Quaternion(orientation.w, orientation.x, orientation.y, orientation.z)
        theta = q.angle # already in radians

        return position.x, position.y, theta

    def get_reference_state(self):
        if self.current_waypoint_index < len(self.waypoints):
            return self.waypoints[self.current_waypoint_index]
        else:
            return None

    def kinematic_model(self, state, control_input):
        x_dot = control_input[0] * np.cos(state[2])
        y_dot = control_input[0] * np.sin(state[2])
        theta_dot = control_input[1]

        return np.array([x_dot, y_dot, theta_dot])

    def cost_function(self, state, control_input, reference_state, prev_control_input=None):
        state_data = tf.convert_to_tensor([state[0], state[1], state[2]], dtype=tf.float64)  # state (x, y, theta)
        ref_data = tf.convert_to_tensor([reference_state[0], reference_state[1], reference_state[2]], dtype=tf.float32)  # reference state (x, y, theta)
        position_error = tf.reduce_sum((state_data[:2] - ref_data[:2]) ** 2)
        angle_error = (state_data[2] - ref_data[2]) ** 2
        control_effort = tf.reduce_sum(control_input ** 2)

        if prev_control_input is not None:
            control_rate_of_change = tf.reduce_sum((control_input - prev_control_input) ** 2)
        else:
            control_rate_of_change = 0

        weight_position_error = 1.0
        weight_angle_error = 1.0
        weight_control_effort = 0.01
        weight_control_rate_of_change = 0.1

        cost = (weight_position_error * position_error +
            weight_angle_error * angle_error +
            weight_control_effort * control_effort +
            weight_control_rate_of_change * control_rate_of_change)

        return tf.convert_to_tensor(cost, dtype=tf.float32)


    def nmpc_optimizer(self, turtlebot_state, reference_state):
        turtlebot_state_np = tf.convert_to_tensor(turtlebot_state, dtype=tf.float32)
        reference_state_np = np.array([reference_state.x, reference_state.y, reference_state.z], dtype=np.float32)
        reference_state_np = tf.convert_to_tensor(reference_state_np, dtype=tf.float32)

        control_input = tf.Variable(initial_value=self.control_input_initial_guess, dtype=tf.float32)
        optimization_step = tf.function(self.optimization_step)

        for _ in range(self.n_iterations):
            optimization_step(turtlebot_state_np, control_input, reference_state_np)

        return control_input.numpy()


def main(args=None):
    rclpy.init(args=args)

    nmpc_controller = NMPCController()

    rclpy.spin(nmpc_controller)

    nmpc_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


