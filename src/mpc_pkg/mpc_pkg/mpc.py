import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import tensorflow as tf
from tf.transformations import euler_from_quaternion

class NMPCController(Node):
    def __init__(self):
        super().__init__('nmpc_controller')

        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.turtlebot_state = None
        self.waypoints = []  # Fill this with your set of waypoints
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
        quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
        _, _, theta = euler_from_quaternion(quaternion)

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
        position_error = np.sum((state[:2] - reference_state[:2])**2)
        angle_error = (state[2] - reference_state[2])**2
        control_effort = np.sum(control_input**2)

        if prev_control_input is not None:
            control_rate_of_change = np.sum((control_input - prev_control_input)**2)
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

        return cost

    def nmpc_optimizer(self, turtlebot_state, reference_state):
        control_input = tf.Variable(np.random.uniform(-0.1, 0.1, size=(2,)), dtype=tf.float64)

        def optimization_step():
            with tf.GradientTape() as tape:
                state_prediction = self.kinematic_model(turtlebot_state, control_input)
                cost = self.cost_function(state_prediction, control_input, reference_state)
            grads = tape.gradient(cost, [control_input])
            self.optimizer.apply_gradients(zip(grads, [control_input]))

        for _ in range(100):  # Adjust the number of iterations as needed
            optimization_step()

        return control_input.numpy()

def main(args=None):
    rclpy.init(args=args)

    nmpc_controller = NMPCController()

    rclpy.spin(nmpc_controller)

    nmpc_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

