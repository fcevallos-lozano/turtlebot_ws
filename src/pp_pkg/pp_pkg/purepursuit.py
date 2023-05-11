import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from math import atan2, sqrt
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import time
import rclpy.qos as qos

fig = plt.figure()

class PurePursuit(Node):

    def __init__(self):
        super().__init__('pure_pursuit')
        self.publisher_ = self.create_publisher(Twist, '/turtlebot4/cmd_vel', 10)
        self.subscription_odom = self.create_subscription(Odometry, '/turtlebot4/odom', self.odom_callback, 
                                    qos.QoSProfile(depth=10, reliability=qos.ReliabilityPolicy.BEST_EFFORT))        # Waypoints initialization
        # Point 21 is closest from Point 0
        # Point 1 is furthes from Point 0
        #self.waypoints =  [Point(x=0.0,y=0.0,z=0.0), Point(x=0.571194595265405, y=-0.4277145118491421, z=0.0), Point(x=1.1417537280142898, y=-0.8531042347260006,z=0.0), Point(x=1.7098876452457967, y=-1.2696346390611464,z=0.0), Point(x=2.2705328851607995, y=-1.6588899151216996,z=0.0), Point(x=2.8121159420106827, y=-1.9791445882187304,z=0.0), Point(x=3.314589274316711, y=-2.159795566252656,z=0.0), Point(x=3.7538316863009027, y=-2.1224619985315876,z=0.0), Point(x=4.112485112342358, y=-1.8323249172947023,z=0.0), Point(x=4.383456805594431, y=-1.3292669972090994,z=0.0), Point(x=4.557386228943757, y=-0.6928302521681386,z=0.0), Point(x=4.617455513800438, y=0.00274597627737883,z=0.0), Point(x=4.55408382321606, y=0.6984486966257434,z=0.0), Point(x=4.376054025556597, y=1.3330664239172116,z=0.0), Point(x=4.096280073621794, y=1.827159263675668,z=0.0), Point(x=3.719737492364894, y=2.097949296701878,z=0.0), Point(x=3.25277928312066, y=2.108933125822431,z=0.0), Point(x=2.7154386886417314, y=1.9004760368018616,z=0.0), Point(x=2.1347012144725985, y=1.552342808106984,z=0.0), Point(x=1.5324590525923942, y=1.134035376721349,z=0.0), Point(x=0.9214084611203568, y=0.6867933269918683,z=0.0), Point(x=0.30732366808208345, y=0.22955002391894264,z=0.0), Point(x=-0.3075127599907512, y=-0.2301742560363831,z=0.0), Point(x=-0.9218413719658775, y=-0.6882173194028102,z=0.0), Point(x=-1.5334674079795052, y=-1.1373288016589413,z=0.0), Point(x=-2.1365993767877467, y=-1.5584414896876835,z=0.0), Point(x=-2.7180981380280307, y=-1.9086314914221845,z=0.0), Point(x=-3.2552809639439704, y=-2.1153141204181285,z=0.0), Point(x=-3.721102967810494, y=-2.0979137913841046,z=0.0), Point(x=-4.096907306768644, y=-1.8206318841755131,z=0.0), Point(x=-4.377088212533404, y=-1.324440752295139,z=0.0), Point(x=-4.555249804461285, y=-0.6910016662308593,z=0.0), Point(x=-4.617336323713965, y=0.003734984720118972,z=0.0), Point(x=-4.555948690867849, y=0.7001491248072772,z=0.0), Point(x=-4.382109193278264, y=1.3376838311365633,z=0.0), Point(x=-4.111620918085742, y=1.8386823176628544,z=0.0), Point(x=-3.7524648889185794, y=2.1224985058331005,z=0.0), Point(x=-3.3123191098095615, y=2.153588702898333,z=0.0), Point(x=-2.80975246649598, y=1.9712114570096653,z=0.0), Point(x=-2.268856462266256, y=1.652958931009528,z=0.0), Point(x=-1.709001159778989, y=1.2664395490411673,z=0.0), Point(x=-1.1413833971013372, y=0.8517589252820573,z=0.0), Point(x=-0.5710732645795573, y=0.4272721367616211,z=0.0)]
        num_points = 50
        x_values = np.linspace(0, 2.5, num_points)
        y_values = 0.625 * np.sin(np.pi * x_values/1.25)

        waypoints_sine_wave = [Point(x=x_values[i], y=y_values[i], z=0.0) for i in range(num_points)]

        self.waypoints = waypoints_sine_wave
       
        self.current_state = np.zeros(3)  # Current state of the robot
        self.lookahead_distance = .15
        self.current_pose = Odometry()
        self.nearest_index = 0
        self.last_visited_index = None
        self.point_history = [] # history of visited points

        # generate reference path
        w_x = [wp.x for wp in self.waypoints]
        w_y = [wp.y for wp in self.waypoints]

        # x_dup = np.unique(waypoints_x, return_counts=True, return_inverse=True)
        # y_dup = np.unique(waypoints_y, return_counts=True, return_inverse=True)
        # print(x_dup)
        # print(y_dup)
        path_x, path_y = self.__interpolate(X=w_x, Y=w_y)
        self.true_path = (path_x, path_y)

        # plot reference path and waypoints
        print(self.waypoints[0])
        print(self.waypoints[1])

        plt.plot(path_x, path_y)
        plt.plot(w_x, w_y, 'ro')
        plt.show()
        self.time_step = 7.5
        self.distance_errors = []
        self.distance_error_time_stamps = []
        self.start_time = time.time()
        self.xstuff = []
        self.ystuff = []
        # self.current_state = None


    def calculate_distance_error(self):
        current_position = np.array([self.current_state[0], self.current_state[1]])
        path_points = np.vstack((self.true_path[0], self.true_path[1])).T
        distances = np.linalg.norm(path_points - current_position, axis=1)
        closest_point_idx = np.argmin(distances)
        #closest_point = path_points[closest_point_idx]
        distance_error = distances[closest_point_idx]
        current_time = time.time()
        self.relative_time = current_time - self.start_time
        #self.get_logger().info(f"Time: {current_time:.2f}, Distance error: {distance_error:.4f}")
        self.distance_errors.append(distance_error)
        self.distance_error_time_stamps.append(self.relative_time)

        

    def __interpolate(self, X, Y):
        tck, u = splprep([X, Y], s=0)
        u_new = np.linspace(u.min(), u.max(), 1000)
        path_x, path_y = splev(u_new, tck)

        return path_x, path_y
    

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        self.xstuff.append(self.current_pose.position.x)
        self.ystuff.append(self.current_pose.position.y)

        self.orientation = msg.pose.pose.orientation 
        #self.current_state = [self.current_pose.position.x, self.current_pose.position.y, self.current_pose.orientation.z]
        #self.get_logger().info(f"pose x: {self.current_pose.position.x}")
        #self.get_logger().info(f"pose y: {self.current_pose.position.y}")

    def pure_pursuit(self):
        #if len(self.waypoints) > 0 and (self.current_pose is not None):
        # find the nearest waypoint
            #min_distance = float('inf')
           # nearest_index = 0
        w_x = [wp.x for wp in self.waypoints]
        w_y = [wp.y for wp in self.waypoints]
        self.passed_values = []
        final_waypoint = self.waypoints[-1]
        distance_to_final = sqrt((final_waypoint.x - self.current_pose.position.x) ** 2 + (final_waypoint.y - self.current_pose.position.y) ** 2)
        #self.get_logger().info(f"final dist: {distance_to_final}")
        self.current_state[0] = self.current_pose.position.x  # Update the current X position
        self.current_state[1] = self.current_pose.position.y  # Update the current Y position
        self.current_state[2] = 2 * np.arctan2(self.orientation.z, self.orientation.w)  # Convert quaternion to yaw and update the current yaw
        self.timer = self.create_timer(self.time_step, self.calculate_distance_error)

                # stop condition (tune the threshold value)
                #stop_threshold = 0.01
        if distance_to_final <= 0.125: #< stop_threshold: tune
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.publisher_.publish(cmd_vel)
            plt.figure()
            plt.plot(self.true_path[0], self.true_path[1])
            plt.plot(self.xstuff, self.ystuff, 'r--')
            plt.show()
            plt.figure()
            plt.plot(self.distance_error_time_stamps, self.distance_errors)  # Plot the waypoints
            plt.grid()
            plt.show()
           # print(self.distance_error_time_stamps)  # Show the plot
            exit(0)
        if self.last_visited_index is None:
            dx = [self.current_pose.position.x - icx for icx in w_x] #list comprehension implied by brackets
            dy = [self.current_pose.position.y - icy for icy in w_y] 
            d = np.hypot(dx, dy) #list of hypotenuses
            ind = np.argmin(d) #finds the smallest in list of d
            self.last_visited_index = ind
        else:
            ind = self.last_visited_index
            distance_this_index = self.__distance(self.current_pose.position.x,
                                                  self.current_pose.position.y, w_x[ind], w_y[ind])
            while True: #keep looping until you find a waypoint that has a smaller distance than the current waypoint
                distance_next_index = self.__distance(self.current_pose.position.x,
                                                  self.current_pose.position.y, w_x[ind+1], w_y[ind+1])
                if distance_this_index < distance_next_index and ind not in self.passed_values:
                    self.passed_values.append(ind)
                    break
                ind = ind + 1 if (ind + 1) < len(w_x) else ind
                distance_this_index = distance_next_index
            self.last_visited_index = ind

        Lf = self.lookahead_distance  # update look ahead distance

        # search look ahead target point index
        while Lf > self.__distance(self.current_pose.position.x, self.current_pose.position.y, w_x[ind], w_y[ind]):
            if (ind + 1) >= len(w_x) and ind not in self.passed_values:
                break  # not exceed goal
            ind += 1
        

        if ind < len(w_x):
            tx = w_x[ind]
            ty = w_y[ind]
        else:  # toward goal
            tx = w_x[-1]
            ty = w_y[-1]
            ind = len(w_x) - 1

        alpha = atan2(ty - self.current_pose.position.y, tx - self.current_pose.position.x) - self.current_pose.orientation.z
        #self.get_logger().info(f"angle: {alpha}")
        #self.get_logger().info(f"angle2: {self.current_pose}")

        cmd_vel = Twist()
        cmd_vel.linear.x = 0.15  # tune this value
        cmd_vel.angular.z = 1.25 * alpha  # tune this value

            # check if the robot is close to the final waypoint
        final_waypoint = self.waypoints[-1]
        distance_to_final = sqrt((final_waypoint.x - self.current_pose.position.x) ** 2 + (final_waypoint.y - self.current_pose.position.y) ** 2)
        #self.get_logger().info(f"final dist: {distance_to_final}")


        self.publisher_.publish(cmd_vel)
        
                        

    def __distance(self, x1, y1, x2, y2):
        """ compute Euclidean Distance """
        d = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        return d


def main(args=None):
    rclpy.init(args=args)
    purepursuit = PurePursuit()

    try:
        while rclpy.ok():
            rclpy.spin_once(purepursuit, timeout_sec=0.1)
            purepursuit.pure_pursuit()
    except KeyboardInterrupt:
        pass
    purepursuit.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


# import rclpy  # ROS2 Python client library
# from rclpy.node import Node  # Base class for ROS2 nodes
# from geometry_msgs.msg import Twist, Point  # Message types for velocity and position
# from nav_msgs.msg import Odometry  # Message type for odometry data
# import numpy as np  # NumPy library for numerical operations
# from scipy.optimize import minimize  # Optimization function from SciPy
# import matplotlib.pyplot as plt  # Matplotlib library for plotting
# from scipy.interpolate import splprep, splev  # Functions for spline interpolation
# import rclpy.qos as qos
# import time

# #fig = plt.figure()  # Initialize a matplotlib figure

# class NMPCController(Node):
#     def __init__(self):
#         super().__init__('nmpc_controller')  # Initialize the Node class
#         self.publisher_ = self.create_publisher(Twist, '/turtlebot4/cmd_vel', 10)
#         self.subscription_odom = self.create_subscription(Odometry, '/turtlebot4/odom', self.odometry_callback, 
#                                     qos.QoSProfile(depth=10, reliability=qos.ReliabilityPolicy.BEST_EFFORT))
#         self.subscription_odom
#         #self.subscription  # prevent unused variable warning
#         self.position_error_weight = 1.0  # Weight for position error in the cost function
#         self.control_input_change_weight = 3.0  # Weight for control input change in the cost function
#         self.T = 0.06  # Time step for the NMPC
#         self.Np = 3  # Prediction horizon
#         self.Nc = 1  # Control horizon
#         self.x_ref = np.zeros(3)  # Reference state (waypoint)
#         self.current_state = np.zeros(3)  # Current state of the robot

#         self.current_waypoint_idx = 0  # Index of the current waypoint
#         num_points = 50  # Number of points in the sine wave path
#         x_values = np.linspace(0, 2.5, num_points)  # X values for the sine wave
#         y_values = 0.625 * np.sin(np.pi * x_values / 1.25)  # Y values for the sine wave
#         waypoints_sine_wave = [Point(x=x_values[i], y=y_values[i], z=0.0) for i in range(num_points)]  # Create waypoints as Point objects
#         self.waypoints = waypoints_sine_wave  # Store the waypoints
#         w_x = [wp.x for wp in self.waypoints]  # Extract X values of waypoints
#         w_y = [wp.y for wp in self.waypoints]  # Extract Y values of waypoints

#         path_x, path_y = self.__interpolate(X=w_x, Y=w_y)  # Interpolate the waypoints to create a smooth path
#         self.true_path = (path_x, path_y)  # Store the true path

#         plt.plot(path_x, path_y)  # Plot the interpolated path
#         plt.plot(w_x, w_y, 'ro')  # Plot the waypoints
#         plt.show()  # Show the plot
#         self.time_step = 4.0
        
#         self.distance_errors = []
#         self.distance_error_time_stamps = []
#         self.start_time = time.time()
#         self.xstuff = []
#         self.ystuff = []


#     def calculate_distance_error(self):
#         current_position = np.array([self.current_state[0], self.current_state[1]])
#         path_points = np.vstack((self.true_path[0], self.true_path[1])).T
#         distances = np.linalg.norm(path_points - current_position, axis=1)
#         closest_point_idx = np.argmin(distances)
#         #closest_point = path_points[closest_point_idx]
#         distance_error = distances[closest_point_idx]
#         current_time = time.time()
#         self.relative_time = current_time - self.start_time
#         self.get_logger().info(f"Time: {current_time:.2f}, Distance error: {distance_error:.4f}")
#         self.distance_errors.append(distance_error)
#         self.distance_error_time_stamps.append(self.relative_time)


#     def __interpolate(self, X, Y):
#         tck, u = splprep([X, Y], s=0)  # Fit a B-spline to the data
#         u_new = np.linspace(u.min(), u.max(), 1000)  # Create new parameter values for the spline
#         path_x, path_y = splev(u_new, tck)  # Evaluate the spline at the new parameter values

#         return path_x, path_y  # Return the interpolated path

#     def odometry_callback(self, msg):
#         position = msg.pose.pose.position  # Extract the position from the
#         # odometry message
#         orientation = msg.pose.pose.orientation  # Extract the orientation from the odometry message
#         self.get_logger().info(f"pose x: {position.x}")  # Log the X position
#         self.get_logger().info(f"pose x: {position.y}")  # Log the Y position
#         self.current_state[0] = position.x  # Update the current X position
#         self.current_state[1] = position.y  # Update the current Y position
#         self.current_state[2] = 2 * np.arctan2(orientation.z, orientation.w)  # Convert quaternion to yaw and update the current yaw
#         self.xstuff.append(position.x)
#         self.ystuff.append(position.y)

#         self.timer = self.create_timer(self.time_step, self.calculate_distance_error)
#         optimal_control = self.calculate_optimal_control()  # Calculate the optimal control input

#         if optimal_control is not None:  # If an optimal control is found
#             twist = Twist()  # Initialize a Twist message
#             twist.linear.x = 0.15  # Set a constant linear velocity
#             twist.angular.z = optimal_control[1]  # Set the angular velocity from the optimal control

#             self.publisher_.publish(twist)  # Publish the velocity command

#         # Check if the robot has reached the current waypoint
#         next_waypoint = self.waypoints[self.current_waypoint_idx]  # Get the next waypoint
#         position_error = np.linalg.norm(self.current_state[:2] - np.array([next_waypoint.x, next_waypoint.y]))  # Calculate the position error

#         if position_error < 0.08:  # If the position error is below the threshold
#             self.current_waypoint_idx += 1  # Move on to the next waypoint
#             if self.current_waypoint_idx >= len(self.waypoints):  # If the last waypoint is reached
#                 self.get_logger().info('Reached the last waypoint. Stopping the robot.')  # Log that the last waypoint is reached
#                 cmd_vel = Twist()  # Initialize a new Twist message
#                 cmd_vel.linear.x = 0.0  # Set linear velocity to 0
#                 cmd_vel.angular.z = 0.0  # Set angular velocity to 0
#                 self.publisher_.publish(cmd_vel)  # Publish the stop command
#                 plt.figure()
#                 plt.plot(self.true_path[0], self.true_path[1])  # Plot the interpolated path
#                 plt.plot(self.xstuff, self.ystuff, 'r--')  # Plot the waypoints
#                 plt.show()  # Show the plot
#                 plt.figure()
#                 plt.plot(self.distance_error_time_stamps, self.distance_errors)  # Plot the waypoints
#                 plt.grid()
#                 plt.show()  # Show the plot
#                 exit(0)  # Exit the program
                
#         self.get_logger().info(f"next waypoint: {self.current_waypoint_idx}")  # Log the index of the next waypoint

#     def calculate_optimal_control(self):
#         # Define your cost function here
#         def cost_function(w):
#             x = self.current_state.copy()  # Copy the current state
#             cost = 0  # Initialize the cost to 0
#             waypoint_idx = self.current_waypoint_idx  # Get the index of the current waypoint
#             for i in range(self.Np):  # For each step in the prediction horizon
#                 u = np.array([0.2, w[min(i, self.Nc - 1)]])  # Get the constant linear velocity and angular velocity from optimization
#                 x += self.T * self.kinematic_model(x, u)  # Update the state based on the kinematic model

#                 # Calculate the next waypoint's position
#                 next_waypoint = self.waypoints[waypoint_idx]  # Get the next waypoint
#                 x_ref_i = np.array([next_waypoint.x, next_waypoint.y, 0])  # Set the reference state (waypoint)

#                 # Update the reference orientation to point towards the next waypoint
#                 delta_position = x_ref_i[:2] - x[:2]  # Calculate the change in position
#                 self.get_logger().info(f"angle: {delta_position}")  # Log the change in position

#                 x_ref_i[2] = np.arctan2(delta_position[1], delta_position[0])  # Update the reference orientation

#                 error = x - x_ref_i  # Calculate the error
#                 cost += self.position_error_weight * np.linalg.norm(error) ** 2  # Add position error to the cost

#                 # Check if the robot is close to the current waypoint and move on to the next waypoint
#                 position_error = np.linalg.norm(x[:2] - x_ref_i[:2])  # Calculate the position error
#                 if position_error < 0.08 and waypoint_idx < len(self.waypoints) - 1:  # If the error is below the threshold and not the last waypoint
#                     waypoint_idx += 1  # Move on to the next waypoint

#             for i in range(self.Nc - 1):  # For each step in the control horizon
#                 delta_w = w[i + 1] - w[i]  # Calculate the change in control input
#                 cost += self.control_input_change_weight * delta_w ** 2  # Add control input change to the cost
#             self.get_logger().info(f"next waypoint: {self.current_waypoint_idx}")  # Log the index of the next waypoint

#             return cost  # Return the final cost

#         # You may need to adjust the bounds and the initial guess depending on your robot's capabilities
#         bounds = [(-2, 2)] * self.Nc  # Set the bounds for the optimization
#         initial_guess = [0] * self.Nc  # Set the initial guess for the optimization

#         result = minimize(cost_function, initial_guess, bounds=bounds)  # Run the optimization
#         optimal_control = np.array([0.2, result.x[0]])  # Return the constant linear ve locity and the first angular velocity
#         return optimal_control  # Return the optimal control input

#     @staticmethod
#     def kinematic_model(x, u):
#         v, w = u  # Extract linear and angular velocities
#         theta = x[2]  # Extract the yaw angle
#         xdot = np.array([v * np.cos(theta), v * np.sin(theta), w])  # Calculate the derivative of the state using the kinematic model
#         return xdot  # Return the derivative of the state

# def main(args=None):
#     rclpy.init(args=args)  # Initialize the ROS2 Python client library
#     nmpc_controller = NMPCController()  # Instantiate the NMPCController node
#     rclpy.spin(nmpc_controller)  # Keep the node running
#     nmpc_controller.destroy_node()  # Destroy the node when finished
#     rclpy.shutdown()  # Shutdown the ROS2 Python client library

# if __name__ == '__main__':
#     main()