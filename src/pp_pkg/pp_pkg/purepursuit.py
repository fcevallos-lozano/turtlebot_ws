import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry, Path
import rclpy.qos as qos
from math import atan2, sqrt
import numpy as np
from std_msgs.msg import Empty
from time import time
from scipy.interpolate import splprep, splev

class PurePursuit(Node):

    def __init__(self):
        super().__init__('pure_pursuit')
        self.publisher_ = self.create_publisher(Twist, '/turtlebot4/cmd_vel', 10)
        self.subscription_odom = self.create_subscription(Odometry, '/turtlebot4/odom', self.odom_callback, 
                                    qos.QoSProfile(depth=10, reliability=qos.ReliabilityPolicy.BEST_EFFORT))
        self.subscription_odom
        #reset odom
            # set up node
        
        # super().__init__('reset_odom')
        # # set up the odometry reset publisher
        # reset_odom = self.create_publisher('/mobile_base/commands/reset_odometry', Empty, 10)

        # # reset odometry (these messages take a few iterations to get through)
        # timer = time()
        # while time() - timer < 0.25:
        #     reset_odom.publish(Empty())
        # Waypoints initialization
        # Point 21 is closest from Point 0
        # Point 1 is furthes from Point 0
        # self.waypoints =  [Point(x=0.0,y=0.0,z=0.0), Point(x=0.571194595265405, y=-0.4277145118491421, z=0.0), Point(x=1.1417537280142898, y=-0.8531042347260006,z=0.0), Point(x=1.7098876452457967, y=-1.2696346390611464,z=0.0), Point(x=2.2705328851607995, y=-1.6588899151216996,z=0.0), Point(x=2.8121159420106827, y=-1.9791445882187304,z=0.0), Point(x=3.314589274316711, y=-2.159795566252656,z=0.0), Point(x=3.7538316863009027, y=-2.1224619985315876,z=0.0), Point(x=4.112485112342358, y=-1.8323249172947023,z=0.0), Point(x=4.383456805594431, y=-1.3292669972090994,z=0.0), Point(x=4.557386228943757, y=-0.6928302521681386,z=0.0), Point(x=4.617455513800438, y=0.00274597627737883,z=0.0), Point(x=4.55408382321606, y=0.6984486966257434,z=0.0), Point(x=4.376054025556597, y=1.3330664239172116,z=0.0), Point(x=4.096280073621794, y=1.827159263675668,z=0.0), Point(x=3.719737492364894, y=2.097949296701878,z=0.0), Point(x=3.25277928312066, y=2.108933125822431,z=0.0), Point(x=2.7154386886417314, y=1.9004760368018616,z=0.0), Point(x=2.1347012144725985, y=1.552342808106984,z=0.0), Point(x=1.5324590525923942, y=1.134035376721349,z=0.0), Point(x=0.9214084611203568, y=0.6867933269918683,z=0.0), Point(x=0.30732366808208345, y=0.22955002391894264,z=0.0), Point(x=-0.3075127599907512, y=-0.2301742560363831,z=0.0), Point(x=-0.9218413719658775, y=-0.6882173194028102,z=0.0), Point(x=-1.5334674079795052, y=-1.1373288016589413,z=0.0), Point(x=-2.1365993767877467, y=-1.5584414896876835,z=0.0), Point(x=-2.7180981380280307, y=-1.9086314914221845,z=0.0), Point(x=-3.2552809639439704, y=-2.1153141204181285,z=0.0), Point(x=-3.721102967810494, y=-2.0979137913841046,z=0.0), Point(x=-4.096907306768644, y=-1.8206318841755131,z=0.0), Point(x=-4.377088212533404, y=-1.324440752295139,z=0.0), Point(x=-4.555249804461285, y=-0.6910016662308593,z=0.0), Point(x=-4.617336323713965, y=0.003734984720118972,z=0.0), Point(x=-4.555948690867849, y=0.7001491248072772,z=0.0), Point(x=-4.382109193278264, y=1.3376838311365633,z=0.0), Point(x=-4.111620918085742, y=1.8386823176628544,z=0.0), Point(x=-3.7524648889185794, y=2.1224985058331005,z=0.0), Point(x=-3.3123191098095615, y=2.153588702898333,z=0.0), Point(x=-2.80975246649598, y=1.9712114570096653,z=0.0), Point(x=-2.268856462266256, y=1.652958931009528,z=0.0), Point(x=-1.709001159778989, y=1.2664395490411673,z=0.0), Point(x=-1.1413833971013372, y=0.8517589252820573,z=0.0), Point(x=-0.5710732645795573, y=0.4272721367616211,z=0.0)]
        num_points = 50
        x_values = np.linspace(0, 1, num_points)
        y_values = 0.5 * np.sin(np.pi * x_values)

        waypoints_sine_wave = [Point(x=x_values[i], y=y_values[i], z=0.0) for i in range(num_points)]

        self.waypoints = waypoints_sine_wave

        """
        For printing out closest points to Point 0

        p1 = self.waypoints[1]
        p21 = self.waypoints[21]        
        p22 = self.waypoints[22]
        p42 = self.waypoints[42]

        d_1 = self.__distance(p1.x, p1.y, 0, 0)
        d_21 = self.__distance(p22.x, p22.y, 0, 0)
        d_22 = self.__distance(p21.x, p21.y, 0, 0)
        d_42 = self.__distance(p42.x, p42.y, 0, 0)
        print(f"1: {d_1}")
        print(f"21: {d_21}")
        print(f"22: {d_22}")
        print(f"42: {d_42}")
        """
        
        self.lookahead_distance = 0.08
        self.current_pose = Odometry().pose.pose
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
        tck, u = splprep([w_x, w_y], s=0)
        u_new = np.linspace(u.min(), u.max(), 1000)
        path_x, path_y = splev(u_new, tck)
        # plot reference path and waypoints
        print(f"waypoint0: {self.waypoints[0]}")
        # print(self.waypoints[1])

        """
        fig, ax = plt.subplots(figsize=(12,8))
        plt.plot(path_x, path_y)
        plt.plot(waypoints_x, waypoints_y, 'ro')

        
        for i, points in enumerate(zip(waypoints_x, waypoints_y)):
            x,y = points
            print(f"{i}: ({x}, {y})")
            ax.text(x + 0.05, y + 0.05, i, size=12)    
        plt.show()
        exit(1)
        """
        # plt.plot(path_x, path_y)
        # plt.plot(w_x, w_y, 'ro')
        # plt.show()

    def odom_callback(self, msg: Odometry):
        # self.current_odom = msg
        self.current_pose = msg.pose.pose
        self.get_logger().info(f"pose x: {self.current_pose.position.x}")
        self.get_logger().info(f"pose y: {self.current_pose.position.y}")

    def pure_pursuit(self):
        #if len(self.waypoints) > 0 and (self.current_pose is not None):
        # find the nearest waypoint
            #min_distance = float('inf')
           # nearest_index = 0
        w_x = [wp.x for wp in self.waypoints]
        w_y = [wp.y for wp in self.waypoints]
        self.passed_values = []

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
        self.get_logger().info(f"lookahead: {ind}")
        if ind < len(w_x):
            tx = w_x[ind]
            ty = w_y[ind]
        else:  # toward goal
            tx = w_x[-1]
            ty = w_y[-1]
            ind = len(w_x) - 1

        alpha = atan2(ty - self.current_pose.position.y, tx - self.current_pose.position.x) - self.current_pose.orientation.z
        self.get_logger().info(f"angle: {alpha}")
        self.get_logger().info(f"angle2: {self.current_pose}")

        cmd_vel = Twist()
        cmd_vel.linear.x = .1  # tune this value
        cmd_vel.angular.z = 1.1 * alpha  # tune this value

            # check if the robot is close to the final waypoint
        final_waypoint = self.waypoints[-1]
        distance_to_final = sqrt((final_waypoint.x - self.current_pose.position.x) ** 2 + (final_waypoint.y - self.current_pose.position.y) ** 2)

            # stop condition (tune the threshold value)
        stop_threshold = 3.0
        if distance_to_final < stop_threshold:
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0

        self.publisher_.publish(cmd_vel)
                
            #for i in range( self.last_visited_index, len(self.waypoints)):
             #   w_x, w_y = self.waypoints[i].x, self.waypoints[i].y
              #  curr_x, curr_y = self.current_pose.position.x, self.current_pose.position.y
               # distance = self.__distance(w_x, w_y, curr_x, curr_y)    # compute euclidean distance

                

            #    if distance < min_distance:
            #        min_distance = distance
            #        self.get_logger().info(f"min distance: {min_distance}")
            #        nearest_index = i
            
            #if min_distance > self.lookahead_distance:
            #    self.last_visited_index = nearest_index - 1
            #else:
            #    self.last_visited_index = nearest_index
            #self.get_logger().info(f"nearest index: {nearest_index}")

        # find the lookahead point
            #distance_sum = 0
            #lookahead_point = None
            #for i in range(nearest_index, len(self.waypoints)):
             #   wf_x, wf_y = self.waypoints[i].x, self.waypoints[i].y       # future waypoint
              #  wi_x, wi_y = self.waypoints[i-1].x, self.waypoints[i-1].y   # current waypoint
               # distance = self.__distance(wf_x, wf_y, wi_x, wi_y)

                #distance_sum += distance
                #if distance_sum > self.lookahead_distance:
                 #   lookahead_point = self.waypoints[i]
            #self.get_logger().info(f"lookahead: {lookahead_point}")

            #if lookahead_point is not None:
            # calculate the control command
                #heading = atan2(lookahead_point.y - self.current_pose.position.y, lookahead_point.x - self.current_pose.position.x)
                #self.get_logger().info(f"--- Heading: {heading} ---")
                #self.get_logger().info(f"--- Orientation: {self.current_pose.orientation.z} ---")
                #angle = heading - self.current_pose.orientation.z
                #self.get_logger().info(f"--- Angle: {angle} ---")
                #angle %= 2*np.pi
    """
                while angle > np.pi:
                    angle -= 2*np.pi
                while angle < -1*np.pi:
                    angle += 2*np.pi
                """
                
            #self.get_logger().info(f"angle: {angle}")

    def __distance(self, x1, y1, x2, y2):
        """ compute Euclidean Distance """
        d = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        return d
    

    #def update(self):
       # self.pure_pursuit()

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



