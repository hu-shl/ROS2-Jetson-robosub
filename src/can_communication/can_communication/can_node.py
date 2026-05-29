import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from px4_msgs.msg import BuoyancyControl
import os
import time

RX  = "receive_log.txt"

class CanNode(Node):
    def __init__(self):
        super().__init__('can_node')
        self.get_logger().info('CAN Node is opgestart!')

        # Onthoud de vorige status van alle 4 de tanks
        self.prev_state = [0.0, 0.0, 0.0, 0.0]

        self.file_handle = open(RX, "r")
        self.file_handle.seek(0, os.SEEK_END) # Go to the end so we only read new lines

        # PX4 Subscription
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscription = self.create_subscription(
            BuoyancyControl,
            '/fmu/out/buoyancy_control', 
            self.buoyancy_callback,
            qos_profile=qos_profile)
         
        # FIX: Instead of a broken subscription, create a timer that runs every 0.1 seconds
        self.timer = self.create_timer(0.1, self.can_receiver)

    def buoyancy_callback(self, msg):
        tanks = msg.tank_command
        send_needed = False

        # Loop door alle 4 de tanks heen
        for i in range(4):
            current_val = tanks[i]
            
            # Controleer of de waarde omhoog is gegaan
            if current_val > self.prev_state[i]:
                # self.get_logger().info(f'Stijging gedetecteerd op Tank {i}: {self.prev_state[i]:.2f} -> {current_val:.2f}')
                send_needed = True
                msg_nr = i
            
            # Update de status voor de volgende vergelijking
            self.prev_state[i] = current_val

        k = 0
        # test software:
        if tanks[0] > 0.5:
            k = 2 # 
        elif tanks[0] < -0.5:                
            k = 4 #
        elif tanks[1] > 0.5:
            k = 5 # 
        elif tanks[1] < -0.5:
            k = 7 # 
        elif tanks[2] > 0.5:
            k = 6 # 
        elif tanks[2] < -0.5:
            k = 8 # 
        elif tanks[3] > 0.5:
            k = 1 # 
        elif tanks[3] < -0.5:
            k = 3 # 

        if k:
            os.system(f"cansend can0 111#0{k}")
            self.get_logger().info(f'ik stuur nu: 111#0{k} naar buoyancy')
            time.sleep(0.05)

        # for j in range(4):
        #     if tanks[j] > 0.5:
        #         os.system(f"cansend can0 123#0{j+1}")
        #         self.get_logger().info(f'ik stuur nu: 123#0{j+1}')
        #         time.sleep(0.05)
        #     elif tanks[j] < -0.5:
        #         os.system(f"cansend can0 123#0{j+5}")
        #         self.get_logger().info(f'ik stuur nu: 123#0{j+5}')
        #         time.sleep(0.05)

    def can_receiver(self):
        """
        Runs every 100ms. Reads any NEW lines added to the text file 
        without blocking the executor.
        """
        # Read a line. If no new line, it returns an empty string instantly.
        message = self.file_handle.readline()

        if message:
            clean_msg = message.strip()
            self.get_logger().info(f'RX is: {clean_msg}')
        

def main(args=None):
    rclpy.init(args=args)
    node = CanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.bus:
            node.bus.shutdown() # Clean up connection on exit
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
