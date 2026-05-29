import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import os
import time


RX  = "receive_log.txt"

class ReceiverNode(Node):
    def __init__(self):
        super().__init__('receiver_node')
        self.get_logger().info('Receiver Node is opgestart!')

        if os.path.exists(RX):
            os.remove(RX)
        
        os.system(f"candump can0 > {RX} &")
        # self.rx_timer = self.create_timer(0.01, self.receive_can_messages)
        time.sleep(0.5)


    def receive_can_messages(self):

        self.get_logger().info('does this run?')
        with open(RX, "r") as f:
            while True: 
                message = f.readline()

                if message:
                    clean_msg = message.strip()
                    self.get_logger().info(f'RX is: {clean_msg}')    
                else:
                    time.sleep(0.1)
       

def main(args=None):
    rclpy.init(args=args)
    node = ReceiverNode()
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

