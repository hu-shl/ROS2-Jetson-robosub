import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
# from px4_msgs.msg import BuoyancyControl
import os
import time
# import can

RX  = "receive_log.txt"

class ReceiverNode(Node):
    def __init__(self):
        super().__init__('receiver_node')
        self.get_logger().info('Receiver Node is opgestart!')

        if os.path.exists(RX):
            os.remove(RX)

        # # 1. Setup Native SocketCAN Interface
        # try:
        #     self.bus = can.interface.Bus(channel='can0', interface='socketcan')
        #     self.get_logger().info('Succesvol verbonden met can0 socket.')
        # except Exception as e:
        #     self.get_logger().error(f'Kon niet verbinden met can0: {e}')
        #     self.bus = None
        
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

        # # RX = subprocess.run(['ls', '-l'], stdout=subprocess.PIPE)
        # command = "ls -l"
        # RX = subprocess.run(command, shell=True, text=True, capture_output=True)
        # self.get_logger().info(f'RX is: {RX.returncode}')
        # self.get_logger().info(f'RX is: {RX.stdout}')

        # """ Asynchronously checks the CAN socket for incoming messages (replaces candump) """
        # if self.bus is None:
        #     return

        # # Read all available messages in the buffer without blocking (timeout=0)
        # while True:
        #     msg = self.bus.recv(timeout=0.0)
        #     if msg is None:
        #         break # Buffer is empty, wait for next timer cycle
            
        #     # Extract data components cleanly
        #     can_id = msg.arbitration_id
        #     data_hex = ' '.join(f'{b:02X}' for b in msg.data)
            
        #     # Print incoming frames straight to your ROS logs!
        #     self.get_logger().info(f'[CAN RX] ID: {can_id:X} | Data: [{data_hex}]')
        

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

