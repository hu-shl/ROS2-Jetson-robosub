import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from px4_msgs.msg import ArmControl
import os
import time

RX  = "receive_log.txt"

class ArmNode(Node):
    def __init__(self):
        super().__init__('arm_node')
        self.get_logger().info('arm Node is opgestart!')

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
            ArmControl,
            '/fmu/out/arm_control', 
            self.arm_callback,
            qos_profile=qos_profile)
         
        # FIX: Instead of a broken subscription, create a timer that runs every 0.1 seconds
        self.timer = self.create_timer(0.1, self.can_receiver)

        self.tank_mapping = {
            0: ('02', '04'),
            1: ('05', '07'),
            2: ('06', '08'),
            3: ('01', '03'),
            4: ('09', '10'),
            5: ('11', '12'),
            6: ('13', '14'),
            7: ('15', '16')
        }

    def arm_callback(self, msg):
        servo = msg.servo_command
        send_needed = False

        # Loop door alle 4 de tanks heen
        for i in range(4):
            current_val = servo[i]
            
            # Controleer of de waarde omhoog is gegaan
            if current_val > self.prev_state[i]:
                # self.get_logger().info(f'Stijging gedetecteerd op Tank {i}: {self.prev_state[i]:.2f} -> {current_val:.2f}')
                send_needed = True
                msg_nr = i
            
            # Update de status voor de volgende vergelijking
            self.prev_state[i] = current_val

        # k = 0
        # # test software:
        # if servo[0] > 0.5:     # rechts rechts
        #     k = '02'  # 
        # elif servo[0] < -0.5:  # rechts links       
        #     k = '04' #
        # elif servo[1] > 0.5:   # rechts omhoog
        #     k = '05' # 
        # elif servo[1] < -0.5:  # rechts omlaag
        #     k = '07' # 
        # elif servo[2] > 0.5:   # links rechts
        #     k = '06' # 
        # elif servo[2] < -0.5:  # rechts rechts
        #     k = '08' # 
        # elif servo[3] > 0.5:   # links omhoog
        #     k = '01' # 
        # elif servo[3] < -0.5:  # links omlaag
        #     k = '03' #
        # elif servo[4] > 0.5:   # rechts rol rechts
        #     k = '09'
        # elif servo[4] < -0.5:  # rechts rol links
        #     k = '10' # 
        # elif servo[5] > 0.5:   # links rol rechts
        #     k = '11' # 
        # elif servo[5] < -0.5:  # rechts rol links
        #     k = '12' # 
        # elif servo[6] > 0.5:   # 
        #     k = '13'
        # elif servo[6] < -0.5:  # 
        #     k = '14' # 
        # elif servo[7] > 0.5:   # 
        #     k = '15' # 
        # elif servo[7] < -0.5:  # 
        #     k = '16' #       


        # if k:
        #     os.system(f"cansend can0 222#{k}")
        #     self.get_logger().info(f'ik stuur nu: 222#{k} naar arm')
        #     time.sleep(0.05)

        commands_to_send = []

        # Check all tanks to see which ones are active
        for i, value in enumerate(tanks):
            if i in self.tank_mapping:
                pos_val, neg_val = self.tank_mapping[i]
                
                if value > 0.5:
                    commands_to_send.append(pos_val)
                elif value < -0.5:
                    commands_to_send.append(neg_val)

        # Send all active commands alternatingly
        for k in commands_to_send:
            os.system(f"cansend can0 111#{k}")
            self.get_logger().info(f'ik stuur nu: 111#{k} naar buoyancy')
            
            # Small delay to prevent saturating the CAN bus 
            # and to allow the receiver to process each message
            time.sleep(0.05)

        # for j in range(4):
        #     if servo[j] > 0.5:
        #         os.system(f"cansend can0 123#0{j+1}")
        #         self.get_logger().info(f'ik stuur nu: 123#0{j+1}')
        #         time.sleep(0.05)
        #     elif servo[j] < -0.5:
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
    node = ArmNode()
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