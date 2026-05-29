#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from px4_msgs.msg import VehicleOdometry, TimesyncStatus #, OpiDetection

from nucleus_driver import NucleusDriver


class NucleusDriverNode(Node):

    def __init__(self):
        super().__init__('nucleus_driver_node')

        self.driver = NucleusDriver()
        self.driver.set_tcp_configuration(host='192.168.144.42')
        self.driver.connect(connection_type='tcp')
        self.driver.commands.set_cur_prof() #coord='VEHICLE'
        print(self.driver.commands.get_cur_prof())
        self.driver.commands.set_mission(range=9)
        self.driver.commands.set_trig(
            src='INTERNAL',
            freq=8
        )
        self.driver.commands.set_imu(ds='ON')
        print(self.driver.commands.get_trig())

        self.driver.start_measurement()

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.publisher_ = self.create_publisher(
            VehicleOdometry,
            '/fmu/in/vehicle_visual_odometry',
            qos_profile
        )

        self.timesync_subscriber = self.create_subscription(
            TimesyncStatus,
            '/fmu/out/timesyncstatus',
            self.timesync_callback,
            10
        )

        self.timer = self.create_timer(0.005, self.timer_callback)  # 200 Hz

        self.quaternion_data = None
        self.velocity_data = None
        self.ins_data = None
        self.gyro_data = None
        self.time_offset = 0

    def timesync_callback(self, msg):
        print(f'Timesync received: {msg.timestamp}')
        self.time_offset = msg.timestamp - self.get_clock().now().nanoseconds // 1000

    def timer_callback(self):
        packet = self.driver.read_packet()
        if(self.driver.parser.packet_queue.qsize() > 100):
            self.get_logger().warn('Packet queue size exceeded, skipping processing.')

        if packet is not None:
            if packet['id'] == 0xd2:  # Quaternion packet
                self.quaternion_data = {
                    'w': packet.get('ahrsData.quaternionW', 0.0),
                    'x': packet.get('ahrsData.quaternionX', 0.0),
                    'y': packet.get('ahrsData.quaternionY', 0.0),
                    'z': packet.get('ahrsData.quaternionZ', 0.0),
                    'fomx': packet.get('ahrsData.fomAhrs', 0.0),
                    'fomy': packet.get('ahrsData.fomAhrs', 0.0),
                    'fomz': packet.get('ahrsData.fomAhrs', 0.0)   
                }
            elif packet['id'] == 0xb4:  # Velocity packet
                #print(f"DEBUG velocity keys: {packet.keys()}") # Zie alle beschikbare namen
                self.velocity_data = {
                    'x': packet.get('velocityX', 0.0),
                    'y': packet.get('velocityY', 0.0),
                    'z': packet.get('velocityZ', 0.0),
                    'fom_x': packet.get('fomX', 0.0),
                    'fom_y': packet.get('fomY', 0.0),
                    'fom_z': packet.get('fomZ', 0.0)
                }
                self.quality_x = fom_to_quality(self.velocity_data['fom_x'])
                self.quality_y = fom_to_quality(self.velocity_data['fom_y'])
                self.quality_z = fom_to_quality(self.velocity_data['fom_z'])
                # print(f'FOM - X: {self.velocity_data["fom_x"]}, Y: {self.velocity_data["fom_y"]}, Z: {self.velocity_data["fom_z"]}')
                # print(f'Quality - X: {self.quality_x}, Y: {self.quality_y}, Z: {self.quality_z}')
                self.min_quality = min(self.quality_x, self.quality_y, self.quality_z)
                if self.velocity_data['x'] <= -32.768 and \
                   self.velocity_data['y'] <= -32.768 and \
                   self.velocity_data['z'] <= -32.768:
                    #self.get_logger().warn('Received invalid velocity data, skipping publishing.')
                    self.velocity_data = None
            elif packet['id'] == 0x82:  # IMU
                self.gyro_data = {
                    'x': packet.get('gyro.x', 0.0),
                    'y': packet.get('gyro.y', 0.0),
                    'z': packet.get('gyro.z', 0.0)
                }

            elif packet['id'] == 0xdc: # INS
                    #print("received INS data")
                    #print(f"DEBUG INS keys: {packet.keys()}") # Zie alle beschikbare namen
                    #print(f"DEBUG INS waarden: {packet}")    # Zie de echte getallen
                    #print(packet.keys())
                    self.ins_data = {
                        'x': packet.get('positionFrameX'),
                        'y': packet.get('positionFrameY', 0.0),
                        'z': packet.get('positionFrameZ', 0.0),
                        'fomX': packet.get('fomIns', 0.05),
                        'fomY': packet.get('fomIns', 0.05),
                        'fomZ': packet.get('fomIns', 0.05)
                        }
                
                    print(f'x: {self.ins_data["x"]}, y: {self.ins_data["y"]}, z: {self.ins_data["z"]}') 
                

        if self.quaternion_data and self.velocity_data and self.gyro_data and self.ins_data:
            msg = VehicleOdometry()
            msg.timestamp = self.get_clock().now().nanoseconds // 1000 + self.time_offset  # Sync with PX4 time
            msg.timestamp_sample = msg.timestamp  # Use the same timestamp for sample

            # Fill quaternion data
            #msg.q = [
            #    self.quaternion_data['w'],
            #    self.quaternion_data['x'],
            #    self.quaternion_data['y'],
            #    self.quaternion_data['z']                
            #]
            msg.q = [0.95674,0.01122,-0.00673,-0.29063]
            print(f"Sending Quaternion: w={msg.q[0]:.2f}, x={msg.q[1]:.2f}, y={msg.q[2]:.2f}, z={msg.q[3]:.2f}")

            #msg.orientation_variance = [
            #    max(self.quaternion_data['fomx']**2, 0.5),
            #    max(self.quaternion_data['fomy']**2, 0.5),
            #    max(self.quaternion_data['fomz']**2, 0.5)
            #]

            # Fill velocity data
            msg.velocity = [
                self.velocity_data['x'],
                self.velocity_data['y'],
                self.velocity_data['z']
            ]
            
            #msg.velocity_variance = [
                #self.velocity_data['fom_x']**2,
                #self.velocity_data['fom_y']**2,
                #self.velocity_data['fom_z']**2
             #   max(self.velocity_data['fom_x']**2, 0.01),
             #   max(self.velocity_data['fom_y']**2, 0.01),
             #   max(self.velocity_data['fom_z']**2, 0.01)
            #]

            msg.angular_velocity = [
                self.gyro_data['x'],
                self.gyro_data['y'],
                self.gyro_data['z']
            ]

            
            msg.position = [
                self.ins_data['x'],
                self.ins_data['y'],
                self.ins_data['z']
            ]

            #msg.position_variance = [
            #    self.ins_data['fomX']**2,
            #    self.ins_data['fomY']**2,
            #    self.ins_data['fomZ']**2
            #]

            msg.orientation_variance = [0.01, 0.01, 0.01]
            msg.position_variance = [0.1, 0.1, 0.1]
            msg.velocity_variance = [0.05, 0.05, 0.05]
   

            msg.quality = self.min_quality
            msg.velocity_frame = VehicleOdometry.VELOCITY_FRAME_BODY_FRD

            self.publisher_.publish(msg)
            # self.get_logger().info(f'Published vehicle_visual_odometry msg: {msg}')
            # print(f'Published vehicle_visual_odometry msg: {msg}')

            # Reset data after publishing
            self.quaternion_data = None
            self.velocity_data = None
            self.gyro_data = None
            self.ins_data = None 


    def destroy_node(self):
        self.driver.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = NucleusDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

def fom_to_quality(fom, fom_min=0.002, fom_max=0.1):
    """
    Convert DVL FOM (standard deviation in m/s) to a quality score [0, 100].

    Parameters:
        fom (float): Figure of Merit from DVL (m/s)
        fom_min (float): Best possible FOM (e.g. < 0.002 m/s)
        fom_max (float): Worst acceptable FOM (e.g. > 0.1 m/s)

    Returns:
        int: Quality value from 0 (poor) to 100 (excellent)
    """
    if fom <= fom_min:
        return 100
    elif fom >= fom_max:
        return 0
    else:
        quality = 100 * (1 - (fom - fom_min) / (fom_max - fom_min))
        return int(round(quality))
