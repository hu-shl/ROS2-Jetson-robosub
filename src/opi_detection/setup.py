from setuptools import find_packages, setup

package_name = 'opi_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('include', [
            'package.xml',
            'opi_detection/include/camera_control.py',
            'opi_detection/include/camera.py',
            'opi_detection/include/config.py',
            'opi_detection/include/edge.py',
            'opi_detection/include/color_Detection.py',
            'opi_detection/include/detector.py',
            'opi_detection/include/image_processing.py',
            'opi_detection/include/__init__.py'
        ]),
    ],
    install_requires=['setuptools', 'numpy', 'opencv-python', 'torch', 'tensorrt'],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='benjamin.boehmer@student.hu.nl',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'v2_with_modules = opi_detection.v2_with_modules:main'
        ],
    },
)
