import sys


def list_usb_cameras():
    """List all USB cameras and their properties"""
    context = pyudev.Context()

    cameras = []
    for device in context.list_devices(subsystem='video4linux'):
        parent = device.parent
        parent_props = parent.properties if parent is not None else {}
        device_props = device.properties
        if parent is not None and 'usb' in parent_props.get('SUBSYSTEM', ''):
            cam_info = {
                'dev_path': device_props.get('DEVPATH', 'N/A'),
                'dev_node': device_props.get('DEVNAME', 'N/A'),
                'serial': parent_props.get('ID_SERIAL_SHORT', 'N/A'),
                'model': parent_props.get('ID_MODEL', 'N/A'),
                'vendor': parent_props.get('ID_VENDOR', 'N/A'),
                'usb_port': parent_props.get('DEVPATH', 'N/A').split('/')[-1]
            }
            cameras.append(cam_info)

    cameras_by_port = {}
    for cam in cameras:
        cameras_by_port.setdefault(cam['usb_port'], []).append(cam)

    print(f"Detected USB ports: {len(cameras_by_port)}")
    for usb_port in sorted(cameras_by_port):
        dev_nodes = sorted(cam['dev_node'] for cam in cameras_by_port[usb_port])
        print(f"USB Port: {usb_port} | Device: {' '.join(dev_nodes)}")

    return cameras


if __name__ == "__main__":
    # Check if pyudev is installed
    try:
        import pyudev
    except ImportError:
        print("Please install pyudev: pip install pyudev")
        sys.exit(1)

    list_usb_cameras()
