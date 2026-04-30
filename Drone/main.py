import time, math, rospy, cv2
import numpy as np
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from dronekit import connect, VehicleMode
from pymavlink import mavutil

iha = connect("udp:127.0.0.1:14550", wait_ready=True, timeout=100)
time.sleep(1)

hedef_lat = None
hedef_lon = None
H = 5.0
ENGEL = 10.0 

bridge = CvBridge()
imgs = {
    "front": np.zeros((240, 320, 3), dtype=np.uint8),
    "back":  np.zeros((240, 320, 3), dtype=np.uint8),
    "left":  np.zeros((240, 320, 3), dtype=np.uint8),
    "right": np.zeros((240, 320, 3), dtype=np.uint8),
    "down":  np.zeros((240, 320, 3), dtype=np.uint8),
    "lidar": np.zeros((240, 320, 3), dtype=np.uint8),
    "depth": np.zeros((240, 320, 3), dtype=np.uint8)
}

depth_min = 30.0  # Derinlik kameranın en yakın mesafesi

def img_cb(msg, cam_name):
    try:
        cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        imgs[cam_name] = cv2.resize(cv_img, (320, 240))
    except:
        pass

def depth_cb(msg):
    global depth_min
    try:
        depth_img = bridge.imgmsg_to_cv2(msg, "32FC1")
        depth_img = cv2.resize(depth_img, (320, 240))
        depth_img = np.where(np.isfinite(depth_img), depth_img, 30.0)
        upper_center = depth_img[30:110, 110:210]
        valid = upper_center[(upper_center > 0.3) & (upper_center < 30.0)]
        if len(valid) > 10:
            depth_min = float(np.percentile(valid, 10))
        else:
            depth_min = 30.0
        disp = np.clip(depth_img, 0, 30)
        norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
        imgs["depth"] = cv2.applyColorMap(np.uint8(norm), cv2.COLORMAP_JET)
    except:
        pass

def vel(vx, vy, vz, yaw=0):
    msg = iha.message_factory.set_position_target_local_ned_encode(
        0, 0, 0, mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000111111000111, 0, 0, 0, vx, vy, vz, 0, 0, 0, 0, yaw)
    iha.send_mavlink(msg)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def takeoff(alt):
    while not iha.is_armable: time.sleep(1)
    iha.mode = VehicleMode("GUIDED")
    iha.armed = True
    while not iha.armed: time.sleep(0.5)
    iha.simple_takeoff(alt)
    while iha.location.global_relative_frame.alt < alt * 0.9: time.sleep(0.5)

def draw_lidar(ranges):
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cx, cy = 160, 200 
    scale = 8.0 
    cv2.circle(img, (cx, cy), 5, (0,0,255), -1) 
    
    n = len(ranges)
    for i, r in enumerate(ranges):
        if 0.15 < r < 30.0:

            angle = (i - n//2) * (2 * math.pi / n) + (math.pi / 2)
            px = int(cx - r * scale * math.sin(angle))
            py = int(cy - r * scale * math.cos(angle))
            if 0 <= px < 320 and 0 <= py < 240:
                img[py, px] = (0, 255, 0) 
                
    cv2.circle(img, (cx, cy), int(ENGEL * scale), (255, 0, 0), 1)
    imgs["lidar"] = img

state = "HAREKET"
brake_time = 0.0

def callback(msg):
    global state, brake_time
    draw_lidar(msg.ranges) 
    
    if iha.mode.name in ("RTL", "LAND") or hedef_lat is None: return
    
    if state == "FREN":
        vel(0, 0, 0, 0) 
        if time.time() - brake_time > 4.0:
            print("Durma tamamlandi. Guvenli bir sekilde Eve Donus (RTL) basliyor.")
            iha.mode = VehicleMode("RTL")
            state = "RTL"
        return
    
    g = iha.location.global_relative_frame
    dist = haversine(g.lat, g.lon, hedef_lat, hedef_lon)
    
    if dist < 3.0:
        print("Hedefe ulasildi. Arac durduruluyor...")
        state = "FREN"
        brake_time = time.time()
        return
        
    brg = bearing(g.lat, g.lon, hedef_lat, hedef_lon)
    rel_goal = (brg - iha.heading + 360) % 360
    if rel_goal > 180: rel_goal -= 360
    
    v = [m if 0.15 < m < 30 else 30.0 for m in msg.ranges]

    front_idx = len(v) // 4
    s, e = int(front_idx-20) % len(v), int(front_idx+20) % len(v)
    
    front_sector = v[s:e] if s < e else v[s:] + v[:e]
    F = min(front_sector) if len(front_sector) > 0 else 30.0

    lidar_engel = F < ENGEL
    depth_engel = depth_min < ENGEL

    if lidar_engel and depth_engel:
        print(f"ENGEL (LiDAR+Derinlik): Lidar={F:.1f}m Derinlik={depth_min:.1f}m")
        state = "FREN"
        brake_time = time.time()
    elif lidar_engel:
        print(f"ENGEL (LiDAR): {F:.1f}m")
        state = "FREN"
        brake_time = time.time()
    elif depth_engel:
        print(f"ENGEL (Derinlik): {depth_min:.1f}m")
        state = "FREN"
        brake_time = time.time()
    else:
        vel(min(2.0, F * 0.3), 0, 0, max(-0.5, min(0.5, rel_goal * 0.025)))

rospy.init_node('lidar_engel')

# Kameralara
rospy.Subscriber("/camera/front/image_raw", Image, img_cb, callback_args="front")
rospy.Subscriber("/camera/back/image_raw", Image, img_cb, callback_args="back")
rospy.Subscriber("/camera/left/image_raw", Image, img_cb, callback_args="left")
rospy.Subscriber("/camera/right/image_raw", Image, img_cb, callback_args="right")
rospy.Subscriber("/camera/down/image_raw", Image, img_cb, callback_args="down")
rospy.Subscriber("/camera/depth/depth", Image, depth_cb)
rospy.Subscriber("/spur/laser/scan", LaserScan, callback)

takeoff(H)

g = iha.location.global_relative_frame
bas_lat, bas_lon = g.lat, g.lon
hr = math.radians(iha.heading)

hedef_lat = bas_lat + (100 * math.cos(hr) / 111000)
hedef_lon = bas_lon + (100 * math.sin(hr) / (111000 * math.cos(math.radians(bas_lat))))

print("Başladı.")

while not rospy.is_shutdown():
    row1 = np.hstack((imgs["left"], imgs["front"], imgs["right"]))
    row2 = np.hstack((imgs["depth"], imgs["down"], imgs["lidar"]))
    grid = np.vstack((row1, row2))
    
    cv2.putText(grid, "Sol Kamera", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(grid, "On Kamera", (330, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(grid, "Sag Kamera", (650, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(grid, "Derinlik", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(grid, "Alt Kamera", (330, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(grid, "LiDAR", (650, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    
    cv2.imshow("DroneQube Kontrol Paneli", grid)
    if cv2.waitKey(30) & 0xFF == 27: 
        break

iha.mode = VehicleMode("LAND")
cv2.destroyAllWindows()
