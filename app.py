from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, RootModel
from typing import Dict, Any
import uvicorn
import json
import cv2
import numpy as np
import mysql.connector
import os
import math
from ultralytics import YOLO
from traffic_brain import TrafficFuzzyController
import asyncio
import base64

# --- GLOBAL TRAFFIC LIGHT STATE ---
LIGHT_STATE = {
    "north": "RED", "south": "RED", "east": "RED", "west": "RED",
    "timer": 0, "current_phase": "NS", # NS (North/South) or EW (East/West)
    "mode": "NORMAL_FUZZY"
}
CURRENT_QUEUE = {"north": 0, "south": 0, "east": 0, "west": 0}
# --- MANUAL EMERGENCY OVERRIDE ---
# Stores the camera_id that needs an immediate green light (e.g., "north")
MANUAL_OVERRIDE = None

# Virtual Sandbox States
SANDBOX_LIGHT_STATE = {
    "north": "RED", "south": "RED", "east": "RED", "west": "RED",
    "timer": 0, "current_phase": "NS", "mode": "NORMAL_FUZZY"
}
SANDBOX_QUEUES = {"north": 0, "south": 0, "east": 0, "west": 0}
SANDBOX_MANUAL_OVERRIDE = None
SANDBOX_EMERGENCIES = {"north": 0, "south": 0, "east": 0, "west": 0}
SANDBOX_INTERSECTION = "tun_perak"

# Sandbox Logs (So the Sandbox Terminal can print Python's thoughts!)
SANDBOX_LOGS = []

# --- ACTIVE INTERSECTION TRACKER ---
ACTIVE_INTERSECTION = "tun_perak" # Default on startup

# --- GLOBAL LIVE POSITIONS FOR 2D SIMULATION ---
LIVE_POSITIONS = {
    "north": [],
    "south": [],
    "east": [],
    "west": []
}

# Initialize the Brain
fuzzy_brain = TrafficFuzzyController()

# --- HARDWARE PRESENTATION TOGGLE ---
# Set 0 = GPU, set to 'cpu' = CPU
PROCESSOR_MODE = 0

# Create violations folder safely
os.makedirs("static/violations", exist_ok=True)

app = FastAPI(title="Smart Intersection System")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- AI & VIDEO SETTINGS ---
MODEL_NAME = "best.pt" 
SPEED_LIMIT_KMH = 80

VIDEO_PATHS = {
    "tun_perak": {
        "north": "videos/north1.mp4",
        "south": "videos/south2.mp4",
        "east": "videos/east1.mp4",
        "west": "videos/west1.mp4"
    },
    "hang_tuah": {
        "north": "videos/south2.mp4", 
        "south": "videos/north1.mp4",
        "east": "videos/west1.mp4"
    }
}

TARGET_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7] 
CLASS_NAMES = {0: 'Ambulance', 1: 'Bicycle', 2: 'Bus', 3: 'Car', 4: 'Fire_Truck', 5: 'Motorcycle', 6: 'Truck', 7: 'Van'}

# --- GLOBAL LIVE STATS FOR DASHBOARD ---
LIVE_STATS = {
    "north": {name: 0 for name in CLASS_NAMES.values()},
    "south": {name: 0 for name in CLASS_NAMES.values()},
    "east": {name: 0 for name in CLASS_NAMES.values()},
    "west": {name: 0 for name in CLASS_NAMES.values()},
    "recent_violations": [] # Stores the last 5 violations
}

# --- DATABASE CONNECTION ---
print("Connecting to MySQL Database...")
try:
    db_connection = mysql.connector.connect(
        host="localhost",
        user="root",          
        password="YOUR_PASSWORD",
        database="fyp_traffic"
    )
    db_cursor = db_connection.cursor()
    print("✅ Database Connected Successfully!")
except mysql.connector.Error as err:
    print(f"❌ Database Connection Failed: {err}")

# --- DATA MODELS (Updated for Pydantic V2) ---
# This tells Python to accept a master dictionary containing "north", "south", etc.
class MasterConfig(RootModel[Dict[str, Any]]):
    pass

def load_config():
    try:
        with open("camera_config.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return None

# --- MATHEMATICAL LINE INTERSECTION ---
def line_intersect(A, B, C, D):
    """Returns True if the vehicle's path (A to B) crosses the Admin's drawn line (C to D)"""
    def ccw(p1, p2, p3):
        return (p3[1]-p1[1]) * (p2[0]-p1[0]) > (p2[1]-p1[1]) * (p3[0]-p1[0])
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# --- DYNAMIC VIDEO GENERATOR ---
async def generate_frames(intersection_id: str, camera_id: str):

    # --- THE 3-WAY SHIELD (Blocks ghost streams on T-Junctions) ---
    if intersection_id == "hang_tuah" and camera_id == "west":
        print(f"🚫 Rejecting WEST stream: {intersection_id.upper()} is a 3-way T-Junction!")
        return
    
    # --- SINGLE AI BRAIN SETUP (Thread Isolated) ---
    print(f"🧠 Loading isolated AI model for {intersection_id.upper()} - {camera_id.upper()}...")
    try:
        local_model = YOLO(MODEL_NAME)
    except Exception as e:
        print(f"❌ FATAL AI LOAD ERROR: {e}")
        return

    # 1. Grab the correct video file dynamically
    video_file = VIDEO_PATHS.get(intersection_id, {}).get(camera_id)
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        print(f"❌ ERROR: Cannot find video for {intersection_id}/{camera_id} at {video_file}")
        return
    
    # --- SAVE PERMANENT BACKGROUND IMAGE ---
    ret, first_frame = cap.read()
    if ret:
        first_frame = cv2.resize(first_frame, (1280, 720))
        # We save the background specifically for this intersection!
        bg_path = f"static/bg_{intersection_id}_{camera_id}.jpg"
        cv2.imwrite(bg_path, first_frame)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    # 2. Grab the nested configuration: master_config -> intersection -> camera
    master_config = load_config()
    inter_config = master_config.get(intersection_id) if master_config else None
    config = inter_config.get(camera_id) if inter_config else None
    
    def parse_pts(key):
        if config and config.get(key):
            pts = config[key]
            if len(pts) > 0: return [tuple(map(int, pt)) for pt in pts]
        return None

    zone_pts = np.array(parse_pts("zone"), np.int32) if parse_pts("zone") else None
    calib_pts = np.array(parse_pts("calibBox"), np.float32) if parse_pts("calibBox") else None
    lineA = parse_pts("lineA")
    lineB = parse_pts("lineB")
    stopLine = parse_pts("stopLine")

    # --- BIRD'S EYE VIEW MATRIX SETUP ---
    perspective_matrix = None
    SCALE = 100 
    
    if calib_pts is not None and len(calib_pts) == 4 and config:
        admin_w = float(config.get("distWidth", 3.5))
        admin_l = float(config.get("distCalib", 20.0)) 
        
        # --- AUTO-SORT POINTS ---
        # Forces the 4 clicked points into strictly: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        pts = sorted(calib_pts, key=lambda x: x[1]) # Sort by Y
        top_pts = sorted(pts[:2], key=lambda x: x[0]) # Top 2 sorted by X
        bottom_pts = sorted(pts[2:], key=lambda x: x[0]) # Bottom 2 sorted by X
        # Rebuild array correctly
        src_pts = np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype=np.float32)
        
        # Create perfect flat map
        dst_pts = np.array([[0, 0], [admin_w*SCALE, 0], [admin_w*SCALE, admin_l*SCALE], [0, admin_l*SCALE]], dtype=np.float32)
        perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    tracked_vehicles = {}
    vehicle_counts = {name: 0 for name in CLASS_NAMES.values()}
    frame_count = 0

    print(f"🎥 Starting Live Stream: {camera_id.upper()}")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                print("🏁 Video Ended.")
                break 
                
            # We MUST add +1 to frame_count for every single frame to keep the Speed Math perfectly accurate!
            frame_count += 1

            # STRICT 1280*720 RESIZE 16:9 widescreen video
            frame = cv2.resize(frame, (1280, 720))

            # UI Lines
            if zone_pts is not None and len(zone_pts) == 4:
                cv2.polylines(frame, [zone_pts], isClosed=True, color=(0, 255, 0), thickness=2)
                # Apply mask so AI ignores everything outside the box
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [zone_pts], 255)
                frame_to_process = cv2.bitwise_and(frame, frame, mask=mask)
            else:
                frame_to_process = frame

            if lineA and len(lineA) == 2:
                cv2.line(frame, lineA[0], lineA[1], (255, 100, 0), 2)
                cv2.putText(frame, "Line A", (lineA[0][0], lineA[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
            if lineB and len(lineB) == 2:
                cv2.line(frame, lineB[0], lineB[1], (255, 255, 0), 2)
                cv2.putText(frame, "Line B", (lineB[0][0], lineB[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            if stopLine and len(stopLine) == 2:
                # 1. Read the actual live traffic light status for this specific camera!
                current_light = LIGHT_STATE.get(camera_id, "RED")
                
                # 2. Change the line color to match the traffic light
                if current_light == "RED":
                    line_color = (0, 0, 255) # Red
                elif current_light == "YELLOW":
                    line_color = (0, 255, 255) # Yellow
                else:
                    line_color = (0, 255, 0) # Green

                cv2.line(frame, stopLine[0], stopLine[1], line_color, 2) 
                cv2.putText(frame, "Stop Line", (stopLine[0][0], stopLine[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)

            # --- THE DOUBLE BUFFER (Prevents flashing on the web map) ---
            current_frame_positions = []

            # AI Tracking (Using frame_to_process, locked to 640 internally for max speed)
            # Change 'model' to 'local_model'
            results = local_model.track(
                frame_to_process, 
                persist=True, 
                classes=TARGET_CLASSES, 
                conf=0.10, 
                iou=0.4, 
                imgsz=640, 
                device=PROCESSOR_MODE, 
                tracker="bytetrack.yaml",
                verbose=False
            )

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    curr_pt = (cx, cy)
                    
                    # ==========================================
                    # 1. THE GATEKEEPER (Ignore adjacent lanes)
                    # ==========================================
                    is_inside = True 
                    if zone_pts is not None:
                        is_inside = cv2.pointPolygonTest(zone_pts, (float(cx), float(cy)), False) >= 0

                    if not is_inside:
                        continue # STOP HERE! Skip this car if it's outside the green zone!

                    # ==========================================
                    # 2. INITIALIZE & VOTE (Define display_class)
                    # ==========================================
                    raw_class = CLASS_NAMES.get(class_id, 'Unknown')

                    if track_id not in tracked_vehicles:
                        tracked_vehicles[track_id] = {
                            'prev_pt': curr_pt, 'frame_A': None, 'frame_B': None, 
                            'speed': 0, 'rt_speed': 0, 'history': [],
                            'saved_db': False, 'crossed_stop': False,
                            'class_votes': {name: 0 for name in CLASS_NAMES.values()}, 
                            'final_class': None
                        }
                    
                    v = tracked_vehicles[track_id]
                    prev_pt = v['prev_pt']
                    
                    if v['final_class'] is None:
                        if raw_class in v['class_votes']: v['class_votes'][raw_class] += 1
                        if sum(v['class_votes'].values()) >= 5:
                            v['final_class'] = max(v['class_votes'], key=v['class_votes'].get)

                    display_class = v['final_class'] if v['final_class'] else raw_class

                    # ==========================================
                    # 3. TRANSLATE TO 2D FLAT MAP
                    # ==========================================
                    real_x, real_y = 0.0, 0.0
                    if perspective_matrix is not None:
                        # MUST use y2 (where tires touch the road) for accurate depth!
                        pt_camera = np.array([[[cx, y2]]], dtype=np.float32)
                        pt_flat = cv2.perspectiveTransform(pt_camera, perspective_matrix)
                        real_x = pt_flat[0][0][0]
                        real_y = pt_flat[0][0][1]

                        # Pack data into the temporary buffer!
                        current_frame_positions.append({
                            "id": int(track_id),
                            "class": display_class,
                            "x": float(real_x),
                            "y": float(real_y)
                        })

                    # ==========================================
                    # 4. REAL-TIME SPEED (High Stability)
                    # ==========================================
                    v['history'].append((real_x, real_y, frame_count))
                    
                    if len(v['history']) >= 15: 
                        p_x, p_y, p_frame = v['history'][-15]
                        
                        virtual_dist = math.hypot(real_x - p_x, real_y - p_y)
                        t_sec = (frame_count - p_frame) / fps
                        
                        if t_sec > 0:
                            dist_meters = virtual_dist / SCALE
                            raw_spd = (dist_meters / t_sec) * 3.6
                            
                            if raw_spd < 150: 
                                if v['rt_speed'] == 0:
                                    v['rt_speed'] = int(raw_spd)
                                else:
                                    v['rt_speed'] = int((v['rt_speed'] * 0.85) + (raw_spd * 0.15))
                            
                    if len(v['history']) > 20:
                        v['history'].pop(0)

                    # ==========================================
                    # 5. THE CROSSING LOGIC (Line A & B)
                    # ==========================================
                    if v['frame_A'] is None and lineA and len(lineA) == 2:
                        if line_intersect(prev_pt, curr_pt, lineA[0], lineA[1]):
                            v['frame_A'] = frame_count

                    if v['frame_B'] is None and lineB and len(lineB) == 2:
                        if line_intersect(prev_pt, curr_pt, lineB[0], lineB[1]):
                            v['frame_B'] = frame_count
                            
                            # --- SPEED CALCULATION ---
                            if v['frame_A'] is not None:
                                time_sec = (v['frame_B'] - v['frame_A']) / fps
                                if time_sec > 0:
                                    dist_ab = float(config.get("distance_ab_meters", 15.0))
                                    v['speed'] = int((dist_ab / time_sec) * 3.6)
                            else:
                                # FAIL-SAFE: If it missed Line A, use its rolling real-time speed!
                                v['speed'] = v['rt_speed']
                                
                            # --- DATABASE SAVING & COUNTING ---
                            if not v['saved_db'] and v['final_class'] is not None and db_connection.is_connected():
                                is_viol = True if v['speed'] > SPEED_LIMIT_KMH else False
                                try:
                                    # Table 1: Volume Logs
                                    sql1 = """INSERT INTO traffic_volume_logs 
                                            (intersection_id, camera_id, vehicle_track_id, vehicle_type, speed_kmh, is_violation) 
                                            VALUES (%s, %s, %s, %s, %s, %s)"""
                                    db_cursor.execute(sql1, (intersection_id, camera_id, int(track_id), v['final_class'], v['speed'], int(is_viol)))
                                    log_id = db_cursor.lastrowid 
                                        
                                    # Table 2: Violation Logs
                                    if is_viol:
                                        # Save image with unique name
                                        img_name = f"viol_{intersection_id}_{camera_id}_{track_id}.jpg"
                                        img_path = f"static/violations/{img_name}"
                                        cv2.imwrite(img_path, frame) 
                                            
                                        sql2 = "INSERT INTO violation_logs (traffic_log_id, violation_type, image_path) VALUES (%s, %s, %s)"
                                        db_cursor.execute(sql2, (log_id, "Overspeeding", img_path))

                                    db_connection.commit()
                                    v['saved_db'] = True
                                    
                                    vehicle_counts[v['final_class']] += 1  

                                    # --- PERMANENTLY ADD TO QUEUE ---
                                    CURRENT_QUEUE[camera_id] += 1

                                    LIVE_STATS[camera_id][v['final_class']] += 1
                                    if is_viol:
                                        alert = f"⚠️ {camera_id.upper()} | ID:{track_id} | {v['speed']}km/h"
                                        LIVE_STATS["recent_violations"].insert(0, alert)
                                        if len(LIVE_STATS["recent_violations"]) > 5:
                                            LIVE_STATS["recent_violations"].pop()
                                except Exception as e:
                                    print(f"❌ DB Error: {e}")

                    # ==========================================
                    # 6. STOP LINE LOGIC (RED LIGHT CAMERA & QUEUE EXIT)
                    # ==========================================
                    if not v.get('crossed_stop', False) and stopLine and len(stopLine) == 2:
                        if line_intersect(prev_pt, curr_pt, stopLine[0], stopLine[1]):
                            v['crossed_stop'] = True 
                            
                            # --- PERMANENTLY REMOVE FROM QUEUE ---
                            if CURRENT_QUEUE[camera_id] > 0:
                                CURRENT_QUEUE[camera_id] -= 1
                            
                            if LIGHT_STATE.get(camera_id, "RED") == "RED":
                                if v['final_class'] is None:
                                    v['final_class'] = display_class
                                
                                if db_connection.is_connected():
                                    try:
                                        sql1 = "INSERT INTO traffic_volume_logs (intersection_id, camera_id, vehicle_track_id, vehicle_type, speed_kmh, is_violation) VALUES (%s, %s, %s, %s, %s, %s)"
                                        db_cursor.execute(sql1, (intersection_id, camera_id, int(track_id), v['final_class'], v['rt_speed'], 1))
                                        log_id = db_cursor.lastrowid 
                                        
                                        img_path = f"static/violations/redlight_{camera_id}_{track_id}.jpg"
                                        cv2.imwrite(img_path, frame) 
                                        
                                        sql2 = "INSERT INTO violation_logs (traffic_log_id, violation_type, image_path) VALUES (%s, %s, %s)"
                                        db_cursor.execute(sql2, (log_id, "Red Light", img_path))
                                        db_connection.commit()
                                        
                                        alert = f"🛑 {camera_id.upper()} | ID:{track_id} | Red Light!"
                                        LIVE_STATS["recent_violations"].insert(0, alert)
                                        if len(LIVE_STATS["recent_violations"]) > 5:
                                            LIVE_STATS["recent_violations"].pop()
                                            
                                    except Exception as e:
                                        print(f"❌ DB Error (Red Light): {e}")

                    v['prev_pt'] = curr_pt

                    # ==========================================
                    # 7. UI DRAWING
                    # ==========================================
                    cv2.circle(frame, curr_pt, 4, (0, 0, 255), -1) 
                    
                    # 1. After crossing Line B -> Show dynamic Real-Time Speed!
                    if v['speed'] > 0 and v['final_class'] is not None:
                        label = f"ID:{track_id} {display_class} | {v['rt_speed']} km/h" # <--- Display rt_speed here!
                        color = (0, 0, 255) if display_class in ['Ambulance', 'Fire_Truck'] else (0, 165, 255)
                    elif v['frame_A'] is not None:
                        label = f"ID:{track_id} {display_class} | Tracking..."
                        color = (255, 100, 0)
                    else:
                        label = f"ID:{track_id} {display_class}"
                        color = (0, 255, 0)
                        
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)

            # ==========================================
            # END OF FRAME LOGIC (Queue & Cleanup)
            # ==========================================
            LIVE_POSITIONS[camera_id] = current_frame_positions

            # 1. GARBAGE COLLECTOR
            active_ids = set(track_ids) if results[0].boxes.id is not None else set()
            for tid in list(tracked_vehicles.keys()):
                if tid not in active_ids:
                    tracked_vehicles[tid]['lost_count'] = tracked_vehicles[tid].get('lost_count', 0) + 1
                    if tracked_vehicles[tid]['lost_count'] > 30: 
                        del tracked_vehicles[tid]
                else:
                    tracked_vehicles[tid]['lost_count'] = 0

            stats = [f"{k}:{val}" for k, val in vehicle_counts.items() if val > 0]
            txt = " | ".join(stats) if stats else "Waiting for vehicles to cross Line B..."
            cv2.rectangle(frame, (10, 10), (790, 35), (0,0,0), -1)
            cv2.putText(frame, txt, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            
            # The Micro-Pause to prevent 4-camera thread deadlock!
            await asyncio.sleep(0.001)
            
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    except Exception as e:
        print(f"❌ FATAL ERROR IN VIDEO STREAM: {e}")
    finally:
        cap.release()

# --- THE UNIFIED TRAFFIC LIGHT ENGINE (Dual-Instance) ---
async def traffic_light_cycle(is_sandbox: bool):
    global MANUAL_OVERRIDE, SANDBOX_MANUAL_OVERRIDE, SANDBOX_LOGS, SANDBOX_INTERSECTION
    print(f"🚦 [{'SANDBOX' if is_sandbox else 'REAL-WORLD'}] Traffic Light Manager Started!")
    
    # Dynamically select which state variables to control
    lights = SANDBOX_LIGHT_STATE if is_sandbox else LIGHT_STATE
    queues = SANDBOX_QUEUES if is_sandbox else CURRENT_QUEUE
    prefix = "[SANDBOX]" if is_sandbox else "[REAL-WORLD]"

    # Helper to log events to the correct terminal
    def log_event(text, color_type):
        if is_sandbox:
            SANDBOX_LOGS.append({"text": text, "color": color_type})
            if len(SANDBOX_LOGS) > 20: SANDBOX_LOGS.pop(0)
        else:
            print(f"🚦 {prefix} {text}")

    async def run_transition(active_camera, green_time):
        global MANUAL_OVERRIDE, SANDBOX_MANUAL_OVERRIDE
        
        # 1. Solid Green Light (Interruptible by Preemption)
        lights[active_camera] = "GREEN"
        solid_green_time = max(0, green_time - 5) 
        
        for i in range(solid_green_time, 0, -1):
            override_val = SANDBOX_MANUAL_OVERRIDE if is_sandbox else MANUAL_OVERRIDE
            if override_val and override_val != active_camera:
                log_event(f"⚡ PREEMPTION TRIGGERED! Aborting {active_camera.upper()} solid green!", "text-red-400 font-bold")
                break # Abort solid green and proceed to yellow transition
            lights["timer"] = i + 5 
            await asyncio.sleep(1)

        # 2. Flashing Green Light (5 seconds - Smooth 10s warning sequence)
        for i in range(10, 0, -1):
            if i % 2 == 0: lights["timer"] = int(i / 2)
            if i % 2 == 0: lights[active_camera] = "GREEN"
            else: lights[active_camera] = "OFF"
            await asyncio.sleep(0.5)

        # 3. Yellow Light (2 seconds)
        lights[active_camera] = "YELLOW"
        for i in range(2, 0, -1):
            lights["timer"] = i
            await asyncio.sleep(1)
            
        # 4. All-Red Clearance (3 seconds)
        lights[active_camera] = "RED" 
        for i in range(3, 0, -1):
            lights["timer"] = i
            await asyncio.sleep(1)

    while True:
        # Build sequence dynamically based on the environment
        current_intersection = SANDBOX_INTERSECTION if is_sandbox else ACTIVE_INTERSECTION
        if current_intersection == "hang_tuah":
            sequence = [
                ("north", "south", "east", "west"), 
                ("east", "north", "south", "west"), 
                ("south", "north", "east", "west")
            ]
        else:
            sequence = [
                ("north", "south", "east", "west"), 
                ("east", "north", "south", "west"), 
                ("south", "north", "east", "west"), 
                ("west", "north", "south", "east")  
            ]

        for active_cam, cross1, cross2, cross3 in sequence:
            override_val = SANDBOX_MANUAL_OVERRIDE if is_sandbox else MANUAL_OVERRIDE
            
            # Preemption Jump Logic
            if override_val and override_val != active_cam:
                allowed = ["north", "east", "south"] if current_intersection == "hang_tuah" else ["north", "east", "south", "west"]
                if override_val in allowed:
                    log_event(f"🚨 Jumping sequence directly to {override_val.upper()}!", "text-red-400")
                    active_cam = override_val
                    cross_cams = [p for p in allowed if p != active_cam]
                    cross1, cross2 = cross_cams[0], cross_cams[1]
                    cross3 = cross_cams[2] if len(cross_cams) > 2 else "west"

            lights["current_phase"] = active_cam.upper()
            for cam in ["north", "south", "east", "west"]: lights[cam] = "RED"

            active_cars = queues.get(active_cam, 0)
            cross_cars = max(queues.get(cross1, 0), queues.get(cross2, 0), queues.get(cross3, 0) if cross3 in queues else 0)
            
            # Emergency inputs are isolated per environment.
            if is_sandbox:
                active_emergencies = SANDBOX_EMERGENCIES.get(active_cam, 0)
                cross_emergencies = (
                    SANDBOX_EMERGENCIES.get(cross1, 0) +
                    SANDBOX_EMERGENCIES.get(cross2, 0) +
                    SANDBOX_EMERGENCIES.get(cross3, 0)
                )
            else:
                active_emergencies = LIVE_STATS[active_cam]["Ambulance"] + LIVE_STATS[active_cam]["Fire_Truck"]
                cross_emergencies = (
                    LIVE_STATS[cross1]["Ambulance"] + LIVE_STATS[cross1]["Fire_Truck"] +
                    LIVE_STATS[cross2]["Ambulance"] + LIVE_STATS[cross2]["Fire_Truck"] +
                    LIVE_STATS[cross3]["Ambulance"] + LIVE_STATS[cross3]["Fire_Truck"]
                )

            # --- 1. EMERGENCY OVERRIDE & DEADLOCK PREVENTION (FIFO + 30s Timer) ---
            is_manual_active = (override_val == active_cam)
            is_manual_cross = (override_val in [cross1, cross2, cross3])

            # Calculate total emergency vehicles across ALL 4 roads
            if is_sandbox:
                total_emergencies = sum(SANDBOX_EMERGENCIES.values())
            else:
                total_emergencies = sum(
                    LIVE_STATS[cam]["Ambulance"] + LIVE_STATS[cam]["Fire_Truck"] 
                    for cam in ["north", "south", "east", "west"]
                )

            # RULE A: Active lane HAS an emergency (or manual force) -> Serve IMMEDIATELY!
            if active_emergencies > 0 or is_manual_active:
                # Reduce to 30s if multiple emergency vehicles are competing!
                emg_time = 30 if (total_emergencies > 1) else 60
                
                log_event(f"🚨 EMERGENCY ON {active_cam.upper()}! Forcing {emg_time}s Green Light! (Total Emg: {total_emergencies})", "text-red-400 font-bold")
                
                await run_transition(active_cam, emg_time)
                
                if is_manual_active:
                    if is_sandbox: SANDBOX_MANUAL_OVERRIDE = None
                    else: MANUAL_OVERRIDE = None
                continue

            # RULE B: Active lane has NO emergency, but a cross lane DOES -> Yield to cross lane!
            elif cross_emergencies > 0 or is_manual_cross:
                await asyncio.sleep(0.1) 
                continue

            # --- 2. NIGHT MODE ---
            if sum(queues.values()) == 0:
                lights["mode"] = "NIGHT_MODE"
                if active_cam == "north":
                    lights["north"] = "GREEN"
                    lights["timer"] = 99 
                    while sum(queues.values()) == 0 and (SANDBOX_MANUAL_OVERRIDE if is_sandbox else MANUAL_OVERRIDE) is None:
                        await asyncio.sleep(1)
                    
                    # FIX: Check if the manual override was for North itself!
                    current_override = SANDBOX_MANUAL_OVERRIDE if is_sandbox else MANUAL_OVERRIDE
                    if current_override == "north":
                        # North is ALREADY Green! Clear override and grant 60s WITHOUT turning red!
                        log_event(f"🚨 EMERGENCY ON NORTH! Extending Green Light for 60s!", "text-red-400 font-bold")
                        if is_sandbox: SANDBOX_MANUAL_OVERRIDE = None
                        else: MANUAL_OVERRIDE = None
                        await run_transition("north", 60)
                    else:
                        # Override was for another road (e.g. East). Safely transition North to Red!
                        await run_transition("north", 0) 
                else:
                    await asyncio.sleep(0.1)
                continue

            # --- SMART SKIP ---
            if active_cars == 0 and cross_cars > 0:
                lights["mode"] = "SMART_SKIP"
                log_event(f"Smart-skipping empty {active_cam.upper()} lane.", "text-gray-500 italic")
                await asyncio.sleep(0.1) 
                continue

            # --- FUZZY LOGIC ---
            lights["mode"] = "NORMAL_FUZZY"
            green_time = fuzzy_brain.calculate_green_time(active_cars, cross_cars)
            log_event(f"Phase {active_cam.upper()}: {active_cars} waiting. -> Green for {green_time}s", "text-green-400 font-bold")
            await run_transition(active_cam, green_time)
            
@app.on_event("startup")
async def startup_event():
    # We spawn TWO independent traffic light managers!
    asyncio.create_task(traffic_light_cycle(is_sandbox=False)) # Real Cameras
    asyncio.create_task(traffic_light_cycle(is_sandbox=True))  # Virtual Sandbox

# --- DYNAMIC ROUTES FOR MULTIPLE INTERSECTIONS & CAMERAS ---
@app.get("/video_feed/{intersection_id}/{camera_id}")
async def video_feed(intersection_id: str, camera_id: str): 
    # Pass both parameters directly to the video generator!
    return StreamingResponse(generate_frames(intersection_id, camera_id), media_type="multipart/x-mixed-replace; boundary=frame")

# --- ROUTE TO FETCH SAVED CONFIGURATION ---
@app.get("/api/get_config")
async def get_config():
    config = load_config()
    if config:
        return config
    # If no file exists yet, return an error so the frontend knows to use defaults
    return {"status": "empty"}

@app.post("/api/save_config")
async def save_config(config: MasterConfig):
    # Get raw JSON from frontend and save it
    config_data = config.model_dump() # Changed to model_dump() for Pydantic v2
    with open("camera_config.json", "w") as f: 
        json.dump(config_data, f, indent=4)
    return {"status": "success", "message": "Saved"}

# --- FIXED: NESTED BACKGROUND SAVING ---
@app.post("/api/save_background/{intersection_id}/{camera_id}")
async def save_background(intersection_id: str, camera_id: str, request: Request):
    data = await request.json()
    image_data = data.get("image")
    
    if image_data:
        encoded_data = image_data.split(',')[1]
        decoded_data = base64.b64decode(encoded_data)
        
        # Save permanently using BOTH intersection and camera name
        filepath = f"static/bg_{intersection_id}_{camera_id}.jpg"
        with open(filepath, "wb") as f:
            f.write(decoded_data)
            
        print(f"📸 Saved permanent background: {intersection_id.upper()} - {camera_id.upper()}")
        return {"status": "success"}
    return {"status": "error"}

@app.get("/api/live_stats")
async def get_live_stats():
    # Return both environments cleanly!
    return {
        "stats": LIVE_STATS,
        "queue": CURRENT_QUEUE,
        "lights": LIGHT_STATE,
        "sandbox_queue": SANDBOX_QUEUES,
        "sandbox_lights": SANDBOX_LIGHT_STATE,
        "sandbox_emergencies": SANDBOX_EMERGENCIES,
        "sandbox_intersection": SANDBOX_INTERSECTION,
        "sandbox_logs": SANDBOX_LOGS
    }

@app.post("/api/trigger_emergency/{camera_id}")
async def trigger_emergency(camera_id: str):
    global MANUAL_OVERRIDE
    
    # Toggle the override: If clicked again, it turns off
    if MANUAL_OVERRIDE == camera_id:
        MANUAL_OVERRIDE = None
        return {"status": "success", "message": f"Canceled override for {camera_id.upper()}"}
    
    # Set the override to the requested road
    MANUAL_OVERRIDE = camera_id
    return {"status": "success", "message": f"Emergency Override Triggered for {camera_id.upper()}!"}

# --- SANDBOX STATE API ---
class SandboxStatePayload(BaseModel):
    queues: Dict[str, int]
    emergencies: Dict[str, int] = {}


def reset_sandbox_state():
    global SANDBOX_MANUAL_OVERRIDE, SANDBOX_LOGS
    for cam in ["north", "south", "east", "west"]:
        SANDBOX_QUEUES[cam] = 0
        SANDBOX_EMERGENCIES[cam] = 0
        SANDBOX_LIGHT_STATE[cam] = "RED"
    SANDBOX_LIGHT_STATE["timer"] = 0
    SANDBOX_LIGHT_STATE["current_phase"] = "NS"
    SANDBOX_MANUAL_OVERRIDE = None
    SANDBOX_LOGS.clear()


@app.post("/api/sandbox/update_state")
async def update_sandbox_state(payload: SandboxStatePayload):
    for cam in ["north", "south", "east", "west"]:
        SANDBOX_QUEUES[cam] = max(0, int(payload.queues.get(cam, 0)))
        SANDBOX_EMERGENCIES[cam] = max(0, int(payload.emergencies.get(cam, 0)))
    return {
        "status": "success",
        "queue": SANDBOX_QUEUES,
        "emergencies": SANDBOX_EMERGENCIES
    }


@app.post("/api/sandbox/update_queues")
async def update_sandbox_queues(queues: dict):
    # Backward-compatible endpoint for older frontend clients.
    for cam in ["north", "south", "east", "west"]:
        if cam in queues:
            SANDBOX_QUEUES[cam] = max(0, int(queues[cam]))
    return {"status": "success", "queue": SANDBOX_QUEUES}


@app.post("/api/sandbox/reset")
async def sandbox_reset():
    reset_sandbox_state()
    return {"status": "success", "queue": SANDBOX_QUEUES, "lights": SANDBOX_LIGHT_STATE}

@app.post("/api/sandbox/trigger_emergency/{camera_id}")
async def sandbox_trigger_emergency(camera_id: str):
    global SANDBOX_MANUAL_OVERRIDE
    if camera_id not in ["north", "south", "east", "west"]:
        return {"status": "error", "message": "Invalid camera_id"}
    if SANDBOX_MANUAL_OVERRIDE == camera_id:
        SANDBOX_MANUAL_OVERRIDE = None
        return {"status": "success", "message": f"Canceled override for {camera_id.upper()}"}
    SANDBOX_MANUAL_OVERRIDE = camera_id
    return {"status": "success", "message": f"Emergency Override Triggered for {camera_id.upper()}!"}

@app.get("/api/live_positions")
async def get_live_positions():
    return LIVE_POSITIONS

# --- MASTER WEB ROUTE (Single Page App) ---
@app.get("/", response_class=HTMLResponse)
async def master_page(request: Request): 
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/set_active/{intersection_id}")
async def set_active_intersection(intersection_id: str):
    global ACTIVE_INTERSECTION, SANDBOX_INTERSECTION
    if intersection_id not in ["tun_perak", "hang_tuah"]:
        return {"status": "error", "message": "Invalid intersection_id"}
    ACTIVE_INTERSECTION = intersection_id
    SANDBOX_INTERSECTION = intersection_id
    reset_sandbox_state()
    
    # RESET all live stats and positions so the new intersection starts 100% fresh!
    for cam in ["north", "south", "east", "west"]:
        LIVE_STATS[cam] = {name: 0 for name in CLASS_NAMES.values()}
        CURRENT_QUEUE[cam] = 0
        LIVE_POSITIONS[cam] = [] # <-- FIX: Clear the 2D map coordinates instantly!
        
    LIVE_STATS["recent_violations"] = []
    
    print(f"🔄 SWITCHED ACTIVE INTERSECTION TO: {intersection_id.upper()}")
    return {"status": "success"}

# --- HISTORICAL ANALYTICS API ---
@app.get("/api/historical_stats")
async def get_historical_stats():
    if not db_connection.is_connected():
        return {"status": "error", "message": "DB disconnected"}
    try:
        # SQL query to calculate the average traffic volume for each hour of the day across the entire month
        query = """
            SELECT camera_id, summary_hour, SUM(total_volume) as total
            FROM hourly_traffic_summary
            GROUP BY camera_id, summary_hour
            ORDER BY summary_hour, camera_id
        """
        db_cursor.execute(query)
        rows = db_cursor.fetchall()
        
        # Structure the data cleanly for Chart.js [0, 0, ... 24 hours]
        stats = {cam: [0]*24 for cam in ["north", "south", "east", "west"]}
        for cam, hour, total in rows:
            if cam in stats:
                stats[cam][hour] = int(total)
                
        return stats
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- DIRECT SANDBOX FUZZY CALCULATION API (QA/DIAGNOSTIC USE) ---
class SandboxPayload(BaseModel):
    active_queue: int
    cross_queue: int


@app.post("/api/sandbox/calculate")
async def sandbox_calculate(payload: SandboxPayload):
    active_queue = max(0, int(payload.active_queue))
    cross_queue = max(0, int(payload.cross_queue))
    green_time = fuzzy_brain.calculate_green_time(active_queue, cross_queue)
    return {
        "active_queue": active_queue,
        "cross_queue": cross_queue,
        "green_time": green_time
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)