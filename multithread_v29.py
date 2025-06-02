import cv2
import numpy as np
from ultralytics import YOLO
import time
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import struct
import socket
import threading

latest_d390 = None       # ตัวแปรที่ thread จะอัพเดท
d390_lock = threading.Lock()  # lock เพื่อกันชนกันระหว่าง thread กับ main loop



model = YOLO("125_300ep.pt")

center_off_gripper_x = 0.315
center_off_gripper_y = 0.27
# === PARAMETERS ===
real_world_width_m = 3.0
real_world_height_m = 6

# robot_offset_x_m = center_off_gripper_x + 0.70
# robot_offset_y_m = center_off_gripper_y + 1.81

robot_offset_x_m = 1.015
robot_offset_y_m = 2.23

# Test

# robot_offset_x_m = 1.90
# robot_offset_y_m = 2.31

center_field_x = real_world_width_m / 2   
center_field_y = real_world_height_m / 2 

center_x_robot = center_field_x - robot_offset_x_m
center_y_robot = center_field_y + robot_offset_y_m
plot_range_x = real_world_width_m + 2
plot_range_y = real_world_height_m + 4
points = []
field_mask_points = []
field_mask_ready = False
H = None
homography_done = False
confi = 0.5
BocciaBallSize = 150
display_scale = 1
camera_input = 0
scale_webcam_display = 1.2
last_jack_pos_sent = None
strategy = ""
last_sent_position = 0

x_sent = 0.0
y_sent = 0.0
strategy_flag = 0.0

last_strategy_flag_sent = None

miss_hit_count = 0

score_own = 0
score_opp = 0




# center_field_x = real_world_width_m / 2   




Sent_center_position = False

throw_logs = []   # list เก็บ log ผลการปา
throw_count = 0   # นับจำนวนครั้ง
prev_d390 = None  # ค่าก่อนหน้า

start_time = time.time()
canon_check_delay = 40  # วินาที

camera_y_offset_segments = []  # เก็บช่วง offset y เป็น [(start_y, end_y, offset), ...]

def get_segmented_y_offset(y_raw):
    """Return offset ตามช่วง y_raw ที่อยู่ในกล้อง"""
    for start_y, end_y, offset in camera_y_offset_segments:
        if start_y <= y_raw < end_y:
            return offset
    return 0.0  # default ไม่มี offset

#======= Server ==============

HOST = '127.0.0.1'
PORT = 9999

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)
print("Wait for MATLAB Connection")

conn, addr = server.accept()

def explain_boccia_reason(strategy_note, throw_count, min_own, min_opp, miss_hit_count, blast_needed):
    if "Forced Place" in strategy_note or "Forced Place after 2 misses" in strategy_note:
        return "เพราะพยายามตีแล้วพลาด 2 ครั้งติดต่อกัน จึงเปลี่ยนเป็นวางลูกแทน"
    elif "No catch-up possible" in strategy_note:
        return "เพราะคะแนนตามคู่ต่อสู้และลูกที่เหลือไม่พอแซง จึงเน้นวางลูก"
    elif "Jack Blast" in strategy_note:
        return "เพราะลูกคู่ต่อสู้ล้อมลูกแจ็ค จึงต้องตีเข้าเป้าเพื่อเปลี่ยนสถานการณ์"
    elif "Secure Lead" in strategy_note:
        return "เพราะเรานำคะแนนคู่ต่อสู้ จึงเลือกวางลูกเพื่อปิดเกม"
    elif "Game Secure" in strategy_note:
        return "เพราะฝ่ายตรงข้ามเหลือลูกน้อย จึงเน้นวางเพื่อควบคุมเกม"
    elif "Place/Block" in strategy_note:
        return "เพราะเรายังใกล้กว่า และต้องวางลูกเพื่อบังคู่ต่อสู้"
    elif "Final Hit" in strategy_note:
        return "ลูกสุดท้าย ต้องตีลูกคู่ต่อสู้เพื่อพลิกสถานการณ์"
    elif "Hit" in strategy_note:
        if min_opp < min_own:
            return "เพราะลูกคู่ต่อสู้ใกล้กว่า จึงเลือกตี"
        elif blast_needed:
            return "เพราะลูกคู่ต่อสู้ล้อมลูกแจ็ค จึงต้องตี"
        else:
            return "เลือกตีเพราะลูกคู่ต่อสู้อยู่ในตำแหน่งสำคัญ"
    elif "Place" in strategy_note:
        if min_own < min_opp:
            return "เพราะลูกเราใกล้กว่าคู่ต่อสู้ จึงเลือกวางลูก"
        else:
            return "เลือกวางลูกเพื่อรักษาตำแหน่ง"
    else:
        return "กลยุทธ์อัตโนมัติตามสถานการณ์"


def update_miss_hit_and_strategy(throw_count, strategy_flag, miss_hit_count, dist_list):
    """
    - ใช้กับลูก 3,4,5,6 (throw_count >= 2)
    - ถ้าตีไม่โดน 2 ครั้ง (ตีแล้วยังมี blocked) ให้เปลี่ยนไปวาง (strategy_flag=0)
    - รีเซ็ต miss_hit_count ถ้าโดน หรือเปลี่ยนไปวาง
    Returns: (strategy_flag, miss_hit_count)
    """

    print(throw_count, strategy_flag, miss_hit_count)
    if throw_count >= 2 and strategy_flag == 1.0:
        # ตรวจสอบว่ายังมี blocked อยู่มั้ย (คือยังตีไม่โดน)
        still_blocked = any(strat.lower() == "hit" for _, _, strat in dist_list)
        print(still_blocked)
        if still_blocked:
            miss_hit_count += 1
            print(f"[❌] ตีไม่โดน {miss_hit_count} ครั้ง")
        # else:
        #     miss_hit_count = 0  # โดนแล้ว รีเซ็ต
        
        # ถ้าพลาด 2 ครั้งติดกัน
        if miss_hit_count == 2:
            print("[⚠️] ครบ 2 ครั้ง ตีไม่โดน! เปลี่ยนกลยุทธ์เป็นวางลูก")
            strategy_flag = 0.0
            miss_hit_count = 0
            # miss_hit_count = 0  # รีเซ็ต
        
    # elif strategy_flag == 0.0:
    #     miss_hit_count = 0  # ทุกครั้งที่เป็น "วาง" รีเซ็ตนับใหม่
    return strategy_flag, miss_hit_count


def decide_boccia_strategy(
    throw_count, ball_distances, jack_pos, own_team, opponent_team,
    balls_per_team=6, lead_margin=0.2, miss_hit_count=0,
    score_own=0, score_opp=0, mode="competition"
):   
    
    boccia_balls = [(label, x, y) for (label, x, y) in ball_distances if not label.lower().startswith("white")]
    opp_balls = [(label, x, y) for (label, x, y) in ball_distances if label.startswith(opponent_team)]
    opp_dists = [np.linalg.norm([x - jack_pos[0], y - jack_pos[1]]) for _, x, y in opp_balls]

    if throw_count == 7:
        throw_count = 0
        print("Reset")
    if len(boccia_balls) == 0 or len(opp_balls) == 0:
        return jack_pos[0], jack_pos[1], 0.0, f"Place (Field Clear, Ball {throw_count+1})"
    if len(opp_dists) > 0 and min(opp_dists) > 0.5:
        return jack_pos[0], jack_pos[1], 0.0, f"Place (Opponent Far, Ball {throw_count+1})"
   
    balls_left = balls_per_team - throw_count
    score_diff = score_opp - score_own   # ถ้าคู่แข่งนำ
    if throw_count >= 3 and score_diff > 0 and score_diff >= balls_left:
        # แซงไม่ทันแล้ว ควรวางอย่างเดียว
        return jack_pos[0], jack_pos[1], 0.0, f"Place (No catch-up possible, score_diff={score_diff}, left={balls_left})"
    
    own_balls = [(label, x, y) for (label, x, y) in ball_distances if label.startswith(own_team)]
    opp_balls = [(label, x, y) for (label, x, y) in ball_distances if label.startswith(opponent_team)]
    dist_own = [np.linalg.norm([x - jack_pos[0], y - jack_pos[1]]) for label, x, y in own_balls]
    dist_opp = [np.linalg.norm([x - jack_pos[0], y - jack_pos[1]]) for label, x, y in opp_balls]
    min_own = min(dist_own) if dist_own else 999
    min_opp = min(dist_opp) if dist_opp else 999

    if mode == "test":
        # โหมดทดสอบ: ปาใส่ลูก Jack เสมอ
        return jack_pos[0], jack_pos[1], 1.0, "Test Mode: Aim Jack"

    # ถ้าพลาดติดกัน 2 ครั้ง บังคับวาง
    if throw_count >= 2 and miss_hit_count >= 2:
        return jack_pos[0], jack_pos[1], 0.0, "Forced Place"
    
    # --- [NEW LOGIC] --- เงื่อนไข blast: ลูกคู่แข่งล้อมแจ๊ค 2 ลูก
    blast_needed = False
    if len(opp_balls) >= 3:
        # จัดเรียง ball ทั้งหมดจากใกล้แจ๊คมาก่อน
        all_balls = [(label, x, y, np.linalg.norm([x-jack_pos[0], y-jack_pos[1]]))
                    for (label, x, y) in own_balls + opp_balls]
        all_balls_sorted = sorted(all_balls, key=lambda tup: tup[3])
        # มีลูกคู่แข่งอยู่ใน 3 อันดับแรกหรือไม่
        top3 = all_balls_sorted[:3]
        opp_close_count = sum([1 for (label, _, _, _) in top3 if label.startswith(opponent_team)])
        if opp_close_count == 3:
            blast_needed = True


    # --- ถ้าต้อง blast ให้ยิงไปที่แจ๊ค โดยใช้ flag=1
    if throw_count >= 2 and blast_needed and miss_hit_count != 2:
        return jack_pos[0], jack_pos[1], 1.0, f"Jack Blast (Ball {throw_count+1})"

    # ลูก 1-2: วางเสมอ
    if throw_count < 2:
        return jack_pos[0], jack_pos[1], 0.0, f"Place (Ball {throw_count+1})"
    # ลูก 3-4: วางถ้าเรายังใกล้กว่า, ตีถ้าคู่แข่งใกล้กว่า
    elif 2 <= throw_count < 6:
    # ===== [NEW LOGIC] =====
        if miss_hit_count >= 2:
            # พลาดติดกัน 2 ครั้ง ให้เปลี่ยนเป็น "วาง" ทันที
            return jack_pos[0], jack_pos[1], 0.0, f"Forced Place after 2 misses (Ball {throw_count+1})"
        # ===== [ORIGINAL LOGIC] =====
        if min_own <= min_opp:
            return jack_pos[0], jack_pos[1], 0.0, f"Place (Ball {throw_count+1})"
        else:
            idx = np.argmin(dist_opp)
            return opp_balls[idx][1], opp_balls[idx][2], 1.0, f"Hit {opp_balls[idx][0]} (Ball {throw_count+1})"
    # ลูก 5-6: ถ้าเรานำมาก ให้วาง block, ถ้าเป็นรองให้ตี หรือเบียดลูกคู่แข่ง
    else:
        opp_left = len(opp_balls)
        # --- NEW: ถ้าเรานำ ให้เน้นวาง/ป้องกัน อย่าเสี่ยงตี ---
        if score_own > score_opp:
            # วางปิดเกมไปเลยถ้าเรานำ และลูกเรายังใกล้ หรือไม่จำเป็นต้องตี
            return jack_pos[0], jack_pos[1], 0.0, f"Place (Secure Lead, Ball {throw_count+1})"
        # --- END NEW ---
        # if min_own < min_opp - lead_margin:
        if min_own < min_opp:
            if opp_left <= 2:
                return jack_pos[0], jack_pos[1], 0.0, f"Place (Game Secure: Opp {opp_left} balls left)"
            else:
                idx_own = np.argmin(dist_own)
                block_x, block_y = own_balls[idx_own][1], own_balls[idx_own][2]
                return block_x, block_y, 0.0, f"Place/Block (Ball {throw_count+1})"
        else:
            if opp_left > 0:
                idx = np.argmin(dist_opp)
                if throw_count == balls_per_team-1:
                    return opp_balls[idx][1], opp_balls[idx][2], 1.0, f"Final Hit {opp_balls[idx][0]}"
                else:
                    return opp_balls[idx][1], opp_balls[idx][2], 1.0, f"Hit {opp_balls[idx][0]} (Ball {throw_count+1})"
            else:
                return jack_pos[0], jack_pos[1], 0.0, f"Place (Opponent Out of Balls)"
            
    
        
        



def receive_d390_thread():
    global latest_d390
    while True:
        try:
            data = conn.recv(4)
            if data and len(data) == 4:
                val = struct.unpack('<i', data)[0]
                with d390_lock:
                    latest_d390 = val
                    
            # print(val)
        except Exception as e:
            print(f"[Recv Thread Error]: {e}")
            break

recv_thread = threading.Thread(target=receive_d390_thread, daemon=True)
recv_thread.start()

def update_action_plan_table(current_throw):
    # ลบข้อมูลเดิม
    for item in tree_action_plan.get_children():
        tree_action_plan.delete(item)
    # เพิ่มข้อความแต่ละลูก
    for i in range(6):  # ลูกที่ 1 ถึง 6
        if i < 2:
            action = "ปาเข้าใกล้ลูก Jack"
        elif i < 4:
            action = "ดูเชิง"
        else:
            action = "เปลี่ยนเกม"
        # ถ้าลูกนี้คือ current_throw ให้ใส่ ⭐
        display_action = f"⭐ {action}" if i == current_throw else action
        tree_action_plan.insert("", "end", values=(f"{i+1}", display_action))
    root.update()


# ==============================

# ฟังก์ชันสร้างตารางพร้อม Scrollbar
def create_treeview_with_scrollbar(parent, columns, headings, width_list=None, height=5):
    frame = tk.Frame(parent)
    tree = ttk.Treeview(frame, columns=columns, show='headings', height=height)
    for col, head in zip(columns, headings):
        tree.heading(col, text=head)
    # กำหนดความกว้างคอลัมน์ ถ้ามี width_list
    if width_list is not None:
        for col, w in zip(columns, width_list):
            tree.column(col, width=w, anchor="center")
    else:
        for col in columns:
            tree.column(col, width=70, anchor="center")  # default width
    tree.pack(side='left', fill='both', expand=True)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side='right', fill='y')
    frame.pack(fill="x", expand=True)
    return tree

def is_canon_logo_brightness(frame, threshold=50):
    """
    ตรวจสอบว่าเป็นภาพกล้องหลุดหรือ Canon Logo หรือไม่
    threshold: ค่า mean ความสว่าง (0-255) ต่ำกว่าค่านี้ถือว่าเป็นโลโก้ canon
    """
    # สามารถ crop เฉพาะส่วนกลางของภาพได้ ถ้าอยากละเอียดขึ้น
    h, w = frame.shape[:2]
    # crop เฉพาะ 60% ตรงกลาง (กันหลุดกรอบ)
    crop = frame[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    # debug: print(f"Mean brightness: {mean_intensity:.2f}")
    return mean_intensity < threshold





# === Tkinter ตาราง ===
root = tk.Tk()



boccia_mode = tk.StringVar(value="competition") 
manual_strategy_mode = tk.BooleanVar(value=False)
manual_strategy_flag = tk.IntVar(value=0)  # 0 = place, 1 = hit



root.title("พิกัดลูกบ็อกเซีย (Offset จาก Robot)")

score_frame = tk.Frame(root)
score_frame.pack(side="top", fill="x")
score_label = tk.Label(score_frame, text="Score", font=("TH SarabunPSK", 20))
score_label.pack()

# === Canvas+Scrollbar รวมหน้า ===
canvas = tk.Canvas(root, borderwidth=0, background="#f0f0f0")
scroll_frame = tk.Frame(canvas)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)

mode_frame = tk.LabelFrame(scroll_frame, text="โหมดการทำงาน", padx=10, pady=5)
mode_frame.pack(fill="x", pady=5)

tk.Radiobutton(mode_frame, text="โหมดแข่ง (อัตโนมัติ)", variable=boccia_mode, value="competition").pack(anchor="w")
tk.Radiobutton(mode_frame, text="โหมดทดสอบ (ทดสอบตีลูกขาว)", variable=boccia_mode, value="test").pack(anchor="w")
strategy_override_frame = tk.LabelFrame(scroll_frame, text="ควบคุมกลยุทธ์เอง", padx=10, pady=5)
strategy_override_frame.pack(fill="x", pady=5)

tk.Checkbutton(strategy_override_frame, text="บังคับกลยุทธ์เอง (Override)", variable=manual_strategy_mode).pack(anchor="w")
tk.Radiobutton(strategy_override_frame, text="วางลูก (Place)", variable=manual_strategy_flag, value=0).pack(anchor="w")
tk.Radiobutton(strategy_override_frame, text="ตีลูก (Hit)", variable=manual_strategy_flag, value=1).pack(anchor="w")


# ======= Robot Offset Manual Control =======
offset_frame = tk.LabelFrame(scroll_frame, text="แก้ไขตำแหน่ง Offset หุ่นยนต์", padx=10, pady=5)
offset_frame.pack(fill="x", pady=5)

tk.Label(offset_frame, text="Offset X (m):").grid(row=0, column=0, sticky="e")
tk.Label(offset_frame, text="Offset Y (m):").grid(row=1, column=0, sticky="e")

entry_offset_x = tk.Entry(offset_frame)
entry_offset_y = tk.Entry(offset_frame)
entry_offset_x.grid(row=0, column=1, padx=5)
entry_offset_y.grid(row=1, column=1, padx=5)

# ใส่ค่าเริ่มต้น
entry_offset_x.insert(0, str(robot_offset_x_m))
entry_offset_y.insert(0, str(robot_offset_y_m))


y_offset_frame = tk.LabelFrame(scroll_frame, text="Camera Y Offset ตามช่วง (แก้ค่าจากภาพ)", padx=10, pady=5)
y_offset_frame.pack(fill="x", pady=5)

offset_tree = ttk.Treeview(y_offset_frame, columns=("start", "end", "offset"), show='headings', height=5)
offset_tree.heading("start", text="Y Start (m)")
offset_tree.heading("end", text="Y End (m)")
offset_tree.heading("offset", text="Y Offset (m)")
offset_tree.pack()

def add_y_offset_entry():
    try:
        start = float(entry_start.get())
        end = float(entry_end.get())
        offset = float(entry_offset.get())
        offset_tree.insert("", "end", values=(f"{start:.2f}", f"{end:.2f}", f"{offset:.2f}"))
        update_y_offset_segments()
    except ValueError:
        print("[❌] ค่าที่กรอกไม่ถูกต้อง")

def delete_y_offset_entry():
    selected = offset_tree.selection()
    for item in selected:
        offset_tree.delete(item)
    update_y_offset_segments()

def update_y_offset_segments():
    global camera_y_offset_segments
    camera_y_offset_segments.clear()
    for row in offset_tree.get_children():
        vals = offset_tree.item(row)['values']
        start = float(vals[0])
        end = float(vals[1])
        offset = float(vals[2])
        camera_y_offset_segments.append((start, end, offset))

entry_start = tk.Entry(y_offset_frame, width=10)
entry_end = tk.Entry(y_offset_frame, width=10)
entry_offset = tk.Entry(y_offset_frame, width=10)
entry_start.pack(side='left', padx=2)
entry_end.pack(side='left', padx=2)
entry_offset.pack(side='left', padx=2)

tk.Button(y_offset_frame, text="เพิ่มช่วง", command=add_y_offset_entry).pack(side='left', padx=2)
tk.Button(y_offset_frame, text="ลบช่วงที่เลือก", command=delete_y_offset_entry).pack(side='left', padx=2)


def apply_offset():
    global robot_offset_x_m, robot_offset_y_m, center_x_robot, center_y_robot
    try:
        robot_offset_x_m = float(entry_offset_x.get())
        robot_offset_y_m = float(entry_offset_y.get())
        center_x_robot = center_field_x - robot_offset_x_m
        center_y_robot = center_field_y + robot_offset_y_m
        print(f"[✅] Updated Robot Offset: X = {robot_offset_x_m:.2f}, Y = {robot_offset_y_m:.2f}")
    except ValueError:
        print("[❌] ค่าที่กรอกไม่ถูกต้อง ต้องเป็นตัวเลข")

tk.Button(offset_frame, text="ปรับค่า Offset", command=apply_offset).grid(row=2, column=0, columnspan=2, pady=5)


scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)
window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

scroll_frame.bind("<Configure>", on_frame_configure)

# Mousewheel scroll
def _on_mousewheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")
canvas.bind_all("<MouseWheel>", _on_mousewheel)

root.title("พิกัดลูกบ็อกเซีย (Offset จาก Robot)")

tree_red = create_treeview_with_scrollbar(
    scroll_frame,
    columns=("label", "x", "y", "confidence"),
    headings=("Red", "X (m)", "Y (m)", "Conf."),
    width_list=[100, 80, 80, 80],   # <--- เพิ่ม
    height=5
)
tree_blue = create_treeview_with_scrollbar(
    scroll_frame,
    columns=("label", "x", "y", "confidence"),
    headings=("Blue", "X (m)", "Y (m)", "Conf."),
    width_list=[100, 80, 80, 80],
    height=5
)
tree_white = create_treeview_with_scrollbar(
    scroll_frame,
    columns=("label", "x", "y", "confidence"),
    headings=("Jack", "X (m)", "Y (m)", "Conf."),
    width_list=[100, 80, 80, 80],
    height=2
)
tree_dist = create_treeview_with_scrollbar(
    scroll_frame,
    columns=("label", "distance", "strategy"),
    headings=("Label", "Distance to Jack (m)", "Strategy"),
    width_list=[120, 150, 120],
    height=10
)
tree_throw_log = create_treeview_with_scrollbar(
    scroll_frame,
    columns=("seq", "x", "y", "strategy"),
    headings=("ครั้งที่", "X (m)", "Y (m)", "กลยุทธ์"),
    width_list=[70, 80, 80, 120],
    height=10
)
tree_action_plan = create_treeview_with_scrollbar(
    scroll_frame,
    columns=("throw_num", "action"),
    headings=("ลูกที่", "สิ่งที่จะเกิดขึ้น"),
    width_list=[60, 170],
    height=6
)
tree_send = create_treeview_with_scrollbar(
    scroll_frame,
    columns=("label", "x", "y", "strategy"),
    headings=("Label", "X (m)", "Y (m)", "Strategy"),
    width_list=[100, 80, 80, 80],
    height=3
)

root.geometry("800x600")  # ขนาดหน้าต่าง (ปรับได้ตามต้องการ)



root.update()

def calculate_score(dist_list, own_team):
    # กรองเฉพาะลูกที่ label ไม่ใช่ jack และ dist มีค่าเป็นตัวเลข
    filtered = [(label, float(dist)) for label, dist, _ in dist_list if not label.lower().startswith("jack") and dist.replace('.','',1).isdigit()]
    # แยกเป็นลูกฝั่งตัวเองและฝั่งตรงข้าม
    own_balls = [dist for label, dist in filtered if label.startswith(own_team)]
    opp_balls = [dist for label, dist in filtered if not label.startswith(own_team)]
    if not own_balls or not opp_balls:
        return 0, 0  # ถ้ายังไม่มีลูกทั้ง 2 ฝั่ง
    # ลูกที่ใกล้ jack ที่สุดของแต่ละฝั่ง
    nearest_own = min(own_balls) if own_balls else 999
    nearest_opp = min(opp_balls) if opp_balls else 999
    # กำหนดฝั่งที่ชนะรอบนี้
    if nearest_own < nearest_opp:
        # นับจำนวนลูกเราที่อยู่ใกล้ jack กว่าลูกแรกของคู่แข่ง
        points = sum(1 for d in own_balls if d < nearest_opp)
        return points, 0
    elif nearest_opp < nearest_own:
        points = sum(1 for d in opp_balls if d < nearest_own)
        return 0, points
    else:
        return 0, 0


def update_throw_log_table(logs):
    for item in tree_throw_log.get_children():
        tree_throw_log.delete(item)
    for entry in logs:
        tree_throw_log.insert("", "end", values=entry)
    root.update()

# === Figure ===
plt.ion()
fig_raw, ax_raw = plt.subplots(figsize=(5, 10))
fig_raw.canvas.manager.set_window_title("Robot Coordinate View (Offset)")
ax_raw.set_xlim(-robot_offset_x_m, plot_range_x - robot_offset_x_m)
ax_raw.set_ylim(-robot_offset_y_m, plot_range_y - robot_offset_y_m)
ax_raw.set_aspect('equal')
ax_raw.grid(True)
ax_raw.plot(0, 0, 'ko', label='Robot (0,0)')

sc_red_raw = ax_raw.scatter([], [], c='red', s=BocciaBallSize, label='Red')
sc_blue_raw = ax_raw.scatter([], [], c='blue', s=BocciaBallSize, label='Blue')
sc_white_raw = ax_raw.scatter([], [], c='white', edgecolors='black', s=BocciaBallSize, label='Jack')
ax_raw.legend()
ax_raw.plot(center_x_robot, center_y_robot, marker='*', markersize=15, color='green', label='Center Field')
ax_raw.legend()
ax_raw.set_title("Real World Coordinates (Offset)")
ax_raw.set_xlabel("X (m)")
ax_raw.set_ylabel("Y (m)")

dist_list = []

def update_tables(red_data, blue_data, white_data):
    for tree in [tree_red, tree_blue, tree_white]:
        for item in tree.get_children():
            tree.delete(item)
    for row in red_data:
        tree_red.insert("", "end", values=row)
    for row in blue_data:
        tree_blue.insert("", "end", values=row)
    for row in white_data:
        tree_white.insert("", "end", values=row)
    root.update()

def update_distance_table(data):
    for item in tree_dist.get_children():
        tree_dist.delete(item)
    for row in data:
        if len(row) == 2:
            row = (*row, "")
        tree_dist.insert("", "end", values=row)
    root.update()

def update_send_table(rows):
    for item in tree_send.get_children():
        tree_send.delete(item)
    for row in rows:
        tree_send.insert("", "end", values=row)
    root.update()

def webcam_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        scaled_x = int(x / scale_webcam_display)
        scaled_y = int(y / scale_webcam_display)
        points.append([scaled_x, scaled_y])

def warped_click(event, x, y, flags, param):
    global field_mask_points, field_mask_ready
    if event == cv2.EVENT_LBUTTONDOWN and len(field_mask_points) < 5:
        field_mask_points.append([x, y])
        print(f"[MASK] Selected: ({x}, {y})")
    if len(field_mask_points) == 5 and not field_mask_ready:
        field_mask_ready = True
        print("[MASK] Field mask ready.")

cap = cv2.VideoCapture(camera_input)
cv2.namedWindow("Webcam")
cv2.setMouseCallback("Webcam", webcam_click)

def send_position_once(x_sent, y_sent):
    global last_sent_position
    try:
        if last_sent_position is None or np.linalg.norm(np.array([x_sent, y_sent]) - np.array(last_sent_position)) > 0.001:

            if miss_hit_count >= 2:
                strategy_flag = 0.0

            data = struct.pack('<fff', float(x_sent), float(y_sent), float(strategy_flag))
            
            
            conn.sendall(data)
            
            
            print(f"[📤] ส่งพิกัดไป MATLAB: x={x_sent:.2f}, y={y_sent:.2f}")
            last_sent_position = (x_sent, y_sent)
    except Exception as e:
        print(f"[❌] ส่งข้อมูลไม่ได้: {e}")

# === เลือกสีฝั่งตัวเอง ===
team_color = tk.StringVar(value="Red")
color_select_window = tk.Toplevel(root)
color_select_window.title("เลือกสีทีมของคุณ")
tk.Label(color_select_window, text="เลือกสีฝั่งของคุณ:").pack(pady=10)
tk.Radiobutton(color_select_window, text="Red", variable=team_color, value="Red").pack()
tk.Radiobutton(color_select_window, text="Blue", variable=team_color, value="Blue").pack()
def confirm_color():
    color_select_window.destroy()
tk.Button(color_select_window, text="ยืนยัน", command=confirm_color).pack(pady=10)
root.wait_window(color_select_window)
print(f"[🎯] คุณเลือกทีม: {team_color.get()}")





while True:
    ret, frame = cap.read()

    # Fail Safe Function
    if time.time() - start_time > canon_check_delay:   
        if is_canon_logo_brightness(frame):
            print("[Canon Logo Detected] กล้องหลุด/ภาพดำ")
            # ส่งค่า fallback, วางลูกแทน, ส่งค่า x,y ล่าสุด
            strategy_flag = 0.0
            try:
                data = struct.pack('<fff', float(x_sent), float(y_sent), float(strategy_flag))
                conn.sendall(data)
                print(f"[📤][Fallback] ส่งพิกัดล่าสุด x={x_sent:.2f}, y={y_sent:.2f}, STRATEGY=Place (Fallback: Canon logo)")
            except Exception as e:
                print(f"[❌] ส่ง fallback ไม่ได้: {e}")
            # แสดงข้อความบนภาพ
            cv2.putText(frame, "Camera LOST! (Canon Logo Detected)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
            cv2.imshow("Webcam", frame)
            key = cv2.waitKey(20)
            if key == 27: break
            continue  # skip เฟรมนี้ ไม่ต้อง process ต่อ

    
    if not ret:
        break

    display = frame.copy()
    for pt in points:
        cv2.circle(display, tuple(pt), 2, (0, 0, 255), -1)

    if len(points) == 4 and not homography_done:
        pts_src = np.array(points, dtype=np.float32)
        dst_w = int(real_world_width_m * 100)
        dst_h = int(real_world_height_m * 100)
        pts_dst = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=np.float32)
        H, _ = cv2.findHomography(pts_src, pts_dst)
        homography_done = True
        print("[INFO] Homography ready.")

    if throw_count == 0:
        # เริ่มรอบใหม่ รีเซ็ต state
        miss_hit_count = 0
        strategy_flag = 0.0
    
    # print('Miss hit count',miss_hit_count)

    red_data, blue_data, white_data = [], [], []
    red_xy_raw, blue_xy_raw, white_xy_raw = [], [], []
    ball_distances = []
    jack_pos = None

    if homography_done and H is not None:
        warped = cv2.warpPerspective(frame, H, (int(real_world_width_m * 100), int(real_world_height_m * 100)))
        warped_h, warped_w = warped.shape[:2]

        # === วาดเส้นกริดแนว X และ Y ทุก 1 เมตร (100 px) ===
        grid_spacing_px = 100  # 100 pixels = 1 meter
        line_color = (0, 255, 0)  # สีเขียว
        line_thickness = 0.5

        scale_m_per_px_x = real_world_width_m / warped_w
        scale_m_per_px_y = real_world_height_m / warped_h

        px = int((x_sent + robot_offset_x_m) / scale_m_per_px_x)
        py = int((real_world_height_m - (y_sent - robot_offset_y_m)) / scale_m_per_px_y)    

        # === วาดเส้นแนว x = 0 (หลัง offset) ===
        x_world = robot_offset_x_m  # ตำแหน่ง x ที่ world หลัง offset → คือ 0
        y_vals = np.linspace(0, real_world_height_m, num=100)  # สร้างค่าตามแกน y

        pts_px = []
        for y in y_vals:
            x_px = int(x_world / scale_m_per_px_x)
            y_px = int((real_world_height_m - y) / scale_m_per_px_y)  # y นับจากล่าง
            pts_px.append((x_px, y_px))

        if field_mask_ready:
            mask_poly = np.array([field_mask_points], dtype=np.int32)
            mask = np.zeros(warped.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, mask_poly, 255)
            warped = cv2.bitwise_and(warped, warped, mask=mask)

        results = model(warped, conf=confi,verbose=False)[0]
        red_id = blue_id = white_id = 1

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            box_height = y2 - y1
            box_width = x2 - x1
            aspect_ratio = box_height / box_width if box_width > 0 else 1

            if aspect_ratio < 0.6:  # หากลูกดูแบนมาก (ใกล้ปลายสนาม)
                correction = int(box_height * 0.3)  # ปรับขึ้น 30% ของความสูง
                cy -= correction  # เลื่อนขึ้นบนในภาพ (Y ลดลง)

            cls_id = int(box.cls[0])
            label_name = results.names[cls_id]
            label_upper = label_name.capitalize()
            conf = float(box.conf[0])

            x_raw = cx * scale_m_per_px_x
            x_m = x_raw - robot_offset_x_m
            y_raw = (warped_h - cy) * scale_m_per_px_y
            # y_m = y_raw + robot_offset_y_m
            camera_y_offset = get_segmented_y_offset(y_raw)
            y_m = y_raw + robot_offset_y_m + camera_y_offset
            

            label_text = f"{label_upper} {red_id if label_upper=='Red' else blue_id if label_upper=='Blue' else white_id}"

            if label_upper == "Red":
                red_id += 1
                red_xy_raw.append((x_m, y_m))
                red_data.append((label_text, f"{x_m:.2f}", f"{y_m:.2f}", f"{conf:.2f}"))
                ball_distances.append((label_text, x_m, y_m))
            elif label_upper == "Blue":
                blue_id += 1
                blue_xy_raw.append((x_m, y_m))
                blue_data.append((label_text, f"{x_m:.2f}", f"{y_m:.2f}", f"{conf:.2f}"))
                ball_distances.append((label_text, x_m, y_m))
            elif label_upper == "White":
                white_id += 1
                white_xy_raw.append((x_m, y_m))
                white_data.append((label_text, f"{x_m:.2f}", f"{y_m:.2f}", f"{conf:.2f}"))
                jack_pos = (x_m, y_m)

            color = (0, 0, 255) if label_upper == "Red" else (255, 0, 0) if label_upper == "Blue" else (255, 255, 255)
            cv2.rectangle(warped, (x1, y1), (x2, y2), color, 1)
            cv2.circle(warped, (cx, cy), 3, color, -1)

        sc_red_raw.set_offsets(np.array(red_xy_raw) if red_xy_raw else np.empty((0, 2)))
        sc_blue_raw.set_offsets(np.array(blue_xy_raw) if blue_xy_raw else np.empty((0, 2)))
        sc_white_raw.set_offsets(np.array(white_xy_raw) if white_xy_raw else np.empty((0, 2)))

        ax_raw.cla()  # ล้างแกน plot ใหม่

        # 1. วาดสนาม ฯลฯ (เหมือนในตัวอย่าง)
        ax_raw.set_xlim(-robot_offset_x_m, plot_range_x - robot_offset_x_m)
        ax_raw.set_ylim(-robot_offset_y_m, plot_range_y - robot_offset_y_m)
        ax_raw.set_aspect('equal')
        ax_raw.grid(True)
        ax_raw.plot(0, 0, 'ko', label='Robot (0,0)')
        ax_raw.plot(center_x_robot, center_y_robot, marker='*', markersize=15, color='green', label='Center Field')

        if red_xy_raw:
            ax_raw.scatter(*zip(*red_xy_raw), c='red', s=BocciaBallSize, label='Red')
        if blue_xy_raw:
            ax_raw.scatter(*zip(*blue_xy_raw), c='blue', s=BocciaBallSize, label='Blue')
        if white_xy_raw:
            ax_raw.scatter(*zip(*white_xy_raw), c='white', edgecolors='black', s=BocciaBallSize, label='Jack')

        # วาดกรอบและข้อความกลยุทธ์
        if jack_pos is not None:
            # ... (copy ส่วนนี้มาตามตัวอย่างด้านบน)
            own_team = team_color.get()
            opponent_team = "Red" if own_team == "Blue" else "Blue"
            x_sent, y_sent, strategy_flag, strategy_note = decide_boccia_strategy(
            throw_count, ball_distances, jack_pos, own_team, opponent_team,
            balls_per_team=6, lead_margin=0.2, miss_hit_count=miss_hit_count,
            score_own=score_own, score_opp=score_opp ,
            mode=boccia_mode.get()
            )

            if manual_strategy_mode.get():
                strategy_flag = float(manual_strategy_flag.get())
                strategy_note = "Manual Override: Hit" if strategy_flag == 1.0 else "Manual Override: Place"
                print(f"[⚙️] ใช้ Manual Override -> กลยุทธ์: {'Hit' if strategy_flag else 'Place'}")
            color = 'yellow' if strategy_flag == 1 else 'lime'
            label_txt = "HIT" if strategy_flag == 1 else "PLACE"
            ax_raw.add_patch(plt.Rectangle(
                (x_sent - 0.125, y_sent - 0.125), 0.25, 0.25,
                fill=False, edgecolor=color, linewidth=3
            ))
            ax_raw.text(x_sent, y_sent+0.16, f"{label_txt}\n({x_sent:.2f},{y_sent:.2f})",
                        color=color, fontsize=12, ha='center', va='bottom', bbox=dict(facecolor='black', alpha=0.2, pad=2))
            ax_raw.text(0.2, plot_range_y-0.3, strategy_note, fontsize=13, color='deepskyblue', bbox=dict(facecolor='black', alpha=0.12))

        ax_raw.legend()
        ax_raw.set_title("Real World Coordinates (Offset)")
        ax_raw.set_xlabel("X (m)")
        ax_raw.set_ylabel("Y (m)")
        fig_raw.canvas.draw()
        fig_raw.canvas.flush_events()
        # --- จบส่วนนี้ ---

        # === นิยามข้อมูลทีมตัวเองและฝ่ายตรงข้าม ===
        own_team = team_color.get()  # "Red" หรือ "Blue"
        opponent_team = "Red" if own_team == "Blue" else "Blue"

        own_data = red_data if own_team == "Red" else blue_data
        opp_data = red_data if opponent_team == "Red" else blue_data

        dist_list.clear()  # 🧹 ล้างข้อมูลแผนในรอบก่อน เพื่อคำนวณใหม่ทุกเฟรม

        

        if jack_pos is not None:           
            for label, x, y in ball_distances:
                dist = np.sqrt((x - jack_pos[0])**2 + (y - jack_pos[1])**2)

                # วางกลยุทธ์
                if abs(x - jack_pos[0]) < 0.1 and y < jack_pos[1] and label.startswith(opponent_team):
                    row_strategy = "Hit"
                elif label.startswith(own_team):
                    row_strategy = "Protect"
                else:
                    row_strategy = ""

                dist_list.append((label, f"{dist:.2f}", row_strategy))

            update_distance_table(dist_list)

                        # --- Calculate Score ---
            score_own, score_opp = calculate_score(dist_list, own_team)
            score_text = f"Score - {own_team}: {score_own}, {opponent_team}: {score_opp}"
            # print(score_text)
            # หากต้องการแสดงใน Tkinter, เพิ่ม Label:
            if not hasattr(root, "score_label"):
                root.score_label = tk.Label(root, text=score_text, font=("TH SarabunPSK", 20))
                root.score_label.pack()
            else:
                score_label.config(text=score_text)





        if not hasattr(ax_raw, "field_plotted"):
            origin_px = np.array([[0, 0]], dtype=np.float32).reshape(-1, 1, 2)
            origin_transformed = cv2.perspectiveTransform(origin_px, H)[0][0]
            origin_m_x = origin_transformed[0] * scale_m_per_px_x
            origin_m_y = origin_transformed[1] * scale_m_per_px_y
            origin_offset_x = origin_m_x - robot_offset_x_m
            origin_offset_y = origin_m_y + robot_offset_y_m

            ax_raw.plot([origin_offset_x, origin_offset_x + 4], [origin_offset_y, origin_offset_y], 'k--')
            ax_raw.plot([origin_offset_x, origin_offset_x + 4], [origin_offset_y + 6, origin_offset_y + 6], 'k--')
            ax_raw.plot([origin_offset_x, origin_offset_x], [origin_offset_y, origin_offset_y + 6], 'k--')
            ax_raw.plot([origin_offset_x + 4, origin_offset_x + 4], [origin_offset_y, origin_offset_y + 6], 'k--')
            ax_raw.field_plotted = True

        fig_raw.canvas.draw()
        fig_raw.canvas.flush_events()
        center_label = "Center Field"
        center_x_str = f"{center_x_robot:.2f}"
        center_y_str = f"{center_y_robot:.2f}"
        center_conf_str = ""
        white_data.append((center_label, center_x_str, center_y_str, center_conf_str))
        update_tables(red_data, blue_data, white_data)

        warped_display = cv2.resize(warped, (warped.shape[1]*display_scale, warped.shape[0]*display_scale))
        cv2.imshow("Top View (Warped)", warped_display)
        cv2.setMouseCallback("Top View (Warped)", warped_click)

    display_to_show = cv2.resize(display, None, fx=scale_webcam_display, fy=scale_webcam_display)
    cv2.imshow("Webcam", display_to_show)

    root.update()
    
    time.sleep(0.5)

    key = cv2.waitKey(20)
    if key == ord('r'):
        points.clear()
        field_mask_points.clear()
        field_mask_ready = False
        H = None
        homography_done = False
        if hasattr(ax_raw, "field_plotted"):
            delattr(ax_raw, "field_plotted")
        cv2.destroyWindow("Top View (Warped)")

    elif key == ord('x'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"snapshot_raw_{timestamp}.jpg", frame)
        if homography_done and H is not None:
            warped_save = cv2.warpPerspective(frame, H, (int(real_world_width_m * 100), int(real_world_height_m * 100)))
            cv2.imwrite(f"snapshot_warped_{timestamp}.jpg", warped_save)
        print(f"[\U0001f4f8] Saved snapshot_raw_{timestamp}.jpg and snapshot_warped_{timestamp}.jpg (if homography ready)")
    elif key == 27:
        break
    elif key == ord('e'):
        points.clear()
        homography_done = False
        print("[🔄] ล้างจุด Homography แล้ว กรุณาคลิกใหม่อีกครั้ง (4 จุด)")


    blocked_balls = []
    for row in dist_list:
        label, dist, strat = row
        if strat.lower() == "hit":
            # หาพิกัดจาก ball_distances
            for label2, x, y in ball_distances:
                if label2 == label:
                    blocked_balls.append((label2, x, y))
                    break



    # ================Sent Data To MATLAB============
    
    if Sent_center_position == False:
        data = struct.pack('<ff', float(center_x_robot), float(center_y_robot))
        conn.sendall(data) ## Comment this when want to Maual
        Sent_center_position = True

    

    update_action_plan_table(throw_count)


    if jack_pos is not None:
        own_team = team_color.get()
        opponent_team = "Red" if own_team == "Blue" else "Blue"
        x_sent, y_sent, strategy_flag, strategy_note = decide_boccia_strategy(
        throw_count, ball_distances, jack_pos, own_team, opponent_team,
        balls_per_team=6, lead_margin=0.2, miss_hit_count=miss_hit_count)

        # --- คำนวณค่าสำหรับ reason (สำคัญมาก) ---
        own_balls = [(label, x, y) for (label, x, y) in ball_distances if label.startswith(own_team)]
        opp_balls = [(label, x, y) for (label, x, y) in ball_distances if label.startswith(opponent_team)]
        dist_own = [np.linalg.norm([x - jack_pos[0], y - jack_pos[1]]) for label, x, y in own_balls]
        dist_opp = [np.linalg.norm([x - jack_pos[0], y - jack_pos[1]]) for label, x, y in opp_balls]
        min_own = min(dist_own) if dist_own else 999
        min_opp = min(dist_opp) if dist_opp else 999
        blast_needed = False
        if len(opp_balls) >= 3:
            all_balls = [(label, x, y, np.linalg.norm([x-jack_pos[0], y-jack_pos[1]]))
                        for (label, x, y) in own_balls + opp_balls]
            all_balls_sorted = sorted(all_balls, key=lambda tup: tup[3])
            top3 = all_balls_sorted[:3]
            opp_close_count = sum([1 for (label, _, _, _) in top3 if label.startswith(opponent_team)])
            if opp_close_count == 3:
                blast_needed = True

        reason = explain_boccia_reason(strategy_note, throw_count, min_own, min_opp, miss_hit_count, blast_needed)
        print(f"แผนลูกที่ {throw_count+1}: {reason}")


        if homography_done and H is not None:
            px = int((x_sent + robot_offset_x_m) / scale_m_per_px_x)
            py = int((real_world_height_m - (y_sent - robot_offset_y_m)) / scale_m_per_px_y)
            color = (0, 255, 255) if strategy_flag == 1 else (0, 200, 0)
            label_txt = "HIT" if strategy_flag == 1 else "PLACE"
            cv2.rectangle(warped, (px-25, py-25), (px+25, py+25), color, 3)
            cv2.putText(warped, f"{label_txt} ({x_sent:.2f},{y_sent:.2f})",
                        (px-35, py-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            cv2.putText(warped, f"{strategy_note}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
            
            cv2.putText(warped, f"Mode: {boccia_mode.get()}", (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # print(f"[STRATEGY] {strategy_note} -> x={x_sent:.2f}, y={y_sent:.2f}, flag={strategy_flag}")
        # (ใช้ค่าที่ได้ไปส่งต่อให้ robot/Matlab และ update ตารางตามเดิม)

        try:
            # print(f"[DEBUG] last_jack_pos_sent={last_jack_pos_sent}, x_sent={x_sent}, y_sent={y_sent}, "
            # f"last_strategy_flag_sent={last_strategy_flag_sent}, strategy_flag={strategy_flag}")
            if (
                last_jack_pos_sent is None or
                np.linalg.norm(np.array([x_sent, y_sent]) - np.array(last_jack_pos_sent)) > 0.001 or
                strategy_flag != last_strategy_flag_sent
            ):
                
                if manual_strategy_mode.get():
                    strategy_flag = float(manual_strategy_flag.get())
                    strategy_note = "Manual Override: Hit" if strategy_flag == 1.0 else "Manual Override: Place"

                data = struct.pack('<fff', float(x_sent), float(y_sent), float(strategy_flag))
                conn.sendall(data)
                update_send_table([
                ("Last Sent", f"{x_sent:.2f}", f"{y_sent:.2f}", f"{strategy_flag:.0f}"),
                ("Center Field", f"{center_x_robot:.2f}", f"{center_y_robot:.2f}", "-"),
                ("Input PLC", str(current_d390), "", ""),      # << เพิ่มแถวนี้
                ("Miss Hit", str(miss_hit_count), "", ""),     # << เพิ่มแถวนี้
            ])
                last_jack_pos_sent = (x_sent, y_sent)
                last_strategy_flag_sent = strategy_flag
                print(f"[📤] กลยุทธ์: {'Hit' if strategy_flag == 1 else 'Place'} (flag={strategy_flag})")
        except Exception as e:
            print(f"[❌] ส่งข้อมูลไม่ได้: {e}")
    with d390_lock:
        current_d390 = latest_d390

    if prev_d390 is not None and current_d390 is not None:
        if current_d390 != prev_d390 and current_d390 > prev_d390:
            # Log โดยใช้ x_sent, y_sent, strategy_flag ที่อัปเดตล่าสุด
            this_strategy = "Hit" if strategy_flag == 1 else "Place"
            throw_count += 1
            throw_logs.append((
                throw_count,
                f"{x_sent:.2f}",
                f"{y_sent:.2f}",
                this_strategy
            ))
            print(f"[LOG] ปาครั้งที่ {throw_count}: x={x_sent:.2f}, y={y_sent:.2f}, strategy={this_strategy}")
            update_throw_log_table(throw_logs)
            strategy_flag, miss_hit_count = update_miss_hit_and_strategy(
            throw_count, strategy_flag, miss_hit_count, dist_list
        )

        if current_d390 == 0 and throw_logs:
            throw_logs.clear()
            throw_count = 0
            update_throw_log_table(throw_logs)
            print("[RESET] d390=0: Reset log ตารางแล้ว")
    prev_d390 = current_d390


    #=======================================================

cap.release()
cv2.destroyAllWindows()
root.destroy()
plt.close('all')
