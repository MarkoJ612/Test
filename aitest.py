from ultralytics import YOLO
import cv2
import random

# ========================
# 1. UCITAJ MODEL
# ========================
model = YOLO("yolov8n.pt")

# ========================
# 2. UCITAJ SLIKU
# ========================
image_path = "test3.jpg"  # promeni ako hoces svoju sliku

results = model(image_path)[0]

# ========================
# 3. FAKE V2X PODACI
# ========================
v2x = {
    "car_ahead_braking": random.choice([True, False]),
    "road_condition": random.choice(["dry", "wet"]),
    "traffic_light": random.choice(["green", "yellow", "red"])
}

print("\n--- V2X DATA ---")
print(v2x)

# ========================
# 4. DETEKCIJA + RISK
# ========================
risk = 0
detected_objects = []

for box in results.boxes:
    cls = int(box.cls[0])
    label = results.names[cls]
    detected_objects.append(label)

    # rizik po objektu
    if label == "person":
        risk += 60
    elif label in ["car", "bus", "truck"]:
        risk += 30
    elif label == "stop sign":
        risk += 40
    elif label == "traffic light":
        risk += 20

print("\n--- DETECTED OBJECTS ---")
print(detected_objects)

# ========================
# 5. V2X UTICAJ
# ========================
if v2x["car_ahead_braking"]:
    print("⚠️ Car ahead braking!")
    risk += 50

if v2x["road_condition"] == "wet":
    print("⚠️ Road is wet!")
    risk += 20

if v2x["traffic_light"] == "red":
    print("🚦 Red light!")
    risk += 40

# ========================
# 6. FINALNA ODLUKA
# ========================
print("\n--- TOTAL RISK ---")
print(risk)

if risk > 80:
    decision = "🟥 BRAKE IMMEDIATELY"
elif risk > 40:
    decision = "🟨 SLOW DOWN"
else:
    decision = "🟩 SAFE TO DRIVE"

print("\n--- FINAL DECISION ---")
print(decision)

# ========================
# 7. PRIKAZ SLIKE
# ========================
annotated = results.plot()
cv2.imshow("AI Driver View", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()