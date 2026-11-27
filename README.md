# 🚦 Smart Intersection Camera System for Urban Traffic

An AI-powered urban traffic monitoring, violation detection, and adaptive traffic signal control system built with **YOLOv8**, **FastAPI**, **OpenCV**, **MySQL**, and **Fuzzy Logic (`scikit-fuzzy`)**.

## ✨ Key Features
* **Multi-Camera Computer Vision (`YOLOv8` + `ByteTrack`):** Real-time detection, tracking, and 8-class vehicle classification (`Car`, `Motorcycle`, `Van`, `Bus`, `Truck`, `Bicycle`, `Ambulance`, `Fire_Truck`) with majority-voting stabilization.
* **Perspective-Corrected Speed Estimation:** Dual-line virtual speed trap (`Line A` $\rightarrow$ `Line B`) combined with Bird's-Eye View homography transformation (`cv2.getPerspectiveTransform`) for real-time speed estimation.
* **Automated Violation Logging:** Captures snapshots and logs overspeeding and red-light violations to MySQL (`traffic_volume_logs` & `violation_logs`).
* **Malaysian 4-Phase & 3-Phase Fuzzy Traffic Brain:** Adaptive signal timing using `scikit-fuzzy` with Smart Skip (skips empty lanes), Night Mode (resting main green), 5s Flashing Green + 2s Yellow + 3s All-Red clearance, and Emergency Preemption.
* **Unified Single-Page Command Center:**
  * **Live Matrix:** 4-camera streaming grid + Chart.js live volume analytics.
  * **Digital Twin & AI Terminal:** Top-down 2D real-time projection, manual emergency override, and 30-day historical hourly analytics.
  * **Calibration Hub:** Interactive HTML5 Canvas for configuring tracking masks, calibration boxes, speed lines, and stop lines per intersection.
  * **SIL Sandbox Simulator:** Independent 2D kinematic traffic physics simulator to stress-test fuzzy logic and multi-emergency preemption.

## 🛠️ Tech Stack
* **Backend:** Python, FastAPI, Uvicorn, OpenCV, Ultralytics YOLOv8, `scikit-fuzzy`
* **Frontend:** HTML5 Canvas, TailwindCSS, Chart.js, Vanilla JS
* **Database:** MySQL 8.0
* **Closed-Loop Validation:** Eclipse SUMO (`traci`)

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/smart-intersection-traffic-system.git
cd smart-intersection-traffic-system