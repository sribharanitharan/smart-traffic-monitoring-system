# 🏗️ Smart Traffic Monitoring System (STMS) — Architecture & Data Flow

This document outlines the end-to-end architectural design, mathematical foundations, data pipelines, and state transitions of the **Smart Traffic Monitoring System**.

---

## 1. High-Level End-to-End Architecture Flow

```mermaid
flowchart TD
    %% Source Node
    subgraph INGESTION ["1. Ingestion Layer"]
        A1[Live RTSP / CCTV Feed] --> B[Video Ingestion Engine]
        A2[Recorded MP4 / AVI File] --> B
        A3[USB Webcam Device Index 0] --> B
    end

    %% Preprocessing
    subgraph CONDITIONING ["2. Optical Conditioning Layer"]
        B --> C1[Frame Decimation & Resizing]
        C1 --> C2[CLAHE Contrast Enhancement <br/> L-Channel in CIE LAB Space]
        C2 --> C3[Gaussian Denoising Filter]
        C3 --> C4[Polygonal ROI Geometric Masking]
    end

    %% Neural Detection & Tracking
    subgraph INFERENCE ["3. Deep Learning & Association Layer"]
        C4 --> D1[YOLOv8 Detection Engine <br/> Anchor-Free Decoupled Head]
        D1 -->|Bounding Boxes, Classes, Confidences| D2[Confidence & NMS Filter]
        D2 --> D3[DeepSORT Multi-Object Tracker <br/> 8-State Kalman Filter + Re-ID]
        D3 -->|Persistent Track IDs & Centroids| D4[Motion History Ring Buffer]
    end

    %% Parallel Telemetry Processing
    subgraph ANALYTICS ["4. Multi-Branch Telemetry Engine"]
        D4 --> E1[Homography Speed Estimator <br/> 3x3 Projective Matrix + EMA]
        D4 --> E2[Spatial KDE Heatmap Engine <br/> 2D Gaussian Kernel Splatting]
        D4 --> E3[Flow Rate & Virtual Tripwire <br/> Directional Counting & Density]
        D4 --> E4[Incident Detection State Machine <br/> Stalled Vehicle & Wrong-Way Vector]
    end

    %% Persistence & Alerts
    subgraph PERSISTENCE ["5. Persistence & Event Dispatcher Layer"]
        E1 --> F1[Thread-Safe SQLite Engine <br/> traffic.db]
        E3 --> F1
        E4 -->|Hazard Triggers| F2[Real-Time Alert Dispatcher]
        F2 --> F1
        F1 --> F3[CSV Automated Telemetry Exporter]
    end

    %% Presentation & Operations
    subgraph PRESENTATION ["6. Presentation & Dashboard Layer"]
        E1 --> G1[OpenCV HUD Live Overlay Renderer]
        E2 --> G1
        E3 --> G1
        E4 --> G1
        G1 --> G2[Annotated Video Stream / MP4 Export]
        
        F1 --> H1[Streamlit Cyberpunk Operations HUD]
        H1 --> H2[Live Velocity & Flow Rate Charts]
        H1 --> H3[Congestion & Bottleneck Heatmap]
        H1 --> H4[Incident Audit Log & CSV Downloader]
    end

    %% Styling
    style INGESTION fill:#111927,stroke:#00e5ff,stroke-width:2px,color:#fff
    style CONDITIONING fill:#1a1c2e,stroke:#7928ca,stroke-width:2px,color:#fff
    style INFERENCE fill:#0d233a,stroke:#0070f3,stroke-width:2px,color:#fff
    style ANALYTICS fill:#1b2d20,stroke:#00ff88,stroke-width:2px,color:#fff
    style PERSISTENCE fill:#2d1b24,stroke:#ff0055,stroke-width:2px,color:#fff
    style PRESENTATION fill:#231a34,stroke:#f5a623,stroke-width:2px,color:#fff
```

---

## 2. Pipeline Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as Operator / Camera
    participant Main as Pipeline Coordinator (main.py)
    participant Pre as Preprocessor (modules/preprocessor.py)
    participant Det as YOLOv8 Detector (modules/detector.py)
    participant Trk as DeepSORT Tracker (modules/tracker.py)
    participant Ana as Telemetry & Speed (modules/)
    participant Inc as Incident Detector (modules/incident_detector.py)
    participant DB as SQLite DB (database/traffic_db.py)
    participant UI as Cyberpunk HUD (dashboard/app.py)

    User->>Main: Provide Video Stream / File
    loop Every Frame t
        Main->>Pre: process_frame(raw_frame)
        Pre-->>Main: conditioned_frame, roi_mask
        
        Main->>Det: detect(conditioned_frame)
        Det-->>Main: [bbox, confidence, class_id]
        
        Main->>Trk: update(detections, frame)
        Trk-->>Main: [track_id, bbox, centroid, trajectory]
        
        par Parallel Analytics
            Main->>Ana: compute_speed(track_id, trajectory, H_matrix)
            Ana-->>Main: speed_kmh, smoothed_speed
        and
            Main->>Ana: update_heatmap(centroid)
            Ana-->>Main: density_heatmap
        and
            Main->>Inc: evaluate_incident(track_id, velocity, trajectory)
            Inc-->>Main: incident_status (Normal / Stalled / Wrong-Way)
        end
        
        Main->>DB: log_frame_telemetry(metrics, alerts)
        Main->>Main: render_hud_annotations()
    end
    
    UI->>DB: Poll telemetry every 2 seconds
    DB-->>UI: Live KPIs, Incident history, Flow trends
    UI-->>User: Render Cyberpunk Operations Dashboard
```

---

## 3. Incident Detection State Machine (FSM)

The incident detection engine identifies anomalous vehicular behavior through a deterministic state machine:

```mermaid
stateDiagram-v2
    [*] --> Tracking_Active: Vehicle enters ROI & receives Track ID
    
    Tracking_Active --> Moving_Normal: Speed >= 5 km/h & Cos(theta) >= 0.0
    
    Moving_Normal --> Potential_Stall: Speed < 3 km/h for > 30 consecutive frames
    Potential_Stall --> Stalled_Incident: Stalled duration exceeds threshold (e.g. 5 sec)
    Stalled_Incident --> Moving_Normal: Speed recovers >= 5 km/h
    
    Moving_Normal --> Potential_Wrong_Way: Trajectory direction opposite to lane vector (Cos(theta) < -0.5)
    Potential_Wrong_Way --> Wrong_Way_Incident: Reversed displacement sustained over N frames
    
    Moving_Normal --> Exited_ROI: Vehicle leaves frame boundary
    Stalled_Incident --> Exited_ROI: Towed / Out of view
    Wrong_Way_Incident --> Exited_ROI: Vehicle leaves frame
    
    Exited_ROI --> [*]: Track memory decommissioned
```

---

## 4. Mathematical Model: Homography Speed Estimation

Monocular cameras suffer from perspective distortion where pixels near the horizon represent vastly greater real-world distances than pixels in the foreground. 

### Perspective Projection Matrix
A $3 \times 3$ planar homography matrix $\mathbf{H}$ maps image plane coordinates $(x_i, y_i, 1)^T$ to real-world ground-plane coordinates $(X_w, Y_w, 1)^T$:

$$\begin{bmatrix} X_w \\ Y_w \\ 1 \end{bmatrix} \sim \mathbf{H} \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix}$$

### Instantaneous and Smoothed Velocity
For track $k$ between frame $t - \Delta t$ and frame $t$:

$$\Delta D = \sqrt{(X_{w,t} - X_{w,t-\Delta t})^2 + (Y_{w,t} - Y_{w,t-\Delta t})^2} \quad [\text{meters}]$$

$$v_{\text{inst}} = \frac{\Delta D}{\Delta t / \text{FPS}} \times 3.6 \quad [\text{km/h}]$$

To eliminate high-frequency jitter, Exponential Moving Average (EMA) smoothing is applied:

$$v_{\text{smooth}}(t) = \alpha \cdot v_{\text{inst}}(t) + (1 - \alpha) \cdot v_{\text{smooth}}(t - 1), \quad \alpha \in [0.15, 0.30]$$

---

## 5. Spatial Density Heatmap (Gaussian KDE Splatting)

Density accumulation models vehicular congestion through continuous 2D Gaussian Kernel splatting coupled with an exponential temporal decay parameter:

$$D_{t}(x, y) = \gamma \cdot D_{t-1}(x, y) + \sum_{i=1}^{N_t} \exp\left( -\frac{(x - x_i)^2 + (y - y_i)^2}{2\sigma^2} \right)$$

* $\gamma = 0.96$: Temporal decay factor (prevents saturation over time)
* $\sigma = 15 \text{ px}$: Spatial spread kernel radius
* $(x_i, y_i)$: Ground contact centroid of vehicle $i$ at frame $t$

---

## 6. Database Entity-Relationship (ER) Schema

```mermaid
erDiagram
    VEHICLE_LOGS {
        int id PK
        int frame_number
        float timestamp
        int track_id
        string vehicle_class
        float speed_kmh
        float position_x
        float position_y
    }
    
    TRAFFIC_METRICS {
        int id PK
        float timestamp
        int total_count
        int current_density
        float flow_rate_vpm
        float average_speed_kmh
        string congestion_level
    }

    INCIDENT_ALERTS {
        int id PK
        float timestamp
        int track_id
        string incident_type
        float speed_kmh
        int location_x
        int location_y
        string severity
    }

    TRAFFIC_METRICS ||--o{ VEHICLE_LOGS : "aggregates"
    VEHICLE_LOGS ||--o{ INCIDENT_ALERTS : "generates"
```
