# Embedded System for Voice Control based on TinyML

## Overview

This project implements a real-time, offline **Keyword Spotting (KWS)** system deployed on the **Arduino Nano 33 BLE Sense**. By leveraging **TinyML** (Tiny Machine Learning) and **TensorFlow Lite for Microcontrollers**, the system allows users to control peripheral devices (such as LEDs and Fans) using voice commands without requiring an internet connection.

The project demonstrates the capability of running complex deep learning models (DS-CNN) on resource-constrained microcontrollers, featuring advanced techniques like **continuous inference** and **semantic command parsing**.

## Key Features

  * **Efficient TinyML Model:**

      * Utilizes a **DS-CNN (Depthwise Separable Convolutional Neural Network)** architecture optimized for embedded devices.
      * **Int8 Quantization:** The model is quantized to 8-bit integers to minimize memory usage (Flash/RAM) while maintaining high accuracy.
      * **Fast Inference:** Achieves an inference time of approximately **35ms** on the Cortex-M4F processor.

  * **Robust Post-Processing:**

      * **Debouncing:** Filters out spurious or repeated detections to prevent erratic switching behavior.
      * **Thresholding:** Applies confidence thresholds to filter out background noise and uncertain predictions.

  * **Hardware Integration:**

      * **Audio Input:** Captures high-fidelity audio using the onboard MP34DT05 PDM microphone.
      * **Actuator Control:** Direct GPIO manipulation to control external relays, fans, or the onboard RGB LED based on recognized intent.

## 🛠️ Hardware & Software Stack

### Hardware

  * **Microcontroller:** [Arduino Nano 33 BLE Sense](https://docs.arduino.cc/hardware/nano-33-ble-sense) (nRF52840 - ARM Cortex-M4F @ 64MHz).
  * **Sensors:** Onboard Digital Microphone (PDM).
  * **Peripherals:** LEDs, Relays, DC Fans (connected via GPIO).

### Software & Tools

  * **Development Environment:** [VS Code](https://code.visualstudio.com/) with [PlatformIO IDE](https://platformio.org/).
  * **ML Platform:** [Edge Impulse](https://www.edgeimpulse.com/) (for data acquisition, MFCC feature extraction, and model training).
  * **Inference Engine:** TensorFlow Lite for Microcontrollers (TFLite Micro).
  * **Framework:** Arduino (mbed OS).

## 📂 Project Structure

```
├── include/                 # Header files for model and logic
├── lib/                     # Edge Impulse SDK & TFLite Micro library
│   └── Speech_Recognition_inferencing/
├── src/                     # Main source code
│   └── main.cpp             # Main logic (Setup, Sliding Window, Inference Loop)
├── platformio.ini           # PlatformIO project configuration (dependencies, baud rate)
└── README.md                # Project documentation
```

## Installation & Setup

1.  **Prerequisites:**

      * Install VS Code.
      * Install the PlatformIO Extension within VS Code.

2.  **Clone the Repository:**

    ```bash
    git clone https://github.com/huypopper-tech/TinyML-Voice-Control-Arduino.git
    cd TinyML-Voice-Control-Arduino
    ```

3.  **Open in PlatformIO:**

      * Open VS Code.
      * Select "File" \> "Open Folder" and choose the cloned directory.
      * PlatformIO will automatically initialize and download the necessary toolchains (Nordic nRF52).

4.  **Build and Upload:**

      * Connect your Arduino Nano 33 BLE Sense via USB.
      * Click the **PlatformIO: Upload** button (right arrow icon) in the bottom status bar.

5.  **Monitor Output:**

      * Click the **PlatformIO: Serial Monitor** button (plug icon).
      * Set the baud rate to `115200` if not automatically detected.

## Usage Example

Once the system is running, the Serial Monitor will display the initialization status. You can then issue voice commands:

  * **Command:** *"Turn on LED"*

      * **Log:** `Predictions: LED: 0.98, ON: 0.95 -> Action: LED ON`
      * **Result:** The Red LED lights up.

  * **Command:** *"Fan off"*

      * **Log:** `Predictions: FAN: 0.92, OFF: 0.99 -> Action: FAN OFF`
      * **Result:** The Fan stops.

## Performance Metrics

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Model Architecture** | DS-CNN | Depthwise Separable CNN |
| **Input Features** | MFCC | Mel-Frequency Cepstral Coefficients |
| **Quantization** | Int8 | Optimized for MCU |
| **Flash Usage** | \~81 KB | Model weights + Program code |
| **RAM Usage** | \~22.8 KB | Tensor Arena + Audio Buffers |
| **Inference Time** | \~35 ms | Per window |


-----

*This project is part of a scientific research report on Embedded Systems and TinyML.*
