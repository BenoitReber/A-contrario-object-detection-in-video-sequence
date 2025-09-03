# A-Contrario Object Detection in Video Sequences

This project implements an **a-contrario approach for object detection** in video sequences, based on the article by M. Ammar and S. Le Hégarat-Mascle. The method is designed to find objects without needing manual thresholds, by identifying statistically significant clusters of anomalous pixels.

The method is highly dependent on the quality of the background model ($B_t$). As shown in the experiments, a sudden global change (e.g., in lighting) can cause the system to fail, highlighting the importance of a robust model.

---

## Key Features of the method

* **Threshold-Free**: Uses the Number of False Alarms (NFA) principle to avoid manual parameter tuning.
* **Two-Stage Detection**:
    1.  **Pixel-Level**: Identifies individual pixels with significant change from a background model.
    2.  **Object-Level**: Groups these "object pixels" into rectangular zones with high statistical density.
* **Flexible**: The choice of background model allows the system to be adapted for different tasks, such as **intruder detection** or **motion detection**.

---

## Structure of the repository

* `algos.py`: The core Python implementation of the a-contrario detection algorithms.
* `experiments.py`: Scripts to test the method in different scenarios.
* `report.pdf`: A detailed report explaining the theory, models, and experimental results.

---
