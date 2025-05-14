
# AR Glasses 

This project uses **OpenCV** and **MediaPipe** to overlay virtual glasses onto a user's face in real-time using a webcam. The glasses are dynamically rotated based on the angle between the eyes, making the overlay look natural. Users can cycle through different glasses images by pressing 'd' (next) and 'a' (previous).

## Requirements

### Hardware
- A system with a **compatible camera** (webcam or external camera).
- A **GPU** (Optional, for faster processing and TensorFlow GPU support).

### Software
- **Python 3.7+**
- **TensorFlow GPU** (if you want to run it on the GPU)
- **OpenCV** (for image and video processing)
- **MediaPipe** (for face mesh detection)

### Dependencies
Install the required Python libraries:

```bash
pip install opencv-python mediapipe numpy tensorflow-gpu
````

### Optional: GPU Support (for TensorFlow)

If you want to use GPU acceleration, make sure you have the following installed:

1. **NVIDIA GPU** with CUDA support.
2. **CUDA Toolkit** and **CuDNN** installed.

Install TensorFlow GPU by running:

```bash
pip install tensorflow-gpu
```

## How to Run

1. Clone or download the repository.
2. Make sure you have a webcam connected to your system.
3. Place your transparent glasses PNG files in the `glasses/` directory.
4. Run the script:

```bash
python3 test.py
```

The script will open a webcam window where it will detect your face in real-time. The glasses will be overlaid on your face based on the eye landmarks, and you can cycle through glasses by pressing the following keys:

* `d`: Next glasses
* `a`: Previous glasses
* `ESC`: Exit the program

## Script Explanation

1. **Face Mesh Detection**:

   * The script uses **MediaPipe** to detect and extract the landmarks of a person's face.
   * The left and right eye landmarks are used to calculate the rotation angle of the glasses.

2. **Overlaying Glasses**:

   * Transparent PNG glasses images are loaded and resized to fit the face's dimensions.
   * The glasses are rotated based on the angle between the eyes and overlaid on the video frame.

3. **Key Bindings**:

   * Press `a` to cycle backward through glasses.
   * Press `d` to cycle forward through glasses.
   * Press `ESC` to exit the application.

## How to Add More Glasses

1. Place your glasses PNG files in the `glasses/` directory.
2. The script automatically detects new glasses and cycles through them.

### Example Glasses Images

Make sure that your glasses images:

* Have **transparent backgrounds**.
* Are in **PNG format**.
* The image size should fit within the dimensions of a person's face.

## Troubleshooting

### 1. If the glasses are not aligning well:

* Ensure that the **eye landmarks** are being detected properly.
* Check the alignment and size of the glasses relative to the face size.

### 2. If the webcam is not detected:

* Make sure the camera is properly connected and accessible.
* Test the camera using simple OpenCV code (`cv2.VideoCapture(0)`).

### 3. If TensorFlow is not using GPU:

Ensure that you have installed `tensorflow-gpu` and the necessary **CUDA** and **CuDNN** versions are compatible with TensorFlow.

## License

This project is open source and available under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

### Project by: **Gaurav Pandit**

For questions or suggestions, feel free to reach out to me on [LinkedIn](https://www.linkedin.com/in/gaurav-pandit/).

```

### Key Sections:
- **Project Description**: Explains the purpose of the project.
- **Requirements**: Lists the hardware and software requirements for running the project.
- **Dependencies**: Python libraries needed for the project.
- **How to Run**: Instructions for setting up and running the project.
- **Script Explanation**: Describes how the script works and what each part does.
- **How to Add More Glasses**: Instructions for adding new glasses images.
- **Troubleshooting**: Tips for common issues.
- **License**: Open-source license info.

You can copy-paste this into a `README.md` file in your project directory. Let me know if you need any further adjustments!
```
