import os
from flask import Flask, render_template, request, redirect, send_from_directory
import cv2
import numpy as np
import base64

from ultralytics import YOLO

app = Flask(__name__)

IMAGES_DIR = os.path.join('.', 'images')  # Directory containing input images
OUTPUT_DIR = os.path.join('.', 'output_images')  # Directory to save annotated images

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Initialize YOLO model
model_path = os.path.join('.', 'runs', 'detect', 'train43', 'weights', 'best.pt')
model = YOLO(model_path)

threshold = 0.5  # Detection threshold

# Function to apply RGB effect
def apply_rgb_effect(image):
    # Apply a color filter to create an RGB effect
    rgb_effect = cv2.applyColorMap(image, cv2.COLORMAP_RAINBOW)

    return rgb_effect

# Function to apply X-ray effect
def apply_xray_effect(image):
    # Get the original image dimensions
    height, width, channels = image.shape

    # Resize the image to the desired size while preserving the original format
    resized_img = cv2.resize(image, (800, int(800 * height / width)))

    # Convert the resized image to grayscale
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Apply contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Apply a sharpening filter to enhance edges
    sharpened = cv2.filter2D(enhanced_gray, -1, np.array([[-0.5, -0.5, -0.5], [-1, 11, -1], [-1, -1, -2]]))

    # Threshold the sharpened image to emphasize veins
    _, thresholded = cv2.threshold(sharpened, 220, 160, cv2.THRESH_BINARY)

    # Create the final X-ray effect by blending the thresholded image with the original
    xray_effect = cv2.divide(enhanced_gray, thresholded, scale=250.0)

    return xray_effect


@app.route('/')
def index():
    return render_template('index.html', message='')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        file_path = os.path.join(IMAGES_DIR, filename)
        file.save(file_path)
        
        # Read the uploaded image
        frame = cv2.imread(file_path)
        if frame is None:
            return "Error: Could not read the uploaded image."

        # Perform object detection
        results = model(frame)[0]

        # Check the day of incubation
        day = int(request.form['day'])
        if 1 <= day <= 21:
            status = "NORMAL"
            color = (0, 255, 0)  # Green font color
        elif day >= 23:
            status = "DELAYED"
            color = (0, 0, 255)  # Red font color
        else:
            status = "UNKNOWN"
            color = (255, 255, 255)  # Default font color

        # Draw bounding boxes and labels on the image
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                # Use a thinner rectangle
                thickness = 2
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness)

                # Use a different font type and make the text smaller
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6

                # Calculate the percentage of each category
                percentage = score * 100

                if class_id == 1:  # Fertile
                    label = f"Fertile: {percentage:.2f}%"
                elif class_id == 0:  # Infertile
                    label = f"Infertile: {percentage:.2f}%"
                elif class_id == 2:  # Hatching
                    label = f"Hatching: {percentage:.2f}%"
                elif class_id == 3:  # Hatched
                    label = f"Hatched: {percentage:.2f}%"
                else:
                    label = f"{status}: {percentage:.2f}%"  # New labels for NORMAL and DELAYED

                cv2.putText(frame, label, (int(x1), int(y1) - 5), font, font_scale, color, thickness, cv2.LINE_AA)

        # Save the annotated image
        output_image_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_image_path, frame)

        # Encode image to base64
        with open(output_image_path, "rb") as img_file:
            encoded_img = base64.b64encode(img_file.read()).decode('utf-8')

        return render_template('result.html', image=encoded_img, status=status)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)

# Add a new route to process the image again
@app.route('/process_image', methods=['POST'])
def process_image():
    # Get the base64 encoded image data from the form
    image_data = request.form['image_data']

    # Decode the base64 image data
    decoded_img = base64.b64decode(image_data)

    # Save the decoded image to a temporary file
    temp_img_path = os.path.join(OUTPUT_DIR, 'temp_image.png')
    with open(temp_img_path, 'wb') as img_file:
        img_file.write(decoded_img)

    # Load the temporary image for processing
    processed_img = cv2.imread(temp_img_path)

    # Apply image processing (e.g., apply RGB or X-ray effect)
    rgb_img = apply_rgb_effect(processed_img)
    xray_img = apply_xray_effect(processed_img)

    # Save the processed images
    rgb_img_path = os.path.join(OUTPUT_DIR, 'rgb_image.png')
    xray_img_path = os.path.join(OUTPUT_DIR, 'xray_image.png')
    cv2.imwrite(rgb_img_path, rgb_img)
    cv2.imwrite(xray_img_path, xray_img)

    # Encode the processed images to base64
    with open(rgb_img_path, "rb") as img_file:
        encoded_rgb_img = base64.b64encode(img_file.read()).decode('utf-8')

    with open(xray_img_path, "rb") as img_file:
        encoded_xray_img = base64.b64encode(img_file.read()).decode('utf-8')

    return render_template('processed.html', rgb_image=encoded_rgb_img, xray_image=encoded_xray_img)

if __name__ == '__main__':
    app.run(debug=True)
