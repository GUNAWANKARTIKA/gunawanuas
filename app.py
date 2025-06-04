from flask import Flask, render_template, request, send_file, url_for
import os
import cv2
import numpy as np
from datetime import datetime
import json

app = Flask(__name__)
# Define paths relative to the static folder for web access
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
HISTORY_FILE = 'static/history.json'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Function to apply image filters (effects)
def apply_filter(image_np, effect):
    """Applies a specified effect to a NumPy image array."""
    if image_np is None:
        return None # Handle case where image loading failed

    if effect == 'grayscale':
        # Convert to grayscale if not already
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            return cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        return image_np # Already grayscale or single channel
    elif effect == 'blur':
        return cv2.GaussianBlur(image_np, (15, 15), 0)
    elif effect == 'cartoon':
        # Ensure image is BGR for color processing in cartoon effect
        if len(image_np.shape) == 2: # If grayscale, convert to BGR
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

        # Convert to grayscale for edge detection for cartoon effect
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        # Adaptive thresholding for edges
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        # Bilateral filter for color smoothing
        color = cv2.bilateralFilter(image_np, 9, 300, 300)
        # Combine color and edges
        return cv2.bitwise_and(color, color, mask=edges)
    return image_np # 'none' effect, return original image

# Function to detect edges
def detect_edges(image_np, method):
    """Detects edges in a NumPy image array using the specified method."""
    if image_np is None:
        return None # Handle case where image loading failed

    # Ensure image is grayscale for edge detection algorithms
    if len(image_np.shape) == 3:
        img_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image_np # Already grayscale

    if method == 'canny':
        return cv2.Canny(img_gray, 100, 200)
    elif method == 'sobel':
        edges_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
        edges_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
        # Combine x and y gradients
        edges = cv2.magnitude(edges_x, edges_y)
        return cv2.convertScaleAbs(edges)
    elif method == 'laplacian':
        edges = cv2.Laplacian(img_gray, cv2.CV_64F)
        return cv2.convertScaleAbs(edges)
    return img_gray # Default, return grayscale if no method matches

# Function to save processing history
def save_to_history(original_rel_path, result_rel_path, method, effect):
    """Saves the processing details to a JSON history file."""
    entry = {
        "original": original_rel_path,
        "result": result_rel_path,
        "method": method,
        "effect": effect,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    data = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {HISTORY_FILE} is corrupted or empty. Starting new history.")
            data = []
    
    data.insert(0, entry) # Add new entry at the beginning
    # Keep only the last 20 entries to prevent the file from growing too large
    with open(HISTORY_FILE, "w") as f:
        json.dump(data[:20], f, indent=4)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if a file was uploaded
        if 'image' not in request.files:
            return "No image file part", 400
        
        file = request.files['image']
        
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return "No selected file", 400

        if file:
            # Generate unique filenames
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            original_filename = f"original_{timestamp}_{file.filename}"
            result_filename = f"result_{timestamp}_{file.filename}"

            # Define full paths for saving
            original_upload_path = os.path.join(UPLOAD_FOLDER, original_filename)
            result_save_path = os.path.join(RESULT_FOLDER, result_filename)

            # Save the original uploaded file
            file.save(original_upload_path)

            # Read the original image using OpenCV
            img_original = cv2.imread(original_upload_path)
            if img_original is None:
                return "Failed to load original image. Please check file format.", 500

            # Get method and effect from form
            method = request.form.get("method", "canny")
            effect = request.form.get("effect", "none")

            # Apply effect to the original image
            img_with_effect = apply_filter(img_original, effect)
            if img_with_effect is None:
                return "Failed to apply effect. Please check image data.", 500

            # Detect edges on the image with applied effect
            final_processed_image = detect_edges(img_with_effect, method)
            if final_processed_image is None:
                return "Failed to detect edges. Please check image data.", 500

            # Convert final_processed_image to BGR if it's grayscale for saving
            if len(final_processed_image.shape) == 2: # If it's grayscale
                final_processed_image = cv2.cvtColor(final_processed_image, cv2.COLOR_GRAY2BGR)

            # Save the final processed image
            cv2.imwrite(result_save_path, final_processed_image)

            # Get relative paths for history and template rendering
            original_rel_path = os.path.join('uploads', original_filename).replace('\\', '/')
            result_rel_path = os.path.join('results', result_filename).replace('\\', '/')

            # Save to history
            save_to_history(original_rel_path, result_rel_path, method, effect)

            # Render result page with correct relative paths
            return render_template("result.html",
                                   original_image=original_rel_path,
                                   result_image=result_rel_path,
                                   method=method)
    
    # For GET requests, render the main index page
    return render_template("index.html")

@app.route("/download/<filename>")
def download(filename):
    """Allows downloading of processed images."""
    file_path = os.path.join(RESULT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404

@app.route("/history")
def history():
    """Displays the history of processed images."""
    data = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {HISTORY_FILE} is corrupted or empty. Displaying empty history.")
            data = []
    return render_template("history.html", history=data)

if __name__ == "__main__":
    app.run(debug=True)