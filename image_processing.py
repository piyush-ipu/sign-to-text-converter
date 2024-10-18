import numpy as np
import cv2
import os

minValue = 70

def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read the image {image_path}. Check the file path.")
        return None
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)

    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return res  

def process_images_in_folder(folder_path, output_folder_path):
    # Create the output folder if it doesn't exist
    if output_folder_path and not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Check if the file is an image
            image_path = os.path.join(folder_path, filename)  # Create full path
            processed_image = process_image(image_path)
            
            if processed_image is not None:
                # Determine the output path
                if output_folder_path:
                    processed_image_path = os.path.join(output_folder_path, f"processed_{filename}")
                else:
                    processed_image_path = os.path.join(folder_path, f"processed_{filename}")
                
                # Save the processed image
                cv2.imwrite(processed_image_path, processed_image)
                print(f"Processed image saved as: {processed_image_path}")

# Specify the folder path containing the images
folder_path = r"E:\Hackathon Project\Smart Attendance System\data\train\E"
# Specify the output folder path (change or set to None as needed)
output_folder_path = r"import numpy as np"
import cv2
import os

minValue = 70

def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read the image {image_path}. Check the file path.")
        return None
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)

    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return res  

def process_images_in_folder(folder_path, output_folder_path):
    # Create the output folder if it doesn't exist
    if output_folder_path and not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Check if the file is an image
            image_path = os.path.join(folder_path, filename)  # Create full path
            processed_image = process_image(image_path)
            
            if processed_image is not None:
                # Determine the output path
                if output_folder_path:
                    processed_image_path = os.path.join(output_folder_path, f"processed_{filename}")
                else:
                    processed_image_path = os.path.join(folder_path, f"processed_{filename}")
                
                # Save the processed image
                cv2.imwrite(processed_image_path, processed_image)
                print(f"Processed image saved as: {processed_image_path}")

# Specify the folder path containing the images
folder_path = r""
# Specify the output folder path (change or set to None as needed)
output_folder_path = r"D:\Sign languange project\data\Processed Image"

process_images_in_folder(folder_path, output_folder_path)

process_images_in_folder(folder_path, output_folder_path)