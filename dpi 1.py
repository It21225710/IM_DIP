import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

class ImageProcessor:
    def _init_(self, root):
        self.root = root
        self.root.title("Image Processing Tool")

        self.original_image = None
        self.image = None
        self.input_image_label = tk.Label(self.root)
        self.input_image_label.grid(row=0, column=0, padx=10, pady=10, rowspan=9)

        self.transformed_image_label = tk.Label(self.root)
        self.transformed_image_label.grid(row=0, column=1, padx=10, pady=10, rowspan=9)

        self.load_button = tk.Button(self.root, text="Upload Image", command=self.load_image)
        self.load_button.grid(row=0, column=2, pady=10, padx=10)

        self.status_label = tk.Label(self.root, text="", padx=10)
        self.status_label.grid(row=10, column=0, columnspan=3, pady=10)

        self.process_label = tk.Label(self.root, text="Image Processing")
        self.process_label.grid(row=11, column=0, pady=10, padx=10)

        self.rotation_label = tk.Label(self.root, text="Rotation Angle:")
        self.rotation_label.grid(row=12, column=0, padx=10)
        self.rotation_entry = tk.Entry(self.root)
        self.rotation_entry.grid(row=12, column=1, pady=10)
        self.rotation_button = self.create_button("Rotate", self.rotate, 12, column=2)

        self.crop_label = tk.Label(self.root, text="Crop (x, y, w, h):")
        self.crop_label.grid(row=13, column=0, padx=10)
        self.crop_entry = tk.Entry(self.root)
        self.crop_entry.grid(row=13, column=1, pady=10)
        self.crop_button = self.create_button("Crop", self.crop, 13, column=2)

        self.black_white_button = self.create_button("Black & White", self.black_and_white, 14, column=0)
        self.grayscale_button = self.create_button("Grayscale", self.grayscale, 14, column=1)
        self.reset_button = self.create_button("Reset", self.reset, 14, column=2)
        
        # Adding horizontal and vertical flip buttons
        self.horizontal_flip_button = self.create_button("Horizontal Flip", self.horizontal_flip, 15, column=0)
        self.vertical_flip_button = self.create_button("Vertical Flip", self.vertical_flip, 15, column=1)

        # Adding the color inversion button
        self.invert_color_button = self.create_button("Invert Colors", self.invert_colors, 15, column=2)


        self.filter_label = tk.Label(self.root, text="Image Filters")
        self.filter_label.grid(row=16, column=0, pady=10, padx=10)

        self.sharpen_button = self.create_button("Sharpen", self.sharpen, 17, column=0)
        self.smooth_button = self.create_button("Smooth", self.smooth, 17, column=1)
        self.edge_button = self.create_button("Edge Detection", self.edge_detection, 17, column=2)
        self.emboss_button = self.create_button("Emboss", self.emboss, 17, column=3)

        self.brightness_label = tk.Label(self.root, text="Brightness:")
        self.brightness_label.grid(row=18, column=0, padx=10)
        self.brightness_entry = tk.Entry(self.root)
        self.brightness_entry.grid(row=18, column=1, pady=10)
        self.brightness_button = self.create_button("Adjust Brightness", self.adjust_brightness, 18, column=2)

        self.contrast_label = tk.Label(self.root, text="Contrast:")
        self.contrast_label.grid(row=19, column=0, padx=10)
        self.contrast_entry = tk.Entry(self.root)
        self.contrast_entry.grid(row=19, column=1, pady=10)
        self.contrast_button = self.create_button("Adjust Contrast", self.adjust_contrast, 19, column=2)





#lahiranga
    def create_button(self, text, command, row, column):

        button = tk.Button(self.root, text=text, command=command)

        button.grid(row=row, column=column, pady=10, padx=10)

        return button

    def load_image(self):

        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])

        if file_path:

            self.original_image = cv2.imread(file_path)

            self.image = self.original_image.copy()  # Make a copy for processing

            self.display_input_image()

            self.display_transformed_image()

    def display_input_image(self):

        if self.original_image is not None:

            image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

            image_pil = Image.fromarray(image_rgb)

            self.input_photo = ImageTk.PhotoImage(image=image_pil)

            self.input_image_label.configure(image=self.input_photo)

            self.input_image_label.image = self.input_photo

    def display_transformed_image(self):

        if self.image is not None:

            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

            image_pil = Image.fromarray(image_rgb)

            self.transformed_photo = ImageTk.PhotoImage(image=image_pil)

            self.transformed_image_label.configure(image=self.transformed_photo)

            self.transformed_image_label.image = self.transformed_photo

    def update_status(self, text):

        self.status_label.config(text=text)

    def reset(self):

        if self.original_image is not None:

            self.image = self.original_image.copy()

            self.display_transformed_image()

            self.update_status("Image Reset")

    def black_and_white(self):

        if self.image is not None:

            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            self.display_transformed_image()

            self.update_status("Applied Black & White")

    def grayscale(self):

        if self.image is not None:

            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            self.display_transformed_image()

            self.update_status("Applied Grayscale")

    def rotate(self):

        rotation_angle = self.rotation_entry.get()

        try:

            angle = float(rotation_angle)

            rows, cols, _ = self.image.shape

            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

            self.image = cv2.warpAffine(self.image, M, (cols, rows))

            self.display_transformed_image()

            self.update_status("Rotated Image")

        except ValueError:

            self.update_status("Invalid input. Please enter a valid angle.")

#Ravindu
    def crop(self):
        crop_coords = self.crop_entry.get()
        try:
            x, y, w, h = map(int, crop_coords.split(','))
            self.image = self.image[y:y + h, x:x + w]
            self.display_transformed_image()
            self.update_status("Cropped Image")
        except ValueError:
            self.update_status("Invalid input. Please enter valid coordinates.")

    def horizontal_flip(self):
        if self.image is not None:
            self.image = cv2.flip(self.image, 1)  # 1 for horizontal flip
            self.display_transformed_image()
            self.update_status("Applied Horizontal Flip")

    def vertical_flip(self):
        if self.image is not None:
            self.image = cv2.flip(self.image, 0)  # 0 for vertical flip
            self.display_transformed_image()
            self.update_status("Applied Vertical Flip")

    def invert_colors(self):
        if self.image is not None:
            self.image = cv2.bitwise_not(self.image)
            self.display_transformed_image()
            self.update_status("Colors Inverted")

    def sharpen(self):
        if self.image is not None:
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            self.image = cv2.filter2D(self.image, -1, kernel)
            self.display_transformed_image()
            self.update_status("Applied Sharpen")

    def smooth(self):
        if self.image is not None:
            self.image = cv2.GaussianBlur(self.image, (5, 5), 0)
            self.display_transformed_image()
            self.update_status("Applied Smooth")

    def edge_detection(self):
        if self.image is not None:
            self.image = cv2.Canny(self.image, 100, 200)
            self.display_transformed_image()
            self.update_status("Edge Detection Applied")

    def emboss(self):
        if self.image is not None:
            kernel = np.array([[-2, -1, 0],
                               [-1, 1, 1],
                               [0, 1, 2]])
            self.image = cv2.filter2D(self.image, -1, kernel)
            self.display_transformed_image()
            self.update_status("Applied Emboss")

    def adjust_brightness(self):
        brightness_value = self.brightness_entry.get()
        try:
            brightness = float(brightness_value)
            self.image = cv2.convertScaleAbs(self.image, alpha=1, beta=brightness)
            self.display_transformed_image()
            self.update_status(f"Brightness adjusted ({brightness})")
        except ValueError:
            self.update_status("Invalid input. Please enter a valid brightness value.")

    def adjust_contrast(self):
        contrast_value = self.contrast_entry.get()
        try:
            contrast = float(contrast_value)
            self.image = cv2.convertScaleAbs(self.image, alpha=contrast, beta=0)
            self.display_transformed_image()
            self.update_status(f"Contrast adjusted ({contrast})")
        except ValueError:
            self.update_status("Invalid input. Please enter a valid contrast value.")

#sujithra
def color_balance(self):
        if self.image is not None:
            self.image = apply_color_balance(self.image)
            self.display_transformed_image()
            self.update_status("Color Balance Applied")

    def show_histogram(self):
        if self.image is not None:
            self.plot_histogram(self.image)
            
    def plot_histogram(self, image):
        # Convert the image to grayscale if it's in color
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate the histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        
        # Plot the histogram using matplotlib
        plt.figure()
        plt.title("Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.hist(image.ravel(), 256, [0, 256])
        plt.show()

    def region_segmentation(self):
        if self.image is not None:
            self.image = perform_region_segmentation(self.image)
            self.display_transformed_image()
            self.update_status("Region-Based Segmentation Applied")

def apply_color_balance(image):
    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # Merge the CLAHE enhanced L channel with the original A and B channels
    balanced_lab_image = cv2.merge((cl, a_channel, b_channel))

    # Convert the LAB image back to BGR color space
    balanced_bgr_image = cv2.cvtColor(balanced_lab_image, cv2.COLOR_LAB2BGR)

    return balanced_bgr_image

def perform_region_segmentation(image):
    # Convert the image to grayscale if it's in color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to create a binary image
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original image
    segmented_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

    return segmented_image

if name == 'main':
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()

  
