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









  
