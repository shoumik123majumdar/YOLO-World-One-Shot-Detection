from ultralytics import YOLO

# Initialize a YOLO-World model
model = YOLO("yolov8l-world.pt")  # or choose yolov8m/l-world.pt

# Define custom classes with very specific descriptions
model.set_classes([
    "gray HVAC condenser unit",
])

# Save the model with the defined offline vocabulary
model.save("hvac_detector_yolov8l.pt")