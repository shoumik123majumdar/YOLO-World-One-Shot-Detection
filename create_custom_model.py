from ultralytics import YOLO

# Initialize a YOLO-World model
model = YOLO("yolov8l-world.pt")  # or choose yolov8m/l-world.pt

# Define custom classes with very specific descriptions
model.set_classes([
    "outdoor air conditioning unit with metal housing",
    "gray HVAC condenser unit outside a house",
    "residential cooling system with ventilation grills",
    "external air conditioner with fan on top",
    "outdoor heat pump system"
])

# Save the model with the defined offline vocabulary
model.save("hvac_detector_yolov8l.pt")