from ultralytics import YOLO

# Load the model
model = YOLO('models/best.pt')
# model = YOLO('yolov8x')

try:
    # Predict on the input video
    results = model.predict('input_videos/calPoly.mp4', save=True)
    
    # Print the first result
    print(results[0])
    print('=====================================')
    
    # Iterate through and print each box in the first result
    for box in results[0].boxes:
        print(box)

        
except Exception as e:
    print(f"An error occurred: {e}")
