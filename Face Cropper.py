import cv2
from ultralytics import solutions
import os

# --- Configuration ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream. Please check camera index, drivers, or if camera is in use.")
    exit()

# --- Initialize ObjectCropper ---
objectcropper = solutions.ObjectCropper(
    show=False,  # <--- IMPORTANT CHANGE: Set show to False
    model="best.pt",
    classes=[0],
    conf=0.5,
    crop_dir=r"E:\Buffer\Applied data science and AI specialization\NCL open cv tasks\Fine tune yolo8n\image database",
)
print(rf"ObjectCropper initialized. Cropped faces will be saved to E:\Buffer\Applied data science and AI specialization\NCL open cv tasks\Fine tune yolo8n\image database")

# --- Process Video Frames ---
frame_count = 0
while cap.isOpened():
    success, im0 = cap.read() # Read a frame from the video
    if not success:
        print("Video frame is empty or processing is complete. Exiting.")
        break # Exit loop if no more frames or error

    frame_count += 1
    # Process the frame with ObjectCropper.
    # When show=False, objectcropper(im0) returns a SolutionResults object.
    solution_results = objectcropper(im0)

    # --- Access the annotated image ---
    # The annotated image is typically available through the .plot_im attribute
    # of the SolutionResults object for many Ultralytics solutions.
    annotated_frame = solution_results.plot_im

    # --- Manual Display ---
    cv2.imshow('Face Detection and Cropper', annotated_frame) # Use the annotated frame for display

    # To gracefully exit the display window:
    if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit the display
        print(" 'q' pressed, stopping video processing.")
        break

# --- Release Resources ---
cap.release() # Release the video capture object
cv2.destroyAllWindows() # Close all OpenCV windows
print("Video processing complete. All resources released.")