import cv2
import numpy as np

# Load the reference image once (make sure the path is correct)
reference_block = cv2.imread('wooden_block_reference.jpg', cv2.IMREAD_GRAYSCALE)
if reference_block is None:
    raise FileNotFoundError("Reference image 'wooden_block_reference.jpg' not found.")

w, h = reference_block.shape[::-1]

def detect_wooden_blocks(frame):
    # Convert the frame to grayscale for template matching
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform template matching using normalized correlation
    res = cv2.matchTemplate(gray, reference_block, cv2.TM_CCOEFF_NORMED)
    
    # Print the max matching score for debugging
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print("Max match value:", max_val)
    
    # Define threshold to determine a valid match (adjust this value as needed)
    threshold = 0.8  
    loc = np.where(res >= threshold)
    
    detected_blocks = []
    for pt in zip(*loc[::-1]):  # Switch columns and rows
        # Draw a green rectangle around the matched region
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
        cv2.putText(frame, "Wooden Block", (pt[0], pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        detected_blocks.append((pt[0], pt[1], w, h))
    
    # If no wooden block is detected, display a message for clarity
    if not detected_blocks:
        cv2.putText(frame, "No wooden block detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame, detected_blocks

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        output_frame, blocks = detect_wooden_blocks(frame)
        cv2.imshow("Wooden Block Detection", output_frame)
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
