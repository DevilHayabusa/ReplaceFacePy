import cv2
import insightface
import os
import time
import threading
from insightface.app import FaceAnalysis

# Threaded Camera Class 
class CameraStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.ret, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

#Main App 
def start_threaded_face_swap():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_image_path = os.path.join(base_dir, "..", "assets", "image.jpg")
    model_path = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")

    
    if not os.path.exists(target_image_path):
        print(f"ERROR: Target image is MISSING at -> {target_image_path}")
        return
        
    if not os.path.exists(model_path):
        print(f"ERROR: AI Model is MISSING at -> {model_path}")
        return

    # Initialize Engine
    print("Loading AI Models... Please wait.")
    face_analyzer = FaceAnalysis(name='buffalo_l')
    face_analyzer.prepare(ctx_id=-1, det_size=(480, 480))
    face_swapper = insightface.model_zoo.get_model(model_path, download=False)

    target_image = cv2.imread(target_image_path)
    target_faces = face_analyzer.get(target_image)
    if not target_faces:
        print("CRITICAL ERROR: No face in target image.")
        return
    target_face_data = target_faces[0]

    # Start the threaded camera
    print("Starting camera stream...")
    video_stream = CameraStream(src=0).start()
    time.sleep(1.0) # Allow camera to warm up

    print("Session started. Press 'Q' to exit.")
    
    # State variables
    processed_frame = video_stream.read()
    is_processing = False

    def process_ai(frame_to_process):
        nonlocal processed_frame, is_processing
        
        # Detect and swap
        source_faces = face_analyzer.get(frame_to_process)
        result_frame = frame_to_process.copy()
        
        if len(source_faces) > 0:
            for source_face in source_faces:
                result_frame = face_swapper.get(result_frame, source_face, target_face_data, paste_back=True)
        
        processed_frame = result_frame
        is_processing = False

    # Main Loop 
    while True:
        current_frame = video_stream.read()
        
        # If AI is free, send the current frame to be processed in a background thread
        if not is_processing:
            is_processing = True
            threading.Thread(target=process_ai, args=(current_frame.copy(),), daemon=True).start()

        # Render the last available processed frame
        cv2.imshow('AI Live Swap - Pro Multithreading', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    video_stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_threaded_face_swap()