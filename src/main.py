"""Main Streamlit application for face detection."""
import streamlit as st
import cv2
import numpy as np
import os
import time
import logging
from typing import Dict, Any, Optional
from PIL import Image
from numpy.typing import NDArray

from .detector import FaceDetector
from .config import Config
from .utils.visualization import draw_detections, draw_facial_features

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FaceDetectionApp:
    """Main application class for face detection UI."""
    
    def __init__(self) -> None:
        """Initialize the face detection application."""
        self.config = Config()
        self.detector = FaceDetector(self.config)
        self.original_image: Optional[NDArray[np.uint8]] = None
        self.processed_image: Optional[NDArray[np.uint8]] = None
        
        logger.info("Face Detection App initialized")

        # Initialize session state for webcam and application mode
        if "webcam_active" not in st.session_state:
            st.session_state.webcam_active = False

        if "app_mode" not in st.session_state:
            st.session_state.app_mode = "image"  # 'image', 'webcam', or 'video'

        self.setup_gui()

    def setup_gui(self) -> None:
        """Setup the Streamlit GUI components"""
        st.title("Face Detection Application")

        # Sidebar for controls
        st.sidebar.header("Application Mode")

        # Mode selection buttons - three columns
        col1, col2, col3 = st.sidebar.columns(3)

        with col1:
            if st.button(
                "📷 Image",
                use_container_width=True,
                disabled=st.session_state.app_mode == "image",
            ):
                # Switch to image mode
                st.session_state.app_mode = "image"
                # Stop webcam if active
                if st.session_state.webcam_active:
                    st.session_state.webcam_active = False
                st.rerun()

        with col2:
            if st.button(
                "🎥 Webcam",
                use_container_width=True,
                disabled=st.session_state.app_mode == "webcam",
            ):
                # Switch to webcam mode
                st.session_state.app_mode = "webcam"
                st.rerun()
        
        with col3:
            if st.button(
                "🎬 Video",
                use_container_width=True,
                disabled=st.session_state.app_mode == "video",
            ):
                # Switch to video mode
                st.session_state.app_mode = "video"
                if st.session_state.webcam_active:
                    st.session_state.webcam_active = False
                st.rerun()

        # Mode indicator
        mode_name = {"image": "Image Upload", "webcam": "Webcam", "video": "Video File"}.get(
            st.session_state.app_mode, "Unknown"
        )
        st.sidebar.info(f"Current Mode: {mode_name}")

        # Separator
        st.sidebar.markdown("---")

        # Detection parameters (common to both modes)
        st.sidebar.subheader("Detection Parameters")
        scale_factor = st.sidebar.slider("Scale Factor", 1.05, 1.5, 1.1, 0.01)
        min_neighbors = st.sidebar.slider("Min Neighbors", 1, 10, 3, 1)

        # DIP techniques
        st.sidebar.subheader("DIP Techniques")
        dip_technique = st.sidebar.radio(
            "Select Technique",
            ["None", "Gaussian Blur", "Edge Detection", "Brightness & Contrast"],
        )

        # Dynamic filter parameters based on selected technique
        filter_params = {}

        if dip_technique == "Gaussian Blur":
            st.sidebar.subheader("Blur Parameters")
            filter_params["kernel_size"] = st.sidebar.slider(
                "Kernel Size",
                1,
                31,
                5,
                2,
                help="Must be odd number. Larger values increase blur effect",
            )
            # Ensure kernel size is odd
            if filter_params["kernel_size"] % 2 == 0:
                filter_params["kernel_size"] += 1

            filter_params["sigma"] = st.sidebar.slider(
                "Sigma",
                0.1,
                10.0,
                0.0,
                0.1,
                help="Standard deviation. 0 means auto-calculated based on kernel size",
            )

        elif dip_technique == "Edge Detection":
            st.sidebar.subheader("Edge Detection Parameters")
            filter_params["threshold1"] = st.sidebar.slider(
                "Threshold 1", 0, 255, 100, 5, help="Lower threshold for edge detection"
            )
            filter_params["threshold2"] = st.sidebar.slider(
                "Threshold 2", 0, 255, 200, 5, help="Upper threshold for edge detection"
            )
            filter_params["edge_mode"] = st.sidebar.radio(
                "Edge Mode", ["Canny", "Sobel", "Laplacian"]
            )

        elif dip_technique == "Brightness & Contrast":
            st.sidebar.subheader("Brightness & Contrast Parameters")
            filter_params["brightness"] = st.sidebar.slider(
                "Brightness", 0.0, 3.0, 1.0, 0.1
            )
            filter_params["contrast"] = st.sidebar.slider(
                "Contrast", 0.1, 3.0, 1.0, 0.1
            )
            filter_params["saturation"] = st.sidebar.slider(
                "Saturation", 0.0, 3.0, 1.0, 0.1, help="Adjust color saturation"
            )

        # Status area
        status = st.empty()

        # Main area for image display
        image_display = st.empty()

        # Mode-specific controls
        if st.session_state.app_mode == "image":
            # IMAGE MODE
            st.sidebar.subheader("Image Controls")
            uploaded_file = st.sidebar.file_uploader(
                "Choose an image...", type=["jpg", "jpeg", "png"]
            )

            # Process buttons for image
            col1, col2 = st.sidebar.columns(2)
            with col1:
                detect_faces = st.button("Detect Faces", use_container_width=True)
            with col2:
                reset_image = st.button("Reset Image", use_container_width=True)

            # Handle file upload
            if uploaded_file is not None:
                # Read image
                image = Image.open(uploaded_file)
                self.original_image = np.array(image)
                # Convert RGB to BGR for OpenCV processing
                self.original_image = cv2.cvtColor(
                    self.original_image, cv2.COLOR_RGB2BGR
                )
                self.processed_image = self.original_image.copy()

                # Update the display
                display_img = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
                image_display.image(display_img, caption="Uploaded Image")
                status.success("Image uploaded successfully!")

                # Update detector parameters
                self.detector.set_params(
                    scale_factor=scale_factor, min_neighbors=min_neighbors
                )

                # Process image based on DIP technique
                if dip_technique != "None":
                    self.processed_image = self.apply_filter(
                        self.processed_image.copy(), dip_technique, filter_params
                    )
                    display_img = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
                    image_display.image(
                        display_img,
                        caption=f"Processed with {dip_technique}",
                    )

                # Detect faces if button is clicked
                if detect_faces:
                    status.info("Detecting faces...")
                    faces = self.detector.detect_faces(self.processed_image)
                    result_image = draw_detections(self.processed_image.copy(), faces)
                    display_img = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    image_display.image(display_img, caption="Detected Faces")
                    status.success(f"Detected {len(faces)} faces!")

                # Reset image if button is clicked
                if reset_image and self.original_image is not None:
                    self.processed_image = self.original_image.copy()
                    display_img = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
                    image_display.image(display_img, caption="Original Image")
                    status.info("Image reset to original!")

            elif self.original_image is None:
                # Show instructions if no image is loaded
                st.markdown("""
                ## Image Mode 

                Upload an image using the sidebar controls.

                You can then apply filters and detect faces in the uploaded image.
                """)
                status.info("Please upload an image to begin.")

        elif st.session_state.app_mode == "webcam":
            # WEBCAM MODE
            st.sidebar.subheader("Webcam Controls")

            # Webcam control buttons
            col1, col2 = st.sidebar.columns(2)

            with col1:
                start_webcam = st.button(
                    "Start Webcam",
                    use_container_width=True,
                    disabled=st.session_state.webcam_active,
                )

            with col2:
                stop_webcam = st.button(
                    "Stop Webcam",
                    use_container_width=True,
                    disabled=not st.session_state.webcam_active,
                )

            # Add webcam mode selector (only shown when webcam is active or about to be)
            if st.session_state.webcam_active or start_webcam:
                webcam_mode = st.sidebar.radio(
                    "Webcam Processing Mode",
                    ["Continuous Detection", "Detection on Demand", "Filter Only"],
                )
                filter_params["webcam_mode"] = webcam_mode

                if webcam_mode == "Detection on Demand":
                    filter_params["detect_button"] = st.sidebar.button(
                        "Detect Now", use_container_width=True
                    )

            # Handle webcam controls
            if start_webcam:
                st.session_state.webcam_active = True
                status.info("Starting webcam...")
                self.process_webcam(
                    image_display,
                    status,
                    dip_technique,
                    filter_params,
                    scale_factor,
                    min_neighbors,
                )

            if stop_webcam:
                st.session_state.webcam_active = False
                status.info("Webcam stopped.")

            # Show instructions if webcam is not active
            if not st.session_state.webcam_active:
                st.markdown("""
                ## Webcam Mode

                Click the "Start Webcam" button to begin real-time face detection.

                You can choose different processing modes:
                - **Continuous Detection**: Constantly detect faces in each frame
                - **Detection on Demand**: Only detect faces when you click a button
                - **Filter Only**: Apply filters without face detection
                """)
                status.info("Click 'Start Webcam' to begin.")
        
        else:
            # VIDEO MODE
            st.sidebar.subheader("Video File Controls")
            
            uploaded_video = st.sidebar.file_uploader(
                "Choose a video file...", type=["mp4", "avi", "mov", "mkv"]
            )
            
            if uploaded_video is not None:
                # Process buttons for video
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    process_video = st.button("Process Video", use_container_width=True)
                with col2:
                    download_video = st.button("Download Result", use_container_width=True)
                
                # Show video info
                file_details = {
                    "Filename": uploaded_video.name,
                    "FileSize": f"{uploaded_video.size / 1024:.2f} KB"
                }
                st.sidebar.json(file_details)
                
                # Update detector parameters
                self.detector.set_params(
                    scale_factor=scale_factor, min_neighbors=min_neighbors
                )
                
                if process_video:
                    status.info("Processing video...")
                    self.process_video_file(
                        uploaded_video,
                        image_display,
                        status,
                        dip_technique,
                        filter_params,
                    )
                
                if download_video and "processed_video_path" in st.session_state:
                    with open(st.session_state.processed_video_path, "rb") as f:
                        st.sidebar.download_button(
                            label="Download Processed Video",
                            data=f,
                            file_name="processed_" + uploaded_video.name,
                            mime="video/mp4",
                        )
            else:
                st.markdown("""
                ## Video Mode

                Upload a video file to process it with face detection.

                Supported formats: MP4, AVI, MOV, MKV

                The video will be processed frame by frame with your selected filters and detection parameters.
                """)
                status.info("Please upload a video file to begin.")

        # Add troubleshooting expander (common to all modes)
        with st.sidebar.expander("Troubleshooting"):
            st.markdown("""
            **Webcam Issues:**
            - Check camera connections
            - Try different camera index
            - Use uploaded image for testing

            **Performance:**
            - Try smaller images/videos
            - Use simpler filters
            - Lower video resolution
            
            **Video Processing:**
            - Large videos may take time
            - Progress is shown during processing
            - Results are saved temporarily
            """)

    def apply_filter(
        self, frame: NDArray[np.uint8], dip_technique: str, params: Dict[str, Any]
    ) -> NDArray[np.uint8]:
        """Apply the selected filter with parameters to an image frame.
        
        Args:
            frame: Input image in BGR format
            dip_technique: Name of the technique to apply
            params: Dictionary of filter parameters
            
        Returns:
            Processed image
        """
        processed_frame = frame.copy()

        if dip_technique == "Gaussian Blur":
            kernel_size = params.get("kernel_size", 5)
            sigma = params.get("sigma", 0)
            processed_frame = cv2.GaussianBlur(
                processed_frame, (kernel_size, kernel_size), sigma
            )

        elif dip_technique == "Edge Detection":
            edge_mode = params.get("edge_mode", "Canny")

            # Convert to grayscale first
            gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)

            if edge_mode == "Canny":
                threshold1 = params.get("threshold1", 100)
                threshold2 = params.get("threshold2", 200)
                edges = cv2.Canny(gray, threshold1, threshold2)

            elif edge_mode == "Sobel":
                # Apply Sobel in both directions
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

                # Compute magnitude
                magnitude = np.sqrt(sobelx**2 + sobely**2)

                # Normalize and scale for display
                edges = cv2.normalize(
                    magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )

            elif edge_mode == "Laplacian":
                # Apply Laplacian
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)

                # Convert back to uint8
                edges = cv2.normalize(
                    laplacian, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )

            # Convert back to BGR for consistency
            processed_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        elif dip_technique == "Brightness & Contrast":
            brightness = params.get("brightness", 1.0)
            contrast = params.get("contrast", 1.0)
            saturation = params.get("saturation", 1.0)

            # Apply brightness and contrast
            processed_frame = cv2.convertScaleAbs(
                processed_frame, alpha=contrast, beta=(brightness * 50 - 50)
            )

            # If saturation is different from 1.0, adjust it
            if saturation != 1.0:
                # Convert to HSV to adjust saturation
                hsv = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255).astype(
                    np.uint8
                )
                processed_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return processed_frame

    def process_webcam(
        self,
        image_display: Any,
        status: Any,
        dip_technique: str,
        filter_params: Dict[str, Any],
        scale_factor: float,
        min_neighbors: int,
    ) -> None:
        """Process webcam feed with face detection and DIP techniques.
        
        Args:
            image_display: Streamlit placeholder for image display
            status: Streamlit placeholder for status messages
            dip_technique: Selected DIP technique
            filter_params: Dictionary of filter parameters
            scale_factor: Detection scale factor
            min_neighbors: Detection min neighbors parameter
        """
        # Update detector parameters
        self.detector.set_params(scale_factor=scale_factor, min_neighbors=min_neighbors)

        # Get webcam mode
        webcam_mode = filter_params.get("webcam_mode", "Continuous Detection")
        last_detection_time = time.time()  # For detection timing

        # Store the current frame for saving
        self.current_frame = None

        try:
            # First try to use the default camera
            status.info("Attempting to access webcam...")
            cap = cv2.VideoCapture(0)

            # If default camera fails, try the system's video capture API
            if not cap.isOpened():
                status.warning(
                    "Default camera access failed. Trying alternative methods..."
                )

                # Try with cv2.CAP_DSHOW on Windows
                cap = cv2.VideoCapture(
                    0, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
                )

                # If still not working, try other indices
                if not cap.isOpened():
                    for camera_idx in range(1, 5):  # Try camera indices 1 through 4
                        status.info(f"Trying to access camera index {camera_idx}...")
                        cap = cv2.VideoCapture(camera_idx)
                        if cap.isOpened():
                            status.success(
                                f"Successfully opened camera at index {camera_idx}"
                            )
                            break
                        cap.release()  # Release if it was created but not opened

            if not cap.isOpened():
                status.error("Error: Could not access any webcam.")
                st.session_state.webcam_active = False
                return

            status.success(f"Webcam active. Mode: {webcam_mode}")
            detecting = True  # Initial state

            # Add save button below the webcam feed (use session state for proper handling)
            if 'save_frame_requested' not in st.session_state:
                st.session_state.save_frame_requested = False
                
            save_col1, save_col2 = st.columns(2)
            with save_col1:
                if st.button("📸 Save Current Frame", use_container_width=True):
                    st.session_state.save_frame_requested = True
            with save_col2:
                save_status = st.empty()

            # Process frames until stopped
            while st.session_state.webcam_active:
                # Read frame
                ret, frame = cap.read()

                if not ret:
                    status.error("Failed to capture frame from webcam.")
                    time.sleep(0.5)  # Wait a bit before trying again
                    continue

                # Store the current frame for saving
                self.current_frame = frame.copy()

                # Apply selected filter
                if dip_technique != "None":
                    processed_frame = self.apply_filter(
                        frame.copy(), dip_technique, filter_params
                    )
                else:
                    processed_frame = frame.copy()

                # Determine if we should detect faces in this frame
                should_detect = False

                if webcam_mode == "Continuous Detection":
                    should_detect = True
                elif webcam_mode == "Detection on Demand":
                    # Check if detect button was clicked
                    should_detect = filter_params.get("detect_button", False)
                    # Reset button state after processing
                    if should_detect:
                        filter_params["detect_button"] = False
                # "Filter Only" mode never detects faces

                # Detect faces if needed
                if should_detect:
                    faces = self.detector.detect_faces(processed_frame)
                    result_frame = draw_detections(processed_frame.copy(), faces)
                    last_detection_time = time.time()
                    status.success(f"Detected {len(faces)} faces")
                    detecting = True
                else:
                    # If not detecting, just show the processed frame
                    result_frame = processed_frame
                    detecting = False

                # Convert from BGR to RGB for display
                display_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

                # Create caption based on current state
                if webcam_mode == "Filter Only":
                    caption = f"Webcam Feed - {dip_technique if dip_technique != 'None' else 'No Filter'}"
                else:
                    detection_status = (
                        "With Face Detection" if detecting else "No Detection"
                    )
                    caption = f"Webcam Feed - {dip_technique if dip_technique != 'None' else 'No Filter'} - {detection_status}"

                # Display the processed frame
                image_display.image(
                    display_frame,
                    caption=caption,
                    channels="RGB",
                )

                # Handle image saving if button is clicked
                if st.session_state.save_frame_requested:
                    self.save_webcam_image(result_frame, save_status)
                    st.session_state.save_frame_requested = False  # Reset flag
                    time.sleep(0.1)  # Brief delay

                # Small delay to reduce CPU usage
                time.sleep(0.033)  # ~ 30 FPS

        except Exception as e:
            status.error(f"Webcam error: {str(e)}")
            st.error(f"An error occurred with the webcam: {str(e)}")
        finally:
            # Release resources when stopped (in any case)
            if cap is not None:
                if hasattr(cap, "release"):
                    cap.release()
            st.session_state.webcam_active = False
            status.info("Webcam stopped.")

    def save_webcam_image(self, frame: NDArray[np.uint8], status: Any) -> None:
        """Save the current webcam frame as an image file.
        
        Args:
            frame: Image frame to save
            status: Streamlit placeholder for status messages
        """
        try:
            # Create a data directory if it doesn't exist
            os.makedirs("saved_images", exist_ok=True)

            # Define filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"saved_images/webcam_capture_{timestamp}.png"

            # Ensure we have a frame to save
            if frame is None:
                status.error("No frame available to save")
                return

            # Save the image
            cv2.imwrite(filename, frame)
            logger.info(f"Saved webcam frame to {filename}")

            # Update status
            status.success(f"Image saved as {filename}")

            # Show download link
            with open(filename, "rb") as file:
                st.download_button(
                    label="Download saved image",
                    data=file,
                    file_name=f"webcam_capture_{timestamp}.png",
                    mime="image/png",
                )

        except Exception as e:
            status.error(f"Failed to save image: {str(e)}")
            st.error(f"Failed to save image: {str(e)}")
    
    def process_video_file(
        self,
        uploaded_video: Any,
        image_display: Any,
        status: Any,
        dip_technique: str,
        filter_params: Dict[str, Any],
    ) -> None:
        """Process uploaded video file with face detection.
        
        Args:
            uploaded_video: Uploaded video file object
            image_display: Streamlit placeholder for image display
            status: Streamlit placeholder for status messages
            dip_technique: Selected DIP technique
            filter_params: Dictionary of filter parameters
        """
        import tempfile
        
        try:
            # Save uploaded video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                temp_video_path = tmp_file.name
            
            # Open video file
            cap = cv2.VideoCapture(temp_video_path)
            
            if not cap.isOpened():
                status.error("Failed to open video file")
                return
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
            
            # Create output video file
            output_path = os.path.join("saved_images", f"processed_{uploaded_video.name}")
            os.makedirs("saved_images", exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Progress bar
            progress_bar = st.progress(0)
            frame_count = 0
            
            status.info(f"Processing {total_frames} frames...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply filter
                if dip_technique != "None":
                    processed_frame = self.apply_filter(
                        frame.copy(), dip_technique, filter_params
                    )
                else:
                    processed_frame = frame.copy()
                
                # Detect faces
                faces = self.detector.detect_faces(processed_frame)
                result_frame = draw_detections(processed_frame, faces)
                
                # Write to output video
                out.write(result_frame)
                
                # Update progress every 10 frames
                frame_count += 1
                if frame_count % 10 == 0 or frame_count == total_frames:
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    
                    # Show current frame
                    display_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                    image_display.image(
                        display_frame,
                        caption=f"Processing frame {frame_count}/{total_frames}",
                        channels="RGB",
                    )
            
            # Release resources
            cap.release()
            out.release()
            os.unlink(temp_video_path)  # Delete temporary file
            
            # Store output path in session state
            st.session_state.processed_video_path = output_path
            
            progress_bar.progress(1.0)
            status.success(f"Video processed successfully! {frame_count} frames processed.")
            logger.info(f"Video saved to {output_path}")
            
            # Show final frame
            status.info("Click 'Download Result' to save the processed video.")
            
        except Exception as e:
            status.error(f"Video processing error: {str(e)}")
            logger.error(f"Video processing error: {str(e)}")
            st.error(f"An error occurred during video processing: {str(e)}")


if __name__ == "__main__":
    app = FaceDetectionApp()
