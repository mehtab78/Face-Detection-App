"""
Face Detection Application Entry Point
Streamlit-based face detection app using OpenCV Haar Cascades
"""
import streamlit as st

# Set page config first (must be first Streamlit command)
st.set_page_config(
    page_title="Face Detection App",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import from src package
try:
    from src.main import FaceDetectionApp
    from src.style import apply_custom_style

    # Apply custom styling
    apply_custom_style()
    
    # Create and run the application
    app = FaceDetectionApp()
except Exception as e:
    st.error(f"Error loading application: {str(e)}")
    st.error(
        "Please check that all required files are present and dependencies are installed."
    )
