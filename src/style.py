import streamlit as st


def apply_custom_style():
    """Apply custom styling to the application"""
    st.markdown(
        """
    <style>
    /* Main container */
    .main .block-container {
        padding-top: 1rem;
    }
    
    /* App title */
    h1 {
        color: #2563EB;
        text-align: center;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.5rem;
    }
    
    /* Section headers */
    h2 {
        color: #1E40AF;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #1E40AF;
        font-size: 1.2rem;
    }
    
    /* Mode buttons */
    .stButton button {
        font-weight: 500;
        border-radius: 4px;
    }
    
    /* Status messages */
    .success {
        color: #10B981;
        font-weight: 500;
    }
    
    .info {
        color: #3B82F6;
        font-weight: 500;
    }
    
    .warning {
        color: #F59E0B;
        font-weight: 500;
    }
    
    .error {
        color: #EF4444;
        font-weight: 500;
    }
    
    /* Mode indicator */
    .current-mode {
        padding: 0.5rem;
        border-radius: 4px;
        background-color: #EFF6FF;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Sidebar sections */
    .sidebar-section {
        background-color: #F9FAFB;
        padding: 0.75rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    
    /* Image captions */
    .caption {
        text-align: center;
        color: #6B7280;
        font-style: italic;
        margin-top: 0.25rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
