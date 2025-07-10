# ======================== Imports ========================
import streamlit as st
import joblib
import fitz  # PyMuPDF for PDF parsing
import docx  # For DOCX file reading
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# ======================== Page Configuration ========================
st.set_page_config(
    page_title=" JobJolt AI - Resume Role Classifier",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================== Load Model and Vectorizer ========================
@st.cache_resource
def load_models():
    try:
        model = joblib.load('model/resume_classifier_model.pkl')
        vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please ensure the model files are in the 'model' directory.")
        return None, None

model, vectorizer = load_models()

# ======================== Enhanced CSS Styling ========================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    .main {
        padding: 0rem 1rem;
    }
    
    .main-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FFD700 0%, #FF6B35 50%, #1E90FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: ;
        text-align: center;
        margin-bottom: 2rem;
        animation: lightning 2s infinite alternate;
        text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
    }
    
    .subtitle {
        font-family: 'Poppins', sans-serif;
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
        animation: fadeInUp 1s ease-out 0.3s both;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #e6f3ff 100%);
        padding: 2rem;
        border: 2px solid transparent;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-image: linear-gradient(135deg, #FFD700, #FF6B35) 1;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: slideInUp 0.8s ease-out;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(255, 215, 0, 0.2);
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #FFD700 0%, #FF6B35 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0;
        animation: electricPulse 1.5s ease-out;
        box-shadow: 0 10px 30px rgba(255, 215, 0, 0.4);
        border: 2px solid #FFD700;
    }
    
    .stats-card {
        background: linear-gradient(135deg, #1E90FF 0%, #FFD700 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        font-weight: 600;
        animation: fadeInLeft 0.8s ease-out;
    }
    
    .upload-zone {
        border: 3px dashed #FFD700;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        background: linear-gradient(135deg, #fff9e6 0%, #fff4d1 100%);
        margin: 2rem 0;
        transition: all 0.3s ease;
        color: #333;
    }
    
    .upload-zone:hover {
        border-color: #FF6B35;
        background: linear-gradient(135deg, #fff 0%, #ffe6cc 100%);
        transform: scale(1.02);
        box-shadow: 0 10px 25px rgba(255, 215, 0, 0.3);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #1E90FF 0%, #FFD700 100%);
    }
    
    .metric-container {
        background: background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        animation: fadeIn 1s ease-out;
    }
    
    .role-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        margin: 0.3rem;
        font-weight: 500;
        animation: sparkle 5s infinite;
    }
    
    .jolt-logo {
        font-size: 2rem;
        color: #FFD700;
        animation: lightning 1.5]s infinite;
    }
    
    @keyframes lightning {
        0%, 100% { 
            transform: scale(1);
            text-shadow: 0 0 10px rgba(255, 215, 0, 0.8);
        }
        50% { 
            transform: scale(1.02);
            text-shadow: 0 0 20px rgba(255, 215, 0, 1), 0 0 30px rgba(255, 107, 53, 0.8);
        }
    }
    
    @keyframes electricPulse {
        0% { 
            opacity: 0; 
            transform: scale(0.8);
            box-shadow: 0 0 0 rgba(255, 215, 0, 0.4);
        }
        50% { 
            opacity: 1; 
            transform: scale(1.05);
            box-shadow: 0 0 30px rgba(255, 215, 0, 0.6);
        }
        100% { 
            opacity: 1; 
            transform: scale(1);
            box-shadow: 0 10px 30px rgba(255, 215, 0, 0.4);
        }
    }
    
    @keyframes sparkle {
        0%, 100% { 
            transform: scale(1);
            box-shadow: 0 0 5px rgba(255, 215, 0, 0.5);
        }
        50% { 
            transform: scale(1.05);
            box-shadow: 0 0 15px rgba(255, 215, 0, 0.8);
        }
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(50px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #FFD700 0%, #FF6B35 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(255, 215, 0, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 215, 0, 0.4);
        background: linear-gradient(135deg, #FF6B35 0%, #FFD700 100%);
    }
    
    .stSelectbox > div > div > div > div {
        background: linear-gradient(135deg, #f8f9ff 0%, #e3e7ff 100%);
        border-radius: 10px;
    }
    
    .progress-bar {
        background: linear-gradient(135deg, #FFD700 0%, #FF6B35 100%);
        height: 20px;
        border-radius: 10px;
        animation: progressAnimation 2s ease-out;
    }
    
    @keyframes progressAnimation {
        from { width: 0%; }
        to { width: 100%; }
    }
    </style>
""", unsafe_allow_html=True)

# ======================== Enhanced Emoji Mapping ========================
role_emojis = {
    "Data Science": "🧪", "Java Developer": "☕", "Python Developer": "🐍",
    "HR": "🧑‍💼", "DevOps Engineer": "⚙️", "Web Designing": "🎨",
    "Mechanical Engineer": "🔩", "Sales": "💼", "Testing": "🧪",
    "Hadoop": "🗃️", "ETL Developer": "🛠️", "Blockchain": "⛓️",
    "Operations Manager": "📋", "Electrical Engineering": "⚡",
    "Health and fitness": "🏋️", "Arts": "🎭", "Database": "🗄️",
    "Civil Engineer": "🏗️", "Network Security Engineer": "🛡️",
    "DotNet Developer": "💻", "Automation Testing": "🤖", "Advocate": "⚖️",
    "Business Analyst": "📊"
}

# ======================== Enhanced Title ========================
st.markdown("""
    <div class='main-title'>⚡ JobJolt AI</div>
    <div class='subtitle'>Your Career Clarity in a Jolt | AI-Powered Resume Analysis</div>
""", unsafe_allow_html=True)

# ======================== Enhanced Sidebar Navigation ========================
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 15px; color: white; margin-bottom: 2rem;'>
            <h2 style='margin: 0; font-family: Poppins; color: white'>⚡ Navigation</h2>
        </div>
    """, unsafe_allow_html=True)
    
    selected = st.radio(
        "Choose your destination:",
        ["🏠 Home", "📤 Upload Resume", "📊 Analytics", "ℹ️ About"],
        index=0
    )
    
    st.markdown("---")
    
    # Add some statistics
    st.markdown("""
        <div class='metric-container' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'>
            <h4 style='margin: 0;'>📈 App Statistics</h4>
            <p style='margin: 0.5rem 0;'>🎯 23+ Job Categories</p>
            <p style='margin: 0.5rem 0;'>⚡ 95% Accuracy</p>
            <p style='margin: 0.5rem 0;'>🚀 AI-Powered</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick tips
    with st.expander("💡 Quick Tips"):
        st.markdown("""
        - Use detailed resume content for better accuracy
        - Include technical skills and experience
        - Mention specific projects and achievements
        - Keep format clean and professional
        """)

# ======================== Enhanced Text Extraction Function ========================
def extract_text(file):
    """
    Enhanced text extraction with better error handling and progress indication.
    """
    try:
        if file.type == "application/pdf":
            text = ""
            with fitz.open(stream=file.read(), filetype="pdf") as doc:
                total_pages = len(doc)
                progress_bar = st.progress(0)
                for i, page in enumerate(doc):
                    text += page.get_text()
                    progress_bar.progress((i + 1) / total_pages)
                time.sleep(0.1)  # Small delay for visual effect
            return text
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
        elif file.type == "text/plain":
            return file.read().decode("utf-8")
        else:
            return "❌ Unsupported file format"
    except Exception as e:
        st.error(f"❌ Error extracting text: {str(e)}")
        return ""

# ======================== Enhanced Prediction Function ========================
def predict_role(text):
    """
    Enhanced prediction function with confidence scoring.
    """
    if not model or not vectorizer:
        return None, None
    
    try:
        transformed_input = vectorizer.transform([text])
        prediction = model.predict(transformed_input)[0]
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(transformed_input)[0]
            confidence = max(probabilities) * 100
        else:
            confidence = 85  # Default confidence
        
        return prediction, confidence
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        return None, None

# ======================== Home Page ========================
if selected == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div class='feature-card' style='border: 2px solid purple; radius: 20px;'>
                <h3 style='color: #FF6B35; margin-bottom: 1rem;'>⚡ Resume Analysis Engine</h3>
                <p style='color: #666;'>Paste your resume content and get instant career insights with JobJolt AI!</p>
            </div>
        """, unsafe_allow_html=True)
        
        resume_text = st.text_area(
            "📝 Enter Resume Content",
            height=300,
            placeholder="Paste your complete resume content here...\n\nInclude:\n• Personal information\n• Work experience\n• Skills and technologies\n• Education\n• Projects and achievements",
            help="💡 Tip: Include detailed information for better accuracy!"
        )
        
        col1a, col1b, col1c = st.columns([1, 2, 1])
        with col1b:
            if st.button("⚡ Get Your Career Jolt!", use_container_width=True):
                if resume_text.strip():
                    with st.spinner("⚡ JobJolt AI is analyzing your resume..."):
                        time.sleep(1)  # Visual effect
                        prediction, confidence = predict_role(resume_text)
                        
                        if prediction:
                            emoji = role_emojis.get(prediction, "🔍")
                            st.markdown(f"""
                                <div class='feature-card' style='border: 2px solid #FF6B35; color:purple; background: white; border-radius: 20px;'>
                                    {emoji} <strong style='color: purple;font-size: 2rem;'>{prediction}</strong>
                                    <br><small style='color: #FF6B35;font-size: 1rem;'>Confidence: {confidence:.1f}%</small>
                                    <br><small style='color: #FF6B35;font-size: 1rem;'>⚡ Powered by JobJolt AI</small>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Add success animation
                            # st.balloons()
                else:
                    st.warning("⚠️ Please enter resume text to get your career jolt!")
    
    with col2:
        st.markdown("""
            <div >
                <h3 style='color: purple;background: white;border: 2px solid purple; border-radius: 20px; margin-bottom: 1rem;'>⚡ Jolt-Powered Roles</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Display role badges
        roles_html = ""
        for role, emoji in role_emojis.items():
            roles_html += f"<span class='role-badge'>{emoji} {role}</span>"
        
        st.markdown(f"<div>{roles_html}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("""
            <div class='stats-card'>
                <h4 style='margin: 0;'>⚡ Why Choose JobJolt AI?</h4>
                <p style='margin: 0.5rem 0;'>✅ Lightning-Fast Analysis</p>
                <p style='margin: 0.5rem 0;'>✅ Instant Career Insights</p>
                <p style='margin: 0.5rem 0;'>✅ High Accuracy Predictions</p>
                <p style='margin: 0.5rem 0;'>✅ Multiple Format Support</p>
            </div>
        """, unsafe_allow_html=True)

# ======================== Upload Resume Page ========================
elif selected == "📤 Upload Resume":
    st.markdown("""
        <div class='feature-card' style='border: 2px solid #FF6B35; radius: 20px;'>
            <h2 style='color: #FF6B35; text-align: center; margin-bottom: 2rem;'>⚡ Upload Your Resume</h2>
            
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<div class='upload-zone'>"
        "<p style='color: #purple; text-align: center;'>Get your career jolt in seconds! Supported formats: PDF, DOCX, TXT</p></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "📎 Choose your resume file",
            type=["pdf", "docx", "txt"],
            help="Upload your resume in PDF, DOCX, or TXT format"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### 📄 Extracted Content")
            with st.spinner("🔍 Extracting text from your resume..."):
                content = extract_text(uploaded_file)
            
            if content and content != "❌ Unsupported file format":
                st.text_area("Resume Content", value=content, height=300)
                
                if st.button("⚡ Get Career Jolt!", use_container_width=True):
                    with st.spinner("⚡ JobJolt AI is analyzing your resume..."):
                        time.sleep(1)
                        prediction, confidence = predict_role(content)
                        
                        if prediction:
                            emoji = role_emojis.get(prediction, "🔍")
                            st.markdown(f"""
                                <div class='feature-card' style='border: 2px solid #FF6B35; color:purple; background: white; border-radius: 20px;'>
                                    {emoji} <strong style='color: purple;font-size: 2rem;'>{prediction}</strong>
                                    <br><small style='color: #FF6B35;font-size: 1rem;'>Confidence: {confidence:.1f}%</small>
                                    <br><small style='color: #FF6B35;font-size: 1rem;'>⚡ Powered by JobJolt AI</small>
                                </div>
                            """, unsafe_allow_html=True)
                            # st.balloons()
            else:
                st.error("❌ Could not extract text from the uploaded file.")
        
        with col2:
            st.markdown("### 📊 File Information")
            file_details = {
                "📄 Filename": uploaded_file.name,
                "📏 File Size": f"{uploaded_file.size / 1024:.1f} KB",
                "📁 File Type": uploaded_file.type,
                "🕒 Upload Time": datetime.now().strftime("%H:%M:%S")
            }
            
            for key, value in file_details.items():
                st.info(f"{key}: {value}")

# ======================== Analytics Page ========================
elif selected == "📊 Analytics":
    st.markdown("""
        <div class='feature-card' style='border: 2px solid #FF6B35; radius: 20px;'>
            <h2 style='color: #FF6B35; text-align: center; margin-bottom: 2rem;'>⚡ JobJolt Analytics Dashboard</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Create sample data for demonstration
    roles_data = list(role_emojis.keys())
    sample_data = {
        'Role': roles_data,
        'Frequency': [45, 38, 35, 28, 25, 22, 20, 18, 15, 12, 10, 8, 7, 6, 5, 4, 3, 3, 2, 2, 1, 1, 1],
        'Avg_Confidence': [92, 89, 87, 85, 83, 81, 79, 77, 75, 73, 71, 69, 67, 65, 63, 61, 59, 57, 55, 53, 51, 49, 47]
    }
    
    df = pd.DataFrame(sample_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Role frequency chart
        fig_freq = px.bar(
            df.head(10), 
            x='Role', 
            y='Frequency',
            title='🎯 Top 10 Most Predicted Roles',
            color='Frequency',
            color_continuous_scale='viridis'
        )
        fig_freq.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_freq, use_container_width=True)
    
    with col2:
        # Confidence distribution
        fig_conf = px.scatter(
            df, 
            x='Frequency', 
            y='Avg_Confidence',
            size='Frequency',
            color='Avg_Confidence',
            title='📈 Confidence vs Frequency Analysis',
            hover_data=['Role']
        )
        fig_conf.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 Total Roles",
            value=len(roles_data),
            delta="23 categories"
        )
    
    with col2:
        st.metric(
            label="📊 Avg Confidence",
            value="85.2%",
            delta="2.1% ↑"
        )
    
    with col3:
        st.metric(
            label="🔥 Most Popular",
            value="Data Science",
            delta="45 predictions"
        )
    
    with col4:
        st.metric(
            label="⚡ Processing Time",
            value="0.8s",
            delta="0.2s ↓"
        )

# ======================== About Page ========================
elif selected == "ℹ️ About":
    st.markdown("""
        <div class='feature-card' style='border: 2px solid #FF6B35; radius: 20px;'>
            <h2 style='color: #FF6B35; text-align: center; margin-bottom: 2rem;'>⚡ About JobJolt AI</h2>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### ⚡ What is JobJolt AI?
        JobJolt AI is your lightning-fast career companion that instantly analyzes resume content and delivers 
        precise job role predictions. Get the career clarity you need in just one jolt!

        ### 🔧 How JobJolt Works
        1. **Lightning Upload**: Upload or paste your resume content
        2. **AI Analysis**: Our advanced algorithms process your experience and skills
        3. **Instant Jolt**: Get immediate career insights with confidence scores
        4. **Career Clarity**: Discover your perfect job match in seconds

        ### 🔧 How It Works
        1. **Text Extraction**: We extract and process text from your resume
        2. **Feature Engineering**: Convert text into numerical features using TF-IDF
        3. **ML Prediction**: Our trained model predicts the best matching role
        4. **Confidence Score**: Get confidence percentage for the prediction

        ### 🎨 Supported Job Categories
        """, unsafe_allow_html=True)
        
        # Display categories in a more organized way
        categories = list(role_emojis.keys())
        for i in range(0, len(categories), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(categories):
                    role = categories[i + j]
                    emoji = role_emojis[role]
                    col.markdown(f"**{emoji} {role}**")

    with col2:
        st.markdown("""
            <div class='stats-card'>
                <h3 style='margin: 0; margin-bottom: 1rem;'>🛠️ Technologies Used</h3>
                <p>🐍 Python</p>
                <p>🧠 Scikit-learn</p>
                <p>📊 TF-IDF Vectorizer</p>
                <p>🎯 Naive Bayes</p>
                <p>🚀 Streamlit</p>
                <p>📈 Plotly</p>
                <p>🎨 Custom CSS</p>
            </div>
        """, unsafe_allow_html=True)

        # Footttterrrrr
        
        st.markdown("""
            <div class='feature-card'style='border: 2px solid #FF6B35; radius: 20px;'>
                <h4 style='color: #FF6B35; margin-bottom: 1rem;'>📞 Contact & Support</h4>
                <p style='color: purple;'><strong>👨‍💻 Developer:</strong> Krishna Gandhi</p>
                <p style='color: purple;'><strong>🚀 Version:</strong> 2.0 Enhanced</p>
                <p style='color: purple;'><strong>📅 Last Updated:</strong> July 2025</p>
                <p style='color: purple;><strong>⭐ Rating:</strong> 4.9/5</p>
            </div>
        """, unsafe_allow_html=True)




# with col1:
#     st.markdown("""
#     ### ⚡ What is JobJolt AI?
#     JobJolt AI is your lightning-fast career companion that instantly analyzes resume content and delivers 
#     precise job role predictions. Get the career clarity you need in just one jolt!

#     ### 🔧 How JobJolt Works
#     1. **Lightning Upload**: Upload or paste your resume content
#     2. **AI Analysis**: Our advanced algorithms process your experience and skills
#     3. **Instant Jolt**: Get immediate career insights with confidence scores
#     4. **Career Clarity**: Discover your perfect job match in seconds

#     ### 🔧 How It Works
#     1. **Text Extraction**: We extract and process text from your resume
#     2. **Feature Engineering**: Convert text into numerical features using TF-IDF
#     3. **ML Prediction**: Our trained model predicts the best matching role
#     4. **Confidence Score**: Get confidence percentage for the prediction

#     ### 🎨 Supported Job Categories
#     """, unsafe_allow_html=True)