import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)   
        nltk.download('stopwords', quiet=True)
    except:
        pass

download_nltk_data()

# Load model and tokenizer
@st.cache_resource
def load_objects():
    try:
        model = load_model('toxicity_model_BiLSTM.keras')

    except:
            st.error("Model file not found. Please ensure the model is trained and saved.")
            return None, None
    
    try:
        tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
    except:
        st.error("Tokenizer file not found.")
        return model, None
    
    return model, tokenizer

# Text preprocessing functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Prediction function
def predict_toxicity(comment, model, tokenizer):
    # Clean and preprocess
    cleaned = clean_text(comment)
    processed = preprocess(cleaned)
    
    # Tokenize and pad
    sequences = tokenizer.texts_to_sequences([processed])
    X = pad_sequences(sequences, maxlen=100)
    
    # Predict
    prediction = model.predict(X, verbose=0)[0][0]
    
    return prediction

# Load data for insights
@st.cache_data
def load_data():
    try:
        train_data = pd.read_csv('train.csv')
        test_data = pd.read_csv('test.csv')
        return train_data, test_data
    except:
        return None, None

# Set page configuration
st.set_page_config(
    page_title="Comment Toxicity Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 18px;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and header
st.markdown("#  Comment Toxicity Detection System")
st.markdown("Detect toxic comments using advanced Deep Learning models (LSTM, CNN, BiLSTM)")
st.divider()

# Load model and tokenizer
model, tokenizer = load_objects()

if model is None or tokenizer is None:
    st.error("❌ Unable to load model and tokenizer. Please ensure they are saved in the working directory.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("## Dashboard Options")
    
    st.info("""
    **How to use:**
    -  **Predict**: Enter a comment to check if it's toxic
    -  **Insights**: View data statistics and model metrics
    -  **Bulk Upload**: Upload CSV for batch predictions
    """)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([" Predict", " Insights", " Bulk Upload", " About"])

# TAB 1: Single Prediction
with tab1:
    st.header("Comment Toxicity Prediction")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_comment = st.text_area(
            "Enter a comment to analyze:",
            placeholder="Type your comment here...",
            height=150,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### ")
        predict_button = st.button(" Predict", use_container_width=True)
    
    if predict_button and user_comment.strip():
        with st.spinner("Analyzing comment..."):
            toxicity_score = predict_toxicity(user_comment, model, tokenizer)
            
        st.divider()
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Toxicity Score",
                f"{toxicity_score:.2%}",
                delta=None,
                label_visibility="visible"
            )
        
        with col2:
            if toxicity_score > 0.5:
                label = "🚨 TOXIC"
                color = "red"
            else:
                label = "✅ NON-TOXIC"
                color = "green"
            st.markdown(f"### Classification")
            st.markdown(f"<h2 style='color: {color};'>{label}</h2>", unsafe_allow_html=True)
        
        with col3:
            confidence = max(toxicity_score, 1 - toxicity_score) * 100
            st. metric(
                "Confidence",
                f"{confidence:.1f}%",
                label_visibility="visible"
            )
        

        
        # Show original vs cleaned text
        with st.expander(" Text Analysis"):
            cleaned = clean_text(user_comment)
            processed = preprocess(cleaned)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Original")
                st.write(user_comment)
            with col2:
                st.subheader("Cleaned")
                st.write(cleaned)
            with col3:
                st.subheader("Processed")
                st.write(processed)
    
    elif predict_button:
        st.warning("⚠️ Please enter a comment to analyze.")

# TAB 2: Insights and Metrics
with tab2:
        # Model Information
        st.subheader(" Model Architecture")
        model_info = """
        - **Architecture**: BiLSTM (Bidirectional LSTM)
        - **Embedding Layer**: 10,000 vocabulary, 128 dimensions
        - **BiLSTM Units**: 64 with dropout (0.5)
        - **Dense Layers**: 64 units with dropout (0.3) + 1 output unit
        - **Activation**: Sigmoid (Binary Classification)
        - **Loss Function**: Binary Crossentropy
        - **Optimizer**: Adam
        - **Input Length**: 100 tokens (padded)
        """
        st.markdown(model_info)
        
        # Performance metrics section
        st.subheader("📈 Model Performance")
        
        perf_col1, perf_col2 = st.columns(2)
        with perf_col1:
            st.markdown("""
            **Training Configuration:**
            - Epochs: 5
            - Batch Size: 32
            - Train/Test Split: 80/20
            - Test Size: 20 percent of total data
            """)
        
        with perf_col2:
            st.warning("""
            ⚠️ **Note**: For detailed performance metrics on your test set, 
            please check the model training output in the notebook.
            
            You can enhance performance by:
            - Training for more epochs
            - Using larger embedding dimensions
            - Adding more dense layers
            - Using ensemble methods
            """)
    
    else:
        st.error("❌ Could not load training data. Ensure train.csv and test.csv are in the working directory.")

# TAB 3: Bulk Upload
with tab3:
    st.header("📤 Bulk Comment Analysis")
    st.markdown("Upload a CSV file with comments column named 'comment_text' to get predictions for all comments.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check if comment_text column exists
            if 'comment_text' not in df.columns:
                st.error("❌ CSV must contain a 'comment_text' column")
                st.info(f"Available columns: {', '.join(df.columns.tolist())}")
            else:
                st.success(f"✅ File loaded successfully ({len(df)} rows)")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Process button
                if st.button("🚀 Predict All Comments", use_container_width=True):
                    with st.spinner("Processing comments..."):
                        # Add predictions
                        predictions = []
                        toxicity_scores = []
                        
                        progress_bar = st.progress(0)
                        
                        for idx, comment in enumerate(df['comment_text']):
                            try:
                                score = predict_toxicity(str(comment), model, tokenizer)
                                toxicity_scores.append(score)
                                predictions.append("Toxic" if score > 0.5 else "Non-Toxic")
                            except:
                                toxicity_scores.append(np.nan)
                                predictions.append("Error")
                            
                            progress_bar.progress((idx + 1) / len(df))
                        
                        # Create results dataframe
                        results_df = df.copy()
                        results_df['Toxicity_Score'] = toxicity_scores
                        results_df['Prediction'] = predictions
                        results_df['Confidence'] = results_df['Toxicity_Score'].apply(
                            lambda x: max(x, 1-x) * 100 if not np.isnan(x) else np.nan
                        )
                        
                        # Display results
                        st.divider()
                        st.subheader("✅ Prediction Results")
                        
                        # Statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        toxic_count = (results_df['Prediction'] == 'Toxic').sum()
                        non_toxic_count = (results_df['Prediction'] == 'Non-Toxic').sum()
                        
                        with col1:
                            st.metric("Total Analyzed", len(results_df))
                        with col2:
                            st.metric("Toxic Found", toxic_count)
                        with col3:
                            st.metric("Non-Toxic", non_toxic_count)
                        with col4:
                            toxic_percentage = (toxic_count / len(results_df) * 100) if len(results_df) > 0 else 0
                            st.metric("Toxicity Rate", f"{toxic_percentage:.1f}%")
                        
                        st.divider()
                        
                        # Results table
                        st.subheader("Detailed Results")
                        
                        # Sortable results
                        sort_col = st.selectbox(
                            "Sort by:",
                            ["Index", "Toxicity_Score (High to Low)", "Prediction"]
                        )
                        
                        if sort_col == "Toxicity_Score (High to Low)":
                            display_df = results_df.sort_values('Toxicity_Score', ascending=False)
                        elif sort_col == "Prediction":
                            display_df = results_df.sort_values('Prediction', ascending=False)
                        else:
                            display_df = results_df
                        
                        st.dataframe(
                            display_df[['comment_text', 'Prediction', 'Toxicity_Score', 'Confidence']].reset_index(drop=True),
                            use_container_width=True,
                            height=400
                        )
                        
                        st.divider()
                        
                        # Visualization
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig, ax = plt.subplots(figsize=(6, 4))
                            categories = ['Toxic', 'Non-Toxic']
                            values = [toxic_count, non_toxic_count]
                            colors = ['#ff4b4b', '#4CAF50']
                            bars = ax.bar(categories, values, color=colors)
                            ax.set_ylabel('Count')
                            ax.set_title('Prediction Distribution')
                            
                            for bar, value in zip(bars, values):
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                       f'{int(value)}', ha='center', va='bottom', fontweight='bold')
                            
                            st.pyplot(fig, use_container_width=True)
                        
                        with col2:
                            fig, ax = plt.subplots(figsize=(6, 4))
                            sizes = [toxic_count, non_toxic_count]
                            labels = [f'Toxic\n({toxic_count})', f'Non-Toxic\n({non_toxic_count})']
                            colors = ['#ff4b4b', '#4CAF50']
                            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                  shadow=True, startangle=90)
                            ax.set_title('Overall Distribution')
                            st.pyplot(fig, use_container_width=True)
                        
                        # Download results
                        st.divider()
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label=" Download Results as CSV",
                            data=csv,
                            file_name="toxicity_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# TAB 4: About
with tab4:
    st.header(" About This Application")
    
    st.markdown("""
    ## Overview
    This is a **Comment Toxicity Detection System** built using deep learning to automatically detect and classify 
    toxic comments in online platforms. It uses advanced neural network architectures to achieve high accuracy.
    
    ## Features
    - **Real-time Single Comment Analysis**: Instantly predict toxicity of individual comments
    - **Detailed Insights**: View data statistics and model performance metrics
    - **Bulk Processing**: Upload CSV files for batch predictions
    - **Interactive Dashboard**: User-friendly interface with visualizations
    
    ## Technical Details
    
    ### Models Used
    1. **BiLSTM (Bidirectional LSTM)** - Best performer
    2. **LSTM (Long Short-Term Memory)**
    3. **CNN (Convolutional Neural Network)**
    
    ### Text Processing Pipeline
    1. **Text Cleaning**: Convert to lowercase, remove special characters
    2. **Tokenization**: Break text into individual tokens
    3. **Stopword Removal**: Filter out common English stopwords
    4. **Vectorization**: Convert text to numerical sequences using Tokenizer
    5. **Padding**: Pad/truncate sequences to fixed length (100 tokens)
    
    ### Model Architecture (BiLSTM - Currently in Use)
    - **Embedding Layer**: 10,000 vocabulary size, 128 dimensions
    - **Bidirectional LSTM**: 64 units (processes text in both directions)
    - **Dropout**: 0.5 rate for regularization
    - **Dense Layers**: 64 units with 0.3 dropout + 1 output unit
    - **Output Activation**: Sigmoid (binary classification)
    
    ### Dataset Information
    - **Training Samples**: ~159,571
    - **Test Samples**: Variable based on your dataset
    - **Classes**: Binary (Toxic / Non-Toxic)
    - **Imbalance**: ~91% Non-Toxic, ~9% Toxic
    
    ## Usage Tips
    
    ### For Single Predictions
    1. Enter a comment in the text box
    2. Click "Predict"
    3. View the toxicity score and classification
    4. Explore the text analysis section to see preprocessing steps
    
    ### For Bulk Analysis
    1. Prepare a CSV file with a 'comment_text' column
    2. Upload the file in the "Bulk Upload" tab
    3. Click "Predict All Comments"
    4. View results and download predictions
    
    ## Toxicity Score Interpretation
    - **0.0 - 0.5**: Non-Toxic (Low risk)
    - **0.5 - 1.0**: Toxic (High risk)
    
    The confidence score indicates how certain the model is about its prediction.
    
    ## Model Improvement Tips
    - Train for more epochs (10-20)
    - Experiment with different embedding dimensions
    - Use ensemble methods combining multiple models
    - Augment training data
    - Fine-tune hyperparameters
    - Add more dense layers with dropout
    
    ## Limitations
    - Context-dependent toxicity may not be captured
    - Sarcasm and slang handling could be improved
    - Language-specific bias towards English
     """)
