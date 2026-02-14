# Streamlit Deployment Guide: Comment Toxicity Detector

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Prepare Your Project](#prepare-your-project)
3. [GitHub Setup](#github-setup)
4. [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
5. [Configuration & Secrets Management](#configuration--secrets-management)
6. [Post-Deployment Verification](#post-deployment-verification)
7. [Troubleshooting](#troubleshooting)
8. [Maintenance & Updates](#maintenance--updates)

---

## Prerequisites

Before deploying, ensure you have:

- ✅ A **GitHub account** (free account is sufficient)
- ✅ Git installed on your local machine
- ✅ A **Streamlit Community Cloud account** (sign up at [streamlit.io](https://streamlit.io))
- ✅ All project files organized locally
- ✅ Model files (`toxicity_model_BiLSTM.keras`, `tokenizer.pkl`) in your project
- ✅ `requirements.txt` with all dependencies

---

## Prepare Your Project

### 1. **Update requirements.txt**

Ensure your `requirements.txt` includes all necessary packages:

```txt
streamlit>=1.28.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
nltk>=3.6.0
tensorflow>=2.10.0
keras>=2.10.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

**Command to generate/update requirements.txt:**
```bash
pip freeze > requirements.txt
```

### 2. **Organize Project Structure**

Your project directory should look like:

```
Comment_toxicity_project/
├── app.py                          # Main Streamlit app
├── requirements.txt                # Dependencies
├── .gitignore                      # Git ignore file
├── README.md                       # Project documentation
├── toxicity_model_BiLSTM.keras    # Trained model
├── tokenizer.pkl                   # Tokenizer pickle file
├── train.csv                       # Training data (optional)
├── test.csv                        # Test data (optional)
└── .streamlit/
    └── config.toml                 # Streamlit configuration
```

### 3. **Create .gitignore File**

Create a `.gitignore` file to exclude unnecessary/large files:

```plaintext
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Data files (optional - comment out if you want to upload)
# *.csv

# Model backup files
*.h5
*.pkl.bak
```

### 4. **Create Streamlit Config File**

Create `.streamlit/config.toml` for optimal Streamlit settings:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[client]
showErrorDetails = false
toolbarMode = "minimal"

[logger]
level = "info"

[server]
maxUploadSize = 200
headless = true
runOnSave = true
```

### 5. **Test Locally Before Deployment**

```bash
# Run the app locally to verify everything works
streamlit run app.py
```

The app should launch at `http://localhost:8501`. Test all features:
- ✅ Single comment prediction
- ✅ Insights tab with data visualization
- ✅ Bulk upload functionality
- ✅ About section loads correctly

---

## GitHub Setup

### 1. **Initialize Git Repository**

```bash
# Navigate to your project directory
cd "E:\GUVI\Guvi projects\Comment_toxicity_project"

# Initialize git repository
git init

# Add all files to staging
git add .

# Create initial commit
git commit -m "Initial commit: Comment Toxicity Detector with Streamlit app"
```

### 2. **Create GitHub Repository**

1. Go to [GitHub.com](https://github.com) and log in
2. Click **"+"** → **"New repository"**
3. Fill in details:
   - **Repository name**: `comment-toxicity-detector`
   - **Description**: "Comment Toxicity Detection using BiLSTM with Streamlit Web App"
   - **Public** or **Private** (Public recommended for easy Streamlit deployment)
   - Add `.gitignore` for Python (optional, if not already added)
   - Add a `README.md`
4. Click **"Create repository"**

### 3. **Push Project to GitHub**

```bash
# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/comment-toxicity-detector.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

**Note**: Replace `YOUR_USERNAME` with your actual GitHub username.

### 4. **Verify Upload**

Visit `https://github.com/YOUR_USERNAME/comment-toxicity-detector` to confirm all files are pushed.

---

## Streamlit Cloud Deployment

### 1. **Sign Up for Streamlit Cloud**

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"Sign in with GitHub"**
3. Authorize Streamlit to access your GitHub repositories
4. Complete your Streamlit account setup

### 2. **Deploy Your App**

1. Click **"New app"** button in Streamlit Cloud
2. Fill in deployment details:
   - **Repository**: Select your GitHub repository
     - Format: `YOUR_USERNAME/comment-toxicity-detector`
   - **Branch**: `main`
   - **Main file path**: `app.py`

3. Click **"Deploy"**

Streamlit will now:
- Clone your repository
- Install dependencies from `requirements.txt`
- Build and launch your app
- Assign a public URL (typically `https://comment-toxicity-detector.streamlit.app`)

⏱️ **Deployment Time**: First deployment takes 2-5 minutes

### 3. **Monitor Deployment**

You'll see:
- **Deploying** → Building dependencies
- **Running** → App is live and accessible

If there are errors, check the logs for debugging information.

---

## Configuration & Secrets Management

### 1. **Set Up Secrets (if needed)**

For sensitive information like API keys:

1. In Streamlit Cloud dashboard, click your app
2. Click **"Settings"** → **"Secrets"**
3. Paste secrets in TOML format:

```toml
# Example if you were using an API
[database]
connection_string = "your_secret_here"

[api]
key = "your_api_key"
```

4. Click **"Save"**

**Note**: For this project, you typically won't need secrets unless you integrate with external services.

### 2. **Verify File Paths**

Streamlit Cloud runs from the repository root, so ensure file paths in `app.py` are correct:

```python
# ❌ Avoid absolute paths
model = load_model('C:/Users/Shankar/...')

# ✅ Use relative paths
model = load_model('toxicity_model_BiLSTM.keras')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
```

### 3. **Handle Large Files**

If your model files are large (>100MB):

**Option A**: Use Git LFS (Large File Storage)
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.keras"
git lfs track "*.pkl"

# Commit and push
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push
```

**Option B**: Upload to cloud storage and download during app startup
```python
import urllib.request

@st.cache_resource
def load_objects():
    # Download from URL if not exists
    if not os.path.exists('toxicity_model_BiLSTM.keras'):
        url = "https://your-cloud-storage-url/toxicity_model_BiLSTM.keras"
        urllib.request.urlretrieve(url, 'toxicity_model_BiLSTM.keras')
    
    model = load_model('toxicity_model_BiLSTM.keras')
    return model
```

---

## Post-Deployment Verification

### 1. **Test All Features**

Once deployed, test:

- ✅ Navigate to your app URL
- ✅ **Predict Tab**: Enter a test comment and verify prediction works
- ✅ **Insights Tab**: Check if data loads and charts display
- ✅ **Bulk Upload**: Upload test CSV and verify batch processing
- ✅ **About Tab**: Verify documentation displays correctly

### 2. **Performance Check**

- Verify app loads within reasonable time
- Test prediction speed
- Check memory usage (Streamlit Cloud has limits)

### 3. **Share Your App**

Your app is now publicly accessible at:
```
https://comment-toxicity-detector.streamlit.app
```

Or use the custom domain if configured.

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"

**Solution**: Ensure `tensorflow` is in `requirements.txt`:
```bash
pip install tensorflow
echo "tensorflow>=2.10.0" >> requirements.txt
git add requirements.txt
git commit -m "Add tensorflow dependency"
git push
```

Redeploy in Streamlit Cloud.

### Issue: "FileNotFoundError: toxicity_model_BiLSTM.keras not found"

**Solutions**:
1. Verify file names match exactly (case-sensitive on Linux)
2. Ensure files are committed to Git:
   ```bash
   git add toxicity_model_BiLSTM.keras tokenizer.pkl
   git commit -m "Add model files"
   git push
   ```
3. Check file size limits (Streamlit Cloud has storage limits)

### Issue: "Slow App Performance"

**Solutions**:
1. Use `@st.cache_resource` for model loading
2. Use `@st.cache_data` for data loading
3. Optimize text processing
4. Reduce prediction batch size
5. Optional: Use lighter model formats

### Issue: "App keeps rerunning"

**Solution**: Remove st.write() or print() statements that cause reruns. Use `st.write()` with proper caching.

### Issue: "CSV upload fails"

**Solution**: Ensure uploaded CSV has a column named exactly `comment_text` (case-sensitive).

### Issue: "Out of Memory"

**Solution**:
- Streamlit Cloud has memory limits
- Consider uploading only necessary files
- Use streaming for large CSV files
- Reduce batch prediction size

---

## Maintenance & Updates

### 1. **Update Your App**

Make changes locally, then push to GitHub:

```bash
# Make changes to files
# Test locally: streamlit run app.py

# Commit and push
git add .
git commit -m "Update: [describe changes]"
git push origin main
```

Streamlit Cloud automatically redeploys within minutes.

### 2. **Retrain Model**

If you retrain your model:

```bash
# Replace old model files
# cp new_toxicity_model_BiLSTM.keras ./

git add toxicity_model_BiLSTM.keras
git commit -m "Update: Retrained BiLSTM model"
git push
```

### 3. **Monitor App Usage**

In Streamlit Cloud dashboard:
- View app analytics
- Check CPU and memory usage
- View error logs
- Track deployment history

### 4. **Scaling Concerns**

If your app gets heavy traffic:
- **Free tier**: Limited resources, may slow down
- **Upgrade to Pro**: For better performance and features
- **Self-host**: Deploy on AWS, Heroku, or Google Cloud

---

## Alternative Deployment Options

### Option 1: Deploy to Heroku

```bash
# Install Heroku CLI
# Create Procfile:
echo "web: streamlit run app.py --logger.level=error" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

### Option 2: Deploy to AWS/Google Cloud

Use containerization with Docker:

```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### Option 3: Self-Host on Linux Server

```bash
# SSH into server
# Clone repository
git clone https://github.com/YOUR_USERNAME/comment-toxicity-detector.git
cd comment-toxicity-detector

# Install dependencies
pip install -r requirements.txt

# Run with gunicorn
gunicorn --bind 0.0.0.0:8501 app:app
```

---

## Quick Reference

### GitHub Commands
```bash
git status                          # Check changes
git add .                           # Stage changes
git commit -m "message"             # Commit
git push origin main                # Push to GitHub
git pull origin main                # Pull latest
```

### Streamlit Commands
```bash
streamlit run app.py                # Run locally
streamlit config show               # View configuration
streamlit cache clear               # Clear cache
```

### Testing Checklist
- [ ] App runs locally without errors
- [ ] All tabs work correctly
- [ ] Sample predictions are accurate
- [ ] CSV upload processes correctly
- [ ] Visualizations display properly
- [ ] No console errors or warnings
- [ ] Model loads quickly
- [ ] All required files are in Git

---

## Support & Resources

- **Streamlit Documentation**: https://docs.streamlit.io
- **Streamlit Cloud Docs**: https://docs.streamlit.io/streamlit-cloud
- **GitHub Help**: https://docs.github.com
- **TensorFlow/Keras**: https://www.tensorflow.org
- **Streamlit Community Forum**: https://discuss.streamlit.io

---

## Summary

To deploy your Comment Toxicity Detector on Streamlit Cloud:

1. ✅ Prepare project with `requirements.txt`
2. ✅ Initialize Git and push to GitHub
3. ✅ Sign in to Streamlit Cloud with GitHub
4. ✅ Click "New app" and select your repository
5. ✅ Streamlit automatically deploys
6. ✅ Share your public URL
7. ✅ Monitor and maintain through GitHub updates

**Deployment is complete! Your app is now live and accessible to everyone.** 🎉


