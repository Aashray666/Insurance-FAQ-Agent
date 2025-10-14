# 🛡️ Insurance FAQ Agent

A complete AI-powered insurance assistant that answers policy questions, handles claims, and searches through insurance documents using Google Gemini 2.5 Flash.

## � Tuable of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation Guide](#installation-guide)
- [Setting Up Insurance Documents](#setting-up-insurance-documents)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- **🤖 AI-Powered Responses**: Google Gemini 2.5 Flash for natural conversations
- **📋 FAQ System**: Instant answers to common insurance questions
- **🔍 Document Search**: RAG-based search through policy documents
- **📊 Claim Management**: Check claim status with claim numbers
- **🔄 Smart Escalation**: Automatic escalation for complex queries
- **📱 Web Interface**: Clean Streamlit-based user interface

## 🔧 Prerequisites

- **Python 3.8+** (Recommended: Python 3.11)
- **Git** (for cloning)
- **Google AI Studio Account** (for Gemini API key)

## 📦 Installation Guide

### Step 1: Clone or Download Project
```bash
# Option A: Clone with Git
git clone <repository-url>
cd Insurance_FAQ_Agent

# Option B: Download and extract ZIP file
# Then navigate to the extracted folder
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Get Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the API key (starts with `AIza...`)

### Step 5: Configure API Key
The API key is already hardcoded in the application for simplicity. If you want to use your own key:

**Option A: Use Hardcoded Key (Recommended for Demo)**
- The app uses a pre-configured API key
- No additional setup needed

**Option B: Use Your Own Key**
- Open `src/services/response_formatter.py`
- Find line: `self.api_key = "AIzaSyCbCM4jwE8BrDuTAfOlIOUoXjMw2CCzYP0"`
- Replace with your API key: `self.api_key = "your-api-key-here"`

## 📄 Setting Up Insurance Documents

### Current Setup (Ready to Use)
The project comes with pre-loaded insurance documents:
- ✅ **Private Car Policy** (`Reliance_Private_Car_Package_Policy_wording.pdf`)
- ✅ **Commercial Vehicle Policy** (`Reliance_Commercial_Vehicles_Package_Policy_wording.pdf`)
- ✅ **Two-wheeler Policy** (`Reliance_Two_wheeler_Package_Policy_wording.pdf`)

### Adding New Insurance Documents

If you want to add your own insurance PDFs:

1. **Place PDF files** in the root directory
2. **Update the document paths** in `src/services/build_vector_dbs.py`:
   ```python
   POLICY_DOCUMENTS = {
       "Private Car": "your_car_policy.pdf",
       "Commercial Vehicle": "your_commercial_policy.pdf", 
       "Two-wheeler": "your_twowheeler_policy.pdf"
   }
   ```
3. **Rebuild vector databases**:
   ```bash
   python src/services/build_vector_dbs.py
   ```

### Vector Database Setup
The project includes pre-built vector databases in `data/vector_dbs/`. If you need to rebuild them:

```bash
# Navigate to project directory
cd Insurance_FAQ_Agent

# Activate virtual environment
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Rebuild vector databases
python src/services/build_vector_dbs.py
```

## ⚙️ Configuration

### FAQ Database
- **Location**: `data/faq_database.json`
- **Format**: JSON with questions, answers, and keywords
- **Customization**: Edit the JSON file to add/modify FAQs

### Sample Claims Data
- **Location**: `data/sample_claims.json`
- **Purpose**: Demo claim status functionality
- **Sample Claims**: PC2024001, CV2024001, TW2024001

### Escalation System
- **Storage**: `data/escalations.json` (auto-created)
- **Purpose**: Tracks queries that need human intervention

## 🚀 Running the Application

### Start the Application
```bash
# Make sure virtual environment is activated
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Run the Streamlit app
streamlit run app.py
```

### Access the Application
- **URL**: http://localhost:8501
- **Browser**: Opens automatically
- **Interface**: Streamlit web interface

### Using the Application
1. **Select Policy Type**: Choose from Private Car, Commercial Vehicle, or Two-wheeler
2. **Enable AI Formatting**: Check "🤖 Use Gemini AI Formatting" for better responses
3. **Ask Questions**: Type your insurance-related questions
4. **Get Responses**: Receive AI-powered or template-based answers

## 💬 Usage Examples

### Policy Questions
```
"What does my car insurance cover?"
"What are the exclusions in my policy?"
"How do I file a claim?"
"What documents are required for claims?"
```

### Claim Status Queries
```
"Check claim PC2024001"
"Status of claim CV2024001"
"Track claim TW2024001"
```

### Document Search
```
"Deductible amounts and limits"
"Coverage for theft and fire"
"Premium calculation factors"
"No claim bonus information"
```

## 📁 Project Structure

```
Insurance_FAQ_Agent/
├── app.py                           # 🚀 Main Streamlit application
├── requirements.txt                 # 📦 Python dependencies
├── README.md                        # 📖 This documentation
│
├── src/                            # 💻 Source code
│   ├── services/                   # 🔧 Core services
│   │   ├── response_formatter.py   # 🤖 Gemini AI integration
│   │   ├── faq_matcher.py         # ❓ FAQ matching logic
│   │   ├── rag_retriever.py       # 🔍 Document search
│   │   ├── claim_service.py       # 📊 Claim management
│   │   ├── escalation_service.py  # 🔄 Query escalation
│   │   └── ...                    # Other utilities
│   │
│   ├── models/                     # 📊 Data models
│   │   ├── chunk.py               # Document chunks
│   │   ├── response.py            # Response formats
│   │   └── ...                    # Other models
│   │
│   └── agent/                      # 🤖 Agent logic (optional)
│
├── data/                           # 💾 Data files
│   ├── faq_database.json          # ❓ FAQ questions & answers
│   ├── sample_claims.json         # 📊 Demo claim data
│   ├── escalations.json           # 🔄 Escalated queries
│   └── vector_dbs/                # 🔍 Document search indexes
│
└── *.pdf                          # 📄 Insurance policy documents
```

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Solution: Ensure virtual environment is activated
.venv\Scripts\activate
pip install -r requirements.txt
```

**2. Gemini API Errors**
```bash
# Check if API key is valid
# Verify internet connection
# Ensure API key has proper permissions
```

**3. Vector Database Issues**
```bash
# Rebuild vector databases
python src/services/build_vector_dbs.py
```

**4. Streamlit Port Issues**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

### Performance Tips

- **First Run**: May take time to load vector databases
- **Memory**: Requires ~2GB RAM for optimal performance
- **Storage**: Vector databases need ~100MB space

### Dependencies Issues

If you encounter dependency conflicts:
```bash
# Create fresh environment
python -m venv fresh_env
fresh_env\Scripts\activate
pip install -r requirements.txt
```

## 🎯 Key Technologies

- **Frontend**: Streamlit (Web UI)
- **AI Model**: Google Gemini 2.5 Flash
- **Vector Search**: FAISS + Sentence Transformers
- **Document Processing**: PyPDF2, PDFPlumber
- **Data Storage**: JSON files
- **Language**: Python 3.8+

## 📞 Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Verify all dependencies are installed
3. Ensure virtual environment is activated
4. Check that PDF files are in the correct location

---

## 🎓 College Project Notes

This project demonstrates:
- ✅ **Natural Language Processing** with Gemini AI
- ✅ **Retrieval Augmented Generation (RAG)**
- ✅ **Vector Database Implementation**
- ✅ **Intelligent Query Routing**
- ✅ **Professional Web Interface**
- ✅ **Real-world Insurance Domain Application**

**Perfect for demonstrating modern AI applications in the insurance industry!** 🚀