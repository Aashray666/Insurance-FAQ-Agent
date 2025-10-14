# 🧑‍💻 Developer Guide - File Structure Explained

This guide explains what each file does in simple terms for developers who want to understand or modify the code.

## 📁 Root Directory Files

### 🚀 Main Application
- **`app.py`** - The main Streamlit web application
  - Creates the web interface users see
  - Handles user input and displays responses
  - Connects all services together (FAQ, RAG, Claims, Escalation)
  - Contains the chat interface and sidebar controls

### 📦 Configuration Files
- **`requirements.txt`** - List of Python packages needed
  - Contains all dependencies like streamlit, transformers, etc.
  - Use: `pip install -r requirements.txt`

- **`README.md`** - User installation and setup guide
  - How to install and run the project
  - For end users and deployment

- **`DEVELOPER_GUIDE.md`** - This file you're reading
  - Technical explanation for developers
  - Code structure and file purposes

### 📄 Insurance Documents
- **`Reliance_Private_Car_Package_Policy_wording.pdf`**
- **`Reliance_Commercial_Vehicles_Package_Policy_wording.pdf`** 
- **`Reliance_Two_wheeler_Package_Policy_wording.pdf`**
  - Original insurance policy documents
  - Used by RAG system to answer policy questions
  - Processed into vector databases for search

---

## 📂 src/ Directory - Core Source Code

### 🔧 src/services/ - Business Logic Services

#### **`response_formatter.py`** - AI Response Formatting
- **Purpose**: Formats responses using Google Gemini 2.5 Flash
- **What it does**:
  - Takes raw policy text and makes it conversational
  - Hardcoded Gemini API key for simplicity
  - Falls back to template formatting if API fails
- **Key function**: `format_rag_response()` - converts chunks to natural language

#### **`faq_matcher.py`** - FAQ Question Matching
- **Purpose**: Matches user questions to pre-written FAQ answers
- **What it does**:
  - Loads FAQ database from JSON file
  - Uses keyword matching to find best FAQ
  - Returns structured FAQ responses
- **Key function**: `match_faq()` - finds best matching FAQ

#### **`rag_retriever.py`** - Document Search Engine
- **Purpose**: Searches through insurance policy documents
- **What it does**:
  - Uses vector databases to find relevant policy sections
  - Converts user questions to embeddings
  - Returns most similar document chunks
- **Key function**: `retrieve_chunks()` - finds relevant policy text

#### **`claim_service.py`** - Claim Status Management
- **Purpose**: Handles insurance claim status queries
- **What it does**:
  - Loads sample claim data from JSON
  - Looks up claims by claim number (PC2024001, etc.)
  - Returns claim status, amount, dates
- **Key function**: `lookup_claim()` - finds claim by ID

#### **`escalation_service.py`** - Query Escalation System
- **Purpose**: Handles complex queries that need human help
- **What it does**:
  - Detects when user needs human assistance
  - Creates support tickets with unique IDs
  - Stores escalated queries in JSON file
- **Key function**: `escalate_query()` - creates support ticket

#### **Other Service Files**:
- **`build_vector_dbs.py`** - Builds search indexes from PDF documents
- **`pdf_processor.py`** - Extracts text from PDF files
- **`vector_db_manager.py`** - Manages FAISS vector databases
- **`generate_sample_claims.py`** - Creates demo claim data
- **`error_handler.py`** - Handles errors gracefully
- **`interfaces.py`** - Defines data interfaces

### 📊 src/models/ - Data Structures

#### **`response.py`** - Response Format
- **Purpose**: Defines how responses are structured
- **Contains**: `FormattedResponse` class with content, source, confidence

#### **`chunk.py`** - Document Chunk
- **Purpose**: Represents a piece of policy document
- **Contains**: Text content, embeddings, similarity scores

#### **`claim_info.py`** - Claim Data
- **Purpose**: Structure for claim information
- **Contains**: Claim ID, status, amount, dates, customer info

#### **`faq_response.py`** - FAQ Answer
- **Purpose**: Structure for FAQ responses
- **Contains**: Question, answer, keywords, category

#### **Other Model Files**:
- **`agent_response.py`** - Agent response structure
- **`query_intent.py`** - User intent classification

### 🤖 src/agent/ - AI Agent Logic (Optional)

#### **`insurance_agent.py`** - Main Agent Class
- **Purpose**: Orchestrates all services using LangGraph
- **What it does**:
  - Routes queries to appropriate services
  - Handles complex multi-step conversations
  - Currently disabled in favor of direct services

#### **Other Agent Files**:
- **`graph_nodes.py`** - LangGraph node definitions
- **`conversation_state.py`** - Manages conversation context

---

## 💾 data/ Directory - Data Files

### 📋 **`faq_database.json`** - FAQ Questions & Answers
- **Purpose**: Database of common insurance questions
- **Structure**:
  ```json
  {
    "private_car": [
      {
        "question": "How do I file a claim?",
        "answer": "Call our helpline...",
        "keywords": ["claim", "file", "submit"],
        "category": "claims"
      }
    ]
  }
  ```
- **How to modify**: Add new questions/answers in the same format

### 📊 **`sample_claims.json`** - Demo Claim Data
- **Purpose**: Sample insurance claims for testing
- **Structure**:
  ```json
  [
    {
      "claim_id": "PC2024001",
      "status": "Under Review",
      "claim_amount": 25000,
      "customer_name": "John Doe"
    }
  ]
  ```
- **How to modify**: Add new claims with unique IDs

### 🔄 **`escalations.json`** - Escalated Queries
- **Purpose**: Stores queries that need human help
- **Auto-created**: Generated when users escalate queries
- **Contains**: Ticket ID, query, reason, timestamp

### 🔍 **`vector_dbs/`** - Search Indexes
- **Purpose**: Pre-built search indexes for fast document search
- **Contains**:
  - `private_car_vectordb/` - Car policy search index
  - `commercial_vehicle_vectordb/` - Commercial policy search index  
  - `two_wheeler_vectordb/` - Two-wheeler policy search index
- **How created**: By running `build_vector_dbs.py`

---

## 🔄 How Everything Works Together

### 1. **User asks a question** in `app.py`
### 2. **App routes the query**:
   - **Claim number?** → `claim_service.py`
   - **Common question?** → `faq_matcher.py`
   - **Policy question?** → `rag_retriever.py`
### 3. **Service processes the query**:
   - Searches databases/documents
   - Finds relevant information
### 4. **Response gets formatted**:
   - `response_formatter.py` makes it conversational using Gemini
### 5. **User sees the answer** in the web interface

### 🔄 **Escalation Flow**:
- If no good answer found → `escalation_service.py`
- Creates support ticket
- Shows professional escalation message

---

## 🛠️ Common Developer Tasks

### **Adding New FAQs**
1. Edit `data/faq_database.json`
2. Add question, answer, keywords
3. Restart app

### **Adding New Insurance Documents**
1. Place PDF in root directory
2. Update `src/services/build_vector_dbs.py`
3. Run: `python src/services/build_vector_dbs.py`
4. Restart app

### **Modifying Gemini Responses**
1. Edit `src/services/response_formatter.py`
2. Change the `_create_prompt()` method
3. Modify temperature/max_tokens settings

### **Adding New Claim Data**
1. Edit `data/sample_claims.json`
2. Add new claim with unique ID
3. Restart app

### **Customizing UI**
1. Edit `app.py`
2. Modify Streamlit components
3. Change sidebar content, colors, layout

---

## 🧪 Testing Individual Components

### **Test FAQ Matching**
```python
from src.services.faq_matcher import FAQMatcher
faq = FAQMatcher("data/faq_database.json")
result = faq.match_faq("How do I file a claim?", "Private Car")
print(result.answer)
```

### **Test RAG Search**
```python
from src.services.rag_retriever import RAGRetriever
rag = RAGRetriever()
chunks = rag.retrieve_chunks("What is covered?", "Private Car")
print(chunks[0].content)
```

### **Test Gemini Formatting**
```python
from src.services.response_formatter import ResponseFormatter
formatter = ResponseFormatter(use_gemini=True)
# Test with chunks...
```

---

## 🎯 Key Design Decisions

### **Why Direct Services Instead of Agent?**
- **Simpler**: Easier to understand and debug
- **Faster**: No complex routing overhead
- **Reliable**: Less points of failure
- **Transparent**: Clear flow from query to response

### **Why Hardcoded API Key?**
- **College Project**: Simplifies setup for demonstration
- **No Environment Variables**: Works out of the box
- **Easy Deployment**: No configuration needed

### **Why JSON Files Instead of Database?**
- **Simplicity**: Easy to read and modify
- **Portability**: Works anywhere without setup
- **Transparency**: Can see exactly what data exists
- **College-Friendly**: No database installation required

---

## 🚀 Performance Notes

- **First Load**: ~10-15 seconds (loading vector databases)
- **Subsequent Queries**: ~1-3 seconds
- **Memory Usage**: ~2GB RAM (due to transformer models)
- **Storage**: ~100MB (vector databases)

---

**This guide should help any developer understand and modify the codebase!** 🎉