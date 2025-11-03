# Summarization + Named Entity Recognition (NER) Dashboard

A local transformer-based Streamlit dashboard where users can choose between text summarization and named entity recognition (NER). Users input text and get either a concise summary or a list of extracted entities. The dashboard loads fine-tuned models for better performance.

## 🚀 Features

- **Text Summarization**: Generate concise summaries using a fine-tuned DistilBART model
- **Named Entity Recognition**: Extract entities (PERSON, ORG, LOC, DATE, etc.) using a fine-tuned DistilBERT model
- **Interactive Dashboard**: User-friendly Streamlit interface
- **Local Processing**: All processing is done locally, no external API calls required

## 📁 Project Structure

```
├── app.py                      # Main Streamlit application
├── summarization_processing.py # Text summarization module
├── ner_processing.py           # NER processing module
├── requirements.txt            # Required Python packages
├── colab/                      # Jupyter notebooks
│   ├── Named_Entity.ipynb      # NER model training/testing
│   └── summarizer.ipynb        # Summarization model training/testing
└── utils/                      # Fine-tuned models (excluded from git)
```

## 🤖 Models

### Summarization Model
- **Base Model**: `sshleifer/distilbart-cnn-12-6`
- **Type**: Fine-tuned DistilBART model for text summarization
- **Loading**: Uses `AutoModelForSeq2SeqLM.from_pretrained`

### NER Model
- **Base Model**: `distilbert-base-cased`
- **Type**: Fine-tuned DistilBERT model for named entity recognition
- **Loading**: Loaded with custom number of labels

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Shreena88/Summarization-NER.git
cd Summarization-NER
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell
.\venv\Scripts\activate

# Windows Command Prompt
venv\Scripts\activate.bat

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Models
Ensure your fine-tuned models are available locally in the `utils/` folder:
- `utils/models/summarization_model/`
- `utils/models/ner_model/`

### 5. Run the Application
```bash
streamlit run app.py
```

## 🎯 Usage

1. **Launch the Dashboard**: Run `streamlit run app.py`
2. **Choose Option**: Select either "Summarization" or "NER" from the sidebar
3. **Input Text**: Enter your text in the provided text area
4. **Get Results**: 
   - **Summarization**: Receive a concise summary of your text
   - **NER**: View extracted named entities in a structured format

## 📊 Jupyter Notebooks

The `colab/` directory contains Jupyter notebooks for model training and testing:
- `Named_Entity.ipynb`: NER model development and evaluation
- `summarizer.ipynb`: Summarization model development and evaluation

## 🔧 Technical Details

- **Framework**: Streamlit for web interface
- **ML Library**: Transformers (Hugging Face)
- **Models**: Fine-tuned transformer models
- **Processing**: Local inference, no external APIs
- **Environment**: Python virtual environment recommended

## 📝 Notes

- All processing is performed locally for privacy and speed
- Recommended to use a virtual environment to avoid dependency conflicts
- Models should be stored in the `utils/` folder (excluded from version control)
- Can be extended to support multi-language processing or custom models

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for any improvements.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).