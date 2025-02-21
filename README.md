# Real-Time-Language-Translation-Using-Neural-Machine-Translation-
Implementing a real-time language translation system using transformer-based Neural Machine Translation (NMT) models. The focus is on optimizing low-latency, high-quality translations for global communication and travel applications. 
This application allows users to translate text between English and various Indic languages. It supports translations in three directions:

- **English to Indic languages**
- **Indic languages to English**
- **Indic languages to Indic languages**

The application uses models provided by IndicTransToolkit, based on Transformer architectures optimized for translation tasks.

## Features

- Select between three translation models.
- Choose source and target languages based on the model.
- Translate input text with a single click.
- Save translations to history and review them later.
- Clear the translation history if required.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd <repo_name>
   ```

2. Create a Python virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate # For Linux/macOS
   env\\Scripts\\activate # For Windows
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run <file_name>.py
   ```

## Notes

- Ensure you have the models downloaded and placed in the correct paths as defined in the `MODEL_PATHS` dictionary.
- GPU acceleration is supported if CUDA is available.

## Dependencies

- Streamlit: For building the web interface.
- Transformers: For loading and using the Hugging Face translation models.
- IndicTransToolkit: For preprocessing and postprocessing Indic language text.

