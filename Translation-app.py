import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.IndicTransToolkit.processor import IndicProcessor

# Paths to the locally stored models
MODEL_PATHS = {
    "en-indic": "E:/NMT/Models/indictrans2-en-indic-dist-200M/",
    "indic-en": "E:/NMT/Models/indictrans2-indic-en-dist-200M/",
    "indic-indic": "E:/NMT/Models/indictrans2-indic-indic-dist-320M/",
}

# Initialize the application
st.set_page_config(page_title="Translation App", layout="wide")

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox(
    "Select a Translation Model",
    ["en-indic", "indic-en", "indic-indic"],
    index=0,
)

# Update the title based on the selected model
st.title(f"Translation App - {model_choice.replace('-', ' ').title()}")

# Load the selected model
@st.cache_resource
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
    processor = IndicProcessor(inference=True)
    return tokenizer, model, processor

tokenizer, model, processor = load_model(MODEL_PATHS[model_choice])
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Define language options based on the model
LANGUAGE_OPTIONS = {
    "en-indic": {
        "source": ["eng_Latn"],
        "target": [
            "asm_Beng", "ben_Beng", "brx_Deva", "doi_Deva", "gom_Deva", "guj_Gujr",
            "hin_Deva", "kan_Knda", "kas_Arab", "kas_Deva", "mai_Deva", "mal_Mlym",
            "mar_Deva", "mni_Beng", "mni_Mtei", "npi_Deva", "ory_Orya", "pan_Guru",
            "san_Deva", "sat_Olck", "snd_Arab", "snd_Deva", "tam_Taml", "tel_Telu",
            "urd_Arab",
        ],
    },
    "indic-en": {
        "source": [
            "asm_Beng", "ben_Beng", "brx_Deva", "doi_Deva", "gom_Deva", "guj_Gujr",
            "hin_Deva", "kan_Knda", "kas_Arab", "kas_Deva", "mai_Deva", "mal_Mlym",
            "mar_Deva", "mni_Beng", "mni_Mtei", "npi_Deva", "ory_Orya", "pan_Guru",
            "san_Deva", "sat_Olck", "snd_Arab", "snd_Deva", "tam_Taml", "tel_Telu",
            "urd_Arab",
        ],
        "target": ["eng_Latn"],
    },
    "indic-indic": {
        "source": [
            "asm_Beng", "ben_Beng", "brx_Deva", "doi_Deva", "gom_Deva", "guj_Gujr",
            "hin_Deva", "kan_Knda", "kas_Arab", "mai_Deva", "mal_Mlym", "mar_Deva",
            "mni_Mtei", "npi_Deva", "ory_Orya", "pan_Guru", "san_Deva", "sat_Olck",
            "snd_Deva", "tam_Taml", "tel_Telu", "urd_Arab",
        ],
        "target": [
            "asm_Beng", "ben_Beng", "brx_Deva", "doi_Deva", "gom_Deva", "guj_Gujr",
            "hin_Deva", "kan_Knda", "kas_Arab", "mai_Deva", "mal_Mlym", "mar_Deva",
            "mni_Mtei", "npi_Deva", "ory_Orya", "pan_Guru", "san_Deva", "sat_Olck",
            "snd_Deva", "tam_Taml", "tel_Telu", "urd_Arab",
        ],
    },
}

# Input for source and target language selection
if model_choice == "en-indic":
    src_lang = "eng_Latn"
    tgt_lang = st.selectbox("Select Target Language", LANGUAGE_OPTIONS[model_choice]["target"])
elif model_choice == "indic-en":
    src_lang = st.selectbox("Select Source Language", LANGUAGE_OPTIONS[model_choice]["source"])
    tgt_lang = "eng_Latn"
else:
    src_lang = st.selectbox("Select Source Language", LANGUAGE_OPTIONS[model_choice]["source"])
    tgt_lang = st.selectbox("Select Target Language", LANGUAGE_OPTIONS[model_choice]["target"])

# Input for text to translate
st.subheader("Enter Text to Translate")
input_text = st.text_area("Input Text", height=150)

# Translation button
# Translation button
if st.button("Translate"):
    if input_text.strip():
        batch = processor.preprocess_batch([input_text], src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )
        with tokenizer.as_target_tokenizer():
            generated_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        translations = processor.postprocess_batch(generated_tokens, lang=tgt_lang)
        st.session_state.last_translation = translations[0]  # Store the latest translation in session state
        st.success(f"Translation: {translations[0]}")
    else:
        st.error("Please enter text to translate.")

# Ensure translation history is initialized
if "history" not in st.session_state:
    st.session_state.history = []

# Save Translation button
if st.button("Save Translation"):
    if "last_translation" in st.session_state:
        st.session_state.history.append((input_text, st.session_state.last_translation))
        st.success("Translation saved!")
    else:
        st.error("No translation available to save. Please translate first.")

# Clear Translation History button
if st.button("Clear History"):
    st.session_state.history = []
    st.success("Translation history cleared!")

# Display Translation History
st.subheader("Translation History")
for i, (original, translated) in enumerate(st.session_state.history, start=1):
    st.write(f"**{i}.**")
    st.write(f"**Original ({src_lang}):** {original}")
    st.write(f"**Translated ({tgt_lang}):** {translated}")
