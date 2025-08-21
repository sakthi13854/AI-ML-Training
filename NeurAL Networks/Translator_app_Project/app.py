import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AI Language Translator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitTranslator:
    @staticmethod
    @st.cache_resource
    def load_model(target_language):
        model_name = f"Helsinki-NLP/opus-mt-en-{target_language}"
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            return tokenizer, model
        except Exception as e:
            st.error(f"Error loading model for {target_language}: {str(e)}")
            return None, None

    @staticmethod
    def translate_text(text, tokenizer, model):
        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with st.spinner("Translating..."):
                translated_tokens = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
            return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        except Exception as e:
            return f"Translation error: {str(e)}"

def main():
    st.title("AI Language Translator")
    st.markdown("### Powered by Transformer Models")
    st.markdown("---")

    languages = {
        'fr': '🇫🇷 French',
        'es': '🇪🇸 Spanish',
        'de': '🇩🇪 German',
        'it': '🇮🇹 Italian',
        'pt': '🇵🇹 Portuguese',
        'ru': '🇷🇺 Russian',
        'zh': '🇨🇳 Chinese',
        'ja': '🇯🇵 Japanese',
        'ko': '🇰🇷 Korean',
        'nl': '🇳🇱 Dutch'
    }

    selected_lang = st.sidebar.selectbox(
        "Choose target language:",
        options=list(languages.keys()),
        format_func=lambda x: languages[x],
        index=0
    )

    st.sidebar.info(f"**Model**: Helsinki-NLP/opus-mt-en-{selected_lang}")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🇬🇧 English Input")
        input_method = st.radio("Choose input method:", ["Text Area", "Single Line", "Example Sentences"], horizontal=True)

        if input_method == "Text Area":
            input_text = st.text_area("Enter your English text:", height=200, placeholder="Type your English text here...")
        elif input_method == "Single Line":
            input_text = st.text_input("Enter your English text:", placeholder="Type a sentence...")
        else:
            examples = [
                "Hello, how are you today?",
                "I love learning new languages.",
                "The weather is beautiful today.",
                "Thank you for your help.",
                "What time is it?",
                "Where is the nearest restaurant?",
                "I would like to book a hotel room.",
                "How much does this cost?"
            ]
            input_text = st.selectbox("Choose an example sentence:", [""] + examples)

    with col2:
        st.markdown(f"### {languages[selected_lang]} Translation")
        translation_container = st.container()

        if input_text and input_text.strip():
            tokenizer, model = StreamlitTranslator.load_model(selected_lang)
            if tokenizer and model:
                translation = StreamlitTranslator.translate_text(input_text, tokenizer, model)
                st.text_area("Translation:", value=translation, height=200, disabled=True)

                st.markdown("---")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Input Length", len(input_text))
                with col_b:
                    st.metric("Translation Length", len(translation))

                if st.button("Copy Translation", key="copy_btn"):
                    st.success("Translation copied to clipboard! (In a real app)")
            else:
                st.error("Failed to load translation model.")
        else:
            st.info("Enter some English text to see the translation")

if __name__ == "__main__":
    main()
