import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load the trained model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = T5Tokenizer.from_pretrained("./final_model")  # Path to saved tokenizer
    model = T5ForConditionalGeneration.from_pretrained("./final_model")  # Path to saved model
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Streamlit app
st.title("HTML Code Generator")
st.write("Enter a prompt to generate HTML code for a restaurant's webpage.")

# Input field for the prompt
prompt = st.text_input("Enter your prompt:")

# Generate button
if st.button("Generate HTML"):
    if prompt.strip():
        # Tokenize the input prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        
        # Generate HTML code
        output_ids = model.generate(input_ids, max_length=512)
        generated_html = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Display the generated HTML
        st.subheader("Generated HTML Code:")
        st.code(generated_html, language="html")
    else:
        st.warning("Please enter a valid prompt.")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit and Transformers.")
