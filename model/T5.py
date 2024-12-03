from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load T5 model
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

def generate_html_with_t5(prompt: str) -> str:
    """
    Generates HTML code based on the given prompt using T5.
    
    Args:
        prompt (str): Description of the desired HTML code.
    
    Returns:
        str: Generated HTML code.
    """
    try:
        # Prepare the input for T5 model
        input_text = f"generate HTML code: {prompt}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate HTML code
        outputs = model.generate(inputs['input_ids'], max_length=1500, num_return_sequences=1)
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_code.strip()
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Example usage
if __name__ == "__main__":
    user_input = input("Enter your HTML prompt: ")
    generated_code = generate_html_with_t5(user_input)
    print("\nGenerated HTML Code:\n")
    print(generated_code)
