from transformers import pipeline

# Load GPT-Neo model from Hugging Face
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

def generate_html_with_gpt_neo(prompt: str) -> str:
    """
    Generates HTML code based on the given prompt using GPT-Neo.
    
    Args:
        prompt (str): Description of the desired HTML code.
    
    Returns:
        str: Generated HTML code.
    """
    try:
        generated = generator(prompt, max_length=1500, num_return_sequences=1)
        return generated[0]['generated_text'].strip()
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Example usage
if __name__ == "__main__":
    user_input = input("Enter your HTML prompt: ")
    generated_code = generate_html_with_gpt_neo(user_input)
    print("\nGenerated HTML Code:\n")
    print(generated_code)
