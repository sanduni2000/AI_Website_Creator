import pandas as pd

# Load the dataset
file_path = 'ExtendedRestaurantHTMLDataset.csv'  # Update this with your file's path
data = pd.read_csv(file_path)

# Create the 'Prompt' column
data['Prompt'] = data['Restaurant Name'].apply(lambda name: f"Create a webpage using HTML with a header and a footer with contact information for the restaurant named as {name}. Use simple CSS for styling.")

# Save the updated dataset to a new file
output_path = 'UpdatedRestaurantHTMLDataset.csv'  # Update this with your desired output file name
data.to_csv(output_path, index=False)

print(f"Updated dataset saved to {output_path}")
