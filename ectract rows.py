import pandas as pd

# Replace 'your_file.csv' with the path to your CSV file
file_path = "filtered_posts_for_classification.csv"

# Read the CSV file
data = pd.read_csv(file_path)

# Extract the first 50 rows
first_five_rows = data.head(50)

# Save the first 50 rows to a new CSV file
first_five_rows.to_csv('first_five_rows_classification.csv', index=False)

print("The first five rows have been saved to 'first_five_rows.csv'")