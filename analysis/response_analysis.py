import pandas as pd
import matplotlib.pyplot as plt
import os

#Name of the CSV file
file_name = '../code/feedback.csv'

# Check if the file exists
if not os.path.exists(file_name):
    print(f"Error: The file '{file_name}' was not found.")
else:
    # Read the CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(file_name)
    except pd.errors.EmptyDataError:
        print(f"The file '{file_name}' is empty. No data to analyze.")
    else:
        # Check if the DataFrame is empty after reading
        if df.empty:
            print(f"The file '{file_name}' is empty. No data to analyze.")
        else:
            print("Analysis of Feedback Data:")

            # Count the occurrences of each feedback type
            feedback_counts = df['feedback_type'].value_counts()

            # Calculate the total number of feedback entries
            total_feedback = len(df)

            # Check if 'thumbs_down' is in the counts to avoid KeyError
            thumbs_down_count = feedback_counts.get('thumbs_down', 0)

            # Calculate the percentage of "thumbs down" feedback
            thumbs_down_percentage = (thumbs_down_count / total_feedback) * 100 if total_feedback > 0 else 0

            print(f"\nTotal feedback entries: {total_feedback}")
            print(f"Thumbs down count: {thumbs_down_count}")
            print(f"Thumbs up count: {feedback_counts.get('thumbs_up', 0)}")
            print(f"\nPercentage of 'thumbs down' feedback: {thumbs_down_percentage:.2f}%")

            # Create a bar plot to visualize the feedback distribution
            plt.figure(figsize=(8, 6))
            feedback_counts.plot(kind='bar', color=['skyblue', 'salmon'])
            plt.title('Distribution of User Feedback')
            plt.xlabel('Feedback Type')
            plt.ylabel('Count')
            plt.xticks(rotation=0)

            # Add value labels on top of the bars
            for index, value in enumerate(feedback_counts):
                plt.text(index, value + 0.5, str(value), ha='center', va='bottom')

            plt.tight_layout()

            # Save the plot to a file
            plot_file = 'feedback_distribution.png'
            plt.savefig(plot_file)
            print(f"\nBar chart saved to '{plot_file}'")

