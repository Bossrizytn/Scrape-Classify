import streamlit as st
import pandas as pd
from transformers import pipeline
import requests
from bs4 import BeautifulSoup

# Function to scrape website content
def scrape_website(url, limit=500):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extralilllict the text content from the website (you can modify this to target specific tags)
            text = soup.get_text()
            # Limit the text content
            return text[:limit]
        else:
            st.error(f"Failed to fetch the website content. Status code: {response.status_code}")
            return ""
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        return ""

# UI Title
st.title('Zero-Shot Classification with Web Scraping')

# Upload the file
uploaded_file = st.file_uploader("Choose a file", type="tsv")

# If file is uploaded
if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file, sep="\t")
    st.write("Dataset Preview:", df.head())

    # Extract the combined labels
    candidate_labels = df["Content Taxonomy v3.0 Tiered Categories"].dropna().unique().tolist()

    # Option for input method
    input_method = st.radio("Choose input method for sequence to classify", ("Manual Text", "Scrape Website"))

    # If the user selects "Scrape Website"
    if input_method == "Scrape Website":
        # URL input
        url = st.text_input("Enter the URL to scrape")
        limit = st.number_input("Limit the length of scraped content (characters)", min_value=100, max_value=5000, value=500)

        if url:
            sequence_to_classify = scrape_website(url, limit)
            st.write("Scraped content preview:", sequence_to_classify[:500])  # Preview of the scraped content
    else:
        # Manual input text
        sequence_to_classify = st.text_area("Enter text to classify", 
                                            value="""Looking for your next adventure? Discover top travel destinations for 2024, 
                                                     from Europe’s hidden gems to tropical escapes. 
                                                     We’ve got tips on budget-friendly travel, luxury getaways, 
                                                     and the latest must-have gadgets for a comfortable and stylish trip""")

    # Input confidence threshold
    confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.40)

    # Zero-shot classification button
    if st.button('Classify'):
        # Initialize the zero-shot classification pipeline
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        # Perform the classification
        result = classifier(sequence_to_classify, candidate_labels, multi_label=True)

        # Filter results based on confidence threshold
        filtered_results = []
        for label, score in zip(result['labels'], result['scores']):
            if score >= confidence_threshold:
                filtered_results.append({'label': label, 'score': score})
            else:
                break  # Stop when confidence is below threshold

        # Display the results
        st.write("Filtered Results:", filtered_results)

        # Filter the dataframe based on filtered labels
        filtered_labels = [res['label'] for res in filtered_results]
        filtered_df = df[df['Content Taxonomy v3.0 Tiered Categories'].isin(filtered_labels)]
        
        # Display the filtered dataframe
        st.write("Filtered Dataframe", filtered_df)

        # Extract the second set of candidate labels
        if "Unnamed: 2" in filtered_df.columns:
            candidate_labels2 = filtered_df["Unnamed: 2"].dropna().unique().tolist()

            # Perform classification on the second set
            result2 = classifier(sequence_to_classify, candidate_labels2, multi_label=True)

            # Display top 10 results from the second classification
            st.write("Second Classification Results (Top 10):", result2['labels'][:10])
