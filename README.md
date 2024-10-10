# Streamlit Zero-Shot Classification with Web Scraping

This Streamlit app performs **zero-shot classification** using Hugging Face's `transformers` library and the **BART-large-mnli** model. Additionally, the app allows you to **scrape website content** and classify the text from the scraped data, or input your own text manually.

## Features

- **Zero-Shot Classification**: Classifies input text against a list of categories from a `.tsv` file.
- **Web Scraping**: Scrapes text from a user-provided URL and uses it for classification.
- **Interactive UI**: Built with Streamlit for easy use with a simple UI.
- **Confidence Threshold**: Allows users to set a confidence threshold for classification results.

## Demo

- **[Live App Link](https://scrape-classify.streamlit.app/)**

## How to Use

1. **Upload a .tsv File**: Upload a `.tsv` file with the categories you want to classify the text against.
   - The `.tsv` file should have a column named `Content Taxonomy v3.0 Tiered Categories`.
   
2. **Choose Input Method**:
   - **Manual Input**: Enter text manually in the text area.
   - **Web Scraping**: Provide a URL to scrape content from a website. You can limit the number of characters extracted.

3. **Set Confidence Threshold**: Use the slider to set the confidence threshold for the classification.

4. **Classify**: Click the **Classify** button to run the classification and see the results.

## Installation

### Prerequisites

- Python 3.7 or higher
- Git

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
