# Chatbot Companion: Ask Anything

This project is a sophisticated chatbot application that leverages **Natural Language Processing (NLP)** and **Machine Learning** to provide interactive responses to various user queries. It covers topics ranging from jokes, motivational books, and health to cutting-edge technologies like machine learning, artificial intelligence, and large language models (LLMs). The chatbot is built using **Streamlit** for a user-friendly web interface and **scikit-learn** for training the intent classifier.

## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [How It Works](#how-it-works)
- [Customization](#customization)
- [Sample Screenshots](#sample-screenshots)
- [Future Improvements](#future-improvements)
- [License](#license)

## Features
- **Multi-Intent Recognition**: Handles a wide variety of user intents, including:
  - Jokes and fun conversations
  - Health and wellness tips
  - Technology-related queries (AI, ML, Data Science, LLMs)
  - Book recommendations (Motivational, Fun)
  - Budgeting and financial advice
  - General small talk (greetings, farewells, thanks)
- **Interactive Interface**: Provides a clean, responsive web interface using **Streamlit**.
- **Machine Learning**: Uses **Logistic Regression** for text classification and **TF-IDF vectorization** for feature extraction.
- **Randomized Responses**: Delivers dynamic conversations with varied responses based on intent.
- **Extensible and Customizable**: Easily add or modify intents, patterns, and responses via a JSON file.

## Tech Stack
![Python Badge](https://img.shields.io/badge/Python-3.x-blue.svg)
![Streamlit Badge](https://img.shields.io/badge/Streamlit-0.87.0-red.svg)
![Scikit-learn Badge](https://img.shields.io/badge/Scikit--learn-0.24.2-orange.svg)

- **Python**: The core programming language used for building and training the chatbot.
- **Streamlit**: Used for creating a highly interactive and easy-to-use web interface.
- **scikit-learn**: A powerful machine learning library used to classify user inputs.
- **nltk**: Natural Language Toolkit for tokenizing and processing text.

## Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/prajwalk-1/NLP-Powered-Chatbot-Built-with-Streamlit.git
   ```
2. Navigate to the project directory:
   ```bash
   cd NLP-Powered-Chatbot-Built-with-Streamlit
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. If necessary, download the **nltk** data used for tokenization:
   ```bash
   python -m nltk.downloader punkt
   ```

## How to Run

1. Start the chatbot by running the following command:
   ```bash
   streamlit run chatbot.py
   ```

2. Once the server starts, a web interface will open automatically in your default browser at:
   ```
   http://localhost:8501/
   ```

3. Interact with the chatbot by typing your questions in the input box. The chatbot will respond with pre-trained answers based on the identified intent.


### File Descriptions:

- **`chatbot.py`**: Contains the main logic for the chatbot, including text vectorization, model training, and user interaction.
- **`intents.json`**: Stores the chatbot's various intents (e.g., greetings, jokes, AI, etc.), along with corresponding user input patterns and responses.
- **`requirements.txt`**: Lists all Python dependencies required to run the chatbot.

## How It Works

1. **User Input Processing**:
   - The chatbot accepts user input through the Streamlit interface and processes it using **TF-IDF vectorization** to convert the text into numerical features.

2. **Intent Classification**:
   - A **Logistic Regression** model is used to predict the intent of the user's query. It matches the input to one of the predefined intents (e.g., "jokes," "AI," "books").

3. **Response Generation**:
   - Once an intent is identified, the chatbot selects a random response from the available options for that specific intent, keeping the conversation engaging and dynamic.

### Example:
For an input like "Tell me a joke," the chatbot might identify the "jokes" intent and respond with a random joke from its predefined list of responses.

## Customization

To modify the chatbot’s behavior, you can edit the `intents.json` file:
- **Add New Intents**: Include a new intent with patterns and responses.
  ```json
  {
    "tag": "new_intent",
    "patterns": ["User input 1", "User input 2"],
    "responses": ["Response 1", "Response 2"]
  }
  ```

- **Modify Responses**: Update the responses for an existing intent to make the bot’s replies more diverse.

## Sample Screenshots

### Chatbot Web Interface
![Chatbot Interface](https://github.com/user-attachments/assets/2a42d607-bea3-4ab3-a0f8-f8ea06163817)

### Intent and Response Example
![Chatbot Interaction](https://github.com/user-attachments/assets/d442b68e-71da-44ad-aee3-2b5f8ffb52dc)

![Screenshot 2024-10-14 173704](https://github.com/user-attachments/assets/f051ab44-e1ec-4ed9-9ba7-a04e1ca1a987)


## Future Improvements
- **Real-Time Data**: Integrate real-time data for intents such as weather or news using APIs.
- **NLP Enhancements**: Implement more advanced NLP models, such as **BERT** or **GPT** for better language understanding.
- **User Personalization**: Add user authentication and personalized responses based on user history or preferences.

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute the code as per the terms of the license.
