# Chatbot Companion: Ask Anything

import os
import random
import ssl
import nltk
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there!", "Hello!", "Hey!", "I'm fine, thank you!", "Nothing much, how about you?"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye!", "See you later!", "Take care!", "Have a great day!"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome!", "No problem!", "Glad I could help!"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot created to assist you!", "My purpose is to help answer your questions.", "I can provide information and assistance on a variety of topics."]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help! What's the problem?", "How can I assist you today?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age. I'm a chatbot!", "I was just born in the digital world!", "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": ["To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.", "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.", "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": ["A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.", "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."]
    },
    {
        "tag": "jokes",
        "patterns": ["Tell me a joke", "Make me laugh", "I need a joke", "Do you know any jokes?", "Funny jokes"],
        "responses": [
            "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "I'm reading a book on anti-gravity. It's impossible to put down!",
            "Why don't skeletons fight each other? They don't have the guts!",
            "What do you call fake spaghetti? An impasta!",
            "Why did the math book look sad? Because it had too many problems!"
        ]
    },
    {
        "tag": "motivational",
        "patterns": ["Give me a motivational quote", "I need some motivation", "Inspire me", "What should I do today?"],
        "responses": [
            "Believe you can and you're halfway there.",
            "The only way to do great work is to love what you do.",
            "Success is not final, failure is not fatal: It is the courage to continue that counts.",
            "Don't watch the clock; do what it does. Keep going!",
            "What lies behind us and what lies before us are tiny matters compared to what lies within us."
        ]
    },
    {
        "tag": "fun_facts",
        "patterns": ["Tell me a fun fact", "I want to learn something new", "Do you know any interesting facts?", "Give me a trivia."],
        "responses": [
            "Did you know honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3000 years old and still edible!",
            "A group of flamingos is called a 'flamboyance'.",
            "Bananas are berries, but strawberries aren't.",
            "Octopuses have three hearts!",
            "The world's largest desert is actually Antarctica."
        ]
    },
    {
        "tag": "health_tips",
        "patterns": ["Give me a health tip", "How can I stay healthy?", "What should I eat for better health?", "Tell me about healthy habits."],
        "responses": [
            "Stay hydrated! Drinking enough water is crucial for your health.",
            "Incorporate more fruits and vegetables into your meals for a balanced diet.",
            "Regular physical activity can boost your mood and energy levels.",
            "Make sure to get enough sleep; it's vital for your overall health.",
            "Limit your sugar intake to maintain a healthy lifestyle."
        ]
    },
    {
        "tag": "movie_recommendations",
        "patterns": ["Recommend me a movie", "What should I watch?", "Suggest a good film", "I need a movie suggestion."],
        "responses": [
            "If you enjoy action, I recommend 'Mad Max: Fury Road'.",
            "For a great comedy, try 'Superbad'.",
            "If you like drama, 'The Shawshank Redemption' is a classic.",
            "For a romantic movie, watch 'The Notebook'.",
            "If you're into sci-fi, 'Inception' is a mind-bending experience."
        ]
    },
    {
        "tag": "travel_advice",
        "patterns": ["Where should I travel?", "Give me travel tips", "What are some good travel destinations?", "I need travel advice."],
        "responses": [
            "Consider visiting Japan for its unique culture and beautiful landscapes.",
            "Traveling to Europe can be a great experience; try to visit Italy for its cuisine.",
            "For a relaxing getaway, the Maldives is a stunning destination.",
            "Iceland is famous for its breathtaking natural wonders; you should definitely check it out.",
            "New Zealand offers incredible outdoor adventures and stunning scenery."
        ]
    },
    {
        "tag": "technology_trends",
        "patterns": ["What's new in technology?", "Tell me about tech trends", "What are the latest gadgets?", "Give me a tech update."],
        "responses": [
            "AI and machine learning are rapidly evolving fields, impacting various industries.",
            "Electric vehicles are gaining popularity as a sustainable alternative to traditional cars.",
            "Augmented reality is transforming the way we interact with digital content.",
            "5G technology is rolling out, promising faster internet speeds and connectivity.",
            "Blockchain technology is being used in various sectors beyond cryptocurrency."
        ]
    },
    {
        "tag": "book_recommendations",
        "patterns": ["Recommend a book", "What should I read?", "Suggest a good novel", "I need a book suggestion."],
        "responses": [
            "If you're into fantasy, try 'The Hobbit' by J.R.R. Tolkien.",
            "For a great mystery, 'The Girl with the Dragon Tattoo' is a thrilling read.",
            "Check out 'Sapiens: A Brief History of Humankind' for a fascinating look at our history.",
            "If you enjoy classics, 'Pride and Prejudice' by Jane Austen is a must-read.",
            "For a great self-help book, 'Atomic Habits' by James Clear is very insightful."
        ]
    },
    {
        "tag": "motivational_books",
        "patterns": ["Recommend a motivational book", "What are some good motivational books?", "Suggest books that inspire", "I need motivation from books."],
        "responses": [
            "'The Power of Habit' by Charles Duhigg explores the science of habits and how to change them.",
            "'You Are a Badass' by Jen Sincero is a great read for self-confidence and empowerment.",
            "'Mindset: The New Psychology of Success' by Carol S. Dweck focuses on the power of having a growth mindset.",
            "'Can't Hurt Me' by David Goggins tells the story of overcoming adversity and achieving greatness.",
            "'The 7 Habits of Highly Effective People' by Stephen Covey provides insights into personal and professional effectiveness."
        ]
    },
    {
        "tag": "fun_books",
        "patterns": ["Recommend a fun book", "What are some good humorous books?", "Suggest light-hearted reads", "I need a fun book suggestion."],
        "responses": [
            "'The Hitchhiker's Guide to the Galaxy' by Douglas Adams is a hilarious sci-fi classic.",
            "'Good Omens' by Neil Gaiman and Terry Pratchett offers a humorous take on the apocalypse.",
            "'Bossypants' by Tina Fey is a witty memoir full of laughs.",
            "'Me Talk Pretty One Day' by David Sedaris is a collection of funny essays about his life.",
            "'Where'd You Go, Bernadette' by Maria Semple is a quirky and entertaining novel."
        ]
    },
    {
        "tag": "machine_learning",
        "patterns": ["Tell me about machine learning", "What is machine learning?", "Explain machine learning", "How does machine learning work?"],
        "responses": [
            "Machine learning is a subset of artificial intelligence that involves training algorithms to learn from and make predictions based on data.",
            "It's used in various applications, including image recognition, natural language processing, and recommendation systems.",
            "Machine learning can be supervised, unsupervised, or semi-supervised, depending on the type of data and the learning objective.",
            "The goal of machine learning is to enable computers to learn from experience and improve their performance over time without explicit programming."
        ]
    },
    {
        "tag": "artificial_intelligence",
        "patterns": ["What is AI?", "Tell me about artificial intelligence", "How does AI work?", "Explain AI."],
        "responses": [
            "Artificial intelligence (AI) is the simulation of human intelligence in machines programmed to think and learn.",
            "AI encompasses various subfields, including machine learning, natural language processing, and robotics.",
            "AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, and decision-making.",
            "The future of AI holds great promise, with potential applications in healthcare, transportation, finance, and more."
        ]
    },
    {
        "tag": "data_science",
        "patterns": ["What is data science?", "Tell me about data science", "Explain data science", "How does data science work?"],
        "responses": [
            "Data science is a multidisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data.",
            "It combines statistics, computer science, and domain expertise to analyze data and inform decision-making.",
            "Data scientists work with big data, utilizing techniques such as data mining, machine learning, and predictive analytics.",
            "The goal of data science is to turn raw data into actionable insights that can drive business and organizational strategies."
        ]
    },
    {
        "tag": "generative_ai",
        "patterns": ["What is generative AI?", "Tell me about generative AI", "How does generative AI work?", "Explain generative AI."],
        "responses": [
            "Generative AI refers to algorithms that can create new content, including text, images, and audio, based on the patterns learned from existing data.",
            "Examples of generative AI include GANs (Generative Adversarial Networks) and language models like GPT-3.",
            "Generative AI has applications in art, music, writing, and even product design.",
            "It's a rapidly growing field with exciting potential for creativity and innovation."
        ]
    },
    {
        "tag": "large_language_models",
        "patterns": ["What are large language models?", "Tell me about LLMs", "How do large language models work?", "Explain LLMs."],
        "responses": [
            "Large language models (LLMs) are AI models trained on vast amounts of text data to understand and generate human-like text.",
            "Examples of LLMs include GPT-3, BERT, and T5, which can perform a variety of language tasks, including translation, summarization, and question answering.",
            "LLMs leverage deep learning techniques to understand context and generate coherent responses.",
            "They are transforming the field of natural language processing and enabling new applications in chatbots, content generation, and more."
        ]
    }
]

# Prepare training data
training_sentences = []
training_labels = []

for intent in intents:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
        
# Vectorization and model training
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_sentences)
clf = LogisticRegression()
clf.fit(X, training_labels)

def chatbot(input_text):
    input_vector = vectorizer.transform([input_text])
    predicted_tag = clf.predict(input_vector)[0]

    for intent in intents:
        if intent['tag'] == predicted_tag:
            response = random.choice(intent['responses'])
            return response

    return "I'm sorry, I don't understand."
    
# Streamlit interface
st.title("Chatbot Companion: Ask Anything")
user_input = st.text_input("You: ")

if user_input:
    response = chatbot(user_input)
    st.write("Chatbot: ", response)
