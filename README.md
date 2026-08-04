# 🛡️ AI Spam & Phishing Message Identifier

This is my first Machine Learning project! I built an end-to-end Natural Language Processing (NLP) application that evaluates text messages in real-time to flag them as safe or suspicious.

### 🚀 Live Demo
[👉 Click here to try out the web application!](https://nlp-project-hqjc6fkcuymgupyyhtr24j.streamlit.app/)

### ⚙️ How It Works (The Backend)
* **Dataset:** Trained on a collection of historical and augmented modern spam text examples.
* **Feature Extraction:** Uses `TfidfVectorizer` to tokenize text inputs and map contextual word weights.
* **Algorithm:** Implements a `LogisticRegression` pipeline to evaluate the probability of text safety.
* **Strict Thresholding:** Tuned to a strict 15% probability boundary condition to catch contemporary cryptocurrency and stock market edge-cases.

### 🛠️ Tech Stack
* Python
* Pandas
* Scikit-Learn (Machine Learning)
* Streamlit (Frontend Dashboard UI)
