# **AI Fitness and Nutrition Planner**

## **🎯 Problem Statement & Solution**

Most fitness apps give generic workout or diet plans that don’t consider individual student needs, cultural food habits, or available resources. Students require a system that generates personalized routines and meal plans using AI, ensuring they are practical, budget-friendly, and effective.

**The AI Fitness and Nutrition Planner** addresses this by using advanced AI (Gemini and Imagen) to create highly personalized, context-aware plans. By factoring in user goals, diet preferences, fitness level, and macro targets, the app delivers customized workout and diet plans that are far more practical and effective than generic alternatives.

## **🚀 Features**

### **1\. 📝 Plan Generation (AI Powered)**

* **Workout Plans:** Generates a 7-day personalized workout schedule based on user goals, fitness level, time commitment, and BMI.  
* **Diet Plans:** Creates a 7-day meal plan tailored to the user's diet preference and custom daily macronutrient targets (Protein, Fat, Carbs).

### **2\. 🍽️ Recipe Finder**

* Searches for recipes using **Spoonacular API** first, falling back to **Gemini's Structured JSON** generation if the primary API fails.  
* Displays detailed recipes, ingredients, instructions, and a **Plotly Pie Chart** showing the macronutrient breakdown.

### **3\. 🛒 Grocery List**

* Automatically aggregates and normalizes ingredients from all generated recipes into a single, clean grocery list.  
* Allows users to **download the list** as a CSV file.

### **4\. 📈 Progress Tracker (Session-Based)**

* Logs daily weight with date tracking.  
* Visualizes weight logs in a line chart to show progress over time.  
* Allows users to delete old logs dynamically without refreshing the whole app.  
* ***Note:*** *Logs are stored in the current Streamlit session only.*

### **5\. 🌟 Motivation**

* Generates a powerful motivational quote using **Gemini**.  
* Creates a corresponding epic motivational image using the **Imagen API** based on the generated quote.

## **📂 File Structure**

The project is organized to separate the core application logic, dependencies, and data/assets.

C:.  
│   .env                  \# Stores environment variables (API keys)  
│   app.py                \# Main Streamlit application script (All logic in one file)  
│   README.md             \# This readme file  
│   requirements.txt      \# List of Python dependencies (streamlit, requests, pandas, etc.)  
│  
├───assets  
│   ├───images            \# Placeholder for future static images  
│   └───videos            \# Placeholder for future videos  
└───data  
        user\_logs.json    \# Placeholder for potential future database integration

## **🛠️ Installation and Setup**

### **Prerequisites**

You must have Python 3.8+ installed on your system.

### **Step 1: Clone the Repository**

git clone \<your-repository-url\>  
cd ai\_fitness\_planner

### **Step 2: Install Dependencies**

Install all required libraries using the requirements.txt file.

pip install \-r requirements.txt

### **Step 3: API Key Configuration (Mandatory)**

This application requires API keys to access the Google and Spoonacular services. For security, these should be placed in a **.env** file.

1. **Get Your Keys:**  
   * **Gemini/Imagen:** Obtain a key from the \[Google AI Studio\] or similar service.  
   * **Spoonacular:** Obtain a key from the \[Spoonacular API website\].  
2. **Create .env File:** Create a file named **.env** in your project root directory and add your keys as shown below.  
   GEMINI\_API\_KEY="YOUR\_ACTUAL\_GEMINI\_KEY"  
   SPOONACULAR\_API\_KEY="YOUR\_ACTUAL\_SPOONACULAR\_KEY"

   ***Note:*** The app.py script is configured to look for these keys but uses hardcoded values (for initial testing setup) if the environment file is not loaded/available. **Always replace the hardcoded values with environment variables in production.**

## **▶️ Running the Application**

To launch the Streamlit web application, execute the following command from the project root directory:

streamlit run app.py

The application will start and open automatically in your default web browser, usually at http://localhost:8501.

### **Offline / Mock Mode**

The app features robust error handling and can run without valid API keys. If the necessary keys are missing or API calls fail, the application automatically falls back to **Mock Mode**, providing static mock data for plans, recipes, and motivation, allowing the UI and state management logic to be tested locally.

## **💻 How to Interact**

1. **Sidebar (Profile):** Start by entering your personal metrics (Weight, Height) and defining your goals and macro targets. This information drives the AI personalization.  
2. **📝 Plan Tab:** Click the buttons to generate comprehensive, personalized **Workout** and **Diet** plans based on your profile.  
3. **🍽️ Recipe Tab:** Enter a query (e.g., "high-protein dessert") and click "Find Recipes." The app will prioritize real recipes with nutritional data.  
4. **📈 Progress Tab:** Log your weight history to track your progress visually over time.