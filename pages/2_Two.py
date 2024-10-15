import streamlit as st
import os
import pandas as pd
import sqlite3
import google.generativeai as genai

# Configure Gemini API key
myapi_key = "AIzaSyDE7hhHqhn_0KgVzJxh9nyhmlhjZVSuCOA"
genai.configure(api_key=myapi_key)

## Function To Load Google Gemini Model and provide queries as response
def get_gemini_response(question, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([prompt[0], question])
    return response.text.strip()  # Clean up the response

## Function To execute SQL query on DataFrame
def execute_sql_on_df(sql, df):
    try:
        # Create an in-memory SQLite database
        conn = sqlite3.connect(':memory:')
        # Write the DataFrame to the SQLite database
        df.to_sql('main', conn, index=False, if_exists='replace')  # Register DataFrame as 'main' table
        return pd.read_sql_query(sql, con=conn, params=None, index_col=None)
    except Exception as e:
        return str(e)

## Load Excel file into DataFrame
df = pd.read_excel('main.xlsx')

## Drop the PATH column if it exists
if 'PATH' in df.columns:
    df.drop(columns=['PATH'], inplace=True)

## Define Your Prompt
prompt = [
    """
    You are an expert in converting English questions to SQL queries!
    
    Here's the description of the table you are supposed to generate sql queries for:
    <table_description>
    The Excel file contains the following columns: NUMBER, DATE, TYPE, TITLE, LINK, ISO, INTERMEDIARY, INTERMEDIARY_NAME, INTERMEDIARY_CATEGORY, INTERMEDIARY_REG, PENALTY, ACTION, NOTICEE
    The table is called 'main'
    The dataset contains SEBI (Securities and Exchange Board of India) adjudication orders with key attributes for analysis. The data is structured in the following columns:

    1. **NUMBER**: The unique identifier for each SEBI order.
    2. **DATE**: The date when the order was passed, in `YYYY-MM-DD` format.
    3. **TYPE**: Specifies the type of SEBI order, which can either be "AO" (Adjudication Order) or "QJ" (Quasi-Judicial).
    4. **TITLE**: A descriptive title for the order, often containing the name of the case or the entity involved.
    5. **LINK**: The URL link to the official document of the SEBI order.
    6. **ISO**: A flag that indicates if the order pertains to **Illiquid Stock Options** (ISO). This value is sometimes missing (NaN), indicating that the order is unrelated to ISO.
    7. **INTERMEDIARY**: A column that indicates if the noticee (the party against whom the order is passed) is a SEBI registered intermediary. If applicable, it includes the type of intermediary.
    8. **INTERMEDIARY_NAME**: The name of the intermediary entity involved in the order, if applicable.
    9. **INTERMEDIARY_CATEGORY**: The category of the intermediary involved, such as "Stock Broker," "Merchant Banker," "Investment Adviser," etc.
    10. **INTERMEDIARY_REG**: The registration number of the intermediary, if available.
    11. **PENALTY**: The monetary penalty imposed by the SEBI order. If the penalty is zero, it indicates that the proceedings were disposed of.
    12. **ACTION**: This column captures any actions related to the penalty, though its data seems sparse (mostly NaN).
    13. **NOTICEE**: The entity or person against whom the order was passed, often accompanied by an identifying tax number (like PAN).

    ### Key Contextual Information:

    - **Order Types**: There are two primary types of SEBI orders — AO (Adjudication Order) and QJ (Quasi-Judicial).
    - **ISO Flag**: A separate column flags orders related to Illiquid Stock Options (ISO), which are of particular interest for certain types of analysis.
    - **Intermediaries**: If the noticee is a SEBI registered intermediary, the details about their category, registration number, and name are captured. Common intermediary categories include Stock Brokers, Merchant Bankers, and Investment Advisers.
    - **Penalties**: Penalty amounts are recorded, and any case where the penalty is 0 signifies that the proceedings were disposed of without a financial penalty.

    </table_description>
    
    <instructions>
    The SQL code should not have any formatting characters or the word SQL in the output.
    </instructions>
    
    """.format(columns=', '.join(df.columns))
]

## Streamlit App
st.set_page_config(page_title="Basic Database Chatbot")
st.header("Basic Database Chatbot (using Gemini)")

question = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

# if submit is clicked
if submit:
    response = get_gemini_response(question, prompt)
    # Clean the SQL query to remove any unwanted characters
    cleaned_response = response.replace('```', '').strip()  # Remove any code block formatting
    st.write("Generated SQL Query:", cleaned_response)  # Display the cleaned SQL query
    result_df = execute_sql_on_df(cleaned_response, df)  # Execute SQL on DataFrame
    
    # Convert SQL result to text
    sql_output = result_df.to_string(index=False)  # Convert DataFrame to string for display
    
    # Prepare prompt for LLM to generate a descriptive answer
    llm_prompt = f"User question: {question}\nSQL output: {sql_output}\n\nWrite a descriptive but brief text answer to the user's question based on the SQL response."
    llm_response = get_gemini_response(llm_prompt, prompt)  # Get LLM response
    
    st.subheader("The SQL Response is")
    st.write(sql_output)  # Display the SQL result DataFrame
    st.subheader("LLM Generated Answer is")
    st.write(llm_response)  # Display the LLM-generated text response




