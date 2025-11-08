import streamlit as st
import pandas as pd
import sqlite3
import google.generativeai as genai
import os
import time # Added for exponential backoff delay
from google.api_core.exceptions import ResourceExhausted, InternalServerError # Added for specific error handling

# --- Configuration and Core Functions ---

# Configure Gemini API key
# The key is set to an empty string. The execution environment will handle authentication.
myapi_key = ""
genai.configure(api_key=myapi_key)

def get_gemini_response(question: str, prompt: list) -> str:
    """
    Load Google Gemini model and generate a response (SQL query) to the given question.
    
    Includes retry logic with exponential backoff for ResourceExhausted and InternalServerError.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    max_retries = 3
    base_delay = 2 # seconds for the initial delay

    for attempt in range(max_retries):
        try:
            # Generate content based on the prompt and question
            response = model.generate_content([prompt[0], question])
            
            # Clean up and return the response
            return response.text.strip()
        
        except (ResourceExhausted, InternalServerError) as e:
            if attempt < max_retries - 1:
                # Calculate exponential delay (2, 4, 8 seconds, etc.)
                delay = base_delay * (2 ** attempt)
                st.warning(f"API Error (ResourceExhausted/Internal): Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                # Failed after all retries
                return f"Gemini API Error: Failed after {max_retries} attempts due to rate limit or server issue. Please try again later. Original Error: {str(e)}"
        
        except Exception as e:
            # Handle other, non-retryable errors immediately
            return f"Gemini API Error: An unexpected error occurred: {str(e)}"
            
    # Should not be reachable
    return "Failed to get response due to unexpected termination of retry loop."


def execute_sql_on_df(sql: str, df: pd.DataFrame) -> pd.DataFrame | str:
    """
    Execute a SQL query on a pandas DataFrame using an in-memory SQLite database.
    """
    try:
        # Create an in-memory SQLite database
        conn = sqlite3.connect(':memory:')
        
        # Write the DataFrame to the SQLite database. Table name is 'main'.
        df.to_sql('main', conn, index=False, if_exists='replace')
        
        # Execute the SQL query and return the results as a DataFrame
        return pd.read_sql_query(sql, con=conn)
    except Exception as e:
        # If an error occurs, return the error message as a string
        return f"SQL Execution Error: {str(e)}"

# Define the system prompt for the Gemini model
prompt = [
    """
    You are an expert in converting English questions to SQL queries!
    
    Here's the description of the table you are supposed to generate SQL queries for:
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
    """
]

# --- Streamlit Application ---
st.set_page_config(page_title="Basic Database Chatbot")
st.header("SQL Generation Chatbot (using Gemini)")
st.markdown("Upload your Excel file to start querying the data using natural language.")

# 1. FILE UPLOADER (For general runnability, keep file uploader)
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

df = None

# Process file if uploaded
if uploaded_file is not None:
    try:
        # Load Excel file into DataFrame from the uploaded file object
        df = pd.read_excel(uploaded_file)
        
        # Drop the PATH column if it exists (Original logic)
        if 'PATH' in df.columns:
            df.drop(columns=['PATH'], inplace=True)

        st.success("File loaded successfully. Data preview:")
        st.dataframe(df.head()) # Show a preview

    except Exception as e:
        st.error(f"Error reading file: {e}")
        df = None # Ensure df is None if reading fails

# 2. Conditional Chatbot Logic
if df is not None:
    # Create input field for user questions
    question = st.text_input("Input your question (e.g., 'What is the average penalty for AO orders?'):", key="input")

    # Create a submit button
    submit = st.button("Generate SQL and Query Data")

    # Handle form submission
    if submit and question:
        with st.spinner('Generating SQL query and executing...'):
            # Get Gemini's response (SQL query) based on the user's question
            response = get_gemini_response(question, prompt)

            # Clean the SQL query to remove any unwanted characters like markdown fences
            cleaned_response = response.replace('```', '').replace('sql', '').strip()

            # Display the generated SQL query
            st.subheader("Generated SQL Query:")
            st.code(cleaned_response, language='sql')

            # Execute the SQL query on the DataFrame
            result_df = execute_sql_on_df(cleaned_response, df)
            
            # Display the query results
            st.subheader("Query Results")

            if isinstance(result_df, pd.DataFrame):
                st.dataframe(result_df)
            else:
                st.error(f"Execution Failed: {result_df}") # Display error message from execution function
    elif submit and not question:
        st.warning("Please enter a question before asking.")
else:
    st.info("Please upload an Excel file above to enable the chatbot functionality.")
