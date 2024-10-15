import streamlit as st
import os
import pandas as pd
import sqlite3
import google.generativeai as genai

## Configure Genai Key
#genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Configure Gemini API key
myapi_key = "AIzaSyDE7hhHqhn_0KgVzJxh9nyhmlhjZVSuCOA"
genai.configure(api_key=myapi_key)

## Function To Load Google Gemini Model and provide queries as response
def get_gemini_response(question, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([prompt[0], question])
    answer = response.text.strip()  # Clean up the response
    answer = answer.replace("sql", "")  # Remove "sql " from the answer
    return answer.strip()


## Function To execute SQL query on DataFrame
def execute_sql_on_df(sql, df, table_name):
    try:
        # Create an in-memory SQLite database
        conn = sqlite3.connect(':memory:')
        # Write the DataFrame to the SQLite database
        df.to_sql(table_name, conn, index=False, if_exists='replace')  # Register DataFrame as the specified table
        return pd.read_sql_query(sql, con=conn, params=None, index_col=None)
    except Exception as e:
        return str(e)  # Return the error message as a string

## Streamlit App
st.set_page_config(page_title="Basic Database Chatbot")
st.header("Basic Database Chatbot (using Gemini)")

## File Upload Section
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

# Set default table name to 'main'
table_name = 'main'

if uploaded_file is not None:
    # Load the user-uploaded file into a dataframe
    df = pd.read_excel(uploaded_file)
    
    # Set table_name based on the uploaded file's name (without extension)
    table_name = uploaded_file.name.replace('.xlsx', '')
    
    # Ask the user to provide context for the uploaded file
    user_context = st.text_area("Provide context for the uploaded file:")
    
    # If no user context is provided, use a fallback
    if not user_context:
        user_context = "No additional context provided by the user."

    context = f"""
    The dataset has been uploaded by the user and contains the following columns: {', '.join(df.columns)}.
    {user_context}
    """
else:
    # Use default 'main.xlsx' if no file is uploaded
    df = pd.read_excel('main.xlsx')
    
    # Use default context when no file is uploaded
    context = """
    The dataset is structured with the following columns:
    1. **NUMBER**: The unique identifier for each SEBI order.
    2. **DATE**: The date when the order was passed, in `YYYY-MM-DD` format.
    3. **TYPE**: Specifies the type of SEBI order, which can either be "AO" (Adjudication Order) or "QJ" (Quasi-Judicial).
    4. **TITLE**: A descriptive title for the order.
    5. **LINK**: The URL link to the official document of the SEBI order.
    6. **ISO**: A flag that indicates if the order pertains to Illiquid Stock Options (ISO).
    7. **INTERMEDIARY**: Specifies if the noticee is a SEBI-registered intermediary.
    8. **INTERMEDIARY_NAME**: The name of the intermediary involved.
    9. **INTERMEDIARY_CATEGORY**: The category of the intermediary (e.g., Stock Broker, Merchant Banker).
    10. **INTERMEDIARY_REG**: The registration number of the intermediary.
    11. **PENALTY**: The monetary penalty imposed by the SEBI order.
    12. **ACTION**: Captures any action related to the penalty.
    13. **NOTICEE**: The entity or person against whom the order was passed.
    """
    
## Drop the PATH column if it exists
if 'PATH' in df.columns:
    df.drop(columns=['PATH'], inplace=True)

## Define Your Prompt
prompt = [
    f"""
    You are an expert in converting English questions to SQL queries!
    The table is called '{table_name}'. The Excel file contains the following columns: {', '.join(df.columns)}.
    {context}

    <instructions>
    The SQL code should not have any formatting characters or the word SQL in the output.
    </instructions>
    """
]

## Input for the SQL question
question = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

# if submit is clicked
if submit:
    response = get_gemini_response(question, prompt)
    
    # Clean the SQL query to remove any unwanted characters like "sql" or code block indicators
    cleaned_response = response.replace('```', '').strip()  # Remove any code block formatting
    cleaned_response = cleaned_response.lower().replace('sql ', '').strip()  # Remove 'sql' if present
    
    st.write("Generated SQL Query:", cleaned_response)  # Display the cleaned SQL query
    
    result_df = execute_sql_on_df(cleaned_response, df, table_name)  # Execute SQL on DataFrame
    
    # Check if result_df is a DataFrame or an error message
    if isinstance(result_df, pd.DataFrame):
        # Convert SQL result to text
        sql_output = result_df.to_string(index=False)  # Convert DataFrame to string for display
        
        # Prepare prompt for LLM to generate a descriptive answer
        llm_prompt = f"User question: {question}\nSQL output: {sql_output}\n\nWrite a descriptive but brief text answer to the user's question based on the SQL response."
        llm_response = get_gemini_response(llm_prompt, prompt)  # Get LLM response
        
        st.subheader("The SQL Response is")
        st.write(sql_output)  # Display the SQL result DataFrame
        st.subheader("LLM Generated Answer is")
        st.write(llm_response)  # Display the LLM-generated text response
    else:
        # Display the error message
        st.subheader("Error Executing SQL Query")
        st.write(result_df)  # Display the error message
