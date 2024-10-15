## TalkToData Project Documentation

## Overview

TalkToData is a Streamlit-based web application that allows users to interact with and analyze SEBI (Securities and Exchange Board of India) order data through natural language queries and visualizations. The project consists of multiple pages, each serving a specific purpose in data exploration and analysis.

## Frontend

The frontend is built using Streamlit, a Python library for creating web applications with minimal effort. The application is structured into multiple pages, each represented by a separate Python file.

### Main Application (app.py)

```1:136:app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

# Set page configuration
st.set_page_config(
    page_title="TalkToData",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load data from the Excel file
@st.cache_data
def load_data(file_path):
    """
    Load data from an Excel file and cache it for improved performance.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    return pd.read_excel(file_path)

# Load Excel data
file_path = 'main.xlsx'
df = load_data(file_path)

# Data preprocessing
df['DATE'] = pd.to_datetime(df['DATE'])
df['quarter'] = df['DATE'].dt.to_period('Q').astype(str)

# Rename quarters based on the Indian Financial Year system
quarter_mapping = {
    '2024Q2': '2025Q1',
    '2024Q1': '2024Q4',
    '2023Q4': '2024Q3',
    '2023Q3': '2024Q2',
    '2023Q2': '2024Q1'
}
df['quarter'] = df['quarter'].replace(quarter_mapping)

# Sidebar filters
st.sidebar.header('Filters')

# Arrange quarters checkboxes in a grid layout
selected_quarters = []
quarter_columns = st.sidebar.columns(2)
for i, quarter in enumerate(sorted(df['quarter'].unique())):
    if quarter_columns[i % 2].checkbox(quarter, value=True):
        selected_quarters.append(quarter)

st.sidebar.write("---")

# Arrange order types checkboxes in a grid layout
selected_type = []
type_columns = st.sidebar.columns(2)
for i, order_type in enumerate(df['TYPE'].unique()):
    if type_columns[i % 2].checkbox(order_type, value=True):
        selected_type.append(order_type)

# Filter data based on selection
filtered_df = df[df['quarter'].isin(selected_quarters) & df['TYPE'].isin(selected_type)]

# Main content
st.title('SEBI Orders Dashboard')

# Create a 2x2 grid layout for visualizations
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Visualization 1: Bar Chart for Number of Orders by Quarter and Type
with col1:
    st.subheader('Number of Orders by Quarter and Type')
    orders_by_quarter_type = filtered_df.groupby(['quarter', 'TYPE']).size().reset_index(name='count')
    fig_orders = px.bar(
        orders_by_quarter_type,
        x='quarter',
        y='count',
        color='TYPE',
        barmode='group',
        title='Orders by Quarter and Type'
    )
    st.plotly_chart(fig_orders)
# Visualization 2: Penalty Applied Per Quarter
with col2:
    st.subheader('Penalty Applied Per Quarter')
    penalty_per_quarter = filtered_df.groupby('quarter')['PENALTY'].sum().reset_index()
    fig_penalty = px.bar(
        penalty_per_quarter,
        x='quarter',
        y='PENALTY',
        title='Penalty Per Quarter'
    )
    st.plotly_chart(fig_penalty)

# Visualization 3: Regulations Violated Per Quarter
with col3:
    st.subheader('Violations Per Quarter')
    regulations = ['pfutp', 'mb', 'lodr', 'icdr', 'pit', 'ia', 'sast', 'broker', 'circular', 'cis', 'act', 'scr']
    violations_per_quarter = filtered_df.groupby('quarter')[regulations].sum().reset_index()

    fig_violations = go.Figure()
    for regulation in regulations:
        fig_violations.add_trace(go.Scatter(
            x=violations_per_quarter['quarter'],
            y=violations_per_quarter[regulation],
            mode='lines+markers',
            name=regulation.upper()
        ))

    fig_violations.update_layout(
        title='Regulations Violated Per Quarter',
        xaxis_title='Quarter',
        yaxis_title='Number of Violations'
    )
    st.plotly_chart(fig_violations)
# Visualization 4: Interactive Data Table
with col4:
    st.subheader('Interactive Data Table')
    st.dataframe(filtered_df)

# Allow CSV download of the filtered data
with st.sidebar:
    st.divider()
st.sidebar.markdown('### Download Filtered Data')
st.sidebar.download_button(
    label='Download CSV',
    data=filtered_df.to_csv(index=False),
    mime='text/csv'
)
```

This is the main dashboard of the application. It provides an overview of the SEBI orders data through various visualizations.

Key features:

1. Data loading and preprocessing
2. Sidebar filters for quarters and order types
3. Four main visualizations:
   a. Bar chart for number of orders by quarter and type
   b. Bar chart for penalty applied per quarter
   c. Line chart for regulations violated per quarter
   d. Interactive data table
4. CSV download option for filtered data

### Page 1: Basic Database Chatbot (pages/1_One.py)

````1:127:pages/1_One.py
import streamlit as st
import os
import pandas as pd
import sqlite3
import google.generativeai as genai

# Configure Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_gemini_response(question: str, prompt: list) -> str:
    """
    Load Google Gemini model and generate a response to the given question.

    Args:
        question (str): The user's input question.
        prompt (list): A list containing the system prompt.

    Returns:
        str: The generated response from the Gemini model.
    """
    # Initialize the Gemini model
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Generate content based on the prompt and question
    response = model.generate_content([prompt[0], question])

    # Clean up and return the response
    return response.text.strip()

def execute_sql_on_df(sql: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute a SQL query on a pandas DataFrame using an in-memory SQLite database.

    Args:
        sql (str): The SQL query to execute.
        df (pd.DataFrame): The DataFrame to query.

    Returns:
        pd.DataFrame: The result of the SQL query as a DataFrame.
    """
    try:
        # Create an in-memory SQLite database
        conn = sqlite3.connect(':memory:')

        # Write the DataFrame to the SQLite database
        df.to_sql('main', conn, index=False, if_exists='replace')

        # Execute the SQL query and return the results as a DataFrame
        return pd.read_sql_query(sql, con=conn)
    except Exception as e:
        # If an error occurs, return the error message as a string
        return str(e)
# Load Excel file into DataFrame
df = pd.read_excel('main.xlsx')

# Drop the PATH column if it exists
if 'PATH' in df.columns:
    df.drop(columns=['PATH'], inplace=True)

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

# Set up the Streamlit app
st.set_page_config(page_title="Basic Database Chatbot")
st.header("Basic Database Chatbot (using Gemini)")

# Create input field for user questions
question = st.text_input("Input: ", key="input")

# Create a submit button
submit = st.button("Ask the question")

# Handle form submission
if submit:
    # Get Gemini's response (SQL query) based on the user's question
    response = get_gemini_response(question, prompt)

    # Clean the SQL query to remove any unwanted characters
    cleaned_response = response.replace('```', '').strip()

    # Display the generated SQL query
    st.write("Generated SQL Query:", cleaned_response)

    # Execute the SQL query on the DataFrame
    result_df = execute_sql_on_df(cleaned_response, df)

    # Display the query results
    st.subheader("The Response is")
    st.write(result_df)
````

This page implements a basic database chatbot using the Gemini API. It allows users to ask questions about the SEBI orders data in natural language, which are then converted to SQL queries and executed.

Key features:

1. Integration with Google's Gemini API
2. Natural language to SQL query conversion
3. Execution of SQL queries on the loaded DataFrame
4. Display of SQL query and results

### Page 2: Enhanced Database Chatbot (pages/2_Two.py)

````1:100:pages/2_Two.py
import streamlit as st
import os
import pandas as pd
import sqlite3
import google.generativeai as genai

## Configure Genai Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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
````

This page is an enhanced version of the database chatbot. It not only converts natural language queries to SQL and executes them but also generates a descriptive answer based on the SQL results.

Key features:

1. Similar to Page 1, but with an additional step
2. Generates a descriptive answer using the Gemini API based on the SQL results

### Page 3: Gemini-powered Database Chatbot (pages/3_Three.py)

````1:129:pages/3_Three.py
import streamlit as st
import os
import pandas as pd
import sqlite3
import google.generativeai as genai

## Configure Genai Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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
````

This page implements a more advanced database chatbot using the Gemini API. It allows users to upload their own Excel files and provides context for the uploaded data.

Key features:

1. File upload functionality for Excel files
2. Dynamic table name based on uploaded file
3. User-provided context for uploaded files
4. Fallback to default dataset if no file is uploaded
5. Natural language to SQL conversion and execution
6. LLM-generated descriptive answers

### Page 4: LLMware-powered Chatbot (pages/4_Four.py)

```1:341:pages/4_Four.py
import streamlit as st
from llmware.resources import CustomTable
from llmware.models import ModelCatalog
from llmware.prompts import Prompt
from llmware.parsers import Parser
from llmware.configs import LLMWareConfig
from llmware.agents import LLMfx
from llmware.setup import Setup
import os

# Initialize session state for loaded tables
if "loaded_tables" not in st.session_state:
    st.session_state["loaded_tables"] = []
def build_table(db: str = None, table_name: str = None, load_fp: str = None, load_file: str = None) -> int:
    """
    Build a database table from a CSV or JSON/JSONL file.

    Args:
        db (str): Database name.
        table_name (str): Name of the table to create.
        load_fp (str): File path to load from.
        load_file (str): Name of the file to load.

    Returns:
        int: Number of rows inserted, or 0 if table already exists, or -1 if file type is not supported.
    """
    if not table_name:
        return 0

    # Avoid rebuilding existing tables
    if table_name in st.session_state["loaded_tables"]:
        return 0

    custom_table = CustomTable(db=db, table_name=table_name)
    analysis = custom_table.validate_csv(load_fp, load_file)
    print(f"Analysis from validate_csv: {analysis}")

    if load_file.endswith(".csv"):
        output = custom_table.load_csv(load_fp, load_file)
    elif load_file.endswith((".jsonl", ".json")):
        output = custom_table.load_json(load_fp, load_file)
    else:
        print("File type not supported for db load")
        return -1

    print(f"Output from loading file: {output}")

    # Display sample rows
    sample_range = min(10, len(custom_table.rows))
    for x in range(sample_range):
        print(f"Sample row {x}: {custom_table.rows[x]}")

    # Remediate schema
    updated_schema = custom_table.test_and_remediate_schema(samples=20, auto_remediate=True)
    print(f"Updated schema: {updated_schema}")

    # Insert rows into the database
    custom_table.insert_rows()

    st.session_state["loaded_tables"].append(table_name)

    return len(custom_table.rows)
@st.cache_resource
def load_reranker_model():
    """
    Load and cache the reranker model for RAG process.

    Returns:
        object: Loaded reranker model.
    """
    return ModelCatalog().load_model("jina-reranker-turbo")

@st.cache_resource
def load_prompt_model():
    """
    Load and cache the prompt model for RAG process.

    Returns:
        object: Loaded prompt model.
    """
    return Prompt().load_model("bling-phi-3-gguf", temperature=0.0, sample=False)

@st.cache_resource
def load_agent_model():
    """
    Load and cache the agent model for Text2SQL queries.

    Returns:
        object: Loaded agent model.
    """
    agent = LLMfx()
    agent.load_tool("sql", sample=False, get_logits=True, temperature=0.0)
    return agent

@st.cache_resource
def parse_file(fp: str, doc: str):
    """
    Parse a file and cache the result.

    Args:
        fp (str): File path.
        doc (str): Document name.

    Returns:
        object: Parsed file output.
    """
    parser_output = Parser().parse_one(fp, doc, save_history=False)
    st.cache_resource.clear()
    return parser_output
def get_rag_response(prompt: str, parser_output, reranker_model, prompter):
    """
    Generate a RAG (Retrieval-Augmented Generation) response.

    Args:
        prompt (str): User prompt.
        parser_output: Parsed document output.
        reranker_model: Loaded reranker model.
        prompter: Loaded prompt model.

    Returns:
        str: Generated response.
    """
    # Rerank chunks if there are more than 3
    if len(parser_output) > 3:
        output = reranker_model.inference(prompt, parser_output, top_n=10, relevance_threshold=0.25)
    else:
        output = [dict(entry, rerank_score=0.0) for entry in parser_output]

    # Use top 3 chunks
    use_top = 3
    output = output[:use_top]

    # Create source from top chunks
    sources = prompter.add_source_query_results(output)

    # Generate response
    responses = prompter.prompt_with_source(prompt, prompt_name="default_with_context")

    # Perform post-inference checks
    source_check = prompter.evidence_check_sources(responses)
    numbers_check = prompter.evidence_check_numbers(responses)
    nf_check = prompter.classify_not_found_response(responses, parse_response=True, evidence_match=False, ask_the_model=False)

    # Process response
    bot_response = ""
    for i, resp in enumerate(responses):
        bot_response = resp['llm_response']
        print(f"Bot response - llm_response raw: {bot_response}")

        add_sources = True

        if "not_found_classification" in nf_check[i] and nf_check[i]["not_found_classification"]:
            add_sources = False
            bot_response += "\n\nThe answer to the question was not found in the source passage attached - please check the source again, and/or try to ask the question in a different way."

        if add_sources:
            numbers_output = process_numbers_check(numbers_check[i])
            bot_response += "\n\n" + numbers_output if numbers_output else ""

            if not numbers_output:
                source_output = process_source_check(source_check[i])
                bot_response += "\n\n" + source_output if source_output else ""

    prompter.clear_source_materials()

    return bot_response
def process_numbers_check(check):
    """
    Process the numbers check result.

    Args:
        check (dict): Numbers check result.

    Returns:
        str: Processed output string.
    """
    if "fact_check" in check and isinstance(check["fact_check"], list) and check["fact_check"]:
        fc = check["fact_check"][0]
        output = ""
        if "text" in fc:
            output += f"Text: {fc['text']}\n\n"
        if "source" in fc:
            output += f"Source: {fc['source']}\n\n"
        if "page_num" in fc:
            output += f"Page Num: {fc['page_num']}\n\n"
        return output
    return ""

def process_source_check(check):
    """
    Process the source check result.

    Args:
        check (dict): Source check result.

    Returns:
        str: Processed output string.
    """
    if "source_review" in check and isinstance(check["source_review"], list) and check["source_review"]:
        fc = check["source_review"][0]
        output = ""
        if "text" in fc:
            output += f"Text: {fc['text']}\n\n"
        if "match_score" in fc:
            output += f"Match Score: {fc['match_score']}\n\n"
        if "source" in fc:
            output += f"Source: {fc['source']}\n\n"
        if "page_num" in fc:
            output += f"Page Num: {fc['page_num']}\n\n"
        return output
    return ""
def get_sql_response(prompt: str, agent, db: str = None, table_name: str = None):
    """
    Generate a SQL response using the agent model.

    Args:
        prompt (str): User prompt.
        agent: Loaded agent model.
        db (str): Database name.
        table_name (str): Table name.

    Returns:
        str: Generated SQL response.
    """
    show_sql = prompt.endswith(" #SHOW")
    if show_sql:
        prompt = prompt[:-len(" #SHOW")]

    model_response = agent.query_custom_table(prompt, db=db, table=table_name)

    try:
        sql_query = model_response["sql_query"]
        db_response = model_response["db_response"]

        if not show_sql:
            bot_response = db_response
        else:
            bot_response = f"Answer: {db_response}\n\nSQL Query: {sql_query}"
    except:
        bot_response = (f"Sorry I could not find an answer to your question.<br/>"
                        f"Here is the SQL query that was generated by your question: "
                        f"<br/>{model_response.get('sql_query', 'No SQL query generated')}.<br/> If this missed the mark, please try asking "
                        f"the question again with a little more specificity.")

    return bot_response
def biz_bot_ui_app(db: str = "postgres", table_name: str = None, fp: str = None, doc: str = None):
    """
    Main function to run the Biz Bot UI application.

    Args:
        db (str): Database name.
        table_name (str): Table name.
        fp (str): File path.
        doc (str): Document name.
    """
    st.title("Biz Bot")

    parser_output = None

    if fp and doc and os.path.exists(os.path.join(fp, doc)):
        parser_output = Parser().parse_one(fp, doc, save_history=False)

    prompter = load_prompt_model()
    reranker_model = load_reranker_model()
    agent = load_agent_model()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Sidebar configuration
    with st.sidebar:
        st.write("Biz Bot")
        model_type = st.selectbox("Pick your mode", ("RAG", "SQL"), index=0)

        uploaded_doc = st.file_uploader("Upload Document")
        uploaded_table = st.file_uploader("Upload CSV")

        if uploaded_doc:
            fp = LLMWareConfig().get_llmware_path()
            doc = uploaded_doc.name
            with open(os.path.join(fp, doc), "wb") as f:
                f.write(uploaded_doc.getvalue())
            parser_output = parse_file(fp, doc)
            st.write(f"Document Parsed and Ready - {len(parser_output)}")

        if uploaded_table:
            fp = LLMWareConfig().get_llmware_path()
            tab = uploaded_table.name
            with open(os.path.join(fp, tab), "wb") as f:
                f.write(uploaded_table.getvalue())
            table_name = tab.split(".")[0]
            st.write(f"Building Table - {tab}, {table_name}")
            st.write(st.session_state['loaded_tables'])
            row_count = build_table(db=db, table_name=table_name, load_fp=fp, load_file=tab)
            st.write(f"Completed - Table - {table_name} - Rows - {row_count} - is Ready.")
    # User input
    prompt = st.chat_input("Say something")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if model_type == "RAG":
                bot_response = get_rag_response(prompt, parser_output, reranker_model, prompter)
            else:
                bot_response = get_sql_response(prompt, agent, db=db, table_name=table_name)
            st.markdown(bot_response)

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

    return 0
if __name__ == "__main__":
    # Main execution
    db = "sqlite"
    table_name = "main"

    # Load sample CSV file
    local_csv_path = "main.csv"
    build_table(db=db, table_name=table_name, load_fp=local_csv_path, load_file="main.csv")

    # Load sample agreement file
    sample_files_path = Setup().load_sample_files(over_write=False)
    fp = os.path.join(sample_files_path, "Agreements")
    fn = "Nike EXECUTIVE EMPLOYMENT AGREEMENT.pdf"

    biz_bot_ui_app(db=db, table_name=table_name, fp=fp, doc=fn)
```

This page implements a sophisticated chatbot using the LLMware library. It supports both RAG (Retrieval-Augmented Generation) and SQL-based querying.

Key features:

1. File upload for documents and CSV files
2. Support for RAG and SQL-based querying
3. Integration with LLMware library for advanced NLP tasks
4. Caching of models and parsed documents for improved performance
5. Interactive chat interface

## Functionalities

1. **Data Visualization**: The main dashboard (app.py) provides various charts and graphs to visualize the SEBI orders data, including order counts, penalties, and regulation violations over time.

2. **Natural Language Querying**: All chatbot pages (1_One.py, 2_Two.py, 3_Three.py, 4_Four.py) allow users to ask questions about the data in natural language.

3. **SQL Generation**: The application converts natural language queries into SQL queries using AI models (Gemini API or LLMware).

4. **Data Exploration**: Users can explore the SEBI orders data through interactive filters and visualizations on the main dashboard.

5. **File Upload**: Page 3 and Page 4 allow users to upload their own Excel files or documents for analysis.

6. **Contextual Understanding**: The chatbots use predefined or user-provided context to better understand the data structure and generate more accurate responses.

7. **RAG and SQL Querying**: Page 4 supports both Retrieval-Augmented Generation (RAG) and SQL-based querying, providing flexibility in how users can interact with the data.

8. **Data Download**: The main dashboard allows users to download the filtered data as a CSV file.

## Styling

The application uses custom CSS styling defined in the `style.css` file:

```1:18:style.css
[data-testid=metric-container] {
    box-shadow: 0 0 4px #c9d6d6;
    padding: 10px;
}

.plot-container>div {
    box-shadow: 0 0 4px #071021;
    padding: 10px;



}

div[data-testid="stExpander"] div[role="button"] p
{
    font-size: 1.3rem;
    color:rgb(248, 253, 253);
}
```

This CSS file provides custom styling for metric containers, plot containers, and expander elements, enhancing the visual appeal of the application.

## Database Connection

The application includes a MySQL connection setup, although it's not directly used in the main application files:

```1:23:mysql_connection
import mysql.connector
#pip install mysql-connector-python
import streamlit as st

conn = mysql.connector.connect(
host="localhost",
port="3306",
user="root",
passwd="",
db="streamlit_mysql")

c = conn.cursor()


def view_all_data():
	c.execute('SELECT * FROM customers order by id asc')
	data = c.fetchall()
	return data

def view_all_departments():
	c.execute('SELECT Department FROM customers')
	data = c.fetchmany
	return data
```

This file sets up a connection to a MySQL database and defines functions to view all data and departments. It could be integrated into the main application for database operations if needed.

## Conclusion

The TalkToData project is a comprehensive data exploration and analysis tool for SEBI orders. It combines powerful visualization capabilities with natural language processing to provide an intuitive and flexible interface for users to interact with the data. The modular structure of the application, with separate pages for different functionalities, allows for easy maintenance and future expansions.
