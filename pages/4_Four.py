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