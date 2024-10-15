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