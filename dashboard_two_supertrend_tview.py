import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots

dbPath = r"G:\My Drive\Octanom\Supertrend\2_supertrends_combinations.db"
table_name = "backtest_results"
startDate = '2023-01-01'

def read_all_from_db(table_name, db_path):
    """
    Read all contents from a table in SQLite database
    
    Args:
        table_name (str): Name of the table to read from
        db_path (str): Path to the database file
    
    Returns:
        pd.DataFrame or str: DataFrame with all data or error message
    """
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return df
    except Exception as e:
        return f"Error reading from database: {str(e)}"


# Set page config
st.set_page_config(page_title="Two SuperTrend Signal PnL", layout="wide")

@st.cache_data
def load_data():
    db_df = read_all_from_db('backtest_results', dbPath)
    # db_df['entry_time'] = pd.to_datetime(db_df['entry_time'], format='%H:%M:%S.%f').dt.time
    db_df['entry_time'] = pd.to_datetime(db_df['entry_time'], errors='coerce').dt.time
    db_df['exit1_time'] = pd.to_datetime(db_df['exit1_time'], format='%H:%M:%S.%f').dt.time
    db_df['exit2_time'] = pd.to_datetime(db_df['exit2_time'], format='%H:%M:%S.%f').dt.time

    db_df['PnL_Pct'] = round((db_df['PnL'] / db_df['entry_px'])*100, 2)
    buy_df = db_df[(db_df['entry_date'] >= startDate) & (db_df['trade'] == 'Buy')].reset_index(drop=True)
    buy_df = buy_df[['year', 'entry_date', 'PnL', 'PnL_Pct', 'st1_params', 'st2_params']]

    return buy_df

def parse_params(param_str):
    """Parse parameter string like '(8, 2.5)' into tuple"""
    try:
        # Remove parentheses and split by comma
        cleaned = param_str.strip('()')
        parts = cleaned.split(',')
        return (float(parts[0].strip()), float(parts[1].strip()))
    except:
        return (0, 0)
    
def create_pnl_matrix(df):
    """Create PnL matrix with st1_params as rows and st2_params as columns"""
    # Group by parameters and sum PnL
    matrix_data = df.groupby(['st1_params', 'st2_params'])['PnL'].sum().reset_index()
    
    # Create pivot table
    pivot_matrix = matrix_data.pivot(index='st1_params', columns='st2_params', values='PnL')
    pivot_matrix = pivot_matrix.fillna(0)
    
    return pivot_matrix

def calculate_max_drawdown(df, group_cols):
    """Calculate maximum drawdown for each parameter combination"""
    max_dd_data = []
    
    for name, group in df.groupby(group_cols):
        # Sort by entry_date to get chronological order
        group_sorted = group.sort_values('entry_date')
        
        # Calculate cumulative PnL
        group_sorted['cumulative_pnl'] = group_sorted['PnL'].cumsum()
        
        # Calculate running maximum
        group_sorted['running_max'] = group_sorted['cumulative_pnl'].expanding().max()
        
        # Calculate drawdown
        group_sorted['drawdown'] = group_sorted['cumulative_pnl'] - group_sorted['running_max']
        
        # Get maximum drawdown (most negative value)
        max_drawdown = group_sorted['drawdown'].min()
        
        max_dd_data.append({
            'st1_params': name[0],
            'st2_params': name[1], 
            'max_drawdown': max_drawdown
        })
    
    return pd.DataFrame(max_dd_data)

def create_max_dd_matrix(df):
    """Create Max Drawdown matrix with st1_params as rows and st2_params as columns"""
    # Calculate max drawdown for each parameter combination
    max_dd_data = calculate_max_drawdown(df, ['st1_params', 'st2_params'])
    
    # Create pivot table
    pivot_matrix = max_dd_data.pivot(index='st1_params', columns='st2_params', values='max_drawdown')
    pivot_matrix = pivot_matrix.fillna(0)
    
    return pivot_matrix

def main():
    st.title(" Two SupertTrend Strategy PnL Dashboard")
    st.markdown("----")

    # Load data
    df = load_data()

    # Sidebar Filters
    st.sidebar.header(" Filters")

    # Year Filter
    available_years = sorted(df['year'].unique())
    selected_years = st.sidebar.multiselect(
        "Select Year(s)",
        options=available_years,
        default=available_years
    )

    # Flter dataframe
    if selected_years:
        filtered_df = df[df['year'].isin(selected_years)]
    else:
        filtered_df = df
    
    if filtered_df.empty:
        st.warning("No data available for selected years...")
        return
    
    # Display Summary Statistics
    col1, col2, col3, col4 = st.columns(4)

    # with col1:
    #     total_pnl = filtered_df['PnL'].sum()
    #     st.metric("Total PnL", f"{total_pnl:.0f}")

    # with col2:
    #     pnl_pct = filtered_df['PnL_Pct'].sum()
    #     st.metric("PnL %", f"{pnl_pct:.0f}")

    # st.markdown("---")

    # Create and display PnL matrix
    st.subheader(f"💰 Total PnL Matrix - {selected_years}")
    # st.markdown("**Rows: ST1 Parameters | Columns: ST2 Parameters**")
    
    pnl_matrix = create_pnl_matrix(filtered_df)

    if not pnl_matrix.empty:
        
        # Create heatmap with explicit text values
        text_values = [[f"{val:,.0f}" for val in row] for row in pnl_matrix.values]
        
        fig = go.Figure(data=go.Heatmap(
            z=pnl_matrix.values,
            x=pnl_matrix.columns,
            y=pnl_matrix.index,
            text=text_values,
            texttemplate="%{text}",
            textfont={"size": 16, "color": "black"},
            colorscale='RdYlGn',
            zmid=0,
            showscale=True,
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f"PnL Matrix - ST1 (Vertical) vs ST2 (Horizontal) - Year {selected_years}",
            xaxis_title="ST2 Parameters",
            yaxis_title="ST1 Parameters",
            height=600,
            width=800,
            font=dict(size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("No parameter combinations found for the selected year.")
    
    # NEW ADDITION: Max Drawdown Matrix
    st.markdown("---")
    st.subheader(f"📉 Maximum Drawdown Matrix - {selected_years}")
    
    max_dd_matrix = create_max_dd_matrix(filtered_df)
    
    if not max_dd_matrix.empty:
        
        # Create heatmap with explicit text values for Max DD
        text_values_dd = [[f"{val:,.0f}" for val in row] for row in max_dd_matrix.values]
        
        fig_dd = go.Figure(data=go.Heatmap(
            z=max_dd_matrix.values,
            x=max_dd_matrix.columns,
            y=max_dd_matrix.index,
            text=text_values_dd,
            texttemplate="%{text}",
            textfont={"size": 16, "color": "black"},
            colorscale='Reds',  # Red scale since drawdowns are negative
            showscale=True,
            hoverongaps=False
        ))
        
        fig_dd.update_layout(
            title=f"Max Drawdown Matrix - ST1 (Vertical) vs ST2 (Horizontal) - Year {selected_years}",
            xaxis_title="ST2 Parameters",
            yaxis_title="ST1 Parameters",
            height=600,
            width=800,
            font=dict(size=12)
        )
        
        st.plotly_chart(fig_dd, use_container_width=True)
        
        # # Display matrix as a table
        # st.subheader("📋 Max Drawdown Matrix Table")
        
        # formatted_dd_matrix = max_dd_matrix.copy()
        
        # st.dataframe(
        #     formatted_dd_matrix.style.format("{:.2f}"),
        #     use_container_width=True
        # )
        
    else:
        st.info("No parameter combinations found for Max DD calculation.")
    
    # Show detailed breakdown
    st.markdown("---")
    st.subheader("🔍 Detailed Breakdown")
    
    # Parameter combination summary
    param_summary = filtered_df.groupby(['st1_params', 'st2_params']).agg({
        'PnL': ['sum', 'count', 'mean'],
        'PnL_Pct': 'mean'
    }).round(2)
    
    param_summary.columns = ['Total_PnL', 'Trade_Count', 'Avg_PnL', 'Avg_PnL_Pct']
    param_summary = param_summary.reset_index()
    param_summary = param_summary.sort_values('Total_PnL', ascending=False)

    # df_no_index = param_summary.reset_index(drop=True)
    param_summary = param_summary.set_index(['st1_params', 'st2_params'])

    st.dataframe(
        param_summary.style.format({
            'Total_PnL': '{:.2f}',
            'Avg_PnL': '{:.2f}',
            'Avg_PnL_Pct': '{:.2%}'
        }),
        use_container_width=True
    )
    
    # Raw data
    with st.expander("📊 View Raw Data"):
        st.dataframe(
            filtered_df.style.format({
                'PnL': '{:.2f}',
                'PnL_Pct': '{:.2%}'
            }),
            use_container_width=True
        )

if __name__ == "__main__":
    main()
