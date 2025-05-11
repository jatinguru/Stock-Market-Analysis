import dash
from dash import dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pandas_ta as ta
import traceback
import time
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Configure session with retry mechanism for all requests
session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# Pass the session to yfinance
#yf.pdr_override()

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define colors
colors = {
    'background': '#0f1124',
    'text': '#FFFFFF',
    'grid': '#323232',
    'red': '#FF3B30',
    'green': '#34C759',
    'blue': '#007AFF',
    'chart_background': '#1C1C1E'
}

# App layout
app.layout = html.Div(style={'backgroundColor': colors['background'], 'color': colors['text'], 'padding': '20px', 'fontFamily': 'Arial'}, children=[
    html.H1("Stock Market Dashboard", style={'textAlign': 'center', 'marginBottom': '30px'}),
    
    # Error message area
    html.Div(id='error-message', style={'color': colors['red'], 'textAlign': 'center', 'marginBottom': '10px'}),
    
    # Loading indicator
    dcc.Loading(
        id="loading-indicator",
        type="circle",
        children=[html.Div(id="loading-output")]
    ),
    
    # Top row - Stock Selector and Date Range
    html.Div([
        html.Div([
            html.Label("Select Stock Symbol:", style={'marginBottom': '10px', 'fontSize': '16px'}),
            dcc.Input(
                id='stock-input',
                type='text',
                value='AAPL',
                placeholder='Enter stock symbol (e.g. AAPL)',
                style={'width': '100%', 'padding': '10px', 'borderRadius': '5px', 'backgroundColor': colors['chart_background'], 'color': colors['text']}
            ),
            html.Button('Submit', id='submit-button', n_clicks=0, 
                         style={'backgroundColor': colors['blue'], 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'margin': '10px 0', 'borderRadius': '5px', 'cursor': 'pointer'})
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px'}),
        
        html.Div([
            html.Label("Select Date Range:", style={'marginBottom': '10px', 'fontSize': '16px'}),
            dcc.DatePickerRange(
                id='date-range',
                start_date=(datetime.now() - timedelta(days=365)).date(),
                end_date=datetime.now().date(),
                style={'backgroundColor': colors['chart_background']}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px'}),
        
        html.Div([
            html.Label("Select Technical Indicators:", style={'marginBottom': '10px', 'fontSize': '16px'}),
            dcc.Checklist(
                id='indicators',
                options=[
                    {'label': ' SMA (50)', 'value': 'sma50'},
                    {'label': ' SMA (200)', 'value': 'sma200'},
                    {'label': ' Bollinger Bands', 'value': 'bb'},
                    {'label': ' RSI', 'value': 'rsi'}
                ],
                value=['sma50'],
                labelStyle={'display': 'block', 'margin': '5px 0', 'color': colors['text']},
                style={'color': colors['text']}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px'})
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px', 'backgroundColor': colors['chart_background'], 'borderRadius': '10px', 'padding': '10px'}),
    
    # Offline data mode toggle
    html.Div([
        html.Label("Use offline data if network issues occur:", style={'marginRight': '10px'}),
        dcc.RadioItems(
            id='offline-mode',
            options=[
                {'label': ' Yes', 'value': 'yes'},
                {'label': ' No', 'value': 'no'}
            ],
            value='yes',
            labelStyle={'display': 'inline-block', 'margin': '0 10px'}
        )
    ], style={'marginBottom': '20px', 'textAlign': 'center'}),
    
    # Key metrics row
    html.Div(id='key-metrics', style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),
    
    # Price chart
    html.Div([
        html.H3("Stock Price Chart", style={'textAlign': 'center'}),
        dcc.Graph(id='price-chart')
    ], style={'backgroundColor': colors['chart_background'], 'borderRadius': '10px', 'padding': '15px', 'marginBottom': '20px'}),
    
    # Bottom row - Technical indicators and Volume
    html.Div([
        html.Div([
            html.H3("RSI (Relative Strength Index)", style={'textAlign': 'center'}),
            dcc.Graph(id='rsi-chart')
        ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': colors['chart_background'], 'borderRadius': '10px', 'padding': '15px'}),
        
        html.Div([
            html.H3("Trading Volume", style={'textAlign': 'center'}),
            dcc.Graph(id='volume-chart')
        ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': colors['chart_background'], 'borderRadius': '10px', 'padding': '15px'})
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    
    # Stock comparison section
    html.Div([
        html.H3("Stock Comparison", style={'textAlign': 'center', 'marginTop': '30px'}),
        html.Div([
            html.Label("Compare with:", style={'marginBottom': '10px', 'fontSize': '16px'}),
            dcc.Input(
                id='compare-input',
                type='text',
                value='MSFT,GOOG',
                placeholder='Enter symbols separated by commas (e.g. MSFT,GOOG)',
                style={'width': '50%', 'padding': '10px', 'borderRadius': '5px', 'backgroundColor': colors['chart_background'], 'color': colors['text']}
            ),
            html.Button('Compare', id='compare-button', n_clicks=0, 
                         style={'backgroundColor': colors['blue'], 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'margin': '10px 10px', 'borderRadius': '5px', 'cursor': 'pointer'})
        ], style={'marginBottom': '20px'}),
        dcc.Graph(id='comparison-chart')
    ], style={'backgroundColor': colors['chart_background'], 'borderRadius': '10px', 'padding': '15px', 'marginTop': '20px'}),
    
    # Store components
    dcc.Store(id='stock-data-store'),
    dcc.Store(id='offline-data-store')
])

# Function to generate fallback data
def generate_fallback_data(symbol, days=365):
    """Generate synthetic stock data for when network fails"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Generate random price data with a trend
    np.random.seed(hash(symbol) % 2**32)  # Use ticker as seed for consistency
    
    # Start with a base price between $10 and $500
    base_price = np.random.uniform(10, 500)
    
    # Generate random daily returns with a slight positive drift
    daily_returns = np.random.normal(0.0005, 0.02, size=len(date_range))
    
    # Calculate price series
    price_series = base_price * (1 + daily_returns).cumprod()
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open': price_series * np.random.uniform(0.99, 1.01, size=len(date_range)),
        'Close': price_series,
        'High': price_series * np.random.uniform(1.01, 1.03, size=len(date_range)),
        'Low': price_series * np.random.uniform(0.97, 0.99, size=len(date_range)),
        'Volume': np.random.randint(100000, 10000000, size=len(date_range))
    }, index=date_range)
    
    # Calculate indicators
    # SMA
    if len(df) >= 50:
        df['SMA50'] = df['Close'].rolling(window=50).mean()
    if len(df) >= 200:
        df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    # RSI
    if len(df) >= 14:
        df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # Bollinger Bands
    if len(df) >= 20:
        bb_indicator = ta.bbands(df['Close'], length=20, std=2)
        if not bb_indicator.empty:
            df = pd.concat([df, bb_indicator], axis=1)
    
    # Add note that this is synthetic data
    df['Synthetic'] = True
    
    return df

# Fetch stock data
@app.callback(
    [Output('stock-data-store', 'data'),
     Output('offline-data-store', 'data'),
     Output('error-message', 'children'),
     Output('loading-output', 'children')],
    [Input('submit-button', 'n_clicks')],
    [State('stock-input', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('offline-mode', 'value')]
)
def fetch_stock_data(n_clicks, ticker, start_date, end_date, offline_mode):
    print("DEBUG: fetch_stock_data called")
    print("n_clicks:", n_clicks)
    print("ticker:", ticker)
    print("start_date:", start_date)
    print("end_date:", end_date)
    if n_clicks == 0:
        # Initial load - return empty
        return None, None, "", None
    
    # Show loading state
    loading_output = "Loading..."
    
    try:
        # Try to fetch real data first
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Fetch data with the configured session
                df = yf.download(ticker, start=start_date, end=end_date, progress=False, session=session)
                
                if df.empty:
                    # If no data, try fallback if enabled
                    if offline_mode == 'yes':
                        fallback_df = generate_fallback_data(ticker)
                        # Filter to match the date range
                        fallback_df = fallback_df.loc[start_date:end_date]
                        
                        # Convert to dict for JSON storage
                        fallback_json = fallback_df.reset_index()
                        fallback_json['Date'] = fallback_json['Date'].dt.strftime('%Y-%m-%d')
                        
                        return None, fallback_json.to_dict('records'), f"Using offline synthetic data for {ticker} (real data not available)", None
                    else:
                        return None, None, f"No data found for {ticker}. Please check the symbol and date range.", None
                
                # Calculate indicators
                # Calculate 50-day and 200-day SMA
                if len(df) >= 50:
                    df['SMA50'] = df['Close'].rolling(window=50).mean()
                if len(df) >= 200:
                    df['SMA200'] = df['Close'].rolling(window=200).mean()
                
                # Calculate Bollinger Bands
                if len(df) >= 20:
                    bb_indicator = ta.bbands(df['Close'], length=20, std=2)
                    if not bb_indicator.empty:
                        df = pd.concat([df, bb_indicator], axis=1)
                
                # Calculate RSI
                if len(df) >= 14:
                    df['RSI'] = ta.rsi(df['Close'], length=14)
                
                # Add flag for real data
                df['Synthetic'] = False
                
                # Convert dates to string format for JSON serialization
                df_json = df.reset_index()
                df_json['Date'] = df_json['Date'].dt.strftime('%Y-%m-%d')
                
                # Generate fallback data for offline use
                fallback_df = generate_fallback_data(ticker)
                fallback_df = fallback_df.loc[start_date:end_date]
                fallback_json = fallback_df.reset_index()
                fallback_json['Date'] = fallback_json['Date'].dt.strftime('%Y-%m-%d')
                
                return df_json.to_dict('records'), fallback_json.to_dict('records'), "", None
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait 1 second before retrying
                    continue
                else:
                    print(f"Error fetching data for {ticker}: {str(e)}")
                    print(traceback.format_exc())
                    
                    # Use offline data if enabled
                    if offline_mode == 'yes':
                        fallback_df = generate_fallback_data(ticker)
                        # Filter to match the date range
                        fallback_df = fallback_df.loc[start_date:end_date]
                        
                        # Convert to dict for JSON storage
                        fallback_json = fallback_df.reset_index()
                        fallback_json['Date'] = fallback_json['Date'].dt.strftime('%Y-%m-%d')
                        
                        return None, fallback_json.to_dict('records'), f"Network issue detected. Using offline synthetic data for {ticker}", None
                    else:
                        return None, None, f"Error fetching data for {ticker}: {str(e)}. Try enabling offline mode.", None
    
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(traceback.format_exc())
        
        # Use offline data if enabled
        if offline_mode == 'yes':
            fallback_df = generate_fallback_data(ticker)
            # Convert to dict for JSON storage
            fallback_json = fallback_df.reset_index()
            fallback_json['Date'] = fallback_json['Date'].dt.strftime('%Y-%m-%d')
            
            return None, fallback_json.to_dict('records'), f"An unexpected error occurred. Using offline synthetic data.", None
        else:
            return None, None, f"An unexpected error occurred: {str(e)}", None

# Update key metrics
@app.callback(
    Output('key-metrics', 'children'),
    [Input('stock-data-store', 'data'),
     Input('offline-data-store', 'data'),
     Input('offline-mode', 'value')]
)
def update_metrics(data, offline_data, offline_mode):
    # Choose which data to use
    selected_data = data if data is not None else (offline_data if offline_mode == 'yes' else None)
    
    if selected_data is None or len(selected_data) < 2:
        # Create empty metrics placeholders
        return [create_metric_card(title, "N/A", colors['blue']) for title in 
                ["Current Price", "Daily Change", "52-Week High", "52-Week Low", "Avg Volume"]]
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(selected_data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        
        # Check if this is synthetic data
        data_source = "Synthetic" if 'Synthetic' in df.columns and df['Synthetic'].iloc[0] else "Real"
        
        # Calculate metrics
        current_price = df['Close'].iloc[-1]
        
        if len(df) > 1:
            previous_close = df['Close'].iloc[-2]
            price_change = current_price - previous_close
            price_change_pct = (price_change / previous_close) * 100
        else:
            price_change = 0
            price_change_pct = 0
        
        # Calculate 52-week high and low
        week_52_high = df['High'].max()
        week_52_low = df['Low'].min()
        
        # Calculate average volume
        avg_volume = df['Volume'].mean()
        
        # Create metrics cards
        metrics = [
            create_metric_card("Current Price", f"${current_price:.2f}", colors['blue']),
            create_metric_card("Daily Change", f"${price_change:.2f} ({price_change_pct:.2f}%)", 
                            colors['green'] if price_change >= 0 else colors['red']),
            create_metric_card("52-Week High", f"${week_52_high:.2f}", colors['blue']),
            create_metric_card("52-Week Low", f"${week_52_low:.2f}", colors['blue']),
            create_metric_card("Data Source", data_source, colors['blue'] if data_source == "Real" else colors['red'])
        ]
        
        return metrics
    
    except Exception as e:
        print(f"Error updating metrics: {str(e)}")
        print(traceback.format_exc())
        # Create empty metrics placeholders
        return [create_metric_card(title, "Error", colors['red']) for title in 
                ["Current Price", "Daily Change", "52-Week High", "52-Week Low", "Data Source"]]

def create_metric_card(title, value, color):
    return html.Div([
        html.H4(title, style={'margin': '0', 'color': colors['text']}),
        html.H2(value, style={'margin': '5px 0', 'color': color})
    ], style={'backgroundColor': colors['chart_background'], 'borderRadius': '10px', 'padding': '15px', 'width': '18%', 'textAlign': 'center'})

# Update price chart
@app.callback(
    Output('price-chart', 'figure'),
    [Input('stock-data-store', 'data'),
     Input('offline-data-store', 'data'),
     Input('indicators', 'value'),
     Input('stock-input', 'value'),
     Input('offline-mode', 'value')]
)
def update_price_chart(data, offline_data, indicators, ticker, offline_mode):
    # Choose which data to use
    selected_data = data if data is not None else (offline_data if offline_mode == 'yes' else None)
    
    if selected_data is None:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="No data available",
            plot_bgcolor=colors['chart_background'],
            paper_bgcolor=colors['chart_background'],
            font=dict(color=colors['text'])
        )
        return fig
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(selected_data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        
        # Check if this is synthetic data
        data_source = "Synthetic" if 'Synthetic' in df.columns and df['Synthetic'].iloc[0] else "Real"
        
        # Create candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color=colors['green'],
            decreasing_line_color=colors['red']
        )])
        
        # Add data source watermark if synthetic
        if data_source == "Synthetic":
            fig.add_annotation(
                text="SYNTHETIC DATA - NOT REAL MARKET DATA",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(color="rgba(255, 59, 48, 0.3)", size=30),
                textangle=-30,
                opacity=0.7
            )
        
        # Add technical indicators based on selection
        if 'sma50' in indicators and 'SMA50' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['SMA50'],
                mode='lines',
                name='50-day SMA',
                line=dict(color='orange', width=1)
            ))
        
        if 'sma200' in indicators and 'SMA200' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['SMA200'],
                mode='lines',
                name='200-day SMA',
                line=dict(color='purple', width=1)
            ))
        
        if 'bb' in indicators:
            for col in ['BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0']:
                if col in df.columns:
                    visible = True
                else:
                    visible = False
            
            if 'BBU_20_2.0' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['BBU_20_2.0'],
                    mode='lines',
                    name='Upper Bollinger Band',
                    line=dict(color='rgba(173, 204, 255, 0.7)', width=1),
                    visible=visible
                ))
            
            if 'BBM_20_2.0' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['BBM_20_2.0'],
                    mode='lines',
                    name='Middle Bollinger Band',
                    line=dict(color='rgba(173, 204, 255, 0.7)', width=1),
                    visible=visible
                ))
            
            if 'BBL_20_2.0' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['BBL_20_2.0'],
                    mode='lines',
                    name='Lower Bollinger Band',
                    line=dict(color='rgba(173, 204, 255, 0.7)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(173, 204, 255, 0.1)',
                    visible=visible
                ))
        
        # Update layout
        title_suffix = " (Synthetic Data)" if data_source == "Synthetic" else ""
        fig.update_layout(
            title=f"{ticker} Stock Price{title_suffix}",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            plot_bgcolor=colors['chart_background'],
            paper_bgcolor=colors['chart_background'],
            font=dict(color=colors['text']),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    except Exception as e:
        print(f"Error updating price chart: {str(e)}")
        print(traceback.format_exc())
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title=f"Error creating chart: {str(e)}",
            plot_bgcolor=colors['chart_background'],
            paper_bgcolor=colors['chart_background'],
            font=dict(color=colors['text'])
        )
        return fig

# Update RSI chart - similar modifications to support offline mode
@app.callback(
    Output('rsi-chart', 'figure'),
    [Input('stock-data-store', 'data'),
     Input('offline-data-store', 'data'),
     Input('stock-input', 'value'),
     Input('offline-mode', 'value')]
)
def update_rsi_chart(data, offline_data, ticker, offline_mode):
    # Choose which data to use
    selected_data = data if data is not None else (offline_data if offline_mode == 'yes' else None)
    
    if selected_data is None:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="No data available",
            plot_bgcolor=colors['chart_background'],
            paper_bgcolor=colors['chart_background'],
            font=dict(color=colors['text'])
        )
        return fig
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(selected_data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        
        # Check if this is synthetic data
        data_source = "Synthetic" if 'Synthetic' in df.columns and df['Synthetic'].iloc[0] else "Real"
        
        fig = go.Figure()
        
        # Add RSI line if it exists
        if 'RSI' in df.columns:
            # Drop NaN values for better visualization
            rsi_data = df['RSI'].dropna()
            
            if not rsi_data.empty:
                fig.add_trace(go.Scatter(
                    x=rsi_data.index,
                    y=rsi_data,
                    mode='lines',
                    name='RSI',
                    line=dict(color=colors['blue'], width=2)
                ))
                
                # Add overbought and oversold lines
                fig.add_shape(
                    type="line",
                    x0=rsi_data.index[0],
                    y0=70,
                    x1=rsi_data.index[-1],
                    y1=70,
                    line=dict(color="red", width=1, dash="dash")
                )
                
                fig.add_shape(
                    type="line",
                    x0=rsi_data.index[0],
                    y0=30,
                    x1=rsi_data.index[-1],
                    y1=30,
                    line=dict(color="green", width=1, dash="dash")
                )
                
                # Add data source watermark if synthetic
                if data_source == "Synthetic":
                    fig.add_annotation(
                        text="SYNTHETIC DATA",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(color="rgba(255, 59, 48, 0.3)", size=30),
                        textangle=-30,
                        opacity=0.7
                    )
            else:
                fig.add_annotation(
                    text="Insufficient data to calculate RSI",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(color=colors['text'], size=16)
                )
        else:
            fig.add_annotation(
                text="RSI not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(color=colors['text'], size=16)
            )
        
        # Update layout
        title_suffix = " (Synthetic Data)" if data_source == "Synthetic" else ""
        fig.update_layout(
            title=f"{ticker} RSI (14-day){title_suffix}",
            xaxis_title="Date",
            yaxis_title="RSI Value",
            template="plotly_dark",
            plot_bgcolor=colors['chart_background'],
            paper_bgcolor=colors['chart_background'],
            font=dict(color=colors['text']),
            yaxis=dict(range=[0, 100])
        )
        
        return fig
    
    except Exception as e:
        print(f"Error updating RSI chart: {str(e)}")
        print(traceback.format_exc())
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title=f"Error creating RSI chart: {str(e)}",
            plot_bgcolor=colors['chart_background'],
            paper_bgcolor=colors['chart_background'],
            font=dict(color=colors['text'])
        )
        return fig

# Update volume chart - similar modifications to support offline mode
@app.callback(
    Output('volume-chart', 'figure'),
    [Input('stock-data-store', 'data'),
     Input('offline-data-store', 'data'),
     Input('stock-input', 'value'),
     Input('offline-mode', 'value')]
)
def update_volume_chart(data, offline_data, ticker, offline_mode):
    # Choose which data to use
    selected_data = data if data is not None else (offline_data if offline_mode == 'yes' else None)
    
    if selected_data is None:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="No data available",
            plot_bgcolor=colors['chart_background'],
            paper_bgcolor=colors['chart_background'],
            font=dict(color=colors['text'])
        )
def update_volume_chart(data, offline_data, ticker, offline_mode):
    # Choose which data to use
    selected_data = data if data is not None else (offline_data if offline_mode == 'yes' else None)
    
    if selected_data is None:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="No data available",
            plot_bgcolor=colors['chart_background'],
            paper_bgcolor=colors['chart_background'],
            font=dict(color=colors['text'])
        )
        return fig
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(selected_data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        
        # Check if this is synthetic data
        data_source = "Synthetic" if 'Synthetic' in df.columns and df['Synthetic'].iloc[0] else "Real"
        
        # Create volume chart with colors based on price movement
        colors_list = []
        for i in range(len(df)):
            if i > 0:
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    colors_list.append(colors['green'])
                else:
                    colors_list.append(colors['red'])
            else:
                colors_list.append(colors['blue'])  # First bar
        
        fig = go.Figure(data=[
            go.Bar(
                x=df.index,
                y=df['Volume'],
                marker_color=colors_list,
                name='Volume'
            )
        ])
        
        # Add 20-day moving average of volume
        if len(df) >= 20:
            volume_ma = df['Volume'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=volume_ma,
                mode='lines',
                name='20-day MA',
                line=dict(color='yellow', width=2)
            ))
        
        # Add data source watermark if synthetic
        if data_source == "Synthetic":
            fig.add_annotation(
                text="SYNTHETIC DATA",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(color="rgba(255, 59, 48, 0.3)", size=30),
                textangle=-30,
                opacity=0.7
            )
        
        # Update layout
        title_suffix = " (Synthetic Data)" if data_source == "Synthetic" else ""
        fig.update_layout(
            title=f"{ticker} Trading Volume{title_suffix}",
            xaxis_title="Date",
            yaxis_title="Volume",
            template="plotly_dark",
            plot_bgcolor=colors['chart_background'],
            paper_bgcolor=colors['chart_background'],
            font=dict(color=colors['text'])
        )
        
        return fig
    
    except Exception as e:
        print(f"Error updating volume chart: {str(e)}")
        print(traceback.format_exc())
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title=f"Error creating volume chart: {str(e)}",
            plot_bgcolor=colors['chart_background'],
            paper_bgcolor=colors['chart_background'],
            font=dict(color=colors['text'])
        )
        return fig

# Add comparison chart callback
@app.callback(
    Output('comparison-chart', 'figure'),
    [Input('compare-button', 'n_clicks'),
     Input('stock-input', 'value'),
     Input('compare-input', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('offline-mode', 'value')]
)
def update_comparison_chart(n_clicks, main_ticker, comparison_tickers, start_date, end_date, offline_mode):
    if n_clicks == 0:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="Click 'Compare' to see comparison",
            plot_bgcolor=colors['chart_background'],
            paper_bgcolor=colors['chart_background'],
            font=dict(color=colors['text'])
        )
        return fig
    
    try:
        tickers = [main_ticker] + comparison_tickers.split(',')
        tickers = [t.strip() for t in tickers]  # Clean up whitespace
        
        # Initialize figure
        fig = go.Figure()
        
        # Flag to track if we're using synthetic data
        using_synthetic = False
        
        # For each ticker, get data and add normalized line
        for ticker in tickers:
            if not ticker:  # Skip empty tickers
                continue
                
            try:
                # Try to get real data first
                df = None
                try:
                    df = yf.download(ticker, start=start_date, end=end_date, progress=False, session=session)
                except Exception as e:
                    print(f"Error downloading {ticker}: {str(e)}")
                
                # Use synthetic data if needed
                if df is None or df.empty:
                    if offline_mode == 'yes':
                        df = generate_fallback_data(ticker)
                        df = df.loc[start_date:end_date]
                        using_synthetic = True
                    else:
                        print(f"No data available for {ticker}")
                        continue
                
                # Normalize to first day = 100 for comparison
                first_close = df['Close'].iloc[0]
                normalized_close = (df['Close'] / first_close) * 100
                
                # Add to chart
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=normalized_close,
                    mode='lines',
                    name=ticker
                ))
                
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
                continue
        
        # Add synthetic data warning if needed
        if using_synthetic:
            fig.add_annotation(
                text="INCLUDES SYNTHETIC DATA",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(color="rgba(255, 59, 48, 0.3)", size=30),
                textangle=-30,
                opacity=0.7
            )
        
        # Update layout
        data_warning = " (Some Synthetic Data)" if using_synthetic else ""
        fig.update_layout(
            title=f"Normalized Price Comparison{data_warning}",
            xaxis_title="Date",
            yaxis_title="Normalized Price (First Day = 100)",
            template="plotly_dark",
            plot_bgcolor=colors['chart_background'],
            paper_bgcolor=colors['chart_background'],
            font=dict(color=colors['text']),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating comparison chart: {str(e)}")
        print(traceback.format_exc())
        # Return error figure
        fig = go.Figure()
        fig.update_layout(
            title=f"Error creating comparison: {str(e)}",
            plot_bgcolor=colors['chart_background'],
            paper_bgcolor=colors['chart_background'],
            font=dict(color=colors['text'])
        )
        return fig

# Add server definition for WSGI deployment
server = app.server

# Run the app
if __name__ == '__main__':
    app.run(debug=True)