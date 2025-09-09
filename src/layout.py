from dash import dcc, html
from src.config import UPDATE_INTERVAL_MS, EEG_CHANNELS

def build_graph(id, title, y_label):
    """Helper function to build a graph component."""
    return dcc.Graph(id=id, config={"displayModeBar": False}, style={'height': '250px'})

def get_layout():
    """Returns the full layout of the Dash application."""
    return html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div(id='main-page-content')
    ])

def get_page_content():
    """Returns the content of the page, including tabs and stores."""
    return html.Div([
        html.H2("MindBrawl Dashboard"),
        dcc.Interval(id='interval', interval=UPDATE_INTERVAL_MS, n_intervals=0),
        
        # Store for game state
        dcc.Store(id='game-state-store', data={
            'score': 0, 
            'target_pos': (175, 175),
            'last_blink_time': 0,
            'shot_timer': 10.0
        }),

        dcc.Tabs(id="tabs", value='game-tab', children=[
            dcc.Tab(label='Mind Game', value='game-tab'),
            dcc.Tab(label='EEG Time Series', value='eeg-tab'),
            dcc.Tab(label='Motion Sensors', value='motion-tab'),
            dcc.Tab(label='Band Power', value='band-tab'),
        ]),
        
        html.Div(id='tab-content')
    ])

def get_tab_content(tab):
    """Renders the content for the selected tab."""
    if tab == 'eeg-tab':
        return html.Div([
            html.H4("Filter Controls"),
            dcc.Checklist(
                id='filter-checklist',
                options=[
                    {'label': 'Power Line Notch (60Hz)', 'value': 'notch'},
                    {'label': 'High-Pass (>1Hz)', 'value': 'highpass'},
                    {'label': 'Low-Pass (<50Hz)', 'value': 'lowpass'},
                ],
                value=['notch', 'highpass'],
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}
            ),
            html.H4("EEG Channels"),
            *[build_graph(f'eeg-graph-{ch}', f"EEG - {ch}", "µV") for ch in EEG_CHANNELS]
        ])
    
    elif tab == 'motion-tab':
        return html.Div([
            html.H4("Accelerometer"),
            build_graph('accel-graph', 'Accelerometer', 'm/s²'),
            html.H4("Gyroscope"),
            build_graph('gyro-graph', 'Gyroscope', '°/s')
        ])
    
    elif tab == 'band-tab':
        return html.Div([
            html.H4("EEG Band Power"),
            html.Div(id='bandpower-graphs')
        ])
    
    elif tab == 'game-tab':
        return html.Div([
            html.H4("The Zen Archer"),
            html.P("Aim with your head, focus to steady your aim, and blink to shoot!"),
            html.Button('Start / Reset Game', id='start-game-btn', n_clicks=0),
            html.Div(
                id='game-area',
                style={
                    'width': '500px', 'height': '500px', 'border': '2px solid black',
                    'margin': 'auto', 'position': 'relative', 'overflow': 'hidden',
                    'background': '#f0f0f0'
                },
                children=[
                    html.Div(id='target-outer', style={
                        'position': 'absolute', 'width': '150px', 'height': '150px',
                        'borderRadius': '50%', 'backgroundColor': 'lightblue', 'left': '175px', 'top': '175px'
                    }, children=[
                        html.Div(id='target-middle', style={
                            'position': 'absolute', 'width': '100px', 'height': '100px',
                            'borderRadius': '50%', 'backgroundColor': 'lightgreen', 'left': '25px', 'top': '25px'
                        }, children=[
                            html.Div(id='target-inner', style={
                                'position': 'absolute', 'width': '50px', 'height': '50px',
                                'borderRadius': '50%', 'backgroundColor': 'red', 'left': '25px', 'top': '25px'
                            })
                        ])
                    ]),
                    html.Div(id='crosshair', style={
                        'width': '30px', 'height': '30px', 'border': '2px solid cyan',
                        'position': 'absolute', 'top': '235px', 'left': '235px'
                    }),
                    html.Div(id='timer-bar', style={
                        'position': 'absolute', 'bottom': '0', 'left': '0', 'height': '10px',
                        'backgroundColor': 'orange', 'width': '100%'
                    })
                ]
            ),
            html.Div(id='game-state-output', style={'textAlign': 'center', 'marginTop': '20px'})
        ])
    return html.Div("Unknown Tab")
