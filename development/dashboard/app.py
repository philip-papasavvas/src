"""
Created on 25 Jan 2020
Practice making a dashboard with Dash & Plotly
"""
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


df = pd.DataFrame({'ccy':
                       {0: 'GBPAUD', 1: 'GBPNZD', 2: 'GBPUSD',
                        3: 'EURAUD', 4: 'EURNZD', 5: 'EURUSD'},
                   'no_breaches':
                       {0: 69, 1: 292, 2: 434, 3: 411, 4: 392, 5: 33}
                   }
                  )

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#11111',
    'text': '#7FDBFF'
}

app.layout = html.Div(children=[
    html.H4(children="Example Currency Breach DF"),
    generate_table(dataframe=df)
])

# app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
#     html.H1(
#         children='Hello Dash',
#         style={
#             'textAlign': 'center',
#             'color': colors['text']
#         }
#     ),
#     html.Div(children='Dash: A web application framework for Python.', style=
#     {'textAlign':'center',
#      'color': colors['text']
#      }
#     ),
#
#     dcc.Graph(
#         id='example-graph',
#         figure={
#             'data': [
#                 {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
#                 {'x': [1, 2, 3], 'y': [8, 4, 5], 'type': 'bar', 'name': u'Montréal'},
#             ],
#             'layout': {
#                 'title': 'Dash Data Visualization'
#             }
#         }
#     )
# ])

if __name__ == '__main__':
    app.run_server(debug=True)