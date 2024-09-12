import dash
from dash import dcc, html, Dash
from dash.dependencies import Input, Output, State
from dash import dash_table
from converter import HexToDecConverter

# Initialize the app
app = Dash()

user_inputs = []
translated_outputs = []
expected_outputs = []
accuracy = []

# App layout
app.layout = html.Div([
    html.Div([
        html.Div('Welcome to the hexadecimal to decimal translation tool!', 
                 style={'color': 'black', 'fontSize': 20, 'text-align': 'center'}),
        html.P('Please enter your hexadecimal value: ', 
               className='my-class', 
               id='my-p-element', 
               style={'color': 'black', 'fontSize': 20, 'text-align': 'center'})
    ], style={'marginBottom': 50, 'marginTop': 25}),
    
    html.Div([
        dcc.Input(id='input-box', type='text', value='', 
                  style={'margin-bottom': '20px', 'width': '300px'}),
        html.Button('Submit', id='submit-button', n_clicks=0),
        html.Div(id='output')
    ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'justify-content': 'center', 'marginBottom': 25}),
    
    html.Div([
        dash_table.DataTable(
            id='input-table', 
            columns=[{'name': 'Submission Number', 'id': 'number'}, 
                     {'name': 'Input Value', 'id': 'input'},
                     {'name': 'Translated Value', 'id': 'translated_output'},
                     {'name': 'Expected Value', 'id': 'expected_output'},
                     {'name': 'Accuracy', 'id': 'accuracy'}
                     ],
            data=[],
            style_cell={
            'textAlign': 'center',  
            'padding': '10px',      
        },
        )
    ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'justify-content': 'center', 'marginBottom': 25})
])

# Add callback for the interaction
@app.callback(
    [Output('output', 'children'),
     Output('input-table', 'data')],
    [Input('submit-button', 'n_clicks')],
    [State('input-box', 'value')]
)
def update_output(n_clicks, value):
    if n_clicks > 0 and value: 
    
        try:
            converter = HexToDecConverter()
            translated_value = converter.convert_hex_to_dec(value)
            expected_value = str(int(value, 16))

            user_inputs.append(value)
            translated_outputs.append(translated_value)
            expected_outputs.append(expected_value)

            total_chars = 0
            matching_chars = 0
            
            for pred, exp in zip(translated_value, expected_value):
                min_len = min(len(pred), len(exp))
                total_chars += min_len
                matching_chars += sum(1 for p, e in zip(pred, exp) if p == e)

            num_accuracy = round(matching_chars/total_chars, 3) * 100
            str_accuracy = "{:.1f}%".format(num_accuracy)
            accuracy.append(str_accuracy)

            table_data = [
                {
                    'number': i+1, 
                    'input': user_input, 
                    'translated_output': translated_outputs, 
                    'expected_output': expected_outputs,
                    'accuracy': accuracy
                } 
                for i, (user_input, translated_outputs, expected_outputs, accuracy) in enumerate(zip(user_inputs, translated_outputs, expected_outputs, accuracy))
            ]

            output_message = html.Div([
                html.P(f"Hexadecimal value: {value}"),
                html.P(f"Translated decimal value: {translated_value}"),
                html.P(f"Expected decimal value: {expected_value}")
            ])

        except ValueError:
            output_message = html.Div([
                html.P(f"'{value}' is not a valid hexadecimal value.", style={'color': 'red'})
            ])
            table_data = []

        return output_message, table_data

    return html.Div(), []

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)