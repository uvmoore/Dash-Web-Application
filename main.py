import base64
import csv
import textwrap
import os
import io
import re
import PyPDF2
import fitz
import dash_table
from dash import Dash, html, dcc, callback, Input, Output, dash_table as dt, dash
import plotly.express as px
from dash.exceptions import PreventUpdate
import resources.utility
import plotly.graph_objects as go
import pandas as pd
from flask import Flask, send_from_directory
from Preprocessing import Preprocess
from LDA_SRS import SrsLdaModel
from Cosine import CosineSimilarity
from dash.dependencies import Input, Output, State
import numpy as np
import random
import dash_bootstrap_components as dbc


class pdfReader:
    def __init__(self, file_path: str) -> str:
        self.file_path = file_path

    def PDF_one_pager(self) -> str:
        """A function that returns a one line string of the
            pdfReader object.

            Parameters:
            file_path(str): The file path to the pdf.

            Returns:
            one_page_pdf (str): A one line string of the pdf.

        """
        content = ""
        p = open(self.file_path, "rb")
        pdf = PyPDF2.PdfFileReader(p)
        num_pages = pdf.numPages
        for i in range(0, num_pages):
            content += pdf.getPage(i).extractText() + "\n"
        content = " ".join(content.replace(u"\xa0", " ").strip().split())
        page_number_removal = r"\d{1,3} of \d{1,3}"
        page_number_removal_pattern = re.compile(page_number_removal, re.IGNORECASE)
        content = re.sub(page_number_removal_pattern, '', content)

        return content

    def pdf_reader(self) -> str:
        """A function that can read .pdf formatted files
            and returns a python readable pdf.

            Returns:
            read_pdf: A python readable .pdf file.
        """
        opener = open(self.file_path, 'rb')
        read_pdf = PyPDF2.PdfFileReader(opener)

        return read_pdf

    def pdf_info(self) -> dict:
        """A function which returns an information dictionary
        of an object.

        Returns:
        dict(pdf_info_dict): A dictionary containing the meta
        data of the object.
        """
        opener = open(self.file_path, 'rb')
        read_pdf = PyPDF2.PdfFileReader(opener)
        pdf_info_dict = {}
        for key, value in read_pdf.documentInfo.items():
            pdf_info_dict[re.sub('/', "", key)] = value
        return pdf_info_dict

    def pdf_dictionary(self) -> dict:
        """A function which returns a dictionary of
            the object where the keys are the pages
            and the text within the pages are the
            values.

            Returns:
            dict(pdf_dict): A dictionary of the object within the
            pdfReader class.
        """
        opener = open(self.file_path, 'rb')
        read_pdf = PyPDF2.PdfFileReader(opener)
        length = read_pdf.numPages
        pdf_dict = {}
        for i in range(length):
            page = read_pdf.getPage(i)
            text = page.extract_text()
            pdf_dict[i] = text
            return pdf_dict


# Code above found on this link: https://towardsdatascience.com/pdf-parsing-dashboard-with-plotly-dash-256bf944f536


directory = 'C:/Users/uriah/Desktop/Jairens-Site-master'

upload = "/project/app_uploaded_files"

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

server = Flask(__name__)
app = Dash(server=server, external_stylesheets=external_stylesheets)
# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@server.route("/download/<path:path>")
def download(path):
    return send_from_directory(upload, path, as_attachment=True)


def from_rec_engine(lda_top_results):
    """
    This function is imported from LDA_Cosine.py (recommendation engine). It takes a list of the recommended CAPEC ids across all 10 topics along with thier
    corresponding cosine similarity values in the form of a list of tuples and reformatted it into a list of just the CAPEC ids.

    ARGS:
        lda_top_results: a list of tuples where the first element is the CAPEC ID and the second is corresponding cosine similarity value

    RETURNS:
        CAPECids: a list of just the first 50 CAPEC ids from lda_top_results list
    """

    formatlist = [id[0] for id in lda_top_results[:50]]
    CAPECids = pd.DataFrame(formatlist)
    return CAPECids


# the path for the two dataframes below may be different on your local computer

file = pd.read_csv('thisfile2.csv')
first_values = file.iloc[:, 0].str.extract(r"\((\d+)").astype(int).values.flatten().tolist()

df = pd.read_csv('resources/Comprehensive CAPEC Dictionary.csv')
capec_ids = pd.DataFrame(first_values)

filtered_df = df[df['ID'].isin(capec_ids[0].tolist())]
filtered_df = filtered_df.reset_index(drop=True)

color = {'Very High': 'red',
         'High': 'maroon',
         'Medium': 'indigo',
         'Low': 'blue',
         'Very Low': 'turquoise',
         'None': 'lightblue',
         np.nan: 'lightblue'}
severity_color = [color[val] for val in df['Typical Severity']]
weights = {'High': 35,
           'Medium': 27,
           'Low': 18,
           np.nan: 18}
weight_size = [weights[val] for val in df['Likelihood Of Attack']]


@app.callback(
    Output('word-cloud', 'figure'),
    Input('dropdown', 'value')
)
def update_figure(numofIDs):
    """
    This function is a callback function that is triggered when a user selects a value from the dropdown menu. The value that is selected
    via dropdown menu is equivalent to the number of CAPEC IDs that are going to be displayed on the word cloud. This funciton updates the
    word cloud to show the correct amount of CAPEC IDs.

    ARGS:
        numofIDs: a integer (default value = 20) chosen by the user via dropdown menu. Can be a value 20-50.

    RETURNS:
        fig: the updated word cloud with the correct number of CAPEC IDs

    """

    layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                        'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}},
                       width=765,
                       height=765)

    try:
        fig = go.Figure(data=go.Scatter(x=random.sample(list(range(650)), numofIDs),
                                        y=random.sample(list(range(650)), numofIDs),
                                        mode='text',
                                        text=filtered_df['ID'],
                                        hovertext=filtered_df['Name'],
                                        hoverinfo='text',
                                        textfont={'size': weight_size,
                                                  'color': severity_color}),
                        layout=layout)
        return fig
    except UnboundLocalError as e:
        print('Unbound Local Error', e)


# --------------------------------------------------------------------


fig = update_figure(20)

app.layout = url_bar_and_content_div

# cids is going to dictate how many CAPEC ids are shown
# range(num) represents range of values to be randomly selected from, reducing the chances of overlapping points.


home = html.Div(children=html.Center(children=[

    html.H1(id='title', children=[html.Span('TrapTM-', style={'fontSize': '36px'}),
                                  ' A Tool for Recommending Attack Patterns using Topic Modeling'
                                  ], style={'fontSize': '24px'}),
    html.H3(children='''
        
    '''),
    html.Div(children=[
        html.H5('''Step 1: Pick SRS File to Upload'''),
        dcc.Upload(
            # dbc.Spinner(html.Div(id="loading-output")),
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '300px',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin-top': '60px',
                'margin-bottom': '60px'

            },

        ),
        dcc.Loading(
            id='loading-output',
            type='default',
            children=[
                html.Div(id='output-data-upload')
            ]
        ),

        dbc.Tooltip(
            "This is where you can upload a PDF of your SRS document",
            target='upload-data',
            placement='auto',
            style={'width': '200px'
                , 'textAlign': 'center'},
        ),

        dbc.Tooltip(
            "Upload your Software Requirements Specification in pdf file, choose the topic modeling algorithm,"
            " choose the number of topics, and see the recommended CAPEC attack patterns. ",
            target='title',
            placement='auto',
            style={'width': '1200px'},
        ),

    ]),
    html.H5('''Step 2: Choose Algorithm to Use'''),
    dcc.RadioItems(['LDA Algorithm', 'LSA Algorithm']),
    html.H5('''Step 3: Choose Number of Topics'''),
    html.Div(dcc.Input(id="topic_input", type='number', placeholder=1, min=1, max=10)),
    dbc.Tooltip(
        "Use the arrow keys are type in a value to choose a number of topics you want to display",
        target='topic_input',
        placement='right',
        style={'width': '300px'},
    ),
    dcc.Link("Go to Word Cloud", href="/page-2", style={'display': 'block', 'margin-top': '25px'}),
    html.Div(id='button-container'),
    html.Div(children=html.Center(children=[
        html.Div(children=[
            dcc.Graph(id="bar_chart1"),
            html.Div(
                id='tableDiv',
                children=[]
            )
        ])

    ],
        style={'max-width': '85%',
               'margin': 'auto'})),
    html.Div(id='output-datatable'),
    html.Div(id='output-data-upload'),

],

    style={'max-width': '85%',
           'margin': 'auto'}))

word_cloud_layout = html.Div([

    dcc.Link("Go back", href="/page-1"),

    html.P("Select the number of CAPEC IDs"),
    dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': '20', 'value': 20},
            {'label': '30', 'value': 30},
            {'label': '40', 'value': 40},
            {'label': '50', 'value': 50}
        ], value=20
    ),

    html.Div([
        dcc.Graph(
            id='word-cloud',
            figure=fig,
            clickData=None,
            style={'width': '50%', 'display': 'inline-block'}
        ),
        html.Div([
            html.P('Word Cloud Legend', style={'color': 'black', 'fontSize': 16}),
            html.Div([
                html.P(children='Severity:', style={'color': 'black', 'margin': '0px 10px'}),
                html.P('Very High', style={'color': color['Very High'], 'margin': '0px 10px'}),
                html.P('High', style={'color': color['High'], 'margin': '0px 10px'}),
                html.P('Medium', style={'color': color['Medium'], 'margin': '0px 10px'}),
                html.P('Low', style={'color': color['Low'], 'margin': '0px 10px'}),
                html.P('Very Low', style={'color': color['Very Low'], 'margin': '0px 10px'}),
                html.Br(),

            ], style={'display': 'flex', 'fontSize': 25}),
            html.Br(),
            html.Div([
                html.P(children='Likelihood of Attack:', style={'color': 'black', 'margin': '0px 10px'}),
                html.P('High', style={'color': 'black', 'fontSize': weights['High'], 'margin': '0px 15px'}),
                html.P('Medium', style={'color': 'black', 'fontSize': weights['Medium'], 'margin': '0px 15px'}),
                html.P('Low', style={'color': 'black', 'fontSize': weights['Low'], 'margin': '0px 15px'}),
            ], style={'display': 'flex', 'fontSize': 16, 'margin': 'auto'}),
            html.Div(id='point-info')
        ], style={'fontSize': 18, 'width': '50%', 'marginLeft': 40, 'marginTop': 80}
        )], style={'display': 'inline-block', 'align': 'center'}),

    html.Div([
        html.P(
            'Cick on one of the following to learn about the related weaknesses, example instances and mitigations of the corresponding attack pattern '),
        dcc.RadioItems(
            id='radio-items',
            options=[
                {'label': 'Related Weaknesses', 'value': 'weakness'},
                {'label': 'Instances', 'value': 'instance'},
                {'label': 'Mitigation', 'value': 'mitigation'},
            ],
            value='mitigation'),
        html.Div(id='table')
    ])
])

app.validation_layout = html.Div([
    url_bar_and_content_div,
    home,
    word_cloud_layout,
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return home
    elif pathname == '/page-2':
        return word_cloud_layout
    else:
        # If the URL doesn't match any page, redirect to the default page (Page 1)
        return dcc.Location(pathname='/page-1', id='dummy')


textPreprocessing = Preprocess()
cosine_sim_test = CosineSimilarity()


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'pdf' in filename:

            pdf = pdfReader(filename)
            text = pdf.PDF_one_pager()

            print("You have uploaded a file")
            pdf = io.BytesIO(decoded).read()
            print('just want to test something')
            print('Test1')
            textPreprocessing = Preprocess()
            textPreprocessing.set_start_page(1)
            textPreprocessing.set_end_page(13)
            print('Test H')

            textPreprocessing.open_pdf_stream(pdf)
            print('test 1.5')
            print('Test2')
            textPreprocessing.process_pdf()
            textPreprocessing.form_n_grams()
            textPreprocessing.form_topic_model_inputs()
            print(textPreprocessing.get_dictionary())
            print('Test3')

            """CREATE LDA MODELS"""
            lda_model = SrsLdaModel(5, textPreprocessing.get_corpus(), textPreprocessing.get_dictionary(),
                                    textPreprocessing.get_text())
            lda_model.create_models()
            lda_model.select_best_model(5)
            print("LDA MODEL: " + str(lda_model.get_selected_lda_model()))
            print("LDA COHERENCE SCRORE:" + str(lda_model.get_selected_model_coherence_score()))
            print('Test 3.5')
            lda_model.save_lda_model("testmodel")
            print('Test 3.6')
            lda_model.save_model_topic_terms("test_terms.txt")
            print('Test4')

            """SIMILARITY CALCULATION"""
            cosine_sim_test = CosineSimilarity()
            cosine_sim_test.init_lda_data(lda_model.get_selected_lda_model())
            cosine_sim_test.lda_calculate_cos_sim()
            cosine_sim_test.save_lda_cos_results("thisfile2.csv")
            print('past this step')

    except Exception as e:
        print(e)
        return html.Div([
            ' '
        ])

    return html.Div([
        html.H5(filename),  # return the filename
    ])

    # with open("TestFile.py") as f:
    #  exec(f.read())


# ------------------------------Old code may come back later--------------------------------------------------

@app.callback(Output('loading-output', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [parse_contents(list_of_contents, list_of_names, list_of_dates)]
        return children


@callback(Output("bar_chart1", 'figure'),
          Input("topic_input", 'value'))
def createGraph(topic):
    d = resources.utility.format_results('thisfile2.csv')

    # gets the key values as list for x axis in graph
    x_axis_capecs = list(d.keys())
    # gets cosine similarities from specific topic
    y_axis_similarity = []
    count = 1
    for item in d.values():
        if topic is None:
            raise PreventUpdate
        else:
            y_axis_similarity.append(item[int(topic) - 1])
    df = pd.DataFrame({'CAPEC': x_axis_capecs, 'Cosine Similarity': y_axis_similarity})
    fig = px.bar(df, x="CAPEC", y="Cosine Similarity")

    data = []
    trace_close = go.Bar(x=list(x_axis_capecs),
                         y=list(y_axis_similarity))

    data.append(trace_close)

    layout = {'xaxis': {
        'title': 'CAPEC'
    },
        'yaxis': {
            'title': 'Cosine Similarity'
        }}

    return {
        "data": data,
        "layout": layout
    }


@app.callback(
    Output('point-info', 'children'),
    Input('word-cloud', 'clickData')
)
def on_click(clickData):
    """
    This function is a callback function that is triggered when a user clicks on a data point. When a data point is clicked the following data from
     the filtered_df is displayed: Name, Description, link to capec.mitre.org page

    ARGS:
        clickData : a dictionary that contains information about a CAPEC ID on the chart that was selected.
                    ['points'] being a list of all the clicked points
                    [0] being the first point since only one point can be clicked at a time
                    ['pointIndex'] being the index of the first point selected corresponding to the index of the selected CAPEC ID in filtered_df
    RETURNS:
        A HTML Div that contains the name and description of the selected CAPEC ID along with the link to it from the capec.org website
    """

    if clickData is None:
        return html.P('Click on a CAPEC ID to see a description of the attack pattern',
                      style={'color': 'grey', 'fontSize': 15})

    point_index = clickData['points'][0]['pointIndex']
    name = filtered_df.loc[point_index, 'Name']
    description = filtered_df.loc[point_index, 'Description']
    id = filtered_df.loc[point_index, 'ID']
    wraptext = '\n'.join(textwrap.wrap(description, width=100))
    text = f'To learn more follow this link: '
    link = f'https://capec.mitre.org/data/definitions/{id}.html'

    return html.Div([
        html.H3(children=name, style={'color': 'darkgrey', 'fontSize': 25}),
        html.P(children=wraptext, style={'color': 'grey', 'fontSize': 15}),
        html.P(children=[text, html.A(link, href=link, target='_blank')], style={'color': 'grey', 'fontSize': 12})
    ])


@app.callback(
    Output('table', 'children'),
    Input('radio-items', 'value'),
    Input('word-cloud', 'clickData')
)
def update_table(value, clickData):
    """
    This function is a callback function that is triggered when a user clicks on a data point. The selected data point is the same point as
    the function on_click seen above. This function contains radio button items such as related weaknesses, instances, and mitigations.
    Each of its respective information from the comprehensive CAPEC csv file is displayed when corresponding radio item is selected.

    ARGS:
        clickData : a dictionary that contains information about a CAPEC ID on the chart that was selected.
                    ['points'] being a list of all the clicked points
                    [0] being the first point since only one point can be clicked at a time
                    ['pointIndex'] being the index of the first point selected corresponding to the index of the selected CAPEC ID in filtered_df
    RETURNS:
        A HTML Div containing data from comprehensive CAPEC csv file depending on which radio item was selected
    """

    if clickData is None:
        return html.P('If a CAPEC ID is clicked, information will be displayed here.',
                      style={'color': 'grey', 'fontSize': 15})

    point_index = clickData['points'][0]['pointIndex']

    if value == 'weakness':
        weakness = filtered_df.loc[point_index, 'Related Weaknesses']
        if pd.isna(weakness):
            return html.P('No weakness data available', style={'color': 'red', 'fontSize': 15})

        cwe_ids = weakness.split("::")[1:-1]
        list_of_cwe = []
        text = 'Below you will find the link(s) of realted weaknesses from the Common Weakness Enumeration (CWE) catolog'
        for cwe in cwe_ids:
            link = f'https://cwe.mitre.org/data/definitions/{cwe}'
            list_of_cwe.append(
                html.P(html.A(link, href=link, target='_blank'), style={'color': 'grey', 'fontSize': 15}))

        return html.Div([
            html.P(text, style={'color': 'grey', 'fontSize': 15}),
            html.P(list_of_cwe)
        ])

    elif value == 'instance':
        instance = filtered_df.loc[point_index, 'Example Instances']
        if pd.isna(instance):
            return html.P('No example instance data available', style={'color': 'red', 'fontSize': 15})

        text = instance.replace('::', '\n')
        return html.P(text, style={'color': 'grey', 'fontSize': 15})

    elif value == 'mitigation':
        mitigation = filtered_df.loc[point_index, 'Mitigations']
        if pd.isna(mitigation):
            return html.P('No mitigation available', style={'color': 'red', 'fontSize': 15})

        text = mitigation.replace('::', '\n')
        return html.P(text, style={'color': 'grey', 'fontSize': 15})


@callback(Output("tableDiv", 'children'),
          Input("topic_input", 'value'))
def createTable(topic):
    header = ["CAPEC ID", "Description", "Cosine Similarity", "Severity", "Prerequisites"]
    results = resources.utility.format_capeccsv("resources/1000.csv")
    d = resources.utility.format_results("thisfile2.csv")
    capecs = list(d.keys())
    similarity = []
    for item in d.values():
        if topic is None:
            raise PreventUpdate
        else:
            similarity.append(item[int(topic) - 1])
    with open('resources/table.csv', 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        csv_dictionary = {}
        writer.writerow(header)

        csv_description = {}
        csv_severity = {}
        csv_prerequisites = {}

        with open("resources/1000.csv", "r", encoding='utf8') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                csv_description = {row[0]: row[1] for row in reader}

        #              csv_prerequisites = {row[0]:row[10] for row in reader}

        with open("resources/1000.csv", "r", encoding='utf8') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                csv_severity = {row[0]: row[7] for row in reader}

        with open("resources/1000.csv", "r", encoding='utf8') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                csv_prerequisites = {row[0]: row[10] for row in reader}

        prerequisites_size = " "

        for capecs, sim in zip(capecs, similarity):
            for key in csv_description:
                prerequisites_size = (csv_prerequisites[key][:30] + '..') if len(csv_prerequisites[key]) > 75 else \
                    csv_prerequisites[key]

                if key == capecs:
                    writer.writerow(
                        [capecs, csv_description[key], sim[:5], csv_severity[key], csv_prerequisites[key]], )

    df = pd.read_csv('resources/table.csv')
    dff = df.sort_values(by=["Cosine Similarity"], ascending=False)
    return html.Div([dt.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=dff.to_dict("rows"),
        style_table={
            'height': '800px', 'overflowY': 'auto', 'overflowX': 'scroll', 'padding': '5px'
        },
        style_cell={
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'minWidth': '0px', 'width': '5px', 'maxWidth': '10px',

        },
        style_data={
            'height': 'auto',
        },

        style_data_conditional=[{
            'if': {'column_id': 'Cosine Similarity'},
            'width': '12%',

        },
            {'if': {'column_id': 'Severity'},
             'width': '5%',
             },
            {'if': {'column_id': 'CAPEC ID'},
             'width': '5%',
             },
            {'if': {'column_id': 'Description'},
             'width': '25%',
             }
        ],

        tooltip_data=[
            {
                column: {'value': str(value), 'type': 'markdown'}
                for column, value in row.items()
            } for row in dff.to_dict('records')
        ]

    )])


pdf_table = dash_table.DataTable(id='pdf-viewer',
                                 page_action='none',
                                 fixed_rows={'headers': True},
                                 style_table={'height': 500, 'overflowY': 'auto'},
                                 style_header={'overflowY': 'auto'}
                                 )

if __name__ == '__main__':
    app.run_server(debug=True)
