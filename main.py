import base64
import csv
import os
import io
import datetime

import PyPDF2

import SRSLDA

from fpdf import FPDF
import dash_table
from dash import Dash, html, dcc, callback, Input, Output, dash_table as dt, dash
import plotly.express as px
from dash.exceptions import PreventUpdate

import resources.utility
import plotly.graph_objects as go
import pandas as pd
from PyPDF2 import PdfFileReader
from flask import Flask, send_from_directory
from urllib.parse import quote as urlquote
import SRSLDA


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


app.layout = url_bar_and_content_div

home = html.Div(children=html.Center(children=[
    html.H1(children='Website Title'),
    html.H3(children='''
        Website Description
    '''),
    html.Div(children=[
        html.H5('''Step 1: Pick SRS File to Upload'''),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '400px',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
        ),
        html.H5('''Step 2: Choose Algorithm to Use'''),
        dcc.RadioItems(['LDA Algorithm', 'LSA Algorithm']),
        html.H5('''Step 3: Choose Number of Topics'''),
        html.Div(dcc.Input(id="topic_input", type='number', placeholder=1, min=1, max=10)),
        dcc.Link(html.Button('Submit', id='submit-val', n_clicks=0, style={'margin-top': '25px'}), href='/'),
        html.Div(id='button-container'),
        html.Div(children=html.Center(children=[
            html.Div(children=[dcc.Graph(id="bar_chart1"),
                               html.Div(
                                   id='tableDiv',
                                   children=[]
                               )
                               ])

        ], style={'max-width': '85%',
                  'margin': 'auto'})),
        html.Div(id='output-data-upload'),
    ])
], style={'max-width': '85%',
          'margin': 'auto'}))

app.validation_layout = html.Div([
    url_bar_and_content_div,
    home,

])


def parse_contents(contents,filename,date):
    if 'pdf' in filename:
        pdfFileObj = open(filename, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        pdfWriter = PyPDF2.PdfFileWriter()
        newFile = open('WebsitePDF.pdf', 'wb')
        for page in range(pdfReader.numPages):
            pageObj = pdfReader.getPage(page)
            pdfWriter.addPage(pageObj)
        pdfWriter.write(newFile)
        pageCount = pdfReader.numPages
        pdfFileObj.close()
        newFile.close()



@app.callback(Output('button-container', 'children'),
              [Input('submit-val','n_clicks')])
def get_csv(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    script_fn = 'SRSLDA.py'


    exec(open(script_fn).read())

#@app.callback(Output('upload-data', 'children'),
#          Input('submit-val', 'n_clicks'))
#def get_csv(contents, filename, date):
#    website_pdf = FPDF()
#    website_pdf.add_page()
#    website_pdf.set_font("Arial",size = 12)
#    website_pdf.text(10,10,txt=contents)
#    SRSLDA.pdf = website_pdf
#    with open('SRSLDA.py') as infile:
#        exec(infile.read())


# Update the index
@callback(Output('page-content', 'children'),
          [Input('url', 'pathname')])
def display_page(pathname):
    return home



@callback(Output("bar_chart1", 'figure'),
          Input("topic_input", 'value'))
def createGraph(topic):
    d = resources.utility.format_results("resources/results.csv")
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
    df = pd.DataFrame({'CAPEC':x_axis_capecs, 'Cosine Similarity':y_axis_similarity})
    fig = px.bar(df, x="CAPEC", y="Cosine Similarity")

    data = []
    trace_close = go.Bar(x=list(x_axis_capecs),
                         y=list(y_axis_similarity))

    data.append(trace_close)

    layout = {'xaxis':{
                    'title':'CAPEC'
                },
                'yaxis':{
                    'title':'Cosine Similarity'
                }}

    return {
        "data": data,
        "layout": layout
    }


@callback(Output("tableDiv", 'children'),
          Input("topic_input", 'value'))
def createTable(topic):
    header = ["CAPEC ID", "Description", "Cosine Similarity",  "Severity", "Prerequisites"]
    results = resources.utility.format_capeccsv("resources/1000.csv")
    d = resources.utility.format_results("resources/results.csv")
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
                csv_description = {row[0]:row[1] for row in reader}

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
                prerequisites_size = (csv_prerequisites[key][:30] + '..') if len(csv_prerequisites[key]) > 75 else csv_prerequisites[key]

                if key == capecs:
                    writer.writerow([capecs, csv_description[key], sim[:5],  csv_severity[key], csv_prerequisites[key]],)


    df = pd.read_csv('resources/table.csv')
    dff = df.sort_values(by=["Cosine Similarity"], ascending=False)
    return html.Div([dt.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=dff.to_dict("rows"),
        style_table={
            'height': '800px', 'overflowY': 'auto', 'overflowX' : 'scroll', 'padding' : '5px'
        },
        style_cell={
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'minWidth' : '0px', 'width': '5px', 'maxWidth': '10px',

        },
        style_data={
            'height': 'auto',
        },

        style_data_conditional=[{
            'if': {'column_id':'Cosine Similarity'},
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


if __name__ == '__main__':
    app.run_server(debug=True)
