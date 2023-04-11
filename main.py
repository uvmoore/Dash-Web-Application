import base64
import csv
import os
import io
import datetime
import re
import PyPDF2
import fitz

from fpdf import FPDF
import dash_table
from dash import Dash, html, dcc, callback, Input, Output, dash_table as dt, dash
import plotly.express as px
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import resources.utility
import plotly.graph_objects as go
import pandas as pd
from PyPDF2 import PdfFileReader
from flask import Flask, send_from_directory
from urllib.parse import quote as urlquote
from Preprocessing import Preprocess
from LDA_SRS import SrsLdaModel
from Cosine import CosineSimilarity
from gensim import corpora
from TestFile import test
from dash.dependencies import Input, Output, State


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
        # try:
        #    file_path = os.path.exists(self.file_path)
        #    file_path = True
        # break
        # except ValueError:
        #   print('Unidentifiable file path')
        read_pdf = PyPDF2.PdfFileReader(opener)
        length = read_pdf.numPages
        pdf_dict = {}
        for i in range(length):
            page = read_pdf.getPage(i)
            text = page.extract_text()
            pdf_dict[i] = text
            return pdf_dict

    def get_publish_date(self) -> str:
        """A function of which accepts an information dictionray of an object
            in the pdfReader class and returns the creation date of the
            object (if applicable).

            Parameters:
            self (obj): An object of the pdfReader class

            Returns:
            pub_date (str): The publication date which is assumed to be the
            creation date (if applicable).
        """
        info_dict_pdf = self.pdf_info()
        pub_date = 'None'
        try:
            publication_date = info_dict_pdf['CreationDate']
            publication_date = datetime.date.strptime(publication_date.replace("'", ""), "D:%Y%m%d%H%M%S%z")
            pub_date = publication_date.isoformat()[0:10]
        except:
            pass
        return str(pub_date)




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



app.layout = url_bar_and_content_div

home = html.Div(children=html.Center(children=[
    html.H1(children='Website Title'),
    html.H3(children='''
        Website Description
    '''),



    html.Div(children=[
        html.H5('''Step 1: Pick SRS File to Upload'''),
        dcc.Upload(
            #dbc.Spinner(html.Div(id="loading-output")),
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
            html.Div(children=[
                dcc.Graph(id="bar_chart1"),
                               html.Div(
                                   id='tableDiv',
                                   children=[]
                               )
                               ])

        ], style={'max-width': '85%',
                  'margin': 'auto'})),
        html.Div(id='output-datatable'),
        html.Div(id='output-data-upload'),
    ])
], style={'max-width': '85%',
          'margin': 'auto'}))

app.validation_layout = html.Div([
    url_bar_and_content_div,
    home,

])

textPreprocessing = Preprocess()
cosine_sim_test = CosineSimilarity()



#@app.callback(
#    Output("loading-output", "children"), [Input("upload-data", "loading_state")]
#)

#def parse_contents(contents, filename, date):
  #  content_type, content_string = contents.split(',')
  #  decoded = base64.b64decode(content_string)




#def LDA(self):
 #   lda_model = SrsLdaModel(5, textPreprocessing.get_corpus(), textPreprocessing.get_dictionary(), textPreprocessing.get_text())
  #  lda_model.create_models()
   # lda_model.select_best_model(5)
    #print("LDA MODEL: " + str(lda_model.get_selected_lda_model()))
    #print("LDA COHERENCE SCRORE:" + str(lda_model.get_selected_model_coherence_score()))
    #lda_model.save_lda_model("testmodel")
    #lda_model.save_model_topic_terms("test_terms.txt")
    #cosine_sim_test.init_lda_data(lda_model.get_selected_lda_model())
    #cosine_sim_test.lda_calculate_cos_sim()
    #cosine_sim_test.save_lda_cos_results("thisfile.csv")



def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    #test.open_pdf_stream(decoded)

    try:
        if 'pdf' in filename:
            html.center('You have uploaded a file')
            pdf = fitz.Document(stream=io.BytesIO(decoded))
            textPreprocessing = Preprocess()

            textPreprocessing.open_pdf_stream(decoded)
            textPreprocessing.process_pdf()
            textPreprocessing.form_n_grams()
            textPreprocessing.form_topic_model_inputs()

            lda_model = SrsLdaModel(5, textPreprocessing.get_corpus(), textPreprocessing.get_dictionary(),
                                    textPreprocessing.get_text())
            lda_model.create_models()
            lda_model.select_best_model(5)
            print("LDA MODEL: " + str(lda_model.get_selected_lda_model()))
            print("LDA COHERENCE SCRORE:" + str(lda_model.get_selected_model_coherence_score()))
            lda_model.save_lda_model("testmodel")
            lda_model.save_model_topic_terms("test_terms.txt")

            cosine_sim_test = CosineSimilarity()

            cosine_sim_test.init_lda_data(lda_model.get_selected_lda_model())
            cosine_sim_test.lda_calculate_cos_sim()
            cosine_sim_test.save_lda_cos_results("thisfile.csv")

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),
        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

    #with open("TestFile.py") as f:
      #  exec(f.read())

#------------------------------Old code may come back later--------------------------------------------------

#@app.callback(Output('output-data-upload', 'children'),
 #             Input('upload-data', 'contents'),
  #            State('upload-data', 'filename'),
   #           State('upload-data', 'last_modified'))
#def update_output(list_of_contents, list_of_names, list_of_dates):
 #   if list_of_contents is not None:
  #      children = [
   #         parse_contents(c, n, d) for c, n, d in
    #        zip(list_of_contents, list_of_names, list_of_dates)]
     #   return children

#--------------------------------Old code may come back later-------------------------------------------------

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
    if os.path.exists('thisfile.csv') == True:
        d = resources.utility.format_results('thisfile.csv')
    else:
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

