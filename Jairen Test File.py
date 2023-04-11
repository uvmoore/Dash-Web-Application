import base64
from Preprocessing import Preprocess
import datetime
import io
import fitz
import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table
import pandas as pd
from LDS_SRS import SrsLdaModel
from Cosine import CosineSimilarity
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







external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,external_stylesheets=external_stylesheets)
app.title = "TRAP Tool"
app.layout = html.Div([
    html.Center(html.H1("TRAP Tool for Recommending Attack Patterns", id="Website-Title")),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload',),
    html.H2(id='options-instructions',children='Options'),
    html.Div(id='starting-page-div',children=
        html.P(id='starting-page-header',children=['Enter Starting Page Number: ',
        dcc.Input(id='start-page', type='number', placeholder='Starting Page Number')]),
    ),
    html.Div(id='ending-page-div',children=
        html.P(id='ending-page-header',children=['Enter Ending Page Number: ',
        dcc.Input(id='end-page',type='number', placeholder='Ending Page Number')]),
    ),
    html.Button("Continue to next page", id="continue-btn", n_clicks=0),
])

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'pdf' in filename:
            print("You have uploaded a file")
            pdf = fitz.Document(stream=io.BytesIO(decoded))
            page_text = pdf.load_page(5).get_text("text")
            print(page_text)
            textPreprocessing = Preprocess()
            textPreprocessing.set_start_page(1)
            textPreprocessing.set_end_page(5)

            textPreprocessing.open_pdf_stream(decoded)
            textPreprocessing.process_pdf()
            textPreprocessing.form_n_grams()
            textPreprocessing.form_topic_model_inputs()
            print(textPreprocessing.get_dictionary())

            """CREATE LDA MODELS"""
            lda_model = SrsLdaModel(5, textPreprocessing.get_corpus(), textPreprocessing.get_dictionary(), textPreprocessing.get_text())
            lda_model.create_models()
            lda_model.select_best_model(5)
            print("LDA MODEL: " + str(lda_model.get_selected_lda_model()))
            print("LDA COHERENCE SCRORE:" + str(lda_model.get_selected_model_coherence_score()))
            lda_model.save_lda_model("testmodel")
            lda_model.save_model_topic_terms("test_terms.txt")

            """SIMILARITY CALCULATION"""
            cosine_sim_test = CosineSimilarity()
            cosine_sim_test.init_lda_data(lda_model.get_selected_lda_model())
            cosine_sim_test.lda_calculate_cos_sim()
            cosine_sim_test.save_lda_cos_results("thisfile.csv")

        else:
            raise Exception

    except Exception as e:
        print(e)
        return html.Div([
            html.Center('Sorry we can only process PDF files at this time.')
        ],
            style={'color': 'red'}
        )

    return



if __name__ == '__main__':
    app.run_server(debug=True)
