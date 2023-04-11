from Preprocessing import Preprocess
from LDA_SRS import SrsLdaModel
from Cosine import CosineSimilarity

test = Preprocess()


if __name__ == '__main__':
    """TEXT PREPROCESSING"""

    test.set_start_page(1)
    test.set_end_page(10)
    #test.open_pdf('alaska-1.pdf')
    # call test.open_pdf_stream('alaska-1.pdf')
    # make a function to send pdf_stream
    test.process_pdf()
    test.form_n_grams()
    test.form_topic_model_inputs()

    """CREATE LDA MODELS"""
    lda_model = SrsLdaModel(5, test.get_corpus(),test.get_dictionary(),test.get_text())
    lda_model.create_models()
    lda_model.select_best_model(5)
    print("LDA MODEL: "+str(lda_model.get_selected_lda_model()))
    print("LDA COHERENCE SCRORE:"+str(lda_model.get_selected_model_coherence_score()))
    lda_model.save_lda_model("testmodel")
    lda_model.save_model_topic_terms("test_terms.txt")

    """SIMILARITY CALCULATION"""
    cosine_sim_test = CosineSimilarity()

    cosine_sim_test.init_lda_data(lda_model.get_selected_lda_model())
    cosine_sim_test.lda_calculate_cos_sim()
    cosine_sim_test.save_lda_cos_results("thisfile.csv")