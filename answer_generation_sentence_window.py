num_sentence_window_size = 3

import dotenv
import os
import openai

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

chapter_id = 2
# load questions
from utils import get_eval_questions
eval_questions = get_eval_questions(chapter_id)

from llama_index import SimpleDirectoryReader
documents = SimpleDirectoryReader(f"data/USHistory_chapter_{chapter_id}").load_data()
from llama_index import Document
document = Document(text="\n\n".join([doc.text for doc in documents]))


from llama_index.llms import OpenAI
from utils import build_sentence_window_index, get_sentence_window_query_engine, get_prebuilt_trulens_recorder

sentence_index = build_sentence_window_index(
    documents,
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
    embed_model="local:BAAI/bge-small-en-v1.5",
    sentence_window_size=num_sentence_window_size,
    save_dir="sentence_index_" + str(num_sentence_window_size),
)
sentence_window_engine = get_sentence_window_query_engine(
    sentence_index
)
tru_recorder = get_prebuilt_trulens_recorder(
    sentence_window_engine,
    app_id='sentence window engine ' + str(num_sentence_window_size)
)

from trulens_eval import Tru
tru = Tru()
tru.reset_database()

from utils import run_evals
run_evals(eval_questions, 
          tru_recorder, 
          sentence_window_engine)
tru.run_dashboard()