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
print(document)


from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

from utils import build_automerging_index
automerging_index = build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index"
)

from utils import get_automerging_query_engine,get_prebuilt_trulens_recorder
automerging_query_engine = get_automerging_query_engine(
    automerging_index,
)
tru_recorder_automerging = get_prebuilt_trulens_recorder(automerging_query_engine,
                            app_id="Automerging Query Engine")

from trulens_eval import Tru
tru = Tru()
tru.reset_database()

from utils import run_evals
run_evals(eval_questions, 
          tru_recorder_automerging, 
          automerging_query_engine)
tru.run_dashboard()