

import dotenv
import os
import openai

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

chapter_id = 3
# load questions
from utils import get_eval_questions
eval_questions = get_eval_questions(chapter_id, "eval_mcp_questions")

from llama_index import SimpleDirectoryReader, VectorStoreIndex
documents = SimpleDirectoryReader(f"data/USHistory_chapter_{chapter_id}").load_data()
from llama_index import Document
document = Document(text="\n\n".join([doc.text for doc in documents]))

from llama_index import ServiceContext
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model="local:BAAI/bge-small-en-v1.5"
)
index = VectorStoreIndex.from_documents([document],
                                        service_context=service_context)

#index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

from utils import get_prebuilt_trulens_recorder

tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                             app_id="Direct Query Engine")

from trulens_eval import Tru
tru = Tru()
tru.reset_database()

from utils import run_evals
run_evals(eval_questions, 
          tru_recorder, 
          query_engine)
tru.run_dashboard()