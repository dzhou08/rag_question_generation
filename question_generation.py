import logging
import sys
import pandas as pd

import dotenv
import os
import openai

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    Response,
)
from llama_index.llms import OpenAI

# the textbook name and chapter id
textbook_name =  "USHistory"
textbook_chapter_id = 3
reader = SimpleDirectoryReader(f"./data/{textbook_name}_chapter_{textbook_chapter_id}")
documents = reader.load_data()
data_generator = DatasetGenerator.from_documents(documents)

bloom_question_template_prompt =  """\
Context information is below.
---------------------
{context_str}
---------------------
You’re a high-school AP History teacher and you want to make sure your
students remember their history concepts so they can do well on
their AP exam.

Given the context information and not prior knowledge, 
generate one multiple-choice question to the following learning objective. 
1. The question should contain four answer options, with only one correct answer.
2. Indicate the correct answer
3. provide feedback why other three are wrong answers

Learning Objective: Discuss economic, political, and demographic similarities and differences between the Spanish colonies

"""
data_generator.question_gen_query = bloom_question_template_prompt
eval_questions = data_generator.generate_questions_from_nodes(5)
print(eval_questions)

"""bloom_question_template_prompt
prompts = data_generator.get_prompts()
text_question_template = prompts["text_question_template"]
print("\n\n")
print(text_question_template.get_template())

print("\n\n")
text_question_template.template = bloom_question_template_prompt

print(text_question_template.get_template())

data_generator.update_prompts({"text_question_template":text_question_template})
print(data_generator.get_prompts())


"""

