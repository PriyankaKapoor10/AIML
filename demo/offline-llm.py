import os
import time

import torch
from dotenv import load_dotenv
from langchain.llms.base import LLM
from llama_index.core import (
    GPTListIndex,
    PromptHelper,
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,


)

from llama_index.legacy import (

    LLMPredictor,
)
from transformers import pipeline

os.environ["OPENAI_API_KEY"] = ""

#def timeit():

prompt_helper = PromptHelper(
    context_window=2048,
    num_output=256,
    chunk_overlap_ratio=.5,
    
)

class LocalOPT(LLM):

    model_name = "facebook/opt-iml-1.3b" 
    # 2.63 GB Model
    pipeline = pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype":torch.bfloat16}
    )


def _call(self,prompt:str,stop=None) ->str:
    response= self.pipeline(promtpt, max_new_tokens=256)[0]["generated_text"]
    return response[len(prompt) :]


@property
def _identifying_params():
    return {"name_of_model":self.model_name}

@property
def _llm_type(self):
    return "custom"

#@timeit()

def create_index():
    print("Creating index")
    #Wrappper around an LLMChain from Langchain
    llm= LLMPredictor(llm=LocalOPT())
    service_context= ServiceContext.from_defaults(
        llm_predictor= llm,prompt_helper=prompt_helper
    )
    reader = SimpleDirectoryReader(input_files=["budget/abc.pdf"])
    docs = reader.load_data()
    print("Loaded {len(docs)} docs")
    index= GPTListIndex.from_documents(docs, service_context=service_context)
    print("Done creating index",index)
    return create_index

@timeit()

def execute_query():
    query_engine = index.as_query_engine()
    response = query_engine.query(
        "Summarize Australia coal Export in 2023",


    )
    return response

if __name__ == "__main__":
   
    if not os.path.exists("./models/index_store.json"):
        print("No Local cache of model found, downloading from huggingface")
        index = create_index()
        index.storage_context.persist(persist_dir="./models")
    else:
        print("Loading from local cache of model")
        llm= LLMPredictor(llm=LocalOPT())
        service_context = ServiceContext.from_defaults(
            llm_predictor= llm,prompt_helper= prompt_helper
        )
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./models"))
    response = execute_query()
    print(response)
    print(response.source_nodes)






    




