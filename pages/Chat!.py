import streamlit as st


def prepare_llm(token):
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    from llama_index.core import ServiceContext
    from llama_index.embeddings.langchain import LangchainEmbedding
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
    from llama_index.llms.huggingface import HuggingFaceLLM
    from llama_index.core import PromptTemplate
    from huggingface_hub import login
    import torch

    login(token)

    documents = SimpleDirectoryReader("data").load_data()
    print(f'Total number of pages: {len(documents)}')

    system_prompt = """
    You are a Q&A assistant. Your goal is to answer questions as
    accurately as possible based on the instructions and context provided.
    """

    query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        device_map="auto",
        # uncomment this if using CUDA to reduce memory usage
        model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True}
    )

    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embed_model
    )

    index = VectorStoreIndex.from_documents(documents, service_context=service_context)

    return index.as_query_engine()


def ask_question(query_engine):
    query = st.text_input('Ask question')
    response = query_engine.query(query)
    st.write(response)


token = st.text_input('Enter your HuggingFace token:')

if token:
    query_engine = prepare_llm(token=token)
    while True:
        ask_question(query_engine=query_engine)

