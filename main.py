
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
import chromadb
from langchain.memory import ConversationBufferMemory
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os , tempfile
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer



###### Streamlit page configuration #####
st.set_page_config(
    page_title="Ollama PDF RAG Streamlit UI",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

##### create sidebar #####

st.sidebar.title("Session diagnostics")
st.sidebar.subheader(" These details are used to track the chatbots status")

####define and initialize the AI model and the embedings
@st.cache_resource()
def model_embed():
  ai_model = ChatOllama(model="gemma3n", temperature=0.2)
  model_embeddings = "mxbai-embed-large:335m"
  return (ai_model, model_embeddings)

### Create a class to process the user PDF
class process_pdf():
  def __init__(self, embed, ai_model):
    self.ai_embeddings = embed
    self.ai_model = ai_model

   # Create vector DB with document

  def create_vector_db(self,chunked, document_name):
    print(document_name, chunked)

    pdf_vector = Chroma.from_documents(
        documents=chunked,
        embedding=OllamaEmbeddings(model=self.ai_embeddings),
        collection_name="pdf_db",
        persist_directory="./pdf_dbs"
    )
    st.sidebar.write(f'Vector DB created from {document_name}')
    return (pdf_vector)


    # Chunk the splitted pages
 
  def chunk_doc(self, pages, doc_name):
    # print(f'{len(pages)} pages loaded from {doc_name} \n chunking document for vector DB now. \n')
    pdf_chunker = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 200,
    )
    chunks = pdf_chunker.split_documents(pages)
    # print(f'{len(chunks)} chunks created from {doc_name} \n')
    return (self.create_vector_db(chunks , doc_name))


  ### split the uploaded PDF ###
 
  def split_pdf (self, staged_pdf):
    #staged_pdf = "./pdfdoc/Check Fraud Statistics_2025.pdf"
    
    if staged_pdf:
      loader = PyPDFLoader(staged_pdf)
      pages = loader.load()
      doc_name = staged_pdf.split('/')[-1]
      st.sidebar.write(f'{len(pages)} pages loaded from {doc_name}')
    else:
      st.write(f'Please upload a PDF document')

    return(self.chunk_doc(pages, doc_name))



#### Create the Retriever after creating the vector DB

def create_retriever(vector_db, ai_model):
  created_vector_db = vector_db
  retriever = MultiQueryRetriever.from_llm(
      created_vector_db.as_retriever(
          search_type="similarity",
          search_kwargs={"k": 2}
      ),
      llm=ai_model
      )
  return (retriever)


#### Create the system prompt ####

def create_rag(retriever, ai_model):
  # create the system prompt
  system_prompt = """
  You are an assistant for question-answering tasks. use the information from the context to help answer the question.
  context:{context}


  """

  prompt = ChatPromptTemplate.from_messages([
      SystemMessagePromptTemplate.from_template(system_prompt),
      # MessagesPlaceholder(variable_name="chat_history"),
      HumanMessagePromptTemplate.from_template("{question}")
  ])

  #### Define the core RAG chain ####
  rag_chain_core = (
      {"context": retriever,
      "question": RunnablePassthrough(),
      # "chat_history": RunnableLambda(lambda x: x["chat_history"])
      }
      | prompt
      | ai_model
  )
  return (rag_chain_core)

# #### Define the function the runnable will use to get the chat history NOT USED FOR THIS VERSION ####

# def get_chat_history(session_id: str, chat_list) -> InMemoryChatMessageHistory:
#   if session_id not in chat_list:
#     chat_list[session_id] = InMemoryChatMessageHistory()
#   return chat_list[session_id]

# ##### Create the runnable with message history ####
# def create_pipeline(rag_chain_core, chat_func):
#   pipeline_with_history = RunnableWithMessageHistory(
#       rag_chain_core,
#       get_session_history=chat_func,
#       input_message_key="question",
#       history_messages_key="chat_history"
# )
#   return (pipeline_with_history)

#### create a function to generate a 3 digit session id ####
def generate_session_id():
  import random
  import string
  return ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))


############################Streamlit app creation #######################



#### create a main function to run the streamlit app ####

def app_runner():

  st.subheader("PDF chatbot using RAG" , divider="gray", anchor=False)

  ### get the model and embedings ###
  st.session_state.ai_model, st.session_state.model_embeddings = model_embed()

  #### Initialize session state ###
  if "messages" not in st.session_state: # to store chat conversations outside the runnable
      st.session_state["messages"] = []
  if "vector_db" not in st.session_state:
      st.session_state["vector_db"] = None
  if "chat_list" not in st.session_state: # for the runnable to store chat conversations
      st.session_state["chat_list"] = {}
  if "rag_chain_core" not in st.session_state:
    st.session_state.rag_chain_core = None

  ##### Create the runnable with message history NOT USED FOR THIS VERSION ####
  # def create_pipeline(rag_chain_core, chat_func):
  #   pipeline_with_history = RunnableWithMessageHistory(
  #       rag_chain_core,
  #       get_session_history=chat_func,
  #       input_message_key="question",
  #       history_messages_key="chat_history"
  # )
  #   return (pipeline_with_history)


    #### Define the function the runnable will use to get the chat history NOT USED FOR THIS VERSION ####

  # def get_chat_history(session_id: str, chat_list) -> InMemoryChatMessageHistory:
  #   if session_id not in st.session_state.chat_list:
  #     chat_list[session_id] = InMemoryChatMessageHistory()
  #   return chat_list[session_id]


  #####Capture and process document######

  # create the vector_db from the uploaded doc
  st.session_state.user_file = st.file_uploader(" Please upload your PDF doc.", type=["pdf"])

  st.session_state.user_doc = process_pdf(st.session_state.model_embeddings, st.session_state.ai_model)

 
  if st.session_state.user_file:
    tempdir = "temp_dir"
    os.makedirs(tempdir, exist_ok=True)
        
    with open(os.path.join ("temp_dir",st.session_state.user_file.name),"wb") as f:
      f.write(st.session_state.user_file.getbuffer())

    st.sidebar.write(os.path.join ("temp_dir",st.session_state.user_file.name))
    st.sidebar.write("File uploaded successfully!")
    st.sidebar.write(f"File name: {st.session_state.user_file.name}")

    @st.cache_resource()
    def get_vector():
      vect_db = st.session_state.user_doc.split_pdf(os.path.join ("temp_dir",st.session_state.user_file.name))
      return vect_db

    with st.spinner("Vectorizing PDF....."):
      st.session_state["vector_db"] = get_vector()
   

  # create Retriever then system prompt
  if st.session_state["vector_db"] is not None:
    st.session_state["retriever"] = create_retriever(st.session_state["vector_db"], st.session_state.ai_model)
    st.session_state["rag_chain_core"] = create_rag(st.session_state["retriever"], st.session_state.ai_model)
    if "session_id" not in st.session_state:
      st.session_state["session_id"] = generate_session_id()
    # st.session_state["pipeline_with_history"] = create_pipeline(st.session_state["rag_chain_core"], get_chat_history())
  st.sidebar.divider()
  st.sidebar.subheader( "All session state variables")
  st.sidebar.write(st.session_state)

  # Ask the user to enter a question
  if st.session_state.rag_chain_core is not None :
    st.session_state.user_question = st.chat_input("Ask a question about your document:")
    if st.session_state.user_question:
      st.session_state.messages.append({"role":"user","content" :st.session_state.user_question ,"sessionID":st.session_state.session_id})
    st.divider()


    st.subheader(" Your uploaded PDF:")
    st.write(f"File name: {st.session_state.user_file.name}")
    with st.container(height=600):
      pdf_viewer(os.path.join ("temp_dir",st.session_state.user_file.name))

    #### Run the pipeline ####
    if st.session_state.user_question is not None:
      with st.spinner(f"working on your question : {st.session_state.user_question} "):
        st.session_state.response = st.session_state.rag_chain_core.invoke(
            {"question": st.session_state.user_question}
            ,
            config={"session_id": st.session_state.session_id}
        )

        st.session_state.messages.append({"role" :"AI Assistant", "content" :st.session_state.response.content, "sessionID":st.session_state.session_id})
      


    st.subheader(" Chat window")
    with st.container(height=300):
      for message in st.session_state.messages:
        # add logic to filter the session id
        if message["sessionID"] == st.session_state.session_id:
          with st.chat_message(message["role"]):
            st.markdown(message["content"])
            st.markdown(f"**SessionID:** {message["sessionID"]}")
      
    st.divider()

if __name__ == "__main__":
  app_runner()

