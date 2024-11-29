import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from github import fetch_github_issues
from note import note_tool

load_dotenv()

def connect_to_vstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
    ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    desired_namespace = os.getenv("ASTRA_DB_KEYSPACE")
    
    ASTRA_DB_KEYSPACE = desired_namespace if desired_namespace else None
    
    vstore = AstraDBVectorStore(
        embedding=embeddings,
        collection_name="github",
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=ASTRA_DB_KEYSPACE,
    )
    return vstore

vstore = connect_to_vstore()

add_to_vectorstore = input("Do you want to update the issues? (y/N): ").lower() in ["yes", "y"]
if add_to_vectorstore:
    owner = "techwithtim"
    repo = "Flask-Web-App-Tutorial"
    issues = fetch_github_issues(owner, repo)
    try:
        vstore.delete_collection()
    except:
        pass
    vstore = connect_to_vstore()
    vstore.add_documents(issues)

retriever = vstore.as_retriever(search_kwargs={"k": 3})
retriever_tool = create_retriever_tool(
    retriever,
    "github_search",
    "Search for information about github issues. For any questions about github issues, you must use this tool!",
)

prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatGoogleGenerativeAI(model="gemini-pro")
tools = [retriever_tool, note_tool]

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

while (question := input("Ask a question about github issues (q to quit): ")) != "q":
    result = agent_executor.invoke({"input": question})
    print(result["output"])
