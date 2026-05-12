from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable


class Generator:
    def __init__(self, llm):
        print("Generator")
        self.llm = llm
        self.prompt_template = ChatPromptTemplate.from_template("""
        Answer the question using ONLY the context below. 
        Provide citations in the format [Source, Page].

        Context: {context}
        Question: {question}
        """)

    @traceable  # Step 16: Monitoring via LangSmith
    def generate(self, query, docs):
        context = "\n\n".join([f"[{d.metadata['source']}] {d.page_content}" for d in docs])
        chain = self.prompt_template | self.llm
        response = chain.invoke({"context": context, "question": query})
        return response.content