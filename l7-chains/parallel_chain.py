from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

from dotenv import load_dotenv
load_dotenv()

information = """
Linear regression is a statistical method and machine learning algorithm that models the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a straight line, often using the "least squares" method to minimize errors. It predicts continuous outcomes based on the linear relationship established between variables.

"""

model1 = ChatOpenAI()

model2 = ChatOpenAI()

prompt1 = PromptTemplate(
    template='Generate notes on the following topic:\n{information}',
    input_variables=['information']
)

prompt2 = PromptTemplate(
    template='Generate 3 quiz questions on the following topic:\n{information}',
    input_variables=['information']
)

prompt3 = PromptTemplate(
    template='Merge the provided Notes and Quiz in a single document \n notes -> {notes} \n quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = parallel_chain | prompt3 | model1 | parser

result = merge_chain.invoke({'information': information})
print(result)

merge_chain.get_graph().print_ascii()
