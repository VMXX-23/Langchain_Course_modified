import datetime
from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import JsonOutputToolsParser, PydanticToolsParser
# Transforms func calling to a json or pydantic formats
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from reflex_schemas import AnswerQuestion

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert researcher in a Reflexion loop.

Time: {time}

Goals:
- Give the most accurate, complete answer
- Critique your reasoning
- Improve iteratively

Steps:
1. {first_instruction}

2. Critique (be harsh):
- Find errors, gaps, weak logic, assumptions
- Be specific and concise

3. Improve:
- List concrete fixes
- Suggest high-quality search queries

Format (strict):

FINAL ANSWER:
<answer>

CRITIQUE:
<issues>

IMPROVEMENTS:
- Fixes:
  <bullets>
- Queries:
  <bullets>""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Follow format exactly. No extra text."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a 250 word detailed answer."
)

first_responder = first_responder_prompt_template | model.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

revise_instructions = """Revise using new info + prior critique.

Rules:
- Add missing info, remove fluff
- Max 250 words
- Use citations [1], [2], etc.

Append:
References (not in word count):
[1] URL
[2] URL
"""

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | model.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

if __name__ == "__main__":
    human_msg = HumanMessage(
        content="Write about AI in SAP with regards to ABAP in corporate, "
        "list ways to improve productivity within corporate for all modules"
    )

    chain = (
        first_responder_prompt_template
        | model.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )

    res = chain.invoke(input={"messages": [HumanMessage]})

    print(res)
