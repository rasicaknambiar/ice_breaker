from typing import Tuple

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from output_parsers import person_intel_parser, PersonIntel
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent


def ice_break(name: str) -> Tuple[PersonIntel, str]:
    linkedin_profile_url = linkedin_lookup_agent(name="Carl Pei")
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)
    summary_template = """
        Given the Linkedin information {information} about a person from which I want you to create:
        1. a short summary
        2. two interesting facts about them
        3. two creative ice breakers to open a conversation with them
        \n{format_instructions}
        """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={
            "format_instructions": person_intel_parser.get_format_instructions()
        },
    )
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    result1 = chain.run(information=linkedin_data)
    return person_intel_parser.parse(result1), linkedin_data.get("profile_pic_url")


if __name__ == "__main__":
    print("Hello LangChain")
    ice_break(name="Carl Pei")
