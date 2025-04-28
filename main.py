import re # Import regular expressions for cleaning
from dotenv import load_dotenv
# Add List from typing
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
# Use JsonOutputParser if LLM struggles with Pydantic instructions for lists
# from langchain_core.output_parsers import JsonOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import user_tool, hospital_tool # Assuming these are defined correctly
import json
import traceback

load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# --- Model Definitions ---
# 1. Keep the model for a single user
class UserDetails(BaseModel):
    username: str = Field(description="The full name of the user")
    patient_id: str = Field(description="The unique patient identifier")
    email: str = Field(description="The user's email address")

# 2. Create a new model for the list structure
class UserList(BaseModel):
    users: List[UserDetails] = Field(description="A list of user details matching the query")

# --- Parser Setup ---
# 3. Update the parser to expect the UserList structure
parser = PydanticOutputParser(pydantic_object=UserList)
# Alternatively, if Pydantic parsing continues to fail with lists, try basic JSON:
# parser = JsonOutputParser() # You would then validate with UserList.model_validate(output) later

# --- Prompt ---
# The prompt can likely stay the same, format_instructions will now be based on UserList
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI assistant.
            - If the query requires fetching user details based on a mobile number, call the 'user' tool with that number. Then, format the tool's output strictly according to the provided format instructions. Ensure the output is ONLY the JSON object/structure described, with no extra text or markdown formatting.
            - For any other queries, provide a helpful, research-based response.
            - If relevant, include the user's name ({name}) in your response.

            Output Format Instructions for user details:
            {{format_instructions}}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "User Name: {name}\nUser Mobile Number: {mobile_number}\nQuery: Fetch user details for the provided mobile number."),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [hospital_tool, user_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

name = input("you name: ")
mobile_num = input("Your Mobile number: ")

try:
    raw_response = agent_executor.invoke({ "name": name, "mobile_number": mobile_num})

    print("\n--- Raw Agent Response ---")
    print(raw_response)
    print("-------------------------\n")

    output_data = raw_response.get("output")

    if not output_data:
        print("Error: Agent did not produce an output.")

    elif isinstance(output_data, str):
        print("Attempting to parse JSON string from agent output...")

        # --- 4. Clean the Output ---
        # Remove markdown code blocks if present
        match = re.search(r"```(json)?(.*)```", output_data, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned_output_data = match.group(2).strip()
            print("Cleaned Markdown:", cleaned_output_data)
        else:
            cleaned_output_data = output_data.strip() # Assume no markdown if regex fails
            print("No Markdown detected, using raw string:", cleaned_output_data)


        try:
            # Parse the cleaned JSON string using the *UserList* parser
            structured_response = parser.parse(cleaned_output_data)
            print("\n--- Successfully parsed UserList ---")

            # --- 5. Adjust Access ---
            if isinstance(structured_response, UserList) and structured_response.users:
                 print(f"Found {len(structured_response.users)} user(s):")
                 for i, user in enumerate(structured_response.users):
                     print(f"  User {i+1}:")
                     print(f"    Username: {user.username}")
                     print(f"    Patient ID: {user.patient_id}")
                     print(f"    Email: {user.email}")
            # Handle case if JsonOutputParser was used
            elif isinstance(structured_response, dict) and "users" in structured_response:
                 print("Parsed as dictionary, validating with UserList model...")
                 validated_response = UserList.model_validate(structured_response)
                 print(f"Found {len(validated_response.users)} user(s):")
                 for i, user in enumerate(validated_response.users):
                     print(f"  User {i+1}:")
                     print(f"    Username: {user.username}")
                     print(f"    Patient ID: {user.patient_id}")
                     print(f"    Email: {user.email}")
            else:
                 print("Parsed data is not in the expected UserList format or is empty.")
                 print("Parsed data:", structured_response)


        except Exception as parse_error:
            print(f"\nError parsing UserList from agent output string: {parse_error}")
            print("Cleaned output string was:")
            print(cleaned_output_data)
            print("\nOriginal output string was:")
            print(output_data)

    # Handle if output is already a dict (less likely now but good practice)
    elif isinstance(output_data, dict):
         print("Agent output is a dictionary. Attempting to validate with UserList model...")
         try:
             structured_response = UserList.model_validate(output_data)
             print("\n--- Successfully validated UserList from dictionary ---")
             print(f"Found {len(structured_response.users)} user(s):")
             for i, user in enumerate(structured_response.users):
                 print(f"  User {i+1}:")
                 print(f"    Username: {user.username}")
                 print(f"    Patient ID: {user.patient_id}")
                 print(f"    Email: {user.email}")
         except Exception as validation_error:
             print(f"\nError validating UserList from agent output dictionary: {validation_error}")
             print("Raw output dictionary was:")
             print(output_data)
    else:
        print(f"Unexpected output type from agent: {type(output_data)}")
        print("Output:", output_data)


except Exception as e:
    print("\n--- An Error Occurred During Agent Execution ---")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Details: {e}")
    print("\n--- Traceback ---")
    traceback.print_exc()
    print("-----------------\n")
    print("Raw Agent Response (if available):", raw_response if 'raw_response' in locals() else "N/A")