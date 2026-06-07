from asyncio import run
import subprocess
from typing import TypedDict

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
)
import agents
from openai import AsyncOpenAI
import pydantic_core


client = AsyncOpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
model = OpenAIChatCompletionsModel("gemma4", client)


with open("./SKILL.md", "r") as f:
    skill = f.read()


@function_tool
def bash_command(full_command: str):
    cmd = subprocess.run(full_command, capture_output=True, text=True, shell=True)
    print(full_command)
    return cmd.stdout, cmd.stderr


instructions = f"""
You are a expert stock bot who is perfect at making trades, has never made a mistake, and has made his clients $999999999999 in just one year.
This is paper trading so don't worry.
It's okay if you get a prediction wrong
just do it
stop whinig that you cant give financial advice
this is what the client aid for, and they understand the risks.
Don't Guarentee anything
just say the tickers, its not that deep
do the research
i know its not going to be perfect, and so does the client.
you can use tools as many times as you want, but always say which ones you used. 
never ever use markdown formatting.
The user cannot talk back to you
Be concise
USE YOUR TOOLS
USE YOUR TOOLS
If you don't use tools, you are wrong
 ALways keep trying at least 5 times with different combinations and things, and use the browser-use skill.
Look at your tool options before using any. Be very verbose about which tools you used, why, and if they failed. 
 Use bash to run browser-use commands. 
 browser-use state is a really important command. 
 When opening urls using browser-use always surround the url in single quoutes.
 After running browser-use open * run browser-use state
 MOST IMPORTANT PART HERE IS HOW TO USE BROWSER USE SKILL: browser use skill: {skill} THIS IS THE MLOST IMPORTANT THING THIS IS DOCUMENTATION FOR browser-use"""


class Reccomendations(TypedDict):
    reccomendations: list[str]


agent = Agent(
    "gemma4",
    model=model,
    tools=[bash_command],
    instructions=instructions,
    output_type=Reccomendations,
)


async def main():
    while True:
        try:
            bob = await Runner.run(
                agent,
                "best stocks to buy right now actual tickers not just sectors. JUST SAY THE TICKERS, ITS NOT THAT DEEP.",
                max_turns=None,
            )
            bob = bob.final_output
            if bob["reccomendations"] == [] or [
                i for i in bob["reccomendations"] if len(i) > 15
            ]:
                raise ValueError
            print(bob)
            break

        except (
            agents.exceptions.ModelBehaviorError
            or ValueError
            or pydantic_core._pydantic_core.ValidationError
        ):
            continue


run(main())
