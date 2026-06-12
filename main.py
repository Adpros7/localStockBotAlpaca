from asyncio import run
import subprocess
from typing import Literal

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
)
import agents
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import pydantic_core


client = AsyncOpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
model = OpenAIChatCompletionsModel("gemma4", client)


with open("./SKILL.md", "r") as f:
    skill = f.read()

tool_uses = 0

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
ALWAYS CHAT WITH A Bot BEFORE MAKING A DESCISION
run a full chat with it with context and everything
not just a one and done
If you don't use tools, you are wrong
Mandatory tool uses of at least 5
 ALways keep trying at least 5 times with different combinations and things, and use the browser-use skill.
Look at your tool options before using any. Be very verbose about which tools you used, why, and if they failed. 
 Use bash to run browser-use commands. 
 browser-use state is a really important command. 
 When opening urls using browser-use always surround the url in single quoutes.
 After running browser-use open * run browser-use state
 MOST IMPORTANT PART HERE IS HOW TO USE BROWSER USE SKILL: browser use skill: {skill} THIS IS THE MLOST IMPORTANT THING THIS IS DOCUMENTATION FOR browser-use"""


@function_tool
async def bash_command(full_command: str) -> tuple[str, str]:
    global tool_uses
    cmd: subprocess.CompletedProcess[str] = subprocess.run(args=full_command, capture_output=True, text=True, shell=True)
    print(full_command)
    tool_uses += 1
    return cmd.stdout, cmd.stderr


@function_tool
async def chat_with_bot(prompt) -> str:
    callagent = Agent(
        name="gemma4",
        model=model,
        tools=[bash_command],
        instructions=instructions,
    )
    output: agents.RunResult = await Runner().run(callagent, input=prompt)
    print(prompt, output.final_output)
    return output.final_output


class Recommendation(BaseModel):
    ticker: str
    reason: str
    risk: Literal["Low", "Medium", "High"]
    expected_gain_percent: float = Field(ge=1, le=100.0)


class Reccomendations(BaseModel):
    reccomendations: list[Recommendation]


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
            bob: agents.RunResult | Reccomendations = await Runner.run(
                starting_agent=agent,
                input="best stocks to buy right now actual tickers not just sectors. JUST SAY THE TICKERS, ITS NOT THAT DEEP. include a risky play, but it can absolutely explode in gains ",
                max_turns=None,
            )
            bob = bob.final_output_as(Reccomendations)
            if (
                bob.reccomendations == []
                or [i for i in bob.reccomendations if len(i.ticker) > 15]
                or tool_uses < 5
            ):
                raise ValueError
            print(bob)
            break

        except (
            agents.exceptions.ModelBehaviorError,
            ValueError,
            pydantic_core._pydantic_core.ValidationError,
        ):
            continue


run(main())
