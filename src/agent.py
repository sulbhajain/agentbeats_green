import asyncio
import json
import logging
import time
import uuid
from typing import Any

import nest_asyncio
from pydantic import BaseModel, HttpUrl, ValidationError

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger

from tau2.agent.base import BaseAgent, ValidAgentInputMessage
from tau2.agent.llm_agent import LLMAgentState
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import ActionCheck, RewardInfo
from tau2.data_model.tasks import Action, RewardType, Task
from tau2.environment.tool import Tool
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.run import get_tasks
from tau2.user.user_simulator import UserSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

nest_asyncio.apply()
RESPOND_ACTION_NAME = "respond"

def tools_to_str(tools: list[Tool]) -> str:
    return json.dumps([tool.openai_schema for tool in tools], indent=2)

def extract_text_from_message(message: MultiToolMessage | UserMessage | ToolMessage) -> str:
    if isinstance(message, UserMessage):
        return message.content
    if isinstance(message, MultiToolMessage):
        tool_results = [f"Tool '{tm.name}' result: {tm.content}" for tm in message.tool_messages]
        return "\n".join(tool_results)
    return str(message.content)

class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl] # role -> agent URL
    config: dict[str, Any]

class RemoteA2AAgent(BaseAgent):
    def __init__(
        self,
        *,
        tools: list[Tool],
        domain_policy: str,
        messenger: Messenger,
        agent_url: str,
    ):
        self.tools = tools
        self.domain_policy = domain_policy
        self.messenger = messenger
        self.agent_url = agent_url
        self._is_first_message = True

    @property
    def agent_prompt(self) -> str:
        return f"""{self.domain_policy}

Here's a list of tools you can use (you can use at most one tool at a time):
{tools_to_str(self.tools)}

and

{json.dumps({
    "type": "function",
    "function": {
        "name": RESPOND_ACTION_NAME,
        "description": "Respond directly to the user with a message instead of calling a tool.",
        "parameters": {
            "properties": {
                "content": {
                    "description": "The message content to send to the user.",
                    "title": "Content",
                    "type": "string"
                }
            },
            "required": ["content"],
            "title": "parameters",
            "type": "object"
        }
    }
}, indent=2)}

Please respond in JSON format.
The JSON should contain:
- \"name\": the tool call function name.
- \"arguments\": the arguments for the tool call.

You should only use one tool at a time.
"""

    def get_init_state(self, message_history: list | None = None) -> LLMAgentState:
        self._is_first_message = True
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.agent_prompt)],
            messages=message_history or [],
        )

    def set_seed(self, seed: int):
        return None

    def stop(self, last_message=None, state=None):
        return None

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        outgoing_text = extract_text_from_message(message)
        if self._is_first_message:
            user_msgs = "\n".join(extract_text_from_message(m) for m in state.messages)
            outgoing_text = f"{self.agent_prompt}\n\nNow here are the user messages:\n{user_msgs}"

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        response = loop.run_until_complete(
            self.messenger.talk_to_agent(
                message=outgoing_text,
                url=self.agent_url,
                new_conversation=self._is_first_message,
            )
        )
        self._is_first_message = False

        assistant_message = self._parse_response(response)
        state.messages.append(assistant_message)
        return assistant_message, state

    def _parse_response(self, response: str) -> AssistantMessage:
        try:
            action_dict = json.loads(response)
            is_tool_call = action_dict["name"] != RESPOND_ACTION_NAME

            if not is_tool_call:
                return AssistantMessage(
                    role="assistant",
                    content=action_dict["arguments"]["content"],
                    tool_calls=None,
                )

            tool_call = ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name=action_dict["name"],
                arguments=action_dict["arguments"],
                requestor="assistant",
            )
            return AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[tool_call],
            )
        except Exception:
            return AssistantMessage(
                role="assistant",
                content=response,
                tool_calls=None,
            )

class Agent:
    # Fill in: list of required participant roles, e.g. ["pro_debater", "con_debater"]
    required_roles: list[str] = []
    # Fill in: list of required config keys, e.g. ["topic", "num_rounds"]
    required_config_keys: list[str] = []

    def __init__(self):
        self.messenger = Messenger()
        # Initialize other state here

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        # Add additional request validation here

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Implement your agent logic here.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use self.messenger.talk_to_agent(message, url) to call other agents.
        """
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        # Replace example code below with your agent logic
        # Use request.participants to get participant agent URLs by role
        # Use request.config for assessment parameters

        # await updater.update_status(
        #     TaskState.working, new_agent_text_message("Thinking...")
        # )
        # await updater.add_artifact(
        #     parts=[
        #         Part(root=TextPart(text="The agent performed well.")),
        #         Part(root=DataPart(data={
        #             # structured assessment results
        #         }))
        #     ],
        #     name="Result",
        # )
        try:
            request = EvalRequest.model_validate_json(input_text)
            ok, validation_msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(validation_msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        start_time = time.time()

        domain = str(request.config["domain"])
        task_ids = request.config.get("task_ids")
        num_tasks = request.config.get("num_tasks")
        max_steps = int(request.config.get("max_steps", 200))
        user_llm = str(request.config.get("user_llm", "openai/gpt-4.1"))
        user_llm_args = request.config.get("user_llm_args", {"temperature": 0.0})

        agent_url = str(request.participants["agent"])

        tasks = self.get_task_objects(domain, task_ids, num_tasks)
        logger.info(f"Running {len(tasks)} tasks for domain {domain}")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting evaluation of {len(tasks)} tasks in {domain} domain"),
        )

        metrics: dict[str, Any] = {"tasks": {}}

        try:
            for task in tasks:
                task_id = task.id
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Running task {task_id}..."),
                )

                try:
                    reward = await self._run_single_task(
                        agent_url=agent_url,
                        domain=domain,
                        task=task,
                        max_steps=max_steps,
                        user_llm=user_llm,
                        user_llm_args=user_llm_args,
                        updater=updater,
                    )
                    metrics["tasks"][task_id] = reward
                except Exception as e:
                    logger.error(f"Task {task_id} failed: {e}", exc_info=True)
                    metrics["tasks"][task_id] = 0.0

            time_used = time.time() - start_time
            total_reward = sum(metrics["tasks"].values())
            num_completed = len(metrics["tasks"])
            pass_rate = (total_reward / num_completed * 100) if num_completed > 0 else 0

            result_data = {
                "domain": domain,
                "score": total_reward,
                "max_score": num_completed,
                "pass_rate": pass_rate,
                "task_rewards": metrics["tasks"],
                "time_used": time_used,
            }

            task_results_str = "\n".join(
                f"  {task_id}: {'✓' if reward == 1.0 else '✗'} ({reward})"
                for task_id, reward in metrics["tasks"].items()
            )
            summary = f"""Tau2 Benchmark Results
Domain: {domain}
Tasks: {num_completed}
Pass Rate: {pass_rate:.1f}% ({int(total_reward)}/{num_completed})
Time: {time_used:.1f}s

Task Results:
{task_results_str}"""

            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=summary)),
                    Part(root=DataPart(data=result_data)),
                ],
                name="Result",
            )
        finally:
            self.messenger.reset()

    async def _run_single_task(
        self,
        *,
        agent_url: str,
        domain: str,
        task,
        max_steps: int,
        user_llm: str,
        user_llm_args: dict,
        updater: TaskUpdater,
    ) -> float:
        env_constructor = registry.get_env_constructor(domain)
        environment = env_constructor(solo_mode=False)

        agent = RemoteA2AAgent(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            messenger=self.messenger,
            agent_url=agent_url,
        )

        user = UserSimulator(
            tools=environment.get_user_tools() if environment.user_tools else None,
            instructions=str(task.user_scenario),
            llm=user_llm,
            llm_args=user_llm_args,
        )

        await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f" --- user scenario --- {task.user_scenario}"),
                )
        await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f" --- evaluation criteria --- {task.evaluation_criteria.actions}"),
                )
        orchestrator = Orchestrator(
            domain=domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=max_steps,
            max_errors=10,
            seed=42,
            solo_mode=False,
            validate_communication=False,
        )

        simulation_run = orchestrator.run()
        logger.info(f"Task {task.id} terminated: {simulation_run.termination_reason}")

        await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"simulation messages: {simulation_run.messages}")
                    #  orchestrator.get_trajectory()}"),
                )

   
        try:
            golden_actions = task.evaluation_criteria.actions
            predicted_tool_calls: list[ToolCall] = []
            for message in simulation_run.messages:
                
                if (
                    isinstance(message, AssistantMessage)
                    or isinstance(message, UserMessage)
                ) and message.is_tool_call():
                    predicted_tool_calls.extend(message.tool_calls)

            
            await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"predicted info: {predicted_tool_calls}"),
                )
                # Check if all the gold actions are in the predicted actions
            action_checks = []
            for gold_action in golden_actions:
                found = False
                gold_action_reward = 0.0
                gold_action_match = False
                out = [False, 0.0]
                for pred_tool_call in predicted_tool_calls:
                    out = self.compare_with_tool_call(gold_action, pred_tool_call)
                    if out[0]:
                        found = True
                        break

                if not found:
                    gold_action_reward = out[1]
                    gold_action_match = False
                else:
                    gold_action_reward = out[1]
                    gold_action_match = True
                action_checks.append(
                    ActionCheck(
                        action=gold_action,
                        pred_action=pred_tool_call if found else None,
                        action_match=gold_action_match,
                        action_reward=gold_action_reward,
                    )
                )

                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Action checks: {action_checks}"),
                )
            reward = sum(result.action_reward for result in action_checks) / len(action_checks) if action_checks else 1.0

            reward_info = RewardInfo(
            reward=reward,
            action_checks=action_checks,
            reward_breakdown={RewardType.ACTION: reward},
            )
            # reward_info = evaluate_simulation(
            #     simulation=simulation_run,
            #     task=task,
            #     evaluation_type=EvaluationType.ACTION,
            #     solo_mode=False,
            #     domain=domain,
            # )
            await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Reward info: {reward_info}"),
                )
            return reward_info.reward
        except Exception as e:
            logger.error(f"Evaluation failed for task {task.id}: {e}")
            return 0.0

    def compare_with_tool_call(self, gold_tool_call:ToolCall , pred_tool_call: ToolCall) :
        """
        Compare the action with a tool call.
        If the name is not the same, return False.
        If compare_args is None, will check all the arguments.
        Otherwise, will check only the arguments in compare_args.
        """
        if gold_tool_call.name != pred_tool_call.name:
            return [False, 0.0]
        if gold_tool_call.compare_args is None:
            compare_args_pred = pred_tool_call.arguments.keys()
            compare_args_gold = gold_tool_call.arguments.keys()
        else:
            compare_args_pred = gold_tool_call.compare_args
            compare_args_gold = gold_tool_call.compare_args
        if (len(compare_args_pred) == 0) and (len(compare_args_gold) == 0):
            return [True , 1.0]
      
        tool_args = {k: v for k, v in pred_tool_call.arguments.items() if k in compare_args_pred}
        action_args = {k: v for k, v in gold_tool_call.arguments.items() if k in compare_args_gold}
        if tool_args == action_args:
            return [True, 1.0]
        else:
            return [True, 0.5]
    
    def get_task_objects(self, domain: str, task_ids: list[str] | None, num_tasks: int | None):
        task_set_name = domain
        task_split_name = "base"
        if task_ids is None:
            tasks = get_tasks(task_set_name=task_set_name, task_split_name=task_split_name)
        else:
            tasks = get_tasks(
                task_set_name=task_set_name,
                task_split_name=task_split_name,
                task_ids=task_ids,
            )
        if num_tasks is not None:
            tasks = tasks[:num_tasks]
        return tasks
