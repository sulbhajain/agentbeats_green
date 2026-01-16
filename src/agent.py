import asyncio
import json
import logging
import time
import uuid
from typing import Any

from pydantic import BaseModel, HttpUrl, ValidationError

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger

import asyncio
import json
import logging
import time
import uuid
from typing import Any


from pydantic import BaseModel, HttpUrl, ValidationError

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger


from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    UserMessage,
)

from tau2.data_model.simulation import ActionCheck, RewardInfo
from tau2.data_model.tasks import  RewardType


from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.user.user_simulator import UserSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl] # role -> agent URL
    config: dict[str, Any]


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

        tasks = get_task_objects(domain, task_ids, num_tasks)
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
        return [False, 0.0]
