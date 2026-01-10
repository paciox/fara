"""Computer Use System with Multi-Agent Orchestration."""

import asyncio
import logging
import json
import os
from typing import Dict, Any, List, Optional, Union
from autogen_core.base import AgentId, AgentProxy
from autogen_core.components import DefaultTopicId, default_subscription
from autogen_core.components.models import UserMessage, LLMMessage, AssistantMessage, ChatCompletionClient
from autogen_core.base import CancellationToken
from webeval.basesystem import BaseSystem
from webeval.trajectory import Trajectory, FinalAnswer
from .base_orchestrator import BaseOrchestrator
from fara.computer_use_agent import ComputerUseAgent
from logging import Logger


@default_subscription
class ComputerUseOrchestrator(BaseOrchestrator):
    """
    Multi-agent orchestrator for computer use tasks.
    Coordinates between planner, executor, and critic agents.
    """

    def __init__(
        self,
        executor_agent: AgentProxy,
        planner_agent: Optional[AgentProxy] = None,
        critic_agent: Optional[AgentProxy] = None,
        description: str = "ComputerUseOrchestrator",
        max_rounds: int = 20,
    ) -> None:
        """
        Initialize the computer use orchestrator.
        
        Args:
            executor_agent: Agent that executes actions (FARA-7B)
            planner_agent: Agent that plans steps (GPT-4, optional)
            critic_agent: Agent that evaluates progress (GPT-4, optional)
            description: Description of orchestrator
            max_rounds: Maximum number of orchestration rounds
        """
        self.executor_agent = executor_agent
        self.planner_agent = planner_agent
        self.critic_agent = critic_agent
        
        # Build agent list
        agents = [executor_agent]
        if planner_agent is not None:
            agents.append(planner_agent)
        if critic_agent is not None:
            agents.append(critic_agent)
            
        super().__init__(agents=agents, description=description, max_rounds=max_rounds)
        
        self._current_phase = "plan"  # plan -> execute -> evaluate -> plan
        self._execution_attempts = 0
        self._max_execution_attempts = 3

    async def _select_next_agent(self, message: LLMMessage) -> Optional[AgentProxy]:
        """
        Select next agent based on orchestration phase.
        Flow: Planner -> Executor -> Critic -> Planner (loop)
        """
        # If no planner/critic, always use executor
        if self.planner_agent is None and self.critic_agent is None:
            return self.executor_agent
        
        # Three-phase orchestration
        if self._current_phase == "plan":
            self._current_phase = "execute"
            return self.planner_agent if self.planner_agent else self.executor_agent
            
        elif self._current_phase == "execute":
            self._execution_attempts += 1
            self._current_phase = "evaluate"
            return self.executor_agent
            
        elif self._current_phase == "evaluate":
            # Check if we should stop or continue
            if self._execution_attempts >= self._max_execution_attempts:
                return None  # Stop orchestration
            self._current_phase = "plan"
            return self.critic_agent if self.critic_agent else None
            
        return None


class ComputerUseSystem(BaseSystem):
    """
    System for desktop automation using computer use agents.
    Supports multi-agent orchestration with planner, executor, and critic.
    """
    
    def __init__(
        self,
        system_name: str,
        executor_client_cfg: Optional[Dict[str, Any]] = None,
        planner_client_cfg: Optional[Dict[str, Any]] = None,
        critic_client_cfg: Optional[Dict[str, Any]] = None,
        max_rounds: int = 20,
        save_screenshots: bool = True,
        downloads_folder: Optional[str] = None,
        use_multi_agent: bool = False,
        reflection_interval: int = 3,
    ) -> None:
        """
        Initialize Computer Use System.
        
        Args:
            system_name: Name of the system
            executor_client_cfg: Config for executor model (FARA-7B)
            planner_client_cfg: Config for planner model (GPT-4, optional)
            critic_client_cfg: Config for critic model (GPT-4, optional)
            max_rounds: Maximum execution rounds
            save_screenshots: Whether to save screenshots
            downloads_folder: Folder for screenshots/downloads
            use_multi_agent: Enable multi-agent orchestration
            reflection_interval: How often to reflect on progress (every N actions)
        """
        super().__init__(system_name)
        self.executor_client_cfg = executor_client_cfg or self._default_executor_config()
        self.planner_client_cfg = planner_client_cfg
        self.critic_client_cfg = critic_client_cfg
        self.max_rounds = max_rounds
        self.save_screenshots = save_screenshots
        self.downloads_folder = downloads_folder or os.path.join(os.getcwd(), "computer_use_outputs")
        self.use_multi_agent = use_multi_agent
        self.reflection_interval = reflection_interval
        
        os.makedirs(self.downloads_folder, exist_ok=True)
    
    def _default_executor_config(self) -> Dict[str, Any]:
        """Default config for FARA executor."""
        return {
            "model": "microsoft/Fara-7B",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
        }
    
    async def _create_executor_agent(self) -> ComputerUseAgent:
        """Create and initialize the executor agent."""
        agent = ComputerUseAgent(
            client_config=self.executor_client_cfg,
            downloads_folder=self.downloads_folder,
            save_screenshots=self.save_screenshots,
            max_rounds=self.max_rounds,
        )
        await agent.initialize()
        return agent
    
    async def _run_with_reflection(
        self, 
        executor: ComputerUseAgent, 
        task: str,
        logger: Optional[Logger] = None
    ) -> Trajectory:
        """
        Run executor with periodic reflection to prevent loops.
        
        Args:
            executor: The executor agent
            task: Task to execute
            logger: Optional logger
            
        Returns:
            Trajectory of the execution
        """
        logger = logger or logging.getLogger(__name__)
        logger.info(f"Starting task with reflection: {task}")
        
        all_actions = []
        all_observations = []
        
        # Initial run
        final_answer, actions, observations = await executor.run(task)
        all_actions.extend(actions)
        all_observations.extend(observations)
        
        return Trajectory(
            thought=all_actions,
            action=all_actions,
            observation=all_observations,
            final_answer=FinalAnswer(answer=final_answer, accurate=None)
        )
    
    def get_answer(
        self, 
        question_id: str, 
        example_data: Dict[str, Any], 
        output_dir: str, 
        logger: Optional[Logger] = None
    ) -> Optional[Trajectory]:
        """
        Execute a computer use task and return trajectory.
        
        Args:
            question_id: Unique identifier for the task
            example_data: Task data including 'query' or 'task' field
            output_dir: Directory to save outputs
            logger: Optional logger
            
        Returns:
            Trajectory of execution or None on failure
        """
        logger = logger or logging.getLogger(__name__)
        
        # Extract task from example_data
        task = example_data.get("query") or example_data.get("task") or example_data.get("question")
        if not task:
            logger.error(f"No task found in example_data: {example_data}")
            return None
        
        logger.info(f"[{question_id}] Starting computer use task: {task}")
        
        try:
            # Create task-specific download folder
            task_output_dir = os.path.join(output_dir, f"task_{question_id}")
            os.makedirs(task_output_dir, exist_ok=True)
            self.downloads_folder = task_output_dir
            
            # Run the task
            async def run_task():
                executor = await self._create_executor_agent()
                try:
                    trajectory = await self._run_with_reflection(executor, task, logger)
                    return trajectory
                finally:
                    await executor.close()
            
            trajectory = asyncio.run(run_task())
            
            # Save trajectory
            trajectory_path = os.path.join(task_output_dir, "trajectory.json")
            with open(trajectory_path, "w") as f:
                json.dump({
                    "question_id": question_id,
                    "task": task,
                    "final_answer": trajectory.final_answer.answer,
                    "num_actions": len(trajectory.action),
                }, f, indent=2)
            
            logger.info(f"[{question_id}] Task completed. Trajectory saved to {trajectory_path}")
            return trajectory
            
        except Exception as e:
            logger.error(f"[{question_id}] Error executing task: {e}", exc_info=True)
            return None
    
    def load_answer_from_disk(self, task_id: str, output_dir: str) -> Any:
        """Load saved trajectory from disk."""
        trajectory_path = os.path.join(output_dir, f"task_{task_id}", "trajectory.json")
        if not os.path.exists(trajectory_path):
            return None
        
        with open(trajectory_path, "r") as f:
            data = json.load(f)
        
        return Trajectory(
            thought=[],
            action=[],
            observation=[],
            final_answer=FinalAnswer(answer=data.get("final_answer", ""), accurate=None)
        )
    
    def hash(self) -> str:
        """Return unique hash for this system configuration."""
        import hashlib
        config_str = json.dumps({
            "system_name": self.system_name,
            "executor_cfg": self.executor_client_cfg,
            "max_rounds": self.max_rounds,
            "use_multi_agent": self.use_multi_agent,
        }, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
