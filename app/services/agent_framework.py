"""
Simple Agent Framework for City Intelligence
Lightweight, robust, and focused on performance
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"

@dataclass
class AgentTask:
    """Base task for agents"""
    task_id: str
    agent_id: str
    data: Dict[str, Any]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class AgentResult:
    """Base result from agents"""
    task_id: str
    agent_id: str
    status: AgentStatus
    data: Dict[str, Any]
    insights: List[str] = None
    errors: List[str] = None
    execution_time: float = 0.0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class AgentMemory:
    """Simple memory system for agents to learn"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.successful_patterns = []
        self.failed_patterns = []
        self.insights = []
    
    def record_success(self, pattern: Dict[str, Any], result: AgentResult):
        """Record successful pattern"""
        self.successful_patterns.append({
            "pattern": pattern,
            "result": result.data,
            "timestamp": datetime.utcnow()
        })
        logger.info(f"Agent {self.agent_id} recorded successful pattern")
    
    def record_failure(self, pattern: Dict[str, Any], error: str):
        """Record failed pattern"""
        self.failed_patterns.append({
            "pattern": pattern,
            "error": error,
            "timestamp": datetime.utcnow()
        })
        logger.warning(f"Agent {self.agent_id} recorded failed pattern: {error}")
    
    def get_successful_patterns(self) -> List[Dict[str, Any]]:
        """Get recent successful patterns"""
        return self.successful_patterns[-10:]  # Last 10 successful patterns
    
    def get_insights(self) -> List[str]:
        """Get learned insights"""
        return self.insights

class BaseAgent(ABC):
    """Base agent class - simple and robust"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.config = config or {}
        self.memory = AgentMemory(agent_id)
        self.status = AgentStatus.IDLE
        self.last_run = None
        
        logger.info(f"Initialized agent: {agent_id}")
    
    @abstractmethod
    async def execute(self, task) -> AgentResult:
        """Execute agent task - must be implemented by subclasses"""
        pass
    
    async def run(self, task) -> AgentResult:
        """Main execution method with error handling"""
        start_time = datetime.utcnow()
        self.status = AgentStatus.RUNNING
        self.last_run = start_time
        
        try:
            logger.info(f"Agent {self.agent_id} starting task {task.task_id}")
            result = await self.execute(task)
            
            # Record success
            if result.status == AgentStatus.SUCCESS:
                self.memory.record_success(task.data, result)
            
            self.status = AgentStatus.SUCCESS
            logger.info(f"Agent {self.agent_id} completed task {task.task_id}")
            
        except Exception as e:
            error_msg = f"Agent {self.agent_id} failed: {str(e)}"
            logger.error(error_msg)
            
            # Record failure
            self.memory.record_failure(task.data, str(e))
            
            result = AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                errors=[str(e)]
            )
            self.status = AgentStatus.FAILED
        
        # Calculate execution time
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        result.execution_time = execution_time
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "last_run": self.last_run,
            "successful_patterns": len(self.memory.successful_patterns),
            "failed_patterns": len(self.memory.failed_patterns)
        }

class AgentOrchestrator:
    """Simple orchestrator for running multiple agents"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.task_queue = asyncio.Queue()
        self.results: List[AgentResult] = []
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id}")
    
    async def run_agent(self, agent_id: str, task) -> AgentResult:
        """Run a single agent"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not registered")
        
        agent = self.agents[agent_id]
        result = await agent.run(task)
        self.results.append(result)
        return result
    
    async def run_pipeline(self, tasks: List) -> List[AgentResult]:
        """Run multiple agents in sequence"""
        results = []
        
        for task in tasks:
            if task.agent_id in self.agents:
                result = await self.run_agent(task.agent_id, task)
                results.append(result)
            else:
                logger.error(f"Agent {task.agent_id} not found for task {task.task_id}")
        
        return results
    
    async def run_parallel(self, tasks: List) -> List[AgentResult]:
        """Run multiple agents in parallel"""
        if not tasks:
            return []
        
        # Group tasks by agent
        agent_tasks = {}
        for task in tasks:
            if task.agent_id not in agent_tasks:
                agent_tasks[task.agent_id] = []
            agent_tasks[task.agent_id].append(task)
        
        # Run agents in parallel
        async def run_agent_tasks(agent_id: str, tasks: List):
            agent = self.agents[agent_id]
            results = []
            for task in tasks:
                result = await agent.run(task)
                results.append(result)
            return results
        
        # Execute all agent tasks in parallel
        coroutines = [
            run_agent_tasks(agent_id, agent_tasks[agent_id])
            for agent_id in agent_tasks.keys()
        ]
        
        all_results = await asyncio.gather(*coroutines)
        
        # Flatten results
        results = []
        for agent_results in all_results:
            results.extend(agent_results)
            self.results.extend(agent_results)
        
        return results
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            agent_id: agent.get_status()
            for agent_id, agent in self.agents.items()
        }
    
    def get_recent_results(self, limit: int = 10) -> List[AgentResult]:
        """Get recent results"""
        return self.results[-limit:]

# Global orchestrator instance
orchestrator = AgentOrchestrator()
