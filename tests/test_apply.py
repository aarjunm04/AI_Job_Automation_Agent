import pytest
from unittest.mock import patch
import agentops
agentops.init(api_key="mock", auto_start_session=False)

from agents.apply_agent import ApplyAgent

def test_apply_no_jobs():
    with patch('agents.apply_agent.LLMInterface'):
        agent = ApplyAgent(run_batch_id="batch-123", user_id="user-456")
        
    result = agent.run(routing_manifest=[]) if not hasattr(agent.run, 'func') else getattr(agent.run, 'func')(agent, routing_manifest=[])
    
    assert result.get("success") is True
    assert result.get("total_processed") == 0
    assert result.get("reason") == "no_auto_apply_jobs"
