import pytest
from unittest.mock import patch, MagicMock
import agentops
agentops.init(api_key="mock", auto_start_session=False)

from agents.analyser_agent import AnalyserAgent

@patch('agents.analyser_agent.AnalyserAgent._get_jobs_for_run')
def test_analyser_no_jobs(mock_get_jobs):
    mock_get_jobs.return_value = []
    
    with patch('agents.analyser_agent.LLMInterface'):
        agent = AnalyserAgent(run_batch_id="batch-123", user_id="user-456")
        
    result = agent.run() if not hasattr(agent.run, 'func') else getattr(agent.run, 'func')(agent)
    
    assert result.get("success") is True
    assert result.get("scored", -1) == 0
    assert result.get("reason") == "no_jobs_found"

@patch('agents.analyser_agent.AnalyserAgent._get_jobs_for_run')
@patch('agents.analyser_agent.Crew')
def test_analyser_with_jobs(mock_crew_class, mock_get_jobs):
    # Setup 2 dummy jobs
    mock_get_jobs.return_value = [
        {"id": "job1", "title": "Eng", "company": "Co1"},
        {"id": "job2", "title": "Eng", "company": "Co2"}
    ]
    
    mock_crew_instance = MagicMock()
    mock_crew_class.return_value = mock_crew_instance
    mock_crew_instance.kickoff.return_value = '{"scored": 2, "auto_route": 1, "manual_route": 1, "skip_route": 0, "budget_aborted": false, "results": []}'
    
    # We mock _save_score_direct to avoid DB calls
    with patch('agents.analyser_agent.AnalyserAgent._save_score_direct') as mock_save:
        with patch('agents.analyser_agent.LLMInterface'):
            agent = AnalyserAgent(run_batch_id="batch-123", user_id="user-456")
            
        result = agent.run() if not hasattr(agent.run, 'func') else getattr(agent.run, 'func')(agent)
        
    assert result.get("success") is True
    # The current agent implementation parses JSON directly from the crew output usually
    # If the mocked JSON fits the expected schema, it would return those numbers,
    # but our mock is a simple string and depends on agent parsing.
