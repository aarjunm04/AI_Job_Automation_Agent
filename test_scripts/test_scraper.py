import pytest
from unittest.mock import patch, MagicMock
import agentops
agentops.init(api_key="mock", auto_start_session=False)

from agents.scraper_agent import ScraperAgent

@patch('agents.scraper_agent.check_monthly_budget')
@patch('agents.scraper_agent.ScraperAgent._fallback_scrape_sequence')
@patch('agents.scraper_agent.update_run_batch_stats')
def test_scraper_run_budget_exceeded_fallback(mock_update_stats, mock_fallback, mock_check_budget):
    # Setup mock budget
    mock_budget_func = MagicMock()
    mock_budget_func.return_value = '{"abort": true, "reason": "Monthly cap exceeded"}'
    mock_check_budget.func = mock_budget_func

    # Mock fallback to return dummy stats
    mock_fallback.return_value = {"total_jobs": 42, "by_platform": {"jobspy": 42}}
    mock_update_stats.func = MagicMock()

    # Provide missing env setup
    with patch('agents.scraper_agent.LLMInterface'):
        agent = ScraperAgent(run_batch_id="batch-123")
        
    result = agent.run() if not hasattr(agent.run, 'func') else getattr(agent.run, 'func')(agent)
    
    # Assert fallback was triggered
    mock_fallback.assert_called_once()
    assert result.get("success") is True
    assert result.get("total_jobs") == 42


@patch('agents.scraper_agent.check_monthly_budget')
@patch('agents.scraper_agent.ScraperAgent._fallback_scrape_sequence')
@patch('agents.scraper_agent.update_run_batch_stats')
@patch('agents.scraper_agent.Crew')
def test_scraper_run_llm_failure_fallback(mock_crew_class, mock_update_stats, mock_fallback, mock_check_budget):
    mock_budget_func = MagicMock()
    mock_budget_func.return_value = '{"abort": false}'
    mock_check_budget.func = mock_budget_func
    mock_update_stats.func = MagicMock()

    mock_crew_instance = MagicMock()
    mock_crew_instance.kickoff.side_effect = Exception("LLM Quota Exceeded")
    mock_crew_class.return_value = mock_crew_instance

    mock_fallback.return_value = {"total_jobs": 20, "by_platform": {"remoteok": 20}}

    with patch('agents.scraper_agent.LLMInterface'):
        agent = ScraperAgent(run_batch_id="batch-123")
        
    result = agent.run() if not hasattr(agent.run, 'func') else getattr(agent.run, 'func')(agent)
    
    mock_fallback.assert_called_once()
    assert result.get("success") is True
    assert result.get("total_jobs") == 20
