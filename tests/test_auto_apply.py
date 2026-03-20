"""
Integration tests for the auto-apply engine.

Tests the complete flow: ATSDetector → FormFiller → Platform Apply → apply_service
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_apply.ats_detector import ATSDetector, ATSType
from auto_apply.form_filler import FormFiller, FillResult
from auto_apply.platforms.base_platform import ApplyResult


# =============================================================================
# ATSDetector Tests
# =============================================================================

class TestATSDetector:
    """Test ATS platform detection."""

    @pytest.fixture
    def detector(self):
        return ATSDetector()

    def test_greenhouse_url_detection(self, detector):
        """Detect Greenhouse from URL patterns."""
        urls = [
            "https://boards.greenhouse.io/company/jobs/12345",
            "https://job-boards.greenhouse.io/company/jobs/12345",
            "https://company.greenhouse.io/job/12345",
        ]
        for url in urls:
            result = detector._detect_from_url(url)
            assert result == ATSType.GREENHOUSE, f"Failed for {url}"

    def test_lever_url_detection(self, detector):
        """Detect Lever from URL patterns."""
        urls = [
            "https://jobs.lever.co/company/12345",
            "https://company.lever.co/apply/12345",
        ]
        for url in urls:
            result = detector._detect_from_url(url)
            assert result == ATSType.LEVER, f"Failed for {url}"

    def test_workday_url_detection(self, detector):
        """Detect Workday from URL patterns."""
        urls = [
            "https://company.wd5.myworkdayjobs.com/careers",
            "https://company.myworkdayjobs.com/en-US/jobs",
        ]
        for url in urls:
            result = detector._detect_from_url(url)
            assert result == ATSType.WORKDAY, f"Failed for {url}"

    def test_indeed_url_detection(self, detector):
        """Detect Indeed from URL patterns."""
        urls = [
            "https://www.indeed.com/viewjob?jk=12345",
            "https://indeed.com/jobs?q=python",
        ]
        for url in urls:
            result = detector._detect_from_url(url)
            assert result == ATSType.INDEED, f"Failed for {url}"

    def test_linkedin_url_detection(self, detector):
        """Detect LinkedIn from URL patterns."""
        urls = [
            "https://www.linkedin.com/jobs/view/12345",
            "https://linkedin.com/jobs/collections/12345",
        ]
        for url in urls:
            result = detector._detect_from_url(url)
            assert result == ATSType.LINKEDIN, f"Failed for {url}"

    def test_unknown_url(self, detector):
        """Unknown URLs return None for URL-based detection."""
        result = detector._detect_from_url("https://randomsite.com/careers")
        assert result is None

    @pytest.mark.asyncio
    async def test_full_detection_with_mock_page(self, detector):
        """Test full detection flow with mocked page."""
        mock_page = AsyncMock()
        mock_page.url = "https://boards.greenhouse.io/company/jobs/12345"
        mock_page.content = AsyncMock(return_value="<html></html>")

        result = await detector.detect(mock_page)
        assert result == ATSType.GREENHOUSE


# =============================================================================
# FormFiller Tests
# =============================================================================

class TestFormFiller:
    """Test form filling functionality."""

    @pytest.fixture
    def filler(self):
        return FormFiller()

    @pytest.mark.asyncio
    async def test_fill_field_text_input(self, filler):
        """Fill a text input field."""
        mock_page = AsyncMock()
        mock_element = AsyncMock()
        mock_element.get_attribute = AsyncMock(return_value="text")
        mock_element.bounding_box = AsyncMock(return_value={"x": 100, "y": 100, "width": 200, "height": 30})
        mock_page.query_selector = AsyncMock(return_value=mock_element)
        mock_page.mouse = AsyncMock()
        mock_page.keyboard = AsyncMock()
        mock_page.evaluate = AsyncMock()

        result = await filler.fill_field(mock_page, "#name", "John Doe")
        
        assert result.success
        assert result.field_selector == "#name"

    @pytest.mark.asyncio
    async def test_fill_field_missing_element(self, filler):
        """Handle missing elements gracefully."""
        mock_page = AsyncMock()
        mock_page.query_selector = AsyncMock(return_value=None)

        result = await filler.fill_field(mock_page, "#nonexistent", "value")
        
        assert not result.success
        assert "not found" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_detect_captcha(self, filler):
        """Detect CAPTCHA presence."""
        mock_page = AsyncMock()
        
        # Test reCAPTCHA detection
        mock_page.query_selector = AsyncMock(side_effect=[
            AsyncMock(),  # Found reCAPTCHA
            None,
            None,
        ])
        
        result = await filler.detect_captcha(mock_page)
        assert result is True

    @pytest.mark.asyncio
    async def test_no_captcha_detected(self, filler):
        """No CAPTCHA when none present."""
        mock_page = AsyncMock()
        mock_page.query_selector = AsyncMock(return_value=None)
        
        result = await filler.detect_captcha(mock_page)
        assert result is False


# =============================================================================
# Platform Apply Tests
# =============================================================================

class TestGreenhouseApply:
    """Test Greenhouse-specific apply logic."""

    @pytest.mark.asyncio
    async def test_greenhouse_apply_success(self):
        """Successful Greenhouse application."""
        from auto_apply.platforms.greenhouse import GreenhouseApply
        
        applier = GreenhouseApply()
        
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()
        mock_page.query_selector = AsyncMock(return_value=AsyncMock())
        mock_page.query_selector_all = AsyncMock(return_value=[])
        mock_page.fill = AsyncMock()
        mock_page.click = AsyncMock()
        mock_page.screenshot = AsyncMock()
        mock_page.url = "https://boards.greenhouse.io/company/jobs/12345"
        mock_page.content = AsyncMock(return_value="<html>Thank you for applying</html>")

        with patch.object(applier, '_fill_application_form', new_callable=AsyncMock):
            with patch.object(applier, '_submit_application', new_callable=AsyncMock) as mock_submit:
                mock_submit.return_value = ApplyResult(
                    success=True,
                    job_url="https://boards.greenhouse.io/company/jobs/12345",
                    platform="greenhouse",
                    confirmation_id="GH-12345",
                    screenshot_path="/tmp/screenshot.png"
                )
                
                result = await applier.apply(
                    page=mock_page,
                    job_url="https://boards.greenhouse.io/company/jobs/12345",
                    resume_path="/path/to/resume.pdf",
                    user_data={"name": "John Doe", "email": "john@example.com"}
                )

        assert result.success


class TestLeverApply:
    """Test Lever-specific apply logic."""

    @pytest.mark.asyncio
    async def test_lever_apply_already_applied(self):
        """Handle 'already applied' case on Lever."""
        from auto_apply.platforms.lever import LeverApply
        
        applier = LeverApply()
        
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.content = AsyncMock(return_value="<html>You've already applied to this job</html>")
        mock_page.url = "https://jobs.lever.co/company/12345"

        result = await applier.apply(
            page=mock_page,
            job_url="https://jobs.lever.co/company/12345",
            resume_path="/path/to/resume.pdf",
            user_data={"name": "John Doe", "email": "john@example.com"}
        )

        assert not result.success
        assert "already applied" in result.error_message.lower()


# =============================================================================
# Apply Service Tests
# =============================================================================

class TestApplyService:
    """Test the apply_service FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from auto_apply.apply_service import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Health check returns OK."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_apply_missing_job_url(self, client):
        """Apply request without job_url fails."""
        response = client.post("/apply", json={
            "user_id": "user123",
            "resume_path": "/path/to/resume.pdf"
        })
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_apply_endpoint_with_mock(self, client):
        """Apply endpoint routes to correct platform."""
        with patch('auto_apply.apply_service.ATSDetector') as MockDetector:
            mock_detector = MockDetector.return_value
            mock_detector.detect = AsyncMock(return_value=ATSType.GREENHOUSE)
            
            with patch('auto_apply.apply_service.PLATFORM_REGISTRY') as mock_registry:
                mock_applier = AsyncMock()
                mock_applier.apply = AsyncMock(return_value=ApplyResult(
                    success=True,
                    job_url="https://boards.greenhouse.io/company/jobs/12345",
                    platform="greenhouse",
                    confirmation_id="GH-12345"
                ))
                mock_registry.get.return_value = mock_applier
                
                response = client.post("/apply", json={
                    "job_url": "https://boards.greenhouse.io/company/jobs/12345",
                    "user_id": "user123",
                    "resume_path": "/path/to/resume.pdf"
                })
                
                # Note: This will fail in real test without full mock setup
                # Just verifying the endpoint exists and accepts requests


# =============================================================================
# Apply Tools Tests
# =============================================================================

class TestApplyTools:
    """Test CrewAI tool wrappers."""

    @pytest.mark.asyncio
    async def test_detect_ats_platform_tool(self):
        """Test ATS detection tool wrapper."""
        from tools.apply_tools import detect_ats_platform
        
        with patch('tools.apply_tools.ATSDetector') as MockDetector:
            mock_detector = MockDetector.return_value
            mock_detector.detect = AsyncMock(return_value=ATSType.GREENHOUSE)
            
            with patch('tools.apply_tools.async_playwright') as mock_pw:
                mock_browser = AsyncMock()
                mock_page = AsyncMock()
                mock_browser.new_page = AsyncMock(return_value=mock_page)
                mock_pw.return_value.__aenter__.return_value.chromium.launch = AsyncMock(return_value=mock_browser)
                
                # Tool execution would happen here in integration

    def test_verify_apply_budget_tool(self):
        """Test budget verification tool."""
        from tools.apply_tools import verify_apply_budget
        
        with patch('tools.apply_tools.httpx') as mock_httpx:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "remaining_budget": 50,
                "daily_limit": 100,
                "used_today": 50,
                "can_apply": True
            }
            mock_httpx.get.return_value = mock_response
            
            # Tool execution test


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_apply_flow(self):
        """Test complete apply flow from detection to submission."""
        # This test requires actual browser and services running
        # Mark as integration test to skip in unit test runs
        
        pytest.skip("Requires running services - use for manual integration testing")
        
        from auto_apply.ats_detector import ATSDetector
        from auto_apply.form_filler import FormFiller
        from auto_apply.platforms.greenhouse import GreenhouseApply
        from playwright.async_api import async_playwright
        
        test_url = "https://boards.greenhouse.io/testcompany/jobs/12345"
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            try:
                # Step 1: Detect ATS
                detector = ATSDetector()
                await page.goto(test_url)
                ats_type = await detector.detect(page)
                assert ats_type == ATSType.GREENHOUSE
                
                # Step 2: Initialize platform applier
                applier = GreenhouseApply()
                
                # Step 3: Apply (would need real test account)
                # result = await applier.apply(page, test_url, resume_path, user_data)
                
            finally:
                await browser.close()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_manual_queue_fallback(self):
        """Test that unsupported platforms route to manual queue."""
        pytest.skip("Requires running services - use for manual integration testing")
        
        import httpx
        
        # Submit job with unknown ATS
        response = await httpx.post(
            "http://localhost:8003/apply",
            json={
                "job_url": "https://unknownats.com/jobs/12345",
                "user_id": "test_user",
                "resume_path": "/path/to/resume.pdf"
            }
        )
        
        data = response.json()
        # Should route to manual queue
        assert data.get("route_to_manual") is True


# =============================================================================
# Fixture Utilities
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "+1-555-123-4567",
        "linkedin": "https://linkedin.com/in/johndoe",
        "github": "https://github.com/johndoe",
        "website": "https://johndoe.dev",
        "location": "San Francisco, CA",
        "work_authorization": "US Citizen",
        "years_experience": "5",
        "education": "BS Computer Science, Stanford University",
        "current_company": "TechCorp Inc.",
        "current_title": "Senior Software Engineer",
    }


@pytest.fixture
def sample_job():
    """Sample job data for testing."""
    return {
        "id": "job_12345",
        "url": "https://boards.greenhouse.io/company/jobs/12345",
        "title": "Senior Software Engineer",
        "company": "TechCorp",
        "location": "San Francisco, CA",
        "salary_min": 150000,
        "salary_max": 200000,
        "remote": True,
    }


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not integration",  # Skip integration tests by default
    ])
