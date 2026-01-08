"""
=============================================================================
AI JOB AUTOMATION AGENT - CORE MODULE INITIALIZATION
=============================================================================
Core module initialization for the AI Job Automation system.
Provides centralized imports and initialization for all core components.

This module exposes:
- AI Engine (OpenAI + Perplexity integration)
- Notion Engine (Database operations)  
- Scraper Engine (Multi-platform job scraping)
- Resume Engine (Dynamic resume generation)
- Automation Engine (Playwright browser automation)
- MCP Client (Model Context Protocol)

Author: AI Job Automation Team
Version: 1.0.0
Last Updated: October 2025
=============================================================================
"""

import logging
from typing import Optional

# Setup core module logging
logger = logging.getLogger(__name__)

# =============================================================================
# VERSION AND METADATA
# =============================================================================

__version__ = "1.0.0"
__author__ = "AI Job Automation Team"
__email__ = "team@ai-job-automation.com"
__description__ = "Core engines for AI-powered job automation"

# =============================================================================
# CORE MODULE IMPORTS
# =============================================================================

# Import core engines (lazy loading to avoid circular imports)
_ai_engine = None
_notion_engine = None
_scraper_engine = None
_resume_engine = None
_automation_engine = None

def get_ai_engine():
    """Get AI Engine instance (lazy loading)"""
    global _ai_engine
    if _ai_engine is None:
        from .ai_engine import AIEngine
        _ai_engine = AIEngine()
        logger.info("AI Engine initialized")
    return _ai_engine

def get_notion_engine():
    """Get Notion Engine instance (lazy loading)"""
    global _notion_engine
    if _notion_engine is None:
        from .notion_engine import NotionEngine
        _notion_engine = NotionEngine()
        logger.info("Notion Engine initialized")
    return _notion_engine

def get_scraper_engine():
    """Get Scraper Engine instance (lazy loading)"""
    global _scraper_engine
    if _scraper_engine is None:
        from .scraper_engine import ScraperEngine
        _scraper_engine = ScraperEngine()
        logger.info("Scraper Engine initialized")
    return _scraper_engine

def get_resume_engine():
    """Get Resume Engine instance (lazy loading)"""
    global _resume_engine
    if _resume_engine is None:
        from .resume_engine import ResumeEngine
        _resume_engine = ResumeEngine()
        logger.info("Resume Engine initialized")
    return _resume_engine

def get_automation_engine():
    """Get Automation Engine instance (lazy loading)"""
    global _automation_engine
    if _automation_engine is None:
        from .automation_engine import AutomationEngine
        _automation_engine = AutomationEngine()
        logger.info("Automation Engine initialized")
    return _automation_engine

# =============================================================================
# CORE SYSTEM INITIALIZATION
# =============================================================================

async def initialize_core_system() -> dict:
    """
    Initialize all core engines and return system status
    
    Returns:
        dict: System initialization status
    """
    try:
        logger.info("Initializing AI Job Automation Core System...")
        
        # Initialize MCP Client first (dependency for other engines)
        from mcp_client import get_mcp_client
        mcp_client = get_mcp_client()
        
        # Initialize core engines
        engines = {
            "ai_engine": get_ai_engine(),
            "notion_engine": get_notion_engine(),
            "scraper_engine": get_scraper_engine(),
            "resume_engine": get_resume_engine(),
            "automation_engine": get_automation_engine(),
            "mcp_client": mcp_client
        }
        
        # Verify all engines are working
        system_status = await verify_system_health(engines)
        
        if system_status["overall_health"] == "healthy":
            logger.info("✅ Core System initialized successfully")
        else:
            logger.warning("⚠️ Core System initialized with some issues")
            
        return system_status
        
    except Exception as e:
        logger.error(f"❌ Core System initialization failed: {e}")
        raise

async def verify_system_health(engines: dict) -> dict:
    """
    Verify health of all core engines
    
    Returns:
        dict: Health status of each component
    """
    health_status = {
        "timestamp": "2025-10-04T23:56:00Z",
        "overall_health": "healthy",
        "engines": {},
        "issues": []
    }
    
    try:
        # Check AI Engine
        try:
            await engines["ai_engine"].health_check()
            health_status["engines"]["ai_engine"] = "healthy"
        except Exception as e:
            health_status["engines"]["ai_engine"] = "unhealthy"
            health_status["issues"].append(f"AI Engine: {str(e)}")
        
        # Check Notion Engine
        try:
            await engines["notion_engine"].health_check()
            health_status["engines"]["notion_engine"] = "healthy"
        except Exception as e:
            health_status["engines"]["notion_engine"] = "unhealthy"
            health_status["issues"].append(f"Notion Engine: {str(e)}")
        
        # Check Scraper Engine
        try:
            engines["scraper_engine"].health_check()
            health_status["engines"]["scraper_engine"] = "healthy"
        except Exception as e:
            health_status["engines"]["scraper_engine"] = "unhealthy"
            health_status["issues"].append(f"Scraper Engine: {str(e)}")
        
        # Check Resume Engine
        try:
            await engines["resume_engine"].health_check()
            health_status["engines"]["resume_engine"] = "healthy"
        except Exception as e:
            health_status["engines"]["resume_engine"] = "unhealthy"
            health_status["issues"].append(f"Resume Engine: {str(e)}")
        
        # Check Automation Engine
        try:
            await engines["automation_engine"].health_check()
            health_status["engines"]["automation_engine"] = "healthy"
        except Exception as e:
            health_status["engines"]["automation_engine"] = "unhealthy"
            health_status["issues"].append(f"Automation Engine: {str(e)}")
        
        # Check MCP Client
        try:
            stats = engines["mcp_client"].get_performance_stats()
            health_status["engines"]["mcp_client"] = "healthy"
            health_status["mcp_stats"] = stats
        except Exception as e:
            health_status["engines"]["mcp_client"] = "unhealthy"
            health_status["issues"].append(f"MCP Client: {str(e)}")
        
        # Determine overall health
        unhealthy_engines = [
            engine for engine, status in health_status["engines"].items() 
            if status == "unhealthy"
        ]
        
        if len(unhealthy_engines) == 0:
            health_status["overall_health"] = "healthy"
        elif len(unhealthy_engines) <= 2:
            health_status["overall_health"] = "degraded"
        else:
            health_status["overall_health"] = "unhealthy"
            
    except Exception as e:
        health_status["overall_health"] = "unhealthy"
        health_status["issues"].append(f"Health check failed: {str(e)}")
    
    return health_status

async def shutdown_core_system() -> None:
    """
    Gracefully shutdown all core engines
    """
    try:
        logger.info("Shutting down AI Job Automation Core System...")
        
        # Shutdown in reverse order
        global _automation_engine, _resume_engine, _scraper_engine, _notion_engine, _ai_engine
        
        if _automation_engine:
            await _automation_engine.cleanup()
            _automation_engine = None
            
        if _resume_engine:
            await _resume_engine.cleanup()
            _resume_engine = None
            
        if _scraper_engine:
            await _scraper_engine.cleanup()
            _scraper_engine = None
            
        if _notion_engine:
            await _notion_engine.cleanup()
            _notion_engine = None
            
        if _ai_engine:
            await _ai_engine.cleanup()
            _ai_engine = None
        
        # Shutdown MCP Client
        from mcp_client import close_mcp_client
        await close_mcp_client()
        
        logger.info("✅ Core System shutdown completed")
        
    except Exception as e:
        logger.error(f"❌ Core System shutdown failed: {e}")
        raise

# =============================================================================
# CORE UTILITIES
# =============================================================================

def get_system_info() -> dict:
    """Get core system information"""
    return {
        "module": "ai_job_automation_core",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "engines": [
            "ai_engine",
            "notion_engine", 
            "scraper_engine",
            "resume_engine",
            "automation_engine"
        ],
        "dependencies": [
            "mcp_client",
            "openai",
            "notion_client",
            "playwright",
            "requests",
            "asyncio"
        ]
    }

def get_engine_status() -> dict:
    """Get current status of all engines"""
    return {
        "ai_engine": _ai_engine is not None,
        "notion_engine": _notion_engine is not None,
        "scraper_engine": _scraper_engine is not None,
        "resume_engine": _resume_engine is not None,
        "automation_engine": _automation_engine is not None
    }

# =============================================================================
# ERROR HANDLING
# =============================================================================

class CoreSystemError(Exception):
    """Base exception for core system errors"""
    pass

class EngineInitializationError(CoreSystemError):
    """Raised when an engine fails to initialize"""
    pass

class SystemHealthError(CoreSystemError):
    """Raised when system health check fails"""
    pass

# =============================================================================
# EXPORT ALL
# =============================================================================

__all__ = [
    # Version info
    '__version__',
    '__author__', 
    '__email__',
    '__description__',
    
    # Engine getters
    'get_ai_engine',
    'get_notion_engine',
    'get_scraper_engine', 
    'get_resume_engine',
    'get_automation_engine',
    
    # System management
    'initialize_core_system',
    'verify_system_health',
    'shutdown_core_system',
    
    # Utilities
    'get_system_info',
    'get_engine_status',
    
    # Exceptions
    'CoreSystemError',
    'EngineInitializationError', 
    'SystemHealthError'
]
