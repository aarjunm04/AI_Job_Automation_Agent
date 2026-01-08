"""
=============================================================================
AI JOB AUTOMATION AGENT - CORE CONFIGURATION MANAGEMENT
=============================================================================
Centralized configuration management for all system components.
Handles environment variables, validation, and configuration loading.

Author: AI Job Automation Team
Version: 1.0.0
Last Updated: October 2025
=============================================================================
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv
import yaml
import json
from pydantic import BaseSettings, Field, validator
from enum import Enum

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION ENUMS
# =============================================================================

class LogLevel(str, Enum):
    """Logging levels enum"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AIProvider(str, Enum):
    """AI provider options"""
    OPENAI = "openai"
    PERPLEXITY = "perplexity"

class JobPlatform(str, Enum):
    """Supported job platforms"""
    LINKEDIN = "linkedin"
    INDEED = "indeed"
    GLASSDOOR = "glassdoor"
    MONSTER = "monster"
    DICE = "dice"
    ANGELLIST = "angellist"
    REMOTE_OK = "remote_ok"
    WEWORKREMOTELY = "weworkremotely"
    STACKOVERFLOW = "stackoverflow"
    GITHUB = "github"
    YCOMBINATOR = "ycombinator"
    UPWORK = "upwork"
    TOPTAL = "toptal"
    ZIPRECRUITER = "ziprecruiter"
    SIMPLYHIRED = "simplyhired"

# =============================================================================
# AI & MCP CONFIGURATION
# =============================================================================

@dataclass
class AIConfig:
    """AI providers configuration"""
    # OpenAI Configuration
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"))
    openai_embedding_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    openai_temperature: float = field(default_factory=lambda: float(os.getenv("OPENAI_TEMPERATURE", "0.7")))
    openai_max_tokens: int = field(default_factory=lambda: int(os.getenv("OPENAI_MAX_TOKENS", "4000")))
    
    # Perplexity Configuration
    perplexity_api_key: str = field(default_factory=lambda: os.getenv("PERPLEXITY_API_KEY", ""))
    perplexity_model: str = field(default_factory=lambda: os.getenv("PERPLEXITY_MODEL", "llama-3.1-sonar-huge-128k-online"))
    perplexity_temperature: float = field(default_factory=lambda: float(os.getenv("PERPLEXITY_TEMPERATURE", "0.5")))
    
    # MCP Configuration
    mcp_server_url: str = field(default_factory=lambda: os.getenv("MCP_SERVER_URL", "http://localhost:3001"))
    mcp_client_timeout: int = field(default_factory=lambda: int(os.getenv("MCP_CLIENT_TIMEOUT", "30")))
    mcp_max_retries: int = field(default_factory=lambda: int(os.getenv("MCP_MAX_RETRIES", "3")))
    mcp_context_window_size: int = field(default_factory=lambda: int(os.getenv("MCP_CONTEXT_WINDOW_SIZE", "32000")))

@dataclass
class NotionConfig:
    """Notion integration configuration"""
    api_key: str = field(default_factory=lambda: os.getenv("NOTION_API_KEY", ""))
    applications_db_id: str = field(default_factory=lambda: os.getenv("NOTION_APPLICATIONS_DB_ID", ""))
    job_tracker_db_id: str = field(default_factory=lambda: os.getenv("NOTION_JOB_TRACKER_DB_ID", ""))
    version: str = field(default_factory=lambda: os.getenv("NOTION_VERSION", "2022-06-28"))
    timeout: int = field(default_factory=lambda: int(os.getenv("NOTION_TIMEOUT", "30")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("NOTION_MAX_RETRIES", "3")))

@dataclass
class OverleafConfig:
    """Overleaf integration configuration"""
    api_key: str = field(default_factory=lambda: os.getenv("OVERLEAF_API_KEY", ""))
    project_id: str = field(default_factory=lambda: os.getenv("OVERLEAF_PROJECT_ID", ""))
    compile_url: str = field(default_factory=lambda: os.getenv("OVERLEAF_COMPILE_URL", "https://api.overleaf.com/docs/compile"))
    timeout: int = field(default_factory=lambda: int(os.getenv("OVERLEAF_TIMEOUT", "60")))

@dataclass
class GmailConfig:
    """Gmail integration configuration"""
    client_id: str = field(default_factory=lambda: os.getenv("GMAIL_CLIENT_ID", ""))
    client_secret: str = field(default_factory=lambda: os.getenv("GMAIL_CLIENT_SECRET", ""))
    redirect_uri: str = field(default_factory=lambda: os.getenv("GMAIL_REDIRECT_URI", "http://localhost:8080/auth/callback"))
    scopes: List[str] = field(default_factory=lambda: os.getenv("GMAIL_SCOPES", "https://www.googleapis.com/auth/gmail.readonly").split(","))
    check_interval: int = field(default_factory=lambda: int(os.getenv("GMAIL_CHECK_INTERVAL", "300")))  # 5 minutes

# =============================================================================
# SCRAPING & AUTOMATION CONFIGURATION
# =============================================================================

@dataclass
class ScrapingConfig:
    """Web scraping configuration"""
    delay_min: int = field(default_factory=lambda: int(os.getenv("SCRAPING_DELAY_MIN", "2")))
    delay_max: int = field(default_factory=lambda: int(os.getenv("SCRAPING_DELAY_MAX", "5")))
    max_concurrent_scrapers: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_SCRAPERS", "5")))
    user_agent: str = field(default_factory=lambda: os.getenv("USER_AGENT", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"))
    timeout: int = field(default_factory=lambda: int(os.getenv("SCRAPING_TIMEOUT", "30")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("SCRAPING_MAX_RETRIES", "3")))
    
    # Platform credentials
    linkedin_email: str = field(default_factory=lambda: os.getenv("LINKEDIN_EMAIL", ""))
    linkedin_password: str = field(default_factory=lambda: os.getenv("LINKEDIN_PASSWORD", ""))
    indeed_api_key: str = field(default_factory=lambda: os.getenv("INDEED_API_KEY", ""))

@dataclass
class PlaywrightConfig:
    """Playwright automation configuration"""
    service_url: str = field(default_factory=lambda: os.getenv("PLAYWRIGHT_SERVICE_URL", "http://localhost:3000"))
    headless: bool = field(default_factory=lambda: os.getenv("PLAYWRIGHT_HEADLESS", "true").lower() == "true")
    timeout: int = field(default_factory=lambda: int(os.getenv("PLAYWRIGHT_TIMEOUT", "30000")))
    viewport_width: int = field(default_factory=lambda: int(os.getenv("PLAYWRIGHT_VIEWPORT_WIDTH", "1920")))
    viewport_height: int = field(default_factory=lambda: int(os.getenv("PLAYWRIGHT_VIEWPORT_HEIGHT", "1080")))
    screenshot_on_error: bool = field(default_factory=lambda: os.getenv("SAVE_SCREENSHOTS", "false").lower() == "true")

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

@dataclass
class SystemConfig:
    """System-wide configuration"""
    timezone: str = field(default_factory=lambda: os.getenv("TIMEZONE", "Asia/Kolkata"))
    daily_run_time: str = field(default_factory=lambda: os.getenv("DAILY_RUN_TIME", "09:00"))
    max_applications_per_day: int = field(default_factory=lambda: int(os.getenv("MAX_APPLICATIONS_PER_DAY", "50")))
    min_job_match_score: int = field(default_factory=lambda: int(os.getenv("MIN_JOB_MATCH_SCORE", "70")))
    
    # Priority thresholds (days)
    high_priority_days: int = field(default_factory=lambda: int(os.getenv("HIGH_PRIORITY_DAYS", "3")))
    medium_priority_days: int = field(default_factory=lambda: int(os.getenv("MEDIUM_PRIORITY_DAYS", "7")))
    low_priority_days: int = field(default_factory=lambda: int(os.getenv("LOW_PRIORITY_DAYS", "14")))

@dataclass
class SecurityConfig:
    """Security and encryption configuration"""
    encryption_key: str = field(default_factory=lambda: os.getenv("ENCRYPTION_KEY", ""))
    jwt_secret: str = field(default_factory=lambda: os.getenv("JWT_SECRET", ""))
    session_timeout: int = field(default_factory=lambda: int(os.getenv("SESSION_TIMEOUT", "3600")))
    extension_api_secret: str = field(default_factory=lambda: os.getenv("EXTENSION_API_SECRET", ""))

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    file_path: str = field(default_factory=lambda: os.getenv("LOG_FILE_PATH", "./logs/automation.log"))
    max_file_size: int = field(default_factory=lambda: int(os.getenv("LOG_MAX_FILE_SIZE", "10485760")))  # 10MB
    backup_count: int = field(default_factory=lambda: int(os.getenv("LOG_BACKUP_COUNT", "5")))
    verbose_logging: bool = field(default_factory=lambda: os.getenv("VERBOSE_LOGGING", "false").lower() == "true")

# =============================================================================
# MAIN CONFIGURATION CLASS
# =============================================================================

class Settings:
    """Main configuration class that aggregates all settings"""
    
    def __init__(self):
        """Initialize all configuration sections"""
        self.ai = AIConfig()
        self.notion = NotionConfig()
        self.overleaf = OverleafConfig()
        self.gmail = GmailConfig()
        self.scraping = ScrapingConfig()
        self.playwright = PlaywrightConfig()
        self.system = SystemConfig()
        self.security = SecurityConfig()
        self.logging = LoggingConfig()
        
        # Additional configurations
        self.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
        self.test_mode = os.getenv("TEST_MODE", "false").lower() == "true"
        self.mock_applications = os.getenv("MOCK_APPLICATIONS", "false").lower() == "true"
        
        # Project paths
        self.project_root = Path(__file__).parent.parent
        self.config_dir = self.project_root / "config"
        self.logs_dir = self.project_root / "logs"
        self.data_dir = self.project_root / "data"
        
        # Validate configuration
        self.validate_config()
    
    def validate_config(self) -> None:
        """Validate critical configuration values"""
        errors = []
        
        # Check required API keys
        if not self.ai.openai_api_key and not self.test_mode:
            errors.append("OPENAI_API_KEY is required")
        
        if not self.ai.perplexity_api_key and not self.test_mode:
            errors.append("PERPLEXITY_API_KEY is required")
        
        if not self.notion.api_key and not self.test_mode:
            errors.append("NOTION_API_KEY is required")
        
        if not self.notion.applications_db_id and not self.test_mode:
            errors.append("NOTION_APPLICATIONS_DB_ID is required")
        
        if not self.notion.job_tracker_db_id and not self.test_mode:
            errors.append("NOTION_JOB_TRACKER_DB_ID is required")
        
        # Check security settings
        if not self.security.encryption_key and not self.test_mode:
            errors.append("ENCRYPTION_KEY is required")
        
        if not self.security.jwt_secret and not self.test_mode:
            errors.append("JWT_SECRET is required")
        
        # Create necessary directories
        self.logs_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        if errors and not self.test_mode:
            raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
    
    def get_supported_platforms(self) -> List[str]:
        """Get list of supported job platforms"""
        return [platform.value for platform in JobPlatform]
    
    def get_ai_providers(self) -> List[str]:
        """Get list of available AI providers"""
        return [provider.value for provider in AIProvider]
    
    def load_job_filters(self) -> Dict[str, Any]:
        """Load job filters from YAML file"""
        filters_path = self.config_dir / "job_filters.yaml"
        if filters_path.exists():
            with open(filters_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def load_resume_template(self) -> Dict[str, Any]:
        """Load resume template from JSON file"""
        template_path = self.config_dir / "resume_template.json"
        if template_path.exists():
            with open(template_path, 'r') as f:
                return json.load(f)
        return {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for logging/debugging"""
        config_dict = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                attr_value = getattr(self, attr_name)
                if hasattr(attr_value, '__dict__'):
                    config_dict[attr_name] = attr_value.__dict__
                else:
                    config_dict[attr_name] = attr_value
        
        # Remove sensitive information for logging
        sensitive_keys = ['api_key', 'password', 'secret', 'token', 'client_secret']
        return self._sanitize_dict(config_dict, sensitive_keys)
    
    def _sanitize_dict(self, data: Dict[str, Any], sensitive_keys: List[str]) -> Dict[str, Any]:
        """Remove sensitive information from dictionary"""
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, dict):
                sanitized[key] = self._sanitize_dict(value, sensitive_keys)
            elif any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "***REDACTED***" if value else ""
            else:
                sanitized[key] = value
        return sanitized

# =============================================================================
# GLOBAL SETTINGS INSTANCE
# =============================================================================

# Create global settings instance
settings = Settings()

# =============================================================================
# CONFIGURATION HELPERS
# =============================================================================

def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings

def reload_settings() -> Settings:
    """Reload settings from environment"""
    global settings
    settings = Settings()
    return settings

def setup_logging() -> None:
    """Setup logging configuration based on settings"""
    log_level = getattr(logging, settings.logging.level.upper())
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(settings.logging.file_path),
            logging.StreamHandler()
        ]
    )

# Setup logging on import
setup_logging()

# =============================================================================
# EXPORT ALL
# =============================================================================

__all__ = [
    'Settings',
    'AIConfig',
    'NotionConfig', 
    'OverleafConfig',
    'GmailConfig',
    'ScrapingConfig',
    'PlaywrightConfig',
    'SystemConfig',
    'SecurityConfig',
    'LoggingConfig',
    'JobPlatform',
    'AIProvider',
    'LogLevel',
    'settings',
    'get_settings',
    'reload_settings',
    'setup_logging'
]
