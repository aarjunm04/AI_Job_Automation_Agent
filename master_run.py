"""
=============================================================================
AI JOB AUTOMATION AGENT - MASTER ORCHESTRATION SCRIPT
=============================================================================
Main orchestration script that coordinates all system components:
- Job discovery and scraping across 20+ platforms
- AI-powered job analysis and matching
- Dynamic resume generation and optimization
- Automated application submission
- Real-time progress tracking and status updates

This is the SINGLE ENTRY POINT for the entire automation system.

Usage:
    python master_run.py --mode discover          # Discover new jobs
    python master_run.py --mode apply             # Apply to staged jobs  
    python master_run.py --mode full-automation   # Complete end-to-end automation
    python master_run.py --mode health-check      # System health verification

Author: AI Job Automation Team
Version: 1.0.0
Last Updated: October 2025
=============================================================================
"""

import asyncio
import sys
import argparse
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

# Core system imports
from config.settings import get_settings
from mcp_client import get_mcp_client, close_mcp_client
from core import (
    initialize_core_system, 
    shutdown_core_system,
    get_ai_engine,
    get_notion_engine, 
    get_scraper_engine,
    get_resume_engine,
    get_automation_engine
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# MASTER ORCHESTRATOR CLASS
# =============================================================================

class MasterOrchestrator:
    """
    Master orchestrator that coordinates all AI job automation components
    
    This class manages the complete workflow:
    1. Job Discovery: Scrape jobs from 20+ platforms
    2. Job Analysis: AI-powered matching and scoring
    3. Resume Generation: Dynamic resume optimization
    4. Application Submission: Automated form filling
    5. Progress Tracking: Real-time status updates
    """
    
    def __init__(self):
        """Initialize Master Orchestrator"""
        self.settings = get_settings()
        self.start_time = datetime.now()
        
        # Core engines (initialized later)
        self.ai_engine = None
        self.notion_engine = None
        self.scraper_engine = None
        self.resume_engine = None
        self.automation_engine = None
        self.mcp_client = None
        
        # Execution statistics
        self.stats = {
            'jobs_discovered': 0,
            'jobs_analyzed': 0,
            'resumes_generated': 0,
            'applications_submitted': 0,
            'applications_successful': 0,
            'total_execution_time': 0.0,
            'errors_encountered': 0
        }
        
        logger.info("🚀 AI Job Automation Master Orchestrator initialized")
    
    # =========================================================================
    # SYSTEM INITIALIZATION
    # =========================================================================
    
    async def initialize_system(self) -> bool:
        """Initialize all core system components"""
        try:
            logger.info("🔧 Initializing AI Job Automation System...")
            
            # Initialize core system
            system_status = await initialize_core_system()
            
            if system_status["overall_health"] != "healthy":
                logger.error("❌ System initialization failed or degraded")
                logger.error(f"Issues: {system_status.get('issues', [])}")
                return False
            
            # Get engine references
            self.ai_engine = get_ai_engine()
            self.notion_engine = get_notion_engine()
            self.scraper_engine = get_scraper_engine()
            self.resume_engine = get_resume_engine()
            self.automation_engine = get_automation_engine()
            self.mcp_client = get_mcp_client()
            
            logger.info("✅ All system components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            return False
    
    # =========================================================================
    # MAIN EXECUTION MODES
    # =========================================================================
    
    async def run_job_discovery(self, max_jobs_per_platform: int = 50) -> Dict[str, Any]:
        """
        Mode 1: Job Discovery
        Scrape jobs from all platforms and store in Applications DB
        """
        logger.info("🔍 Starting Job Discovery Mode")
        start_time = time.time()
        
        try:
            # Step 1: Load search configuration
            job_filters = self.settings.load_job_filters()
            search_criteria = job_filters.get('search_criteria', {})
            job_titles = search_criteria.get('job_titles', [])
            
            logger.info(f"📋 Search criteria: {len(job_titles)} job titles configured")
            
            # Step 2: Scrape jobs from all platforms
            logger.info("🕷️ Starting multi-platform job scraping...")
            scraping_results = await self.scraper_engine.scrape_all_platforms(
                search_terms=job_titles[:10],  # Use top 10 job titles
                max_jobs_per_platform=max_jobs_per_platform
            )
            
            # Step 3: Process and convert scraped jobs
            all_scraped_jobs = []
            for result in scraping_results:
                if result.jobs_new > 0:
                    logger.info(f"✅ {result.platform}: {result.jobs_new} new jobs discovered")
                    # Get jobs from scraper engine
                    platform_jobs = await self._get_jobs_from_scraper_result(result)
                    all_scraped_jobs.extend(platform_jobs)
            
            # Step 4: Convert to JobApplication format
            job_applications = self.scraper_engine.convert_to_job_applications(all_scraped_jobs)
            logger.info(f"🔄 Converted {len(job_applications)} jobs to application format")
            
            # Step 5: Batch create in Notion Applications DB
            if job_applications:
                logger.info("💾 Saving jobs to Notion Applications Database...")
                created_ids = await self.notion_engine.batch_create_job_applications(job_applications)
                logger.info(f"✅ Saved {len(created_ids)} jobs to Notion")
                
                self.stats['jobs_discovered'] = len(created_ids)
            
            execution_time = time.time() - start_time
            self.stats['total_execution_time'] += execution_time
            
            summary = {
                'mode': 'job_discovery',
                'success': True,
                'jobs_discovered': self.stats['jobs_discovered'],
                'platforms_scraped': len(scraping_results),
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🎉 Job Discovery completed: {self.stats['jobs_discovered']} jobs discovered in {execution_time:.1f}s")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Job Discovery failed: {e}")
            self.stats['errors_encountered'] += 1
            return {
                'mode': 'job_discovery',
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    async def run_job_analysis(self, limit: int = 20) -> Dict[str, Any]:
        """
        Mode 2: Job Analysis
        Analyze discovered jobs and update with AI insights
        """
        logger.info("🧠 Starting Job Analysis Mode")
        start_time = time.time()
        
        try:
            # Step 1: Get jobs that need analysis
            try:
                from core.notion_engine import ApplicationStatus
            except Exception:
                # Fallback enum when core.notion_engine is not importable (e.g., during static analysis)
                from enum import Enum
                class ApplicationStatus(Enum):
                    DISCOVERED = "DISCOVERED"
                    STAGED_FOR_AUTO_APPLY = "STAGED_FOR_AUTO_APPLY"
                    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"
                    REJECTED = "REJECTED"
            jobs_to_analyze = await self.notion_engine.get_jobs_by_status(ApplicationStatus.DISCOVERED)
            
            if not jobs_to_analyze:
                logger.info("ℹ️ No jobs found for analysis")
                return {
                    'mode': 'job_analysis',
                    'success': True,
                    'jobs_analyzed': 0,
                    'execution_time': time.time() - start_time
                }
            
            logger.info(f"📊 Found {len(jobs_to_analyze)} jobs to analyze")
            
            # Step 2: Load user profile for matching
            user_profile = self._load_user_profile()
            
            # Step 3: Start AI session
            session_id = await self.ai_engine.start_ai_session("master_orchestrator")
            
            analyzed_count = 0
            for job in jobs_to_analyze[:limit]:  # Limit to avoid timeout
                try:
                    logger.info(f"🔍 Analyzing: {job['job_title']} at {job['company']}")
                    
                    # AI-powered job analysis
                    analysis = await self.ai_engine.analyze_job_opportunity(
                        job_data=job,
                        user_profile=user_profile,
                        session_id=session_id
                    )
                    
                    # Determine next status based on match score
                    if analysis.match_score >= 85:
                        next_status = ApplicationStatus.STAGED_FOR_AUTO_APPLY
                        priority = "High"
                    elif analysis.match_score >= 70:
                        next_status = ApplicationStatus.MANUAL_REVIEW_REQUIRED
                        priority = "Medium"
                    else:
                        next_status = ApplicationStatus.REJECTED
                        priority = "Low"
                    
                    # Update job in Notion
                    await self.notion_engine.update_job_application(
                        job['page_id'],
                        {
                            'status': next_status.value,
                            'priority': priority,
                            'match_score': analysis.match_score,
                            'ai_analysis': f"Match Score: {analysis.match_score}%. Key requirements: {', '.join(analysis.key_requirements[:3])}",
                            'application_strategy': analysis.application_strategy[:500]  # Truncate for Notion
                        }
                    )
                    
                    analyzed_count += 1
                    logger.info(f"✅ Analyzed {job['job_title']}: {analysis.match_score}% match -> {next_status.value}")
                    
                    # Small delay between analyses
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to analyze job {job.get('job_title', 'unknown')}: {e}")
                    self.stats['errors_encountered'] += 1
                    continue
            
            # End AI session
            await self.ai_engine.end_ai_session("master_orchestrator")
            
            execution_time = time.time() - start_time
            self.stats['jobs_analyzed'] = analyzed_count
            self.stats['total_execution_time'] += execution_time
            
            summary = {
                'mode': 'job_analysis',
                'success': True,
                'jobs_analyzed': analyzed_count,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🎉 Job Analysis completed: {analyzed_count} jobs analyzed in {execution_time:.1f}s")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Job Analysis failed: {e}")
            self.stats['errors_encountered'] += 1
            return {
                'mode': 'job_analysis', 
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    async def run_automated_applications(self, max_applications: int = 10) -> Dict[str, Any]:
        """
        Mode 3: Automated Applications
        Generate resumes and submit applications for high-priority jobs
        """
        logger.info("🤖 Starting Automated Applications Mode")
        start_time = time.time()
        
        try:
            # Step 1: Get high-priority jobs ready for application
            high_priority_jobs = await self.notion_engine.get_high_priority_jobs(limit=max_applications)
            
            if not high_priority_jobs:
                logger.info("ℹ️ No high-priority jobs found for application")
                return {
                    'mode': 'automated_applications',
                    'success': True,
                    'applications_submitted': 0,
                    'execution_time': time.time() - start_time
                }
            
            logger.info(f"🎯 Found {len(high_priority_jobs)} high-priority jobs for application")
            
            # Step 2: Load user profile and resume template
            user_profile = self._load_user_profile()
            resume_template = self.settings.load_resume_template()
            
            applications_successful = 0
            applications_attempted = 0
            
            for job in high_priority_jobs:
                try:
                    logger.info(f"📝 Processing application: {job['job_title']} at {job['company']}")
                    
                    # Step 3: Generate optimized resume
                    from core.resume_engine import ResumeGenerationRequest, OptimizationLevel, ResumeFormat
                    
                    resume_request = ResumeGenerationRequest(
                        job_title=job['job_title'],
                        job_description=job.get('job_description', ''),
                        company_name=job['company'],
                        user_profile=user_profile,
                        optimization_level=OptimizationLevel.MODERATE,
                        output_format=ResumeFormat.PDF
                    )
                    
                    resume_result = await self.resume_engine.generate_optimized_resume(resume_request)
                    
                    if not resume_result.success:
                        logger.error(f"❌ Resume generation failed for {job['job_title']}: {resume_result.errors}")
                        continue
                    
                    logger.info(f"✅ Resume generated for {job['job_title']} (ATS Score: {resume_result.ats_score})")
                    self.stats['resumes_generated'] += 1
                    
                    # Step 4: Research company for better application
                    company_intel = await self.ai_engine.research_company_intelligence(
                        company_name=job['company'],
                        job_role=job['job_title']
                    )
                    
                    # Step 5: Generate cover letter
                    cover_letter = await self.ai_engine.generate_personalized_cover_letter(
                        job_data=job,
                        company_intel=company_intel,
                        user_profile=user_profile,
                        resume_highlights=resume_result.optimization_applied.skill_prioritization if resume_result.optimization_applied else []
                    )
                    
                    logger.info(f"✅ Cover letter generated for {job['job_title']} ({cover_letter.word_count} words)")
                    
                    # Step 6: Prepare application job data
                    from core.automation_engine import ApplicationJob
                    
                    application_job = ApplicationJob(
                        job_id=job['page_id'],
                        job_title=job['job_title'],
                        company=job['company'],
                        job_url=job['job_url'],
                        platform=job['platform'],
                        resume_path=resume_result.file_path,
                        application_data={
                            'cover_letter_content': cover_letter.content,
                            'user_profile': user_profile,
                            'company_research': company_intel.overview
                        }
                    )
                    
                    # Step 7: Automated application submission
                    logger.info(f"🚀 Submitting application for {job['job_title']}...")
                    
                    application_attempt = await self.automation_engine.apply_to_job(application_job)
                    applications_attempted += 1
                    
                    if application_attempt.success:
                        applications_successful += 1
                        logger.info(f"🎉 Successfully applied to {job['job_title']} at {job['company']}")
                        
                        # Move to Job Tracker DB
                        await self.notion_engine.migrate_application_to_tracker(
                            job['page_id'],
                            job
                        )
                    else:
                        logger.error(f"❌ Application failed for {job['job_title']}: {application_attempt.errors}")
                    
                    # Rate limiting between applications
                    if applications_attempted < len(high_priority_jobs):
                        delay = 60  # 1 minute between applications
                        logger.info(f"⏳ Waiting {delay} seconds before next application...")
                        await asyncio.sleep(delay)
                    
                except Exception as e:
                    logger.error(f"❌ Application process failed for {job.get('job_title', 'unknown')}: {e}")
                    self.stats['errors_encountered'] += 1
                    continue
            
            execution_time = time.time() - start_time
            self.stats['applications_submitted'] = applications_attempted
            self.stats['applications_successful'] = applications_successful
            self.stats['total_execution_time'] += execution_time
            
            summary = {
                'mode': 'automated_applications',
                'success': True,
                'applications_attempted': applications_attempted,
                'applications_successful': applications_successful,
                'success_rate': (applications_successful / max(applications_attempted, 1)) * 100,
                'resumes_generated': self.stats['resumes_generated'],
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🎉 Automated Applications completed: {applications_successful}/{applications_attempted} successful in {execution_time:.1f}s")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Automated Applications failed: {e}")
            self.stats['errors_encountered'] += 1
            return {
                'mode': 'automated_applications',
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    async def run_full_automation(self) -> Dict[str, Any]:
        """
        Mode 4: Full End-to-End Automation
        Complete workflow from discovery to application
        """
        logger.info("🚀 Starting Full End-to-End Automation")
        start_time = time.time()
        
        try:
            # Step 1: Job Discovery
            logger.info("📡 Phase 1: Job Discovery")
            discovery_result = await self.run_job_discovery(max_jobs_per_platform=30)
            
            if not discovery_result['success']:
                return discovery_result
            
            # Small delay between phases
            await asyncio.sleep(30)
            
            # Step 2: Job Analysis  
            logger.info("🧠 Phase 2: Job Analysis")
            analysis_result = await self.run_job_analysis(limit=50)
            
            if not analysis_result['success']:
                return analysis_result
            
            # Small delay between phases
            await asyncio.sleep(30)
            
            # Step 3: Automated Applications
            logger.info("🤖 Phase 3: Automated Applications")
            max_daily_applications = self.settings.system.max_applications_per_day
            application_result = await self.run_automated_applications(max_applications=min(max_daily_applications, 15))
            
            # Combine results
            execution_time = time.time() - start_time
            
            summary = {
                'mode': 'full_automation',
                'success': True,
                'phases': {
                    'discovery': discovery_result,
                    'analysis': analysis_result,
                    'applications': application_result
                },
                'overall_stats': {
                    'jobs_discovered': discovery_result.get('jobs_discovered', 0),
                    'jobs_analyzed': analysis_result.get('jobs_analyzed', 0),
                    'applications_submitted': application_result.get('applications_attempted', 0),
                    'applications_successful': application_result.get('applications_successful', 0),
                    'total_execution_time': execution_time
                },
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("🎉 Full Automation completed successfully!")
            logger.info(f"📊 Summary: {summary['overall_stats']['jobs_discovered']} jobs discovered, {summary['overall_stats']['jobs_analyzed']} analyzed, {summary['overall_stats']['applications_successful']} applications successful")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Full Automation failed: {e}")
            return {
                'mode': 'full_automation',
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    async def run_system_health_check(self) -> Dict[str, Any]:
        """
        Mode 5: System Health Check
        Verify all components are working correctly
        """
        logger.info("🏥 Starting System Health Check")
        
        try:
            health_results = {}
            
            # Check each engine
            engines = {
                'ai_engine': self.ai_engine,
                'notion_engine': self.notion_engine,
                'scraper_engine': self.scraper_engine,
                'resume_engine': self.resume_engine,
                'automation_engine': self.automation_engine
            }
            
            for engine_name, engine in engines.items():
                try:
                    if hasattr(engine, 'health_check'):
                        health_result = await engine.health_check()
                        health_results[engine_name] = health_result
                    else:
                        health_results[engine_name] = {'status': 'no_health_check_available'}
                except Exception as e:
                    health_results[engine_name] = {'status': 'error', 'error': str(e)}
            
            # Check MCP Client
            try:
                mcp_stats = self.mcp_client.get_performance_stats()
                health_results['mcp_client'] = {'status': 'healthy', 'stats': mcp_stats}
            except Exception as e:
                health_results['mcp_client'] = {'status': 'error', 'error': str(e)}
            
            # Overall system health
            unhealthy_components = [
                name for name, result in health_results.items() 
                if result.get('status') != 'healthy'
            ]
            
            overall_status = 'healthy' if len(unhealthy_components) == 0 else 'degraded'
            
            summary = {
                'mode': 'health_check',
                'overall_status': overall_status,
                'component_health': health_results,
                'unhealthy_components': unhealthy_components,
                'execution_stats': self.stats,
                'system_uptime': str(datetime.now() - self.start_time),
                'timestamp': datetime.now().isoformat()
            }
            
            if overall_status == 'healthy':
                logger.info("✅ System Health Check: All components healthy")
            else:
                logger.warning(f"⚠️ System Health Check: {len(unhealthy_components)} components unhealthy: {unhealthy_components}")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Health Check failed: {e}")
            return {
                'mode': 'health_check',
                'overall_status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _load_user_profile(self) -> Dict[str, Any]:
        """Load user profile for job matching and applications"""
        return {
            'name': 'Aarjun Mahule',
            'email': 'aarjun.mahule@example.com',
            'phone': '+91-9921652687',
            'location': 'Nagpur, Maharashtra, India',
            'education': {
                'degree': 'B.TECH Computer Science and Engineering',
                'institution': 'G H Raisoni College Of Engineering, Nagpur',
                'graduation_year': 2025,
                'cgpa': 7.34
            },
            'experience': {
                'total_years': 1,
                'current_role': 'Data Science Intern',
                'previous_roles': ['Student']
            },
            'skills': {
                'programming_languages': ['Python', 'C++', 'JavaScript', 'SQL'],
                'ml_frameworks': ['TensorFlow', 'PyTorch', 'Scikit-learn', 'Pandas', 'NumPy'],
                'tools': ['Docker', 'Git', 'Jupyter', 'VS Code'],
                'cloud_platforms': ['AWS', 'GCP'],
                'databases': ['PostgreSQL', 'MongoDB']
            },
            'preferences': {
                'job_types': ['AI Engineer', 'ML Engineer', 'Data Scientist'],
                'work_arrangement': 'Remote preferred',
                'salary_expectation': '6+ LPA',
                'location_preference': ['Remote', 'Nagpur', 'Pune', 'Mumbai']
            }
        }
    
    async def _get_jobs_from_scraper_result(self, result) -> List:
        """Get actual job data from scraper result (placeholder)"""
        # In a real implementation, this would fetch the actual scraped jobs
        # For now, return empty list as scraper handles conversion
        return []
    
    async def cleanup(self) -> None:
        """Clean up system resources"""
        try:
            logger.info("🧹 Cleaning up system resources...")
            
            # Shutdown core system
            await shutdown_core_system()
            
            # Close MCP client
            await close_mcp_client()
            
            logger.info("✅ System cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

async def main():
    """Main entry point for the AI Job Automation system"""
    parser = argparse.ArgumentParser(
        description="AI Job Automation Agent - Master Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python master_run.py --mode discover                    # Discover new jobs
  python master_run.py --mode analyze                     # Analyze discovered jobs  
  python master_run.py --mode apply                       # Apply to high-priority jobs
  python master_run.py --mode full-automation             # Complete end-to-end automation
  python master_run.py --mode health-check                # System health verification
  
  python master_run.py --mode discover --limit 100        # Discover up to 100 jobs per platform
  python master_run.py --mode apply --limit 5             # Apply to max 5 jobs
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['discover', 'analyze', 'apply', 'full-automation', 'health-check'],
        required=True,
        help='Execution mode'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of items to process (jobs to discover/analyze/apply)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to custom configuration file'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🐛 Debug logging enabled")
    
    # Initialize orchestrator
    orchestrator = MasterOrchestrator()
    
    try:
        # Initialize system
        if not await orchestrator.initialize_system():
            logger.error("❌ System initialization failed")
            sys.exit(1)
        
        # Execute based on mode
        result = None
        
        if args.mode == 'discover':
            limit = args.limit or 50
            result = await orchestrator.run_job_discovery(max_jobs_per_platform=limit)
            
        elif args.mode == 'analyze':
            limit = args.limit or 20
            result = await orchestrator.run_job_analysis(limit=limit)
            
        elif args.mode == 'apply':
            limit = args.limit or 10
            result = await orchestrator.run_automated_applications(max_applications=limit)
            
        elif args.mode == 'full-automation':
            result = await orchestrator.run_full_automation()
            
        elif args.mode == 'health-check':
            result = await orchestrator.run_system_health_check()
        
        # Output results
        if result:
            print("\n" + "="*80)
            print("🎯 EXECUTION SUMMARY")
            print("="*80)
            print(json.dumps(result, indent=2, default=str))
            
            if result.get('success', False):
                logger.info("🎉 Execution completed successfully!")
                sys.exit(0)
            else:
                logger.error("❌ Execution failed!")
                sys.exit(1)
        else:
            logger.error("❌ No result returned")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️ Execution interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)
        
    finally:
        # Always cleanup
        await orchestrator.cleanup()

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Process interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
