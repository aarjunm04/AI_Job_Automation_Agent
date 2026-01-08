#!/bin/bash
# =============================================================================
# AI JOB AUTOMATION AGENT - DAILY CRON JOB SCRIPT
# =============================================================================
# Automated daily execution of job discovery, analysis, and applications
# Designed to run via cron for fully automated job hunting
# =============================================================================

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
CRON_LOG="$LOG_DIR/daily_cron.log"
ERROR_LOG="$LOG_DIR/cron_errors.log"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$CRON_LOG"
    
    if [ "$level" = "ERROR" ]; then
        echo -e "${timestamp} [${level}] ${message}" >> "$ERROR_LOG"
    fi
}

log "INFO" "🚀 Starting daily AI Job Automation cron job"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    log "ERROR" "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Change to project directory
cd "$PROJECT_ROOT" || {
    log "ERROR" "❌ Failed to change to project directory: $PROJECT_ROOT"
    exit 1
}

# Source environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
    log "INFO" "📝 Loaded environment variables from .env"
else
    log "WARN" "⚠️  No .env file found. Using system environment variables."
fi

# Check if services are running
log "INFO" "🔍 Checking service status..."

services_running=true

# Check main services
for service in backend n8n redis postgres; do
    if ! docker-compose ps "$service" | grep -q "Up"; then
        log "WARN" "⚠️  Service $service is not running"
        services_running=false
    fi
done

# Start services if not running
if [ "$services_running" = false ]; then
    log "INFO" "🔄 Starting Docker services..."
    
    if docker-compose up -d; then
        log "INFO" "✅ Services started successfully"
        # Wait for services to be ready
        sleep 30
    else
        log "ERROR" "❌ Failed to start Docker services"
        exit 1
    fi
else
    log "INFO" "✅ All services are running"
fi

# Health check
log "INFO" "🏥 Performing system health check..."

health_check_passed=true

# Check backend API
if ! curl -s -f http://localhost:8080/health > /dev/null; then
    log "ERROR" "❌ Backend API health check failed"
    health_check_passed=false
fi

# Check N8N API
if ! curl -s -f http://localhost:5678/healthz > /dev/null; then
    log "ERROR" "❌ N8N health check failed"
    health_check_passed=false
fi

# Check database connection
if ! docker-compose exec -T postgres pg_isready -U "${POSTGRES_USER:-n8n_user}" > /dev/null; then
    log "ERROR" "❌ Database health check failed"
    health_check_passed=false
fi

if [ "$health_check_passed" = false ]; then
    log "ERROR" "❌ System health check failed. Aborting cron job."
    
    # Send alert notification if configured
    if [ ! -z "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data '{"text":"🚨 AI Job Automation: Daily cron job failed - system health check failed"}' \
            "$SLACK_WEBHOOK_URL" || true
    fi
    
    exit 1
fi

log "INFO" "✅ System health check passed"

# Get current statistics before execution
log "INFO" "📊 Getting current statistics..."

stats_before=$(curl -s http://localhost:8080/api/statistics || echo '{}')
log "INFO" "📈 Current stats: $stats_before"

# Execute the automation pipeline
log "INFO" "🤖 Starting full automation pipeline..."

execution_start_time=$(date +%s)

# Method 1: Use Python master_run.py directly
if [ -f "$PROJECT_ROOT/master_run.py" ]; then
    log "INFO" "🐍 Executing via master_run.py..."
    
    cd "$PROJECT_ROOT"
    
    if python3 master_run.py --mode full-automation >> "$CRON_LOG" 2>&1; then
        log "INFO" "✅ Python automation completed successfully"
        execution_method="python"
    else
        log "WARN" "⚠️  Python automation failed, trying N8N webhook"
        execution_method="fallback"
    fi
else
    log "INFO" "📄 master_run.py not found, using N8N webhook"
    execution_method="webhook"
fi

# Method 2: Trigger N8N workflow via webhook (fallback or primary)
if [ "$execution_method" = "webhook" ] || [ "$execution_method" = "fallback" ]; then
    log "INFO" "🔗 Triggering N8N automation workflow..."
    
    webhook_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d '{"source":"daily_cron","timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'"}' \
        http://localhost:5678/webhook/ai-job-trigger || echo '{"success":false}')
    
    log "INFO" "📥 Webhook response: $webhook_response"
    
    # Check if webhook execution was successful
    if echo "$webhook_response" | grep -q '"success":true'; then
        log "INFO" "✅ N8N workflow triggered successfully"
    else
        log "ERROR" "❌ N8N workflow trigger failed"
        
        # Send failure notification
        if [ ! -z "$SLACK_WEBHOOK_URL" ]; then
            curl -X POST -H 'Content-type: application/json' \
                --data '{"text":"🚨 AI Job Automation: Daily cron job failed - workflow trigger failed"}' \
                "$SLACK_WEBHOOK_URL" || true
        fi
        
        exit 1
    fi
fi

# Calculate execution time
execution_end_time=$(date +%s)
execution_duration=$((execution_end_time - execution_start_time))
log "INFO" "⏱️  Execution completed in ${execution_duration} seconds"

# Wait for processing to complete (give some time for async operations)
log "INFO" "⏳ Waiting for processing to complete..."
sleep 120  # Wait 2 minutes for operations to finish

# Get updated statistics
log "INFO" "📊 Getting updated statistics..."
stats_after=$(curl -s http://localhost:8080/api/statistics || echo '{}')

# Parse statistics (basic parsing)
jobs_discovered_before=$(echo "$stats_before" | grep -o '"jobs_discovered":[0-9]*' | cut -d':' -f2 || echo "0")
jobs_discovered_after=$(echo "$stats_after" | grep -o '"jobs_discovered":[0-9]*' | cut -d':' -f2 || echo "0")

applications_sent_before=$(echo "$stats_before" | grep -o '"applications_sent":[0-9]*' | cut -d':' -f2 || echo "0")
applications_sent_after=$(echo "$stats_after" | grep -o '"applications_sent":[0-9]*' | cut -d':' -f2 || echo "0")

# Calculate differences
new_jobs=$((jobs_discovered_after - jobs_discovered_before))
new_applications=$((applications_sent_after - applications_sent_before))

log "INFO" "📈 Results: $new_jobs new jobs discovered, $new_applications applications sent"

# Generate daily report
report_date=$(date '+%Y-%m-%d')
daily_report="$LOG_DIR/daily_report_${report_date}.json"

cat > "$daily_report" << EOF
{
  "date": "$report_date",
  "execution_time": $execution_duration,
  "execution_method": "$execution_method",
  "results": {
    "jobs_discovered": $new_jobs,
    "applications_sent": $new_applications,
    "total_jobs_discovered": $jobs_discovered_after,
    "total_applications_sent": $applications_sent_after
  },
  "statistics": {
    "before": $stats_before,
    "after": $stats_after
  },
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
}
EOF

log "INFO" "📄 Daily report saved to: $daily_report"

# Send success notification
if [ ! -z "$SLACK_WEBHOOK_URL" ]; then
    log "INFO" "📱 Sending success notification to Slack..."
    
    curl -X POST -H 'Content-type: application/json' \
        --data '{
            "text": "🎉 AI Job Automation Daily Report - '$report_date'",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🤖 Daily Job Automation Report"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": "*Date:* '$report_date'"
                        },
                        {
                            "type": "mrkdwn",
                            "text": "*Execution Time:* '${execution_duration}'s"
                        },
                        {
                            "type": "mrkdwn",
                            "text": "*Jobs Discovered:* '$new_jobs'"
                        },
                        {
                            "type": "mrkdwn",
                            "text": "*Applications Sent:* '$new_applications'"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "✅ Automation completed successfully!"
                    }
                }
            ]
        }' \
        "$SLACK_WEBHOOK_URL" || log "WARN" "⚠️  Failed to send Slack notification"
fi

# Send Discord notification if configured
if [ ! -z "$DISCORD_WEBHOOK_URL" ]; then
    log "INFO" "📱 Sending success notification to Discord..."
    
    curl -X POST -H 'Content-type: application/json' \
        --data '{
            "embeds": [
                {
                    "title": "🤖 AI Job Automation - Daily Report",
                    "color": 3447003,
                    "fields": [
                        {
                            "name": "📅 Date",
                            "value": "'$report_date'",
                            "inline": true
                        },
                        {
                            "name": "⏱️ Execution Time", 
                            "value": "'${execution_duration}' seconds",
                            "inline": true
                        },
                        {
                            "name": "🔍 Jobs Discovered",
                            "value": "'$new_jobs'",
                            "inline": true
                        },
                        {
                            "name": "🚀 Applications Sent",
                            "value": "'$new_applications'",
                            "inline": true
                        }
                    ],
                    "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'",
                    "footer": {
                        "text": "AI Job Automation Agent"
                    }
                }
            ]
        }' \
        "$DISCORD_WEBHOOK_URL" || log "WARN" "⚠️  Failed to send Discord notification"
fi

# Cleanup old logs (keep last 30 days)
log "INFO" "🧹 Cleaning up old logs..."
find "$LOG_DIR" -name "daily_report_*.json" -mtime +30 -delete || true
find "$LOG_DIR" -name "*.log" -mtime +30 -delete || true

# Database maintenance (if needed)
log "INFO" "🔧 Performing database maintenance..."
docker-compose exec -T postgres psql -U "${POSTGRES_USER:-n8n_user}" -d "${POSTGRES_DB:-n8n_ai_job}" -c "VACUUM ANALYZE;" || log "WARN" "⚠️  Database maintenance failed"

# Final success log
log "INFO" "🎉 Daily cron job completed successfully!"
log "INFO" "📊 Summary: $new_jobs jobs discovered, $new_applications applications sent in ${execution_duration}s"

# Exit with success
exit 0
