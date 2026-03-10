// Merged from chrome_extension/ — 2026-03-03
/**
 * =============================================================================
 * AI JOB AUTOMATION AGENT - SIDEBAR UI CONTROLLER
 * =============================================================================
 */

'use strict';

async function getApiBase() {
  try {
    const result = await chrome.storage.local.get('api_base_url');
    const url = result.api_base_url;
    return (url && typeof url === 'string' && url.trim()) ? url.trim().replace(/\/$/, '') : 'http://localhost:8000';
  } catch (_) {
    return 'http://localhost:8000';
  }
}

class SidebarUI {
  constructor() {
    this.state = {
      currentJob: null,
      userProfile: null,
      activityLog: [],
      chatHistory: []
    };

    this.init();
  }

  async init() {
    console.log('[Sidebar] Initializing...');

    // Show loading screen
    const loadingScreen = document.getElementById('loadingScreen');
    if (loadingScreen) {
      loadingScreen.classList.remove('hidden');
    }

    // Set up tab switching
    this.setupTabs();

    // Set up form handlers
    this.setupFormHandlers();

    // Set up chat
    this.setupChat();

    // Load initial data
    await this.loadInitialData();

    // Listen for messages from background
    chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
      this.handleMessage(msg);
    });

    // Hide loading, show app
    setTimeout(() => {
      if (loadingScreen) loadingScreen.style.display = 'none';
      const appContainer = document.getElementById('appContainer');
      if (appContainer) appContainer.style.display = 'flex';
    }, 500);

    console.log('[Sidebar] Ready');
  }

  setupTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const tabId = btn.dataset.tab;

        // Update active states
        tabButtons.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));

        btn.classList.add('active');
        const targetTab = document.getElementById(tabId);
        if (targetTab) targetTab.classList.add('active');
      });
    });
  }

  setupFormHandlers() {
    // Profile form
    const profileForm = document.getElementById('profileForm');
    if (profileForm) {
      profileForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        await this.saveProfile();
      });
    }

    // Automation rules
    const saveRulesBtn = document.getElementById('saveAutomationRules');
    if (saveRulesBtn) {
      saveRulesBtn.addEventListener('click', () => this.saveAutomationRules());
    }

    // Match threshold slider
    const thresholdSlider = document.getElementById('matchThreshold');
    if (thresholdSlider) {
      thresholdSlider.addEventListener('input', (e) => {
        const valueDisplay = document.getElementById('matchThresholdValue');
        if (valueDisplay) valueDisplay.textContent = `${e.target.value}%`;
      });
    }

    // API config
    const saveApiBtn = document.getElementById('saveApiConfig');
    if (saveApiBtn) {
      saveApiBtn.addEventListener('click', () => this.saveApiConfig());
    }

    const testApiBtn = document.getElementById('testApiConnection');
    if (testApiBtn) {
      testApiBtn.addEventListener('click', () => this.testApiConnection());
    }

    // Notion config
    const saveNotionBtn = document.getElementById('saveNotionConfig');
    if (saveNotionBtn) {
      saveNotionBtn.addEventListener('click', () => this.saveNotionConfig());
    }

    // Clear activity log
    const clearLogBtn = document.getElementById('clearActivityLog');
    if (clearLogBtn) {
      clearLogBtn.addEventListener('click', () => this.clearActivityLog());
    }
  }

  setupChat() {
    const sendBtn = document.getElementById('sendChatBtn');
    const clearBtn = document.getElementById('clearChatBtn');
    const input = document.getElementById('chatInput');

    if (sendBtn) sendBtn.addEventListener('click', () => this.sendChatMessage());
    if (clearBtn) clearBtn.addEventListener('click', () => this.clearChat());

    if (input) {
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          this.sendChatMessage();
        }
      });
    }
  }

  async loadInitialData() {
    try {
      // Load user profile
      const profileRes = await this.sendMessage({ type: 'GET_USER_PROFILE' });
      if (profileRes.success) {
        this.state.userProfile = profileRes.profile;
        this.populateProfileForm(profileRes.profile);
      }

      // Load activity log
      const activityRes = await this.sendMessage({ type: 'GET_ACTIVITY_LOG' });
      if (activityRes.success) {
        this.state.activityLog = activityRes.log || [];
        this.renderActivityLog();
      }

      // Test FastAPI connection
      try {
        const apiBase = await getApiBase();
        const healthRes = await fetch(`${apiBase}/health`, { method: 'GET', signal: AbortSignal.timeout(5000) });
        this.updateConnectionStatus(healthRes.ok);
      } catch (_) {
        this.updateConnectionStatus(false);
      }

      // Load Chrome version
      this.loadSystemInfo();

    } catch (err) {
      console.error('[Sidebar] Failed to load initial data', err);
    }
  }

  async sendMessage(message) {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage(message, (response) => {
        resolve(response || { success: false });
      });
    });
  }

  handleMessage(msg) {
    console.log('[Sidebar] Received message:', msg.type);

    switch (msg.type) {
      case 'JOB_DETECTED':
        this.state.currentJob = msg.jobData;
        this.updateCurrentJob(msg.jobData);
        break;
    }
  }

  updateCurrentJob(jobData) {
    const content = document.getElementById('currentJobContent');
    if (!content) return;

    content.innerHTML = `
      <div style="padding: 16px;">
        <h4 style="margin: 0 0 8px 0; font-size: 16px; font-weight: 600;">${jobData.title || 'Unknown'}</h4>
        <div style="font-size: 14px; color: #6b7280; margin-bottom: 4px;">${jobData.company || 'Unknown Company'}</div>
        <div style="font-size: 13px; color: #9ca3af; margin-bottom: 16px;">${jobData.location || 'Location not specified'}</div>
        <button class="btn btn-primary btn-sm" id="analyzeJobBtn">🧠 Analyze Job</button>
      </div>
    `;

    // Attach event listener
    const btn = document.getElementById('analyzeJobBtn');
    if (btn) {
      btn.addEventListener('click', () => this.analyzeCurrentJob());
    }
  }

  async analyzeCurrentJob() {
    if (!this.state.currentJob) {
      alert('No job detected. Navigate to a job posting first.');
      return;
    }

    const analysisContent = document.getElementById('analysisContent');
    try {
      const apiBase = await getApiBase();
      const res = await fetch(`${apiBase}/match`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_url: this.state.currentJob.url }),
        signal: AbortSignal.timeout(15000),
      });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      if (analysisContent) {
        analysisContent.innerHTML = `
          <div style="white-space: pre-wrap; line-height: 1.6;">${data.match_reasoning || JSON.stringify(data, null, 2)}</div>
        `;
        const analysisTab = document.querySelector('[data-tab="analysis"]');
        if (analysisTab) analysisTab.click();
      }
    } catch (err) {
      if (analysisContent) {
        analysisContent.innerHTML = `
          <div style="color: #ef4444;">Analysis failed: ${err.message || 'Unknown error'}</div>
        `;
      }
    }
  }

  async sendChatMessage() {
    const input = document.getElementById('chatInput');
    if (!input) return;

    const message = input.value.trim();
    if (!message) return;

    // Add user message to chat
    this.addChatMessage('user', message);
    input.value = '';

    // Show typing indicator
    const typingId = this.addChatMessage('assistant', '💭 Thinking...');

    // Send to FastAPI /match
    try {
      const apiBase = await getApiBase();
      const res = await fetch(`${apiBase}/match`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_url: this.state.currentJob?.url || '' }),
        signal: AbortSignal.timeout(15000),
      });

      const typingEl = document.getElementById(typingId);
      if (typingEl) typingEl.remove();

      if (res.ok) {
        const data = await res.json();
        this.addChatMessage('assistant', data.match_reasoning || JSON.stringify(data));
      } else {
        this.addChatMessage('assistant', `❌ Error: HTTP ${res.status}`);
      }
    } catch (err) {
      const typingEl = document.getElementById(typingId);
      if (typingEl) typingEl.remove();
      this.addChatMessage('assistant', `❌ Error: ${err.message || 'Unknown error'}`);
    }
  }

  addChatMessage(role, content) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return '';

    const messageId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const isUser = role === 'user';
    const messageEl = document.createElement('div');
    messageEl.id = messageId;
    messageEl.style.cssText = `
      display: flex;
      gap: 12px;
      ${isUser ? 'flex-direction: row-reverse;' : ''}
    `;

    messageEl.innerHTML = `
      <div style="width: 32px; height: 32px; border-radius: 50%; background: ${isUser ? '#2563eb' : '#10b981'}; display: flex; align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0;">
        ${isUser ? '👤' : '🤖'}
      </div>
      <div style="flex: 1; background: ${isUser ? '#eff6ff' : '#f0fdf4'}; padding: 12px; border-radius: 12px; font-size: 14px; line-height: 1.6; white-space: pre-wrap;">
        ${content}
      </div>
    `;

    chatMessages.appendChild(messageEl);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return messageId;
  }

  clearChat() {
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
      chatMessages.innerHTML = `
        <div style="text-align: center; color: #999; font-size: 13px; padding: 20px;">
          💬 Start a conversation with your AI assistant
        </div>
      `;
    }

    this.sendMessage({
      type: 'CLEAR_CHAT_HISTORY',
      payload: {}
    });
  }

  populateProfileForm(profile) {
    const form = document.getElementById('profileForm');
    if (!form) return;

    Object.keys(profile).forEach(key => {
      const input = form.querySelector(`[name="${key}"]`);
      if (input) input.value = profile[key] || '';
    });
  }

  async saveProfile() {
    const form = document.getElementById('profileForm');
    if (!form) return;

    const formData = new FormData(form);
    const profile = Object.fromEntries(formData.entries());

    const result = await this.sendMessage({
      type: 'UPDATE_PROFILE',
      payload: { profile }
    });

    if (result.success) {
      alert('✅ Profile saved successfully!');
      this.state.userProfile = profile;
    } else {
      alert('❌ Failed to save profile');
    }
  }

  async saveAutomationRules() {
    const matchThreshold = document.getElementById('matchThreshold')?.value;
    const dailyLimit = document.getElementById('dailyLimit')?.value;

    const settings = {
      ...this.state.userProfile,
      match_threshold: parseInt(matchThreshold),
      daily_limit: parseInt(dailyLimit)
    };

    const result = await this.sendMessage({
      type: 'UPDATE_PROFILE',
      payload: { profile: settings }
    });

    if (result.success) {
      alert('✅ Automation rules saved!');
    } else {
      alert('❌ Failed to save rules');
    }
  }

  async saveApiConfig() {
    const apiBaseInput = document.getElementById('apiBaseUrl');
    const apiBase = apiBaseInput?.value?.trim();

    if (!apiBase) {
      alert('Please enter an API base URL');
      return;
    }

    await chrome.storage.local.set({ api_base_url: apiBase });
    alert('✅ API base URL saved!');
  }

  async testApiConnection() {
    const btn = document.getElementById('testApiConnection');
    if (btn) {
      btn.textContent = 'Testing...';
      btn.disabled = true;
    }

    let connected = false;
    try {
      const apiBase = await getApiBase();
      const res = await fetch(`${apiBase}/health`, { method: 'GET', signal: AbortSignal.timeout(5000) });
      connected = res.ok;
    } catch (_) {
      connected = false;
    }

    if (btn) {
      btn.textContent = 'Test Connection';
      btn.disabled = false;
    }

    this.updateConnectionStatus(connected);
    alert(connected ? '✅ API Connected!' : '❌ Connection Failed');
  }

  updateConnectionStatus(connected) {
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    const apiStatus = document.getElementById('apiStatus');

    if (connected) {
      if (statusDot) statusDot.className = 'status-dot connected';
      if (statusText) statusText.textContent = 'Connected';
      if (apiStatus) {
        apiStatus.textContent = 'Connected';
        apiStatus.style.color = '#10b981';
      }
    } else {
      if (statusDot) statusDot.className = 'status-dot disconnected';
      if (statusText) statusText.textContent = 'Disconnected';
      if (apiStatus) {
        apiStatus.textContent = 'Disconnected';
        apiStatus.style.color = '#ef4444';
      }
    }
  }

  async saveNotionConfig() {
    const webhookUrl = document.getElementById('notionWebhook')?.value;

    const settings = {
      ...this.state.userProfile,
      notion_webhook_url: webhookUrl
    };

    const result = await this.sendMessage({
      type: 'UPDATE_PROFILE',
      payload: { profile: settings }
    });

    if (result.success) {
      alert('✅ Notion webhook saved!');
    } else {
      alert('❌ Failed to save webhook');
    }
  }

  renderActivityLog() {
    const list = document.getElementById('activityList');
    const fullLog = document.getElementById('fullActivityLog');

    if (!this.state.activityLog.length) {
      if (list) list.innerHTML = '<p style="text-align: center; color: #999; padding: 20px;">No activity yet</p>';
      if (fullLog) fullLog.innerHTML = '<p style="text-align: center; color: #999; padding: 20px;">No activity yet</p>';
      return;
    }

    const renderItem = (item) => `
      <div class="activity-item">
        <div class="activity-icon">🤖</div>
        <div class="activity-content">
          <div class="activity-title">${item.type}</div>
          <div class="activity-subtitle">${JSON.stringify(item).slice(0, 100)}</div>
          <div class="activity-time">${new Date(item.timestamp).toLocaleString()}</div>
        </div>
      </div>
    `;

    if (list) list.innerHTML = this.state.activityLog.slice(0, 5).map(renderItem).join('');
    if (fullLog) fullLog.innerHTML = this.state.activityLog.map(renderItem).join('');
  }

  async clearActivityLog() {
    if (!confirm('Clear all activity logs?')) return;

    await chrome.storage.local.set({ ai_activity_log: [] });
    this.state.activityLog = [];
    this.renderActivityLog();
  }

  async loadSystemInfo() {
    try {
      const chromeVersion = /Chrome\/([0-9.]+)/.exec(navigator.userAgent)?.[1] || 'Unknown';
      const chromeEl = document.getElementById('chromeVersion');
      if (chromeEl) chromeEl.textContent = chromeVersion;

      const estimate = await navigator.storage?.estimate();
      const usageMB = estimate ? (estimate.usage / 1024 / 1024).toFixed(2) : '0';
      const storageEl = document.getElementById('storageUsed');
      if (storageEl) storageEl.textContent = `${usageMB} MB`;
    } catch (err) {
      console.error('Failed to load system info', err);
    }
  }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    window.sidebarUI = new SidebarUI();
  });
} else {
  window.sidebarUI = new SidebarUI();
}
