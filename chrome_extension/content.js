/**
 * =============================================================================
 * AI JOB AUTOMATION AGENT - CONTENT SCRIPT (COMPLETE)
 * =============================================================================
 * 
 * In-page automation layer that runs on job sites.
 * 
 * Features:
 * - Platform detection (LinkedIn, Indeed, Naukri, generic ATS)
 * - Job data extraction with platform-specific selectors
 * - Intelligent form autofill with fuzzy field matching
 * - Resume upload automation
 * - Submit detection (URL change, DOM mutations, success banners)
 * - Minimal overlay UI for quick actions
 * - MCP integration via background messaging
 * 
 * Author: AI Job Automation Team
 * Version: 1.0.0
 * =============================================================================
 */

'use strict';

/**
 * =============================================================================
 * CONFIGURATION & CONSTANTS
 * =============================================================================
 */
const CFG = {
  DEBUG: false,
  FORM_FIELD_MATCH_THRESHOLD: 0.6,
  SUBMIT_DETECTION_TIMEOUT_MS: 25000,
  SUBMIT_DETECTION_POLL_MS: 800,
  MAX_RESUME_SIZE_MB: 10,
  LOG_PREFIX: '[AI-CONTENT]'
};

/**
 * =============================================================================
 * LOGGING UTILITIES
 * =============================================================================
 */
const debug = (...args) => CFG.DEBUG && console.debug(CFG.LOG_PREFIX, ...args);
const info = (...args) => console.info(CFG.LOG_PREFIX, ...args);
const warn = (...args) => console.warn(CFG.LOG_PREFIX, ...args);
const error = (...args) => console.error(CFG.LOG_PREFIX, ...args);

/**
 * =============================================================================
 * MESSAGING BRIDGE (Content <-> Background)
 * =============================================================================
 */
async function sendToBackground(message) {
  return new Promise((resolve) => {
    try {
      chrome.runtime.sendMessage(message, (response) => {
        if (chrome.runtime.lastError) {
          warn('Message send error:', chrome.runtime.lastError.message);
          resolve({ success: false, error: chrome.runtime.lastError.message });
        } else {
          resolve(response || { success: false, error: 'No response' });
        }
      });
    } catch (err) {
      warn('sendToBackground failed:', err);
      resolve({ success: false, error: err.message });
    }
  });
}

/**
 * =============================================================================
 * PLATFORM DETECTOR
 * =============================================================================
 */
class PlatformDetector {
  static detect() {
    const hostname = window.location.hostname.toLowerCase();
    const pathname = window.location.pathname.toLowerCase();

    // LinkedIn
    if (hostname.includes('linkedin.com')) {
      if (pathname.includes('/jobs/view/') || document.querySelector('[data-job-id], .jobs-description, .jobs-unified-top-card')) {
        return { platform: 'linkedin', type: 'job_detail' };
      }
      return { platform: 'linkedin', type: 'listing' };
    }

    // Indeed
    if (hostname.includes('indeed.com')) {
      if (pathname.includes('/viewjob') || document.querySelector('.jobsearch-JobComponent')) {
        return { platform: 'indeed', type: 'job_detail' };
      }
      return { platform: 'indeed', type: 'listing' };
    }

    // Naukri
    if (hostname.includes('naukri.com')) {
      if (document.querySelector('.jd-container') || pathname.includes('/job-listings/')) {
        return { platform: 'naukri', type: 'job_detail' };
      }
      return { platform: 'naukri', type: 'listing' };
    }

    // Wellfound / AngelList
    if (hostname.includes('wellfound.com') || hostname.includes('angel.co')) {
      if (document.querySelector('[data-test="JobDetail"]') || pathname.includes('/jobs/')) {
        return { platform: 'wellfound', type: 'job_detail' };
      }
      return { platform: 'wellfound', type: 'listing' };
    }

    // Generic job site detection
    const jobSelectors = [
      '.job-details', '.job-description', '.posting-headline', 
      '.posting', '[data-job-id]', '#job-desc', '.job-meta'
    ];

    if (jobSelectors.some(sel => document.querySelector(sel))) {
      return { platform: 'generic_ats', type: 'job_detail' };
    }

    return { platform: 'unknown', type: 'other' };
  }
}

/**
 * =============================================================================
 * JOB DATA EXTRACTOR
 * =============================================================================
 */
class JobExtractor {
  static extract(platform) {
    const url = window.location.href;
    let title = '';
    let company = '';
    let location = '';
    let description = '';

    try {
      switch (platform) {
        case 'linkedin':
          title = this._getText('.topcard__title, .jobs-unified-top-card__job-title, h1.t-24');
          company = this._getText('.topcard__org-name-link, .jobs-unified-top-card__company-name, .job-details-jobs-unified-top-card__company-name a');
          location = this._getText('.topcard__flavor--bullet, .jobs-unified-top-card__bullet, .jobs-unified-top-card__workplace-type');
          description = this._getText('.description__text, .jobs-description__content, .jobs-box__html-content');
          break;

        case 'indeed':
          title = this._getText('.jobsearch-JobInfoHeader-title, h1.jobsearch-JobInfoHeader-title');
          company = this._getText('.jobsearch-InlineCompanyRating div, .jobsearch-CompanyInfoContainer a');
          location = this._getText('.jobsearch-JobInfoHeader-subtitle div, .jobsearch-JobInfoHeader-subtitle');
          description = this._getText('#jobDescriptionText, .jobsearch-jobDescriptionText');
          break;

        case 'naukri':
          title = this._getText('.jd-header-title, .jd-header h1');
          company = this._getText('.jd-header .companyName a, .companyInfo a');
          location = this._getText('.jd-header .location, .location');
          description = this._getText('.job-desc, .JDC_content, .dang-inner-html');
          break;

        case 'wellfound':
          title = this._getText('[data-test="JobTitle"], h1');
          company = this._getText('[data-test="StartupLink"], .company-name');
          location = this._getText('[data-test="LocationLink"], .location');
          description = this._getText('[data-test="JobDescription"], .job-description');
          break;

        default:
          // Generic fallback
          title = this._getText('h1, .job-title, [data-job-title]') || document.title;
          company = this._getText('.company, .company-name, [data-company]');
          location = this._getText('.location, .job-location, [data-location]');
          description = this._getText('.job-description, .description, [role="main"]');
      }

      return {
        title: title.trim(),
        company: company.trim(),
        location: location.trim(),
        description: description.slice(0, 5000).trim(), // Limit description length
        url,
        platform,
        extracted_at: new Date().toISOString()
      };

    } catch (err) {
      warn('Job extraction failed', err);
      return null;
    }
  }

  static _getText(selector) {
    const el = document.querySelector(selector);
    return el ? (el.innerText || el.textContent || '').trim() : '';
  }
}

/**
 * =============================================================================
 * FORM AUTOFILL ENGINE
 * =============================================================================
 */
class FormAutofill {
  constructor() {
    this.fieldMap = null;
  }

  async fillForm(userProfile = {}, resumeMode = { enabled: true }) {
    try {
      const form = this._findMainForm();
      if (!form) {
        return { success: false, error: 'No form found on page' };
      }

      this.fieldMap = this._scanFormFields(form);
      const values = this._profileToValues(userProfile);

      // Fill text fields
      let filledCount = 0;
      for (const [logical, val] of Object.entries(values)) {
        if (!val) continue;
        
        const input = this._findBestFieldFor(logical);
        if (input) {
          this._setFieldValue(input, val);
          filledCount++;
          debug(`Filled ${logical} ->`, input.name || input.id);
        }
      }

      // Handle resume upload
      if (resumeMode.enabled) {
        await this._handleResumeUpload(form, userProfile);
      }

      // Trigger change events
      this._triggerFormEvents(form);

      info(`Autofill complete: ${filledCount} fields filled`);
      return { success: true, filledCount };

    } catch (err) {
      error('Autofill failed', err);
      return { success: false, error: err.message };
    }
  }

  _findMainForm() {
    const forms = Array.from(document.forms || []);
    if (!forms.length) return null;

    // Prefer form with file input and most inputs
    let best = forms[0];
    let bestScore = 0;

    forms.forEach(form => {
      const fileInputs = form.querySelectorAll('input[type="file"]').length;
      const inputsCount = form.querySelectorAll('input, textarea, select').length;
      const score = fileInputs * 10 + inputsCount;

      if (score > bestScore) {
        best = form;
        bestScore = score;
      }
    });

    return best;
  }

  _scanFormFields(form) {
    const fields = [];
    const controls = form.querySelectorAll('input:not([type="hidden"]):not([type="submit"]), textarea, select');

    controls.forEach(control => {
      try {
        const info = {
          el: control,
          tag: control.tagName.toLowerCase(),
          type: control.type || 'text',
          name: control.name || '',
          id: control.id || '',
          placeholder: control.getAttribute('placeholder') || '',
          aria: control.getAttribute('aria-label') || '',
          labelText: this._resolveLabel(control),
          visible: !!(control.offsetWidth || control.offsetHeight)
        };
        fields.push(info);
      } catch (err) {
        // Skip problematic fields
      }
    });

    return { all: fields };
  }

  _resolveLabel(control) {
    // Try label[for]
    if (control.id) {
      const label = document.querySelector(`label[for="${CSS.escape(control.id)}"]`);
      if (label) return this._normalize(label.innerText || label.textContent);
    }

    // Try parent label
    const parentLabel = control.closest('label');
    if (parentLabel) return this._normalize(parentLabel.innerText);

    // Try previous sibling
    const prev = control.previousElementSibling;
    if (prev && ['label', 'span', 'div'].includes(prev.tagName.toLowerCase())) {
      return this._normalize(prev.innerText);
    }

    // Fallback to aria-label, placeholder, name
    return this._normalize(
      control.getAttribute('aria-label') || 
      control.getAttribute('placeholder') || 
      control.name || 
      control.id
    );
  }

  _normalize(text) {
    return String(text || '')
      .replace(/[\u00A0]/g, ' ')
      .replace(/[^\w\s.-]/g, '')
      .trim()
      .toLowerCase();
  }

  _profileToValues(profile) {
    return {
      full_name: profile.full_name || `${profile.first_name || ''} ${profile.last_name || ''}`.trim(),
      first_name: profile.first_name || '',
      last_name: profile.last_name || '',
      email: profile.email || '',
      phone: profile.phone || profile.mobile || '',
      linkedin: profile.linkedin || '',
      location: profile.location || profile.city || '',
      address: profile.address || '',
      github: profile.github || '',
      portfolio: profile.portfolio || '',
      cover_letter: profile.cover_letter || ''
    };
  }

  _findBestFieldFor(logicalKey) {
    const canonicalLabels = {
      full_name: ['full name', 'name', 'your name', 'applicant name'],
      first_name: ['first name', 'given name', 'fname'],
      last_name: ['last name', 'surname', 'family name', 'lname'],
      email: ['email', 'e-mail', 'email address', 'your email'],
      phone: ['phone', 'phone number', 'mobile', 'telephone', 'contact'],
      linkedin: ['linkedin', 'linkedin url', 'linkedin profile'],
      location: ['location', 'city', 'address'],
      github: ['github', 'github url'],
      portfolio: ['portfolio', 'website', 'personal website'],
      cover_letter: ['cover letter', 'additional information', 'message']
    };

    const labels = canonicalLabels[logicalKey] || [logicalKey];
    let bestField = null;
    let bestScore = 0;

    this.fieldMap.all.forEach(field => {
      const fieldText = [field.labelText, field.placeholder, field.aria, field.name, field.id]
        .map(t => this._normalize(t))
        .join(' ');

      for (const label of labels) {
        const score = this._similarity(fieldText, label);
        if (score > bestScore) {
          bestScore = score;
          bestField = field;
        }
      }
    });

    return bestScore >= CFG.FORM_FIELD_MATCH_THRESHOLD ? bestField.el : null;
  }

  _similarity(a, b) {
    const tokensA = new Set(a.split(/\s+/).filter(t => t.length > 0));
    const tokensB = new Set(b.split(/\s+/).filter(t => t.length > 0));
    
    const intersection = [...tokensA].filter(t => tokensB.has(t)).length;
    const union = new Set([...tokensA, ...tokensB]).size;
    
    return union === 0 ? 0 : intersection / union;
  }

  _setFieldValue(el, value) {
    if (!el) return;

    try {
      el.focus();

      if (el.tagName.toLowerCase() === 'select') {
        for (const opt of Array.from(el.options || [])) {
          if (this._normalize(opt.text).includes(this._normalize(value))) {
            el.value = opt.value;
            break;
          }
        }
      } else if (el.type === 'checkbox' || el.type === 'radio') {
        el.checked = !!value;
      } else {
        el.value = value;
      }

      // Trigger events
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
      el.blur();

    } catch (err) {
      warn('Failed to set field value', err);
    }
  }

  async _handleResumeUpload(form, userProfile) {
    try {
      const fileInputs = form.querySelectorAll('input[type="file"]');
      if (!fileInputs.length) return;

      // Request dynamic resume from background
      const result = await sendToBackground({
        type: 'REQUEST_DYNAMIC_RESUME',
        payload: { jobTitle: userProfile.targetJobTitle || 'Software Engineer' }
      });

      if (!result.success || !result.resume) {
        warn('Resume generation failed', result.error);
        return;
      }

      // Get resume blob
      const blobResult = await sendToBackground({
        type: 'GET_RESUME_BLOB',
        payload: { resumeId: result.resume.resume_id }
      });

      if (!blobResult.success) {
        warn('Failed to get resume blob');
        return;
      }

      // Convert base64 to File
      const base64 = blobResult.base64;
      const mime = blobResult.meta.mime || 'application/pdf';
      const fileName = blobResult.meta.fileName || 'resume.pdf';

      const byteString = atob(base64);
      const arrayBuffer = new ArrayBuffer(byteString.length);
      const uint8Array = new Uint8Array(arrayBuffer);
      
      for (let i = 0; i < byteString.length; i++) {
        uint8Array[i] = byteString.charCodeAt(i);
      }

      const blob = new Blob([uint8Array], { type: mime });
      const file = new File([blob], fileName, { type: mime });

      // Set file in first file input
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      fileInputs[0].files = dataTransfer.files;

      fileInputs[0].dispatchEvent(new Event('change', { bubbles: true }));
      info('Resume uploaded successfully');

    } catch (err) {
      warn('Resume upload failed', err);
    }
  }

  _triggerFormEvents(form) {
    try {
      form.dispatchEvent(new Event('change', { bubbles: true }));
      form.dispatchEvent(new Event('input', { bubbles: true }));
    } catch (err) {
      // Ignore
    }
  }
}

/**
 * =============================================================================
 * SUBMIT DETECTOR
 * =============================================================================
 */
class SubmitDetector {
  constructor(initialUrl) {
    this.initialUrl = initialUrl;
    this.detected = false;
    this.observer = null;
  }

  async waitForSubmission() {
    return new Promise((resolve, reject) => {
      const deadline = Date.now() + CFG.SUBMIT_DETECTION_TIMEOUT_MS;

      const checkUrl = () => {
        if (window.location.href !== this.initialUrl) {
          this.detected = true;
          cleanup();
          resolve({ method: 'url_change', url: window.location.href });
        }
      };

      const successSelectors = [
        '.application-confirmation', '.application-success', 
        '.thanks-for-applying', '.apply-success', 
        '.success-banner', '.submission-complete'
      ];

      const checkDOM = () => {
        for (const sel of successSelectors) {
          if (document.querySelector(sel)) {
            this.detected = true;
            cleanup();
            resolve({ method: 'success_selector', selector: sel });
            return true;
          }
        }

        const bodyText = (document.body?.innerText || '').toLowerCase();
        if (bodyText.includes('thank you') || bodyText.includes('application received')) {
          this.detected = true;
          cleanup();
          resolve({ method: 'text_signal' });
          return true;
        }

        return false;
      };

      this.observer = new MutationObserver(() => {
        if (!this.detected) checkDOM();
      });

      this.observer.observe(document.body, { childList: true, subtree: true });

      const interval = setInterval(() => {
        if (Date.now() > deadline) {
          cleanup();
          reject(new Error('Submit detection timeout'));
          return;
        }

        checkUrl();
        checkDOM();
      }, CFG.SUBMIT_DETECTION_POLL_MS);

      const cleanup = () => {
        clearInterval(interval);
        if (this.observer) this.observer.disconnect();
      };
    });
  }
}

/**
 * =============================================================================
 * OVERLAY UI
 * =============================================================================
 */
class Overlay {
  constructor() {
    this.root = null;
    this._create();
  }

  _create() {
    try {
      this.root = document.createElement('div');
      this.root.id = 'ai-job-agent-overlay';
      
      Object.assign(this.root.style, {
        position: 'fixed',
        right: '20px',
        bottom: '20px',
        width: '320px',
        maxWidth: '90vw',
        zIndex: '2147483647',
        boxShadow: '0 8px 24px rgba(0,0,0,0.15)',
        borderRadius: '12px',
        background: 'white',
        fontFamily: 'system-ui, -apple-system, sans-serif',
        overflow: 'hidden',
        display: 'none',
        border: '1px solid #e0e0e0'
      });

      document.body.appendChild(this.root);
    } catch (err) {
      warn('Overlay creation failed', err);
    }
  }

  show(html) {
    if (!this.root) return;
    this.root.innerHTML = html;
    this.root.style.display = 'block';
  }

  hide() {
    if (!this.root) return;
    this.root.style.display = 'none';
  }

  showQuickActions(jobData) {
    const html = `
      <div style="padding: 16px;">
        <h3 style="margin: 0 0 12px 0; font-size: 16px; font-weight: 600;">🤖 AI Assistant</h3>
        <div style="font-size: 13px; color: #666; margin-bottom: 12px;">
          Job detected: <strong>${jobData.title || 'Unknown'}</strong>
        </div>
        <div style="display: flex; flex-direction: column; gap: 8px;">
          <button id="ai-analyze-btn" style="padding: 10px; border: none; border-radius: 6px; background: #2563eb; color: white; cursor: pointer; font-size: 14px; font-weight: 500;">
            🧠 Analyze Job
          </button>
          <button id="ai-autofill-btn" style="padding: 10px; border: none; border-radius: 6px; background: #10b981; color: white; cursor: pointer; font-size: 14px; font-weight: 500;">
            📝 Auto-Fill Form
          </button>
          <button id="ai-close-btn" style="padding: 8px; border: 1px solid #e0e0e0; border-radius: 6px; background: white; color: #666; cursor: pointer; font-size: 13px;">
            Close
          </button>
        </div>
      </div>
    `;
    this.show(html);

    // Attach listeners
    setTimeout(() => {
      document.getElementById('ai-analyze-btn')?.addEventListener('click', () => {
        window.aiContentScript?.analyzeCurrentJob();
      });
      document.getElementById('ai-autofill-btn')?.addEventListener('click', () => {
        window.aiContentScript?.autoFillForm();
      });
      document.getElementById('ai-close-btn')?.addEventListener('click', () => {
        this.hide();
      });
    }, 100);
  }
}

/**
 * =============================================================================
 * MAIN CONTENT CONTROLLER
 * =============================================================================
 */
class AIJobContent {
  constructor() {
    this.platformInfo = { platform: 'unknown', type: 'other' };
    this.jobData = null;
    this.formAutofill = new FormAutofill();
    this.overlay = new Overlay();
    this.userProfile = null;
    
    this.init();
  }

  async init() {
    try {
      info('Content script initializing...');

      // Detect platform
      this.platformInfo = PlatformDetector.detect();
      debug('Platform:', this.platformInfo);

      // Load user profile
      this.userProfile = await this.loadUserProfile();

      // If job detail page, extract and notify
      if (this.platformInfo.type === 'job_detail') {
        await this.extractAndNotifyJob();
      }

      // Set up message listener
      chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
        this.handleMessage(msg, sender, sendResponse);
        return true;
      });

      // Expose API for debugging
      window.aiContentScript = {
        analyzeCurrentJob: () => this.analyzeCurrentJob(),
        autoFillForm: () => this.autoFillForm(),
        extractJob: () => this.extractAndNotifyJob()
      };

      info('Content script ready');

    } catch (err) {
      error('Init failed', err);
    }
  }

  async loadUserProfile() {
    const result = await sendToBackground({ type: 'GET_USER_PROFILE' });
    return result.success ? result.profile : {};
  }

  async extractAndNotifyJob() {
    try {
      this.jobData = JobExtractor.extract(this.platformInfo.platform);
      
      if (!this.jobData) {
        debug('No job data extracted');
        return;
      }

      // Notify background
      await sendToBackground({
        type: 'JOB_DETECTED',
        payload: { jobData: this.jobData }
      });

      // Show overlay
      this.overlay.showQuickActions(this.jobData);

      info('Job detected and notified');

    } catch (err) {
      warn('Job extraction failed', err);
    }
  }

  async analyzeCurrentJob() {
    try {
      if (!this.jobData) {
        this.jobData = JobExtractor.extract(this.platformInfo.platform);
      }

      if (!this.jobData) {
        alert('No job data found on this page');
        return;
      }

      // Show loading
      this.overlay.show(`
        <div style="padding: 20px; text-align: center;">
          <div style="font-size: 24px; margin-bottom: 12px;">🧠</div>
          <div style="font-size: 14px; color: #666;">Analyzing job with AI...</div>
        </div>
      `);

      // Build prompt
      const prompt = `Analyze this job for my profile and provide:
1. Match score (0-100)
2. Key matching skills
3. Missing skills I should highlight
4. Suggested improvements

Job Details:
Title: ${this.jobData.title}
Company: ${this.jobData.company}
Location: ${this.jobData.location}
Description: ${this.jobData.description.slice(0, 2000)}

My Profile:
Name: ${this.userProfile.full_name || 'Not set'}
Skills: ${this.userProfile.skills || 'Not specified'}
Experience: ${this.userProfile.experience || 'Not specified'}

Provide a concise analysis in bullet points.`;

      // Call MCP
      const result = await sendToBackground({
        type: 'MCP_COMPLETE',
        payload: {
          sessionName: `job_${this.jobData.url.slice(-20)}`,
          taskType: 'job_analysis',
          prompt,
          meta: { job_url: this.jobData.url }
        }
      });

      if (result.success) {
        this.overlay.show(`
          <div style="padding: 16px; max-height: 400px; overflow-y: auto;">
            <h3 style="margin: 0 0 12px 0; font-size: 16px; font-weight: 600;">📊 AI Analysis</h3>
            <div style="font-size: 13px; line-height: 1.6; white-space: pre-wrap;">${result.completion}</div>
            <button id="ai-close-analysis" style="margin-top: 12px; padding: 8px 16px; border: none; border-radius: 6px; background: #e0e0e0; cursor: pointer; width: 100%;">
              Close
            </button>
          </div>
        `);

        setTimeout(() => {
          document.getElementById('ai-close-analysis')?.addEventListener('click', () => {
            this.overlay.hide();
          });
        }, 100);

      } else {
        this.overlay.show(`
          <div style="padding: 16px;">
            <div style="color: #ef4444; font-size: 14px;">Analysis failed: ${result.error || 'Unknown error'}</div>
            <button id="ai-close-error" style="margin-top: 12px; padding: 8px 16px; border: none; border-radius: 6px; background: #e0e0e0; cursor: pointer; width: 100%;">
              Close
            </button>
          </div>
        `);

        setTimeout(() => {
          document.getElementById('ai-close-error')?.addEventListener('click', () => {
            this.overlay.hide();
          });
        }, 100);
      }

    } catch (err) {
      error('Job analysis failed', err);
      alert('Analysis failed. Check console for details.');
    }
  }

  async autoFillForm() {
    try {
      info('Starting autofill...');
      
      const result = await this.formAutofill.fillForm(this.userProfile, { enabled: true });
      
      if (result.success) {
        alert(`✅ Auto-fill complete! ${result.filledCount} fields filled.`);
      } else {
        alert(`❌ Auto-fill failed: ${result.error}`);
      }

    } catch (err) {
      error('Autofill failed', err);
      alert('Auto-fill failed. Check console for details.');
    }
  }

  handleMessage(msg, sender, sendResponse) {
    switch (msg.type) {
      case 'ANALYZE_JOB':
        this.analyzeCurrentJob();
        sendResponse({ success: true });
        break;

      case 'AUTO_FILL_FORM':
        this.autoFillForm();
        sendResponse({ success: true });
        break;

      case 'TOGGLE_SIDEBAR':
        if (this.overlay.root.style.display === 'none') {
          this.overlay.showQuickActions(this.jobData || {});
        } else {
          this.overlay.hide();
        }
        sendResponse({ success: true });
        break;

      case 'TAB_UPDATED':
        // Re-detect on navigation
        this.platformInfo = PlatformDetector.detect();
        if (this.platformInfo.type === 'job_detail') {
          this.extractAndNotifyJob();
        }
        sendResponse({ success: true });
        break;

      default:
        sendResponse({ success: false, error: 'Unknown message type' });
    }
  }
}

/**
 * =============================================================================
 * INITIALIZE
 * =============================================================================
 */
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    new AIJobContent();
  });
} else {
  new AIJobContent();
}
