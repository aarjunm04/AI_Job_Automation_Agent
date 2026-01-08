// content/content.js
/**
 * =============================================================================
 * AI JOB AUTOMATION AGENT - CONTENT SCRIPT (Production-grade)
 * =============================================================================
 * Responsibilities:
 * - Detect platform & page type
 * - Extract job metadata (title, company, location, url)
 * - Provide an AutoFill engine that maps user profile -> arbitrary forms
 * - Trigger resume retrieval from RAG endpoint and upload blobs into form fields
 * - Detect successful submission reliably (URL change, DOM mutation, success banners)
 * - Communicate with background script (MCP / Notion logging)
 *
 * Principles:
 * - No globals (except window.aiContentScript minimal controlled API)
 * - All background/content interactions through chrome.runtime.sendMessage
 * - Defensive programming: timeouts + retries + error boundaries
 * - Modular, testable functions
 *
 * Author: AI Job Automation Team (upgraded)
 * Version: 2.0.0
 * =============================================================================
 */

(() => {
  'use strict';

  /* ===========================
     CONFIG
     =========================== */
  const CFG = {
    DEBUG: false, // set true temporarily during dev
    RAG_RESUME_ENDPOINT: 'http://localhost:8080/rag/dynamic_resume', // GET ?job_title=...
    RESUME_DOWNLOAD_TIMEOUT_MS: 20000,
    FETCH_RETRY_COUNT: 2,
    FETCH_RETRY_BACKOFF_MS: 800,
    FORM_FIELD_MATCH_THRESHOLD: 0.6, // fuzzy match threshold
    SUBMIT_DETECTION_POLL_MS: 800,
    SUBMIT_DETECTION_TIMEOUT_MS: 25_000,
    MAX_UPLOAD_BYTES: 10 * 1024 * 1024, // 10 MB limit for resume blob
    LOG_PREFIX: '[AI-CONTENT]',
  };

  /* ===========================
     UTILS
     =========================== */
  function debug(...args) {
    if (CFG.DEBUG) console.debug(CFG.LOG_PREFIX, ...args);
  }
  function info(...args) {
    console.info(CFG.LOG_PREFIX, ...args);
  }
  function warn(...args) {
    console.warn(CFG.LOG_PREFIX, ...args);
  }
  function err(...args) {
    console.error(CFG.LOG_PREFIX, ...args);
  }

  function timeoutPromise(ms, promise) {
    return Promise.race([
      promise,
      new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), ms))
    ]);
  }

  async function fetchWithRetries(url, opts = {}, retries = CFG.FETCH_RETRY_COUNT) {
    let attempt = 0;
    while (true) {
      try {
        const result = await timeoutPromise(CFG.RESUME_DOWNLOAD_TIMEOUT_MS, fetch(url, opts));
        if (!result.ok) throw new Error(`HTTP ${result.status}`);
        return result;
      } catch (e) {
        if (attempt >= retries) throw e;
        attempt++;
        await new Promise(r => setTimeout(r, CFG.FETCH_RETRY_BACKOFF_MS * attempt));
      }
    }
  }

  // simple normalized string helper
  function normalizeLabel(s) {
    return String(s || '')
      .replace(/[\u00A0]/g, ' ')
      .replace(/[^\w\s.-]/g, '')
      .trim()
      .toLowerCase();
  }

  // Jaro-Winkler-like or simple similarity: here we implement token overlap ratio
  function tokenSimilarity(a, b) {
    a = normalizeLabel(a);
    b = normalizeLabel(b);
    if (!a || !b) return 0;
    const as = Array.from(new Set(a.split(/\s+/)));
    const bs = Array.from(new Set(b.split(/\s+/)));
    const inter = as.filter(x => bs.includes(x)).length;
    const union = new Set([...as, ...bs]).size;
    return union === 0 ? 0 : inter / union;
  }

  /* ===========================
     PLATFORM DETECTION
     =========================== */
  class PlatformDetector {
    static detect() {
      const hostname = window.location.hostname.toLowerCase();
      const pathname = window.location.pathname.toLowerCase();

      // quick winners
      if (hostname.includes('linkedin.com')) {
        if (pathname.includes('/jobs/view/') || document.querySelector('[data-job-id], .jobs-description')) {
          return { platform: 'linkedin', type: 'job_detail' };
        }
        return { platform: 'linkedin', type: 'other' };
      }

      if (hostname.includes('indeed.com')) {
        if (pathname.includes('/viewjob') || document.querySelector('.jobsearch-JobComponent')) {
          return { platform: 'indeed', type: 'job_detail' };
        }
        return { platform: 'indeed', type: 'other' };
      }

      if (hostname.includes('naukri.com')) {
        if (document.querySelector('.jd-container') || pathname.includes('/job-listings/')) {
          return { platform: 'naukri', type: 'job_detail' };
        }
        return { platform: 'naukri', type: 'other' };
      }

      // generic job-site heuristics (Workday/Greenhouse/Lever etc.)
      const jobSelectors = [
        '.job-details', '.job-description', '.posting-headline', '.posting', '[data-job-id]',
        '#job-desc', '.job-meta'
      ];
      if (jobSelectors.some(sel => document.querySelector(sel))) {
        return { platform: 'generic_job_site', type: 'job_detail' };
      }

      return { platform: 'unknown', type: 'other' };
    }
  }

  /* ===========================
     JOB DATA EXTRACTOR - resilient
     =========================== */
  class JobDataExtractor {
    static extract(platform) {
      try {
        const url = window.location.href;
        let title = '';
        let company = '';
        let location = '';
        let description = '';

        // Platform-specific first (higher reliability)
        if (platform === 'linkedin') {
          title = document.querySelector('.topcard__title, .jobs-unified-top-card__job-title')?.innerText || title;
          company = document.querySelector('.topcard__org-name-link, .jobs-unified-top-card__company-name')?.innerText || company;
          location = document.querySelector('.topcard__flavor--bullet, .jobs-unified-top-card__workplace-type')?.innerText || location;
          description = document.querySelector('.description__text, .jobs-description__container')?.innerText || '';
        } else if (platform === 'indeed') {
          title = document.querySelector('.jobsearch-JobInfoHeader-title')?.innerText || title;
          company = document.querySelector('.jobsearch-InlineCompanyRating div')?.innerText || company;
          location = document.querySelector('.jobsearch-JobInfoHeader-subtitle div')?.innerText || location;
          description = document.querySelector('#jobDescriptionText')?.innerText || '';
        } else if (platform === 'naukri') {
          title = document.querySelector('.jd-header h1')?.innerText || title;
          company = document.querySelector('.jd-header .companyName a')?.innerText || company;
          description = document.querySelector('.job-desc .dang')?.innerText || '';
        } else {
          // Generic fallback
          title = document.querySelector('h1')?.innerText ||
                  document.querySelector('[data-job-title]')?.innerText ||
                  document.title;
          company = document.querySelector('[data-company]')?.innerText ||
                    document.querySelector('.company')?.innerText || '';
          const descrEl = document.querySelector('.job-description, .description, [role="main"]');
          description = descrEl?.innerText || '';
        }

        title = (title || '').trim();
        company = (company || '').trim();
        location = (location || '').trim();

        const jobData = {
          title,
          company,
          location,
          url,
          description,
          extracted_at: new Date().toISOString()
        };

        return jobData;
      } catch (e) {
        warn('JobDataExtractor failed', e);
        return null;
      }
    }
  }

  /* ===========================
     FORM AUTO-FILLER
     - scans for input/textarea/select/file fields
     - builds label -> input mapping using DOM heuristics
     - fuzzy matches label text (placeholder, aria-label, preceding <label>, sibling text)
     - supports resume file upload via RAG endpoint (downloads blob, sets File in input)
     =========================== */
  class FormAutoFiller {
    constructor() {
      this.fieldMap = null;
    }

    async fillForm(userProfile = {}, resumeMode = { mode: 'auto' } ) {
      try {
        const form = this._findMainForm();
        if (!form) throw new Error('No form found on page');

        this.fieldMap = this._scanFormFields(form);

        // Prepare values
        const values = this._profileToValues(userProfile);

        // Fill text fields
        for (const [logical, val] of Object.entries(values)) {
          const input = this._findBestFieldFor(logical);
          if (input) {
            this._setFieldValue(input, val);
            debug('filled field', logical, '->', input, val);
          }
        }

        // Handle resume upload separately (async)
        if (resumeMode && resumeMode.mode) {
          await this._handleResumeUpload(form, resumeMode, userProfile);
        }

        // optional: trigger change events to ensure frameworks pick up
        this._triggerEventsForForm(form);

        return { success: true, filled: true };
      } catch (e) {
        err('AutoFill error', e);
        return { success: false, error: e.message };
      }
    }

    _findMainForm() {
      // heuristics: find largest <form> or the one containing resume/upload, or first form
      const forms = Array.from(document.forms || []);
      if (!forms.length) return null;

      // prefer form with file input or many inputs
      let best = forms[0];
      let bestScore = 0;
      forms.forEach(f => {
        const fileInputs = f.querySelectorAll('input[type="file"]').length;
        const inputsCount = f.querySelectorAll('input, textarea, select').length;
        const score = fileInputs * 10 + inputsCount;
        if (score > bestScore) {
          best = f;
          bestScore = score;
        }
      });
      return best;
    }

    _scanFormFields(form) {
      const fields = [];
      const controls = form.querySelectorAll('input, textarea, select, [contenteditable="true"]');

      controls.forEach(control => {
        try {
          const type = control.tagName.toLowerCase() === 'input' ? (control.type || 'text') : control.tagName.toLowerCase();
          const info = {
            el: control,
            tag: control.tagName.toLowerCase(),
            type,
            name: control.name || '',
            id: control.id || '',
            placeholder: control.getAttribute('placeholder') || '',
            aria: control.getAttribute('aria-label') || '',
            labelText: this._resolveLabelText(control),
            visible: !!(control.offsetWidth || control.offsetHeight || control.getClientRects().length)
          };
          fields.push(info);
        } catch (e) {
          // ignore fields that throw
        }
      });

      // build lookup maps
      const map = {
        all: fields,
        byName: new Map(),
        byId: new Map()
      };
      fields.forEach(f => {
        if (f.name) map.byName.set(f.name, f);
        if (f.id) map.byId.set(f.id, f);
      });
      return map;
    }

    _resolveLabelText(control) {
      // check for label[for]
      try {
        if (control.id) {
          const lbl = document.querySelector(`label[for="${CSS.escape(control.id)}"]`);
          if (lbl) return normalizeLabel(lbl.innerText || lbl.textContent || lbl.innerHTML);
        }
      } catch(_) {}

      // check closest <label> ancestor
      const ancestorLabel = control.closest('label');
      if (ancestorLabel) return normalizeLabel(ancestorLabel.innerText || '');

      // check previous sibling text nodes
      const prev = control.previousElementSibling;
      if (prev && (prev.tagName.toLowerCase() === 'label' || prev.tagName.toLowerCase() === 'span' || prev.tagName.toLowerCase() === 'div')) {
        return normalizeLabel(prev.innerText || '');
      }

      // aria or placeholder
      const aria = control.getAttribute('aria-label');
      if (aria) return normalizeLabel(aria);
      const ph = control.getAttribute('placeholder');
      if (ph) return normalizeLabel(ph);

      // fallback to name/id
      return normalizeLabel(control.name || control.id || '');
    }

    _profileToValues(profile) {
      // transform a user profile into a small set of canonical values the autofiller knows
      // the profile format is expected to be { full_name, first_name, last_name, email, phone, mobile, linkedin, location, city, state, country, resume_files: {...} }
      const v = {};
      v.full_name = profile.full_name || `${profile.first_name || ''} ${profile.last_name || ''}`.trim();
      v.first_name = profile.first_name || (v.full_name ? v.full_name.split(' ')[0] : '');
      v.last_name = profile.last_name || (v.full_name ? v.full_name.split(' ').slice(1).join(' ') : '');
      v.email = profile.email || profile.primary_email || '';
      v.phone = profile.phone || profile.mobile || profile.contact || '';
      v.linkedin = profile.linkedin || profile.linkedin_url || '';
      v.location = profile.location || profile.city || '';
      v.address = profile.address || '';
      v.github = profile.github || '';
      v.portfolio = profile.portfolio || '';
      // cover letter / summary
      v.cover_letter = profile.cover_letter || '';
      return v;
    }

    _findBestFieldFor(logicalKey) {
      if (!this.fieldMap) return null;
      const canonicalLabels = {
        full_name: ['full name', 'name', 'applicant name', 'your name'],
        first_name: ['first name', 'given name'],
        last_name: ['last name', 'surname', 'family name'],
        email: ['email', 'e-mail', 'email address', 'your email'],
        phone: ['phone', 'phone number', 'mobile', 'mobile number', 'contact number'],
        linkedin: ['linkedin', 'linkedin url', 'linkedin profile'],
        location: ['location', 'city', 'address'],
        github: ['github', 'github url'],
        cover_letter: ['cover letter', 'coverletter', 'additional information']
      };

      const candidates = this.fieldMap.all;
      const labels = canonicalLabels[logicalKey] || [logicalKey];

      let best = null;
      let bestScore = 0;

      candidates.forEach(c => {
        const labelCandidates = [c.labelText, c.placeholder, c.aria, c.name, c.id].map(x => normalizeLabel(x)).join(' ');
        for (const lab of labels) {
          const sim = tokenSimilarity(labelCandidates, lab);
          if (sim > bestScore) {
            bestScore = sim;
            best = c;
          }
        }
      });

      if (bestScore >= CFG.FORM_FIELD_MATCH_THRESHOLD) return best.el;
      return null;
    }

    _setFieldValue(el, value) {
      if (!el) return;
      try {
        // handle contenteditable
        if (el.getAttribute && el.getAttribute('contenteditable') === 'true') {
          el.focus();
          el.innerText = value;
        } else if (el.tagName.toLowerCase() === 'select') {
          // try to set option that matches
          for (const opt of Array.from(el.options || [])) {
            if (normalizeLabel(opt.text).includes(normalizeLabel(value))) {
              el.value = opt.value;
              break;
            }
          }
        } else if (el.type === 'checkbox' || el.type === 'radio') {
          el.checked = !!value;
        } else {
          el.focus();
          el.value = value;
        }
        // dispatch input/change events
        ['input','change'].forEach(evt => {
          try { el.dispatchEvent(new Event(evt, { bubbles: true })); } catch(_) {}
        });
      } catch (e) {
        warn('setFieldValue failed', e);
      }
    }

    async _handleResumeUpload(form, resumeMode, profile) {
      // find file input
      try {
        const fileInputInfo = this.fieldMap.all.find(f => f.type === 'file' || (f.labelText && /resume|cv|upload/i.test(f.labelText)));
        if (!fileInputInfo) {
          debug('No file input found on form for resume upload');
          return;
        }
        const fileInput = fileInputInfo.el;

        // Only proceed for file inputs (some sites use custom upload UI; we'll try to handle input[type=file] first)
        if (fileInput.tagName.toLowerCase() !== 'input' || fileInput.type !== 'file') {
          debug('File input found but not a native file input - skip automatic upload');
          return;
        }

        // determine which resume to use
        let resumeBlob;
        if (resumeMode.mode === 'manual' && resumeMode.manualBlob) {
          resumeBlob = resumeMode.manualBlob;
        } else if (resumeMode.mode === 'auto' && profile && profile.preferred_resume) {
          // if user stored resume blob in profile (rare)
          resumeBlob = profile.preferred_resume;
        } else {
          // contact RAG to get resume file for this job title
          const jobTitle = (profile.job_title || profile.target || (document.title || '')) ;
          resumeBlob = await this._downloadResumeFromRag(jobTitle);
        }

        if (!resumeBlob) {
          warn('No resume blob available to upload');
          return;
        }

        // validate size
        if (resumeBlob.size > CFG.MAX_UPLOAD_BYTES) {
          warn('Resume exceeds max allowable size, skipping upload');
          return;
        }

        // create File instance if needed
        let fileObj;
        if (resumeBlob instanceof File) {
          fileObj = resumeBlob;
        } else {
          // guess filename
          const filename = `resume_${(profile && (profile.full_name || 'candidate')).replace(/\s+/g,'_')}.pdf`;
          fileObj = new File([resumeBlob], filename, { type: resumeBlob.type || 'application/pdf', lastModified: Date.now() });
        }

        // programmatically set file input - requires DataTransfer
        const dt = new DataTransfer();
        dt.items.add(fileObj);
        fileInput.files = dt.files;

        // trigger change event
        fileInput.dispatchEvent(new Event('change', { bubbles: true }));

        debug('Resume uploaded to native file input');
        return true;
      } catch (e) {
        warn('Resume upload handler failed', e);
        return false;
      }
    }

    async _downloadResumeFromRag(jobTitle) {
      // Job title should be URL-encoded
      try {
        const q = encodeURIComponent(jobTitle || '');
        const url = `${CFG.RAG_RESUME_ENDPOINT}?job_title=${q}`;

        info('Requesting resume from RAG:', url);
        const res = await fetchWithRetries(url, { method: 'GET' }, CFG.FETCH_RETRY_COUNT);
        // Expect JSON { filename, mime_type, base64 } or direct blob stream
        const contentType = res.headers.get('content-type') || '';
        if (contentType.includes('application/json')) {
          const json = await res.json();
          if (json.base64) {
            const bytes = atob(json.base64);
            const arr = new Uint8Array(bytes.length);
            for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
            const blob = new Blob([arr.buffer], { type: json.mime_type || 'application/pdf' });
            return blob;
          } else if (json.url) {
            // remote file url - attempt to download
            const fileResp = await fetchWithRetries(json.url, {}, CFG.FETCH_RETRY_COUNT);
            return await fileResp.blob();
          }
        } else {
          // If response is a direct file stream
          return await res.blob();
        }
      } catch (e) {
        warn('RAG resume download failed', e);
        return null;
      }
    }

    _triggerEventsForForm(form) {
      try {
        ['input', 'change', 'blur'].forEach(ev => {
          form.querySelectorAll('input, textarea, select').forEach(el => {
            try { el.dispatchEvent(new Event(ev, { bubbles: true })); } catch(_) {}
          });
        });
      } catch (e) {
        // ignore
      }
    }
  }

  /* ===========================
     SUBMISSION DETECTOR
     - uses multiple strategies:
       1) URL change
       2) DOM mutation for success banners / confirmations
       3) presence of known success selectors
     =========================== */
  class SubmissionDetector {
    constructor() {
      this._initialUrl = window.location.href;
      this._mutationObserver = null;
      this._detected = false;
    }

    async waitForSubmission(timeout = CFG.SUBMIT_DETECTION_TIMEOUT_MS) {
      return new Promise((resolve, reject) => {
        const deadline = Date.now() + timeout;
        const checkUrl = () => {
          if (window.location.href !== this._initialUrl) {
            this._detected = true;
            cleanup();
            resolve({ method: 'url_change', url: window.location.href });
          }
        };

        const knownSuccessSelectors = [
          '.application-confirmation', '.application-success', '.thanks-for-applying', 
          '.apply-success', '.success-banner', '.submission-complete', '#application-confirmation'
        ];

        const checkDOM = () => {
          for (const sel of knownSuccessSelectors) {
            if (document.querySelector(sel)) {
              this._detected = true;
              cleanup();
              return resolve({ method: 'success_selector', selector: sel });
            }
          }
          // also check for large visible banners with 'thank' 'appl' etc
          const bodyText = document.body && document.body.innerText ? document.body.innerText.toLowerCase() : '';
          if (bodyText.includes('thank you') || bodyText.includes('application received') || bodyText.includes('we have received')) {
            this._detected = true;
            cleanup();
            return resolve({ method: 'text_signal' });
          }
          return false;
        };

        const onMutations = (mutationsList) => {
          if (this._detected) return;
          // quick heuristic: if modal or success node added
          for (const m of mutationsList) {
            if (m.addedNodes && m.addedNodes.length) {
              if (checkDOM()) return;
            }
          }
        };

        // Mutation Observer
        this._mutationObserver = new MutationObserver(onMutations);
        this._mutationObserver.observe(document.body, { childList: true, subtree: true });

        // Polling URL and DOM
        const interval = setInterval(() => {
          if (Date.now() > deadline) {
            cleanup();
            return reject(new Error('submit_detection_timeout'));
          }
          checkUrl();
          if (checkDOM()) return;
        }, CFG.SUBMIT_DETECTION_POLL_MS);

        function cleanup() {
          try {
            clearInterval(interval);
            if (this && this._mutationObserver) {
              this._mutationObserver.disconnect();
            }
          } catch (_) {}
        }
        // bind cleanup to local closure
        cleanup = cleanup.bind(this);
      });
    }
  }

  /* ===========================
     MESSAGING UTIL (content <-> background)
     =========================== */
  async function sendToBackground(message) {
    try {
      return await new Promise((resolve) => {
        chrome.runtime.sendMessage(message, (resp) => {
          resolve(resp);
        });
      });
    } catch (e) {
      warn('sendToBackground failed', e);
      return null;
    }
  }

  /* ===========================
     INSIGHTS OVERLAY - minimal, non-intrusive
     =========================== */
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
          right: '16px',
          bottom: '16px',
          width: '320px',
          maxWidth: '40%',
          zIndex: '2147483647',
          boxShadow: '0 8px 20px rgba(0,0,0,0.2)',
          borderRadius: '10px',
          background: '#fff',
          color: '#111',
          fontFamily: 'Inter, system-ui, Arial, sans-serif',
          overflow: 'hidden',
          opacity: '0.98',
          display: 'none'
        });
        document.body.appendChild(this.root);
      } catch (e) {
        // skip overlay if we can't create
      }
    }
    show(htmlContent) {
      if (!this.root) return;
      this.root.innerHTML = htmlContent;
      this.root.style.display = 'block';
    }
    hide() {
      if (!this.root) return;
      this.root.style.display = 'none';
    }
  }

  /* ===========================
     MAIN CONTENT CLASS (coordinates everything)
     =========================== */
  class AIJobAutomationContent {
    constructor() {
      this.platformInfo = { platform: 'unknown', type: 'other' };
      this.jobData = null;
      this.formAutoFiller = new FormAutoFiller();
      this.overlay = new Overlay();
      this.init().catch(e => err('init failed', e));
    }

    async init() {
      try {
        info('Content script starting');
        this.platformInfo = PlatformDetector.detect();
        debug('platformInfo', this.platformInfo);

        if (this.platformInfo.platform === 'unknown') {
          info('Platform unknown; content script will still provide form autofill if manual trigger occurs');
        } else {
          info('Detected platform:', this.platformInfo.platform, this.platformInfo.type);
        }

        // expose minimal control API for debugging/testing (safe)
        window.aiContentScript = {
          autoFillForm: async (profile = null, resumeMode = { mode: 'auto' }) => {
            const userProfile = profile || await this._getUserProfile();
            return this.formAutoFiller.fillForm(userProfile, resumeMode);
          },
          extractJobData: () => this._maybeExtractJobData(),
          openSidebar: () => sendToBackground({ type: 'OPEN_SIDEBAR' })
        };

        // if job detail page, extract and inform background
        if (this.platformInfo.type === 'job_detail') {
          await this._maybeExtractJobData();
        }

        // start listening to commands from background/sidebar
        this._setupRuntimeMessageHandler();

        // optionally show a small overlay button allowing manual fill
        this._insertQuickControls();

        info('Content script ready');
      } catch (e) {
        err('AI content init error', e);
      }
    }

    async _maybeExtractJobData() {
      try {
        const data = JobDataExtractor.extract(this.platformInfo.platform);
        if (!data) return null;
        this.jobData = data;
        // notify background that page contains a job
        await sendToBackground({ type: 'JOB_DETECTED', jobData: this.jobData });
        return this.jobData;
      } catch (e) {
        warn('extractJobData failed', e);
        return null;
      }
    }

    async _getUserProfile() {
      try {
        const resp = await sendToBackground({ type: 'GET_USER_PROFILE' });
        return (resp && resp.profile) ? resp.profile : {};
      } catch (e) {
        return {};
      }
    }

    _setupRuntimeMessageHandler() {
      chrome.runtime.onMessage.addListener(async (message, sender, sendResponse) => {
        try {
          debug('MSG received:', message);
          switch (message.type) {
            case 'AUTO_FILL_FORM':
              {
                const profile = await this._getUserProfile();
                const result = await this.formAutoFiller.fillForm(profile, message.resumeMode || { mode: 'auto' });
                sendResponse({ success: true, result });
              }
              break;
            case 'EXTRACT_JOB_DATA':
              {
                const jd = await this._maybeExtractJobData();
                sendResponse({ success: !!jd, jobData: jd });
              }
              break;
            case 'TOGGLE_SIDEBAR':
              {
                await sendToBackground({ type: 'OPEN_SIDEBAR' });
                sendResponse({ success: true });
              }
              break;
            default:
              sendResponse({ success: false, error: 'unknown_message' });
          }
        } catch (e) {
          err('runtime message handler error', e);
          sendResponse({ success: false, error: e.message });
        }
        // indicate we'll call sendResponse asynchronously if needed
        return true;
      });
    }

    _insertQuickControls() {
      try {
        // create unobtrusive button -> shows overlay control to quick autofill/analysis
        const btn = document.createElement('button');
        btn.id = 'ai-quick-autofill-btn';
        btn.title = 'AI Job Agent — quick controls';
        Object.assign(btn.style, {
          position: 'fixed',
          right: '20px',
          bottom: '96px',
          zIndex: 2147483647,
          width: '56px',
          height: '56px',
          borderRadius: '28px',
          border: 'none',
          boxShadow: '0 4px 14px rgba(0,0,0,0.25)',
          background: '#0ea5a4',
          color: '#fff',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          fontSize: '18px'
        });
        btn.innerText = 'AI';
        btn.addEventListener('click', async (e) => {
          e.preventDefault();
          try {
            const profile = await this._getUserProfile();
            // show minimal overlay
            const content = `
              <div style="padding:14px;">
                <div style="font-weight:600;margin-bottom:8px;">AI Job Assistant</div>
                <div style="display:flex; gap:8px;">
                  <button id="ai-autofill-btn" style="flex:1;padding:8px;border-radius:6px;border:none;background:#2563eb;color:#fff;">Auto-fill</button>
                  <button id="ai-toggle-sidebar-btn" style="flex:1;padding:8px;border-radius:6px;border:1px solid #e5e7eb;background:#fff;color:#111;">Sidebar</button>
                </div>
                <div style="margin-top:8px;font-size:12px;color:#6b7280;">Profile: ${profile.full_name ? profile.full_name : 'No profile loaded'}</div>
              </div>
            `;
            this.overlay.show(content);
            document.getElementById('ai-autofill-btn').addEventListener('click', async () => {
              this.overlay.hide();
              await this.formAutoFiller.fillForm(profile, { mode: 'auto' });
            });
            document.getElementById('ai-toggle-sidebar-btn').addEventListener('click', async () => {
              this.overlay.hide();
              await sendToBackground({ type: 'OPEN_SIDEBAR' });
            });
          } catch (e) {
            warn('quick control failed', e);
          }
        });
        document.body.appendChild(btn);
      } catch (e) {
        // ignore if DOM blocked
      }
    }

    // called when user presses 'apply' - we will wait for submit detection
    async handleManualApplyEvent(metadata = {}) {
      try {
        // notify background an apply has started
        await sendToBackground({ type: 'APPLY_STARTED', metadata, jobData: this.jobData });

        // wait for submission detection
        const detector = new SubmissionDetector();
        const result = await detector.waitForSubmission();
        debug('submission detected:', result);

        // Notify background about successful apply
        await sendToBackground({
          type: 'APPLY_COMPLETED',
          metadata: {
            detected: result,
            jobUrl: this.jobData ? this.jobData.url : window.location.href,
            timestamp: new Date().toISOString()
          }
        });

        // attempt to fire post-apply UI updates or toast
        this._showAppliedToast();

        return { success: true, detected: result };
      } catch (e) {
        warn('apply detection failed', e);
        await sendToBackground({
          type: 'APPLY_FAILED',
          metadata: {
            error: e.message,
            jobUrl: this.jobData ? this.jobData.url : window.location.href,
            timestamp: new Date().toISOString()
          }
        });
        return { success: false, error: e.message };
      }
    }

    _showAppliedToast() {
      try {
        const toast = document.createElement('div');
        Object.assign(toast.style, {
          position: 'fixed',
          bottom: '22px',
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 2147483647,
          background: '#111827',
          color: '#fff',
          padding: '10px 16px',
          borderRadius: '8px',
          boxShadow: '0 8px 24px rgba(0,0,0,0.3)'
        });
        toast.innerText = 'Application detected — logged to Notion';
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 4200);
      } catch (_) {}
    }
  }

  // instantiate the content assistant
  const assistant = new AIJobAutomationContent();

  // For debugging ease, expose a small console command if debug true
  if (CFG.DEBUG) {
    window.__ai_content_debug = {
      assistant,
      CFG,
      tokenSimilarity
    };
  }

})();