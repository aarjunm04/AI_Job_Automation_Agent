/**
 * sidebar/sidebar.js
 * AI Job Automation Agent — Sidebar Controller (Enterprise v1, Option B - Self-contained)
 *
 * Purpose:
 * - Heavy, production-ready JS controller for the sidebar UI
 * - Embedded storage + API helpers (no external utils required)
 * - Integrates with background: RAG, MCP, Notion, apply logging, resume retrieval
 *
 * Features:
 * - Profile management (sync with chrome.storage.sync & background)
 * - Resume manager (Auto (RAG) vs Manual 1..7)
 * - Activity log (persistent, paginated)
 * - Status indicators for MCP / RAG / Notion
 * - Quick actions: Auto-fill, Log Apply, Fetch RAG Resume
 * - Defensive: tolerant selectors, offline handling, async/await everywhere
 *
 * Version: 2025.11.v1-optionB (self-contained)
 */
(function () {
  'use strict';

  // =========================
  // CONFIG
  // =========================
  const CFG = {
    DEBUG: false,
    LOG_PREFIX: '[AI-SIDEBAR]',
    STORAGE_KEYS: {
      PROFILE: 'ai_user_profile',
      RESUMES_META: 'ai_resumes_meta',
      RESUME_BLOBS: 'ai_resume_blobs', // in local storage
      ACTIVITY_LOG: 'ai_activity_log',
      EXT_STATE: 'ai_extension_state',
      API_CONFIG: 'ai_api_config'
    },
    ACTIVITY_PAGE_SIZE: 20,
    TOAST_DURATION: 4500,
    MAX_RESUMES_SHOW: 7,
    API_TIMEOUT_MS: 12_000,
    RAG_REFRESH_COOLDOWN_MS: 60_000,
    MAX_ACTIVITY_ENTRIES: 500
  };

  // =========================
  // LOGGING / UTIL
  // =========================
  const $ = (sel, root = document) => (root ? root.querySelector(sel) : null);
  const $$ = (sel, root = document) => (root ? Array.from(root.querySelectorAll(sel)) : []);
  const log = (...args) => { if (CFG.DEBUG) console.log(CFG.LOG_PREFIX, ...args); };
  const warn = (...args) => console.warn(CFG.LOG_PREFIX, ...args);
  const error = (...args) => console.error(CFG.LOG_PREFIX, ...args);

  function nowISO() { return new Date().toISOString(); }

  // safe wrapper for chrome.runtime.sendMessage with timeout
  function safeSendMessage(message, timeout = CFG.API_TIMEOUT_MS) {
    return new Promise((resolve) => {
      let finished = false;
      try {
        chrome.runtime.sendMessage(message, (resp) => {
          if (!finished) { finished = true; resolve(resp); }
        });
      } catch (e) {
        warn('safeSendMessage send error', e);
        if (!finished) { finished = true; resolve(null); }
      }
      // enforce timeout
      setTimeout(() => { if (!finished) { finished = true; resolve(null); } }, timeout);
    });
  }

  // =========================
  // STORAGE HELPERS (embedded)
  // =========================
  const Storage = {
    getSync: (key) => new Promise((res) => {
      try {
        chrome.storage.sync.get([key], (obj) => res(obj ? obj[key] : undefined));
      } catch (e) { warn('storage.sync.get failed', e); res(undefined); }
    }),
    setSync: (obj) => new Promise((res, rej) => {
      try {
        chrome.storage.sync.set(obj, () => {
          if (chrome.runtime.lastError) rej(chrome.runtime.lastError);
          else res();
        });
      } catch (e) { rej(e); }
    }),
    getLocal: (key) => new Promise((res) => {
      try {
        chrome.storage.local.get([key], (obj) => res(obj ? obj[key] : undefined));
      } catch (e) { warn('storage.local.get failed', e); res(undefined); }
    }),
    setLocal: (obj) => new Promise((res, rej) => {
      try {
        chrome.storage.local.set(obj, () => {
          if (chrome.runtime.lastError) rej(chrome.runtime.lastError);
          else res();
        });
      } catch (e) { rej(e); }
    })
  };

  // =========================
  // APIClient (embedded) - proxies to background messages
  // =========================
  const APIClient = {
    getUserProfile: async () => {
      const resp = await safeSendMessage({ type: 'GET_USER_PROFILE' });
      return resp && resp.profile ? { success: true, profile: resp.profile } : { success: false };
    },

    updateProfile: async (profile) => {
      const resp = await safeSendMessage({ type: 'UPDATE_PROFILE', payload: { profile } });
      return resp || { success: false };
    },

    requestDynamicResume: async (jobTitle) => {
      const resp = await safeSendMessage({ type: 'REQUEST_DYNAMIC_RESUME', payload: { jobTitle } }, CFG.API_TIMEOUT_MS * 2);
      return resp || { success: false };
    },

    getResumesMeta: async () => {
      const resp = await safeSendMessage({ type: 'GET_RESUMES_META' });
      return resp || { success: false };
    },

    getResumeBlob: async (resumeId) => {
      const resp = await safeSendMessage({ type: 'GET_RESUME_BLOB', payload: { resumeId } }, CFG.API_TIMEOUT_MS * 2);
      return resp || { success: false };
    },

    requestInjectAndAutofill: async (resumeMode) => {
      const resp = await safeSendMessage({ type: 'REQUEST_INJECT_AND_AUTOFILL', payload: { resumeMode } });
      return resp || { success: false };
    },

    getExtensionStatus: async () => {
      const resp = await safeSendMessage({ type: 'GET_EXTENSION_STATUS' });
      return resp || { success: false };
    },

    // helper to directly ask background to log an apply (not commonly used by sidebar)
    logApply: async (metadata) => {
      const resp = await safeSendMessage({ type: 'APPLY_COMPLETED', metadata });
      return resp || { success: false };
    }
  };

  // =========================
  // UI Utilities - Toasts & Modal (embedded)
  // =========================
  const UI = (function () {
    let toastRoot = null;
    let modalRoot = null;

    function ensureRoots() {
      if (!toastRoot) {
        toastRoot = document.createElement('div');
        toastRoot.id = 'ai-toast-root';
        Object.assign(toastRoot.style, {
          position: 'fixed',
          right: '16px',
          bottom: '16px',
          zIndex: 2147484000,
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
          pointerEvents: 'none'
        });
        document.body.appendChild(toastRoot);
      }
      if (!modalRoot) {
        modalRoot = document.createElement('div');
        modalRoot.id = 'ai-modal-root';
        Object.assign(modalRoot.style, {
          position: 'fixed',
          left: 0,
          top: 0,
          width: '100vw',
          height: '100vh',
          display: 'none',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 2147484090,
          background: 'rgba(0,0,0,0.35)'
        });
        document.body.appendChild(modalRoot);
      }
    }

    function toast(message, opts = {}) {
      ensureRoots();
      const box = document.createElement('div');
      box.className = 'ai-toast';
      Object.assign(box.style, {
        background: '#0f172a',
        color: '#fff',
        padding: '10px 12px',
        borderRadius: '8px',
        boxShadow: '0 8px 20px rgba(2,6,23,0.4)',
        pointerEvents: 'auto',
        maxWidth: '360px',
        fontSize: '13px'
      });
      box.textContent = message;
      toastRoot.appendChild(box);
      setTimeout(() => {
        box.style.transition = 'opacity 300ms, transform 300ms';
        box.style.opacity = '0';
        box.style.transform = 'translateY(8px)';
        setTimeout(() => box.remove(), 350);
      }, opts.duration || CFG.TOAST_DURATION);
    }

    function modal(html, opts = {}) {
      ensureRoots();
      modalRoot.innerHTML = '';
      const panel = document.createElement('div');
      panel.className = 'ai-modal';
      Object.assign(panel.style, {
        background: '#fff',
        borderRadius: '10px',
        padding: '18px',
        width: opts.width || '720px',
        maxWidth: '96%',
        maxHeight: '86vh',
        overflow: 'auto',
        boxShadow: '0 10px 40px rgba(2,6,23,0.45)'
      });
      panel.innerHTML = html;
      modalRoot.appendChild(panel);
      modalRoot.style.display = 'flex';
      function close() { modalRoot.style.display = 'none'; modalRoot.innerHTML = ''; }
      modalRoot.addEventListener('click', (e) => { if (e.target === modalRoot) close(); });
      return { close };
    }

    return { toast, modal, ensureRoots };
  })();

  // =========================
  // Helpers
  // =========================
  function createEl(tag, props = {}, children = []) {
    const el = document.createElement(tag);
    Object.entries(props || {}).forEach(([k, v]) => {
      if (k === 'class') el.className = v;
      else if (k === 'html') el.innerHTML = v;
      else el.setAttribute(k, v);
    });
    (children || []).forEach(c => el.appendChild(c));
    return el;
  }

  function escapeHtml(s) {
    if (!s) return '';
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
  }

  function escapeAttr(s) {
    if (!s) return '';
    return String(s).replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  // =========================
  // Sidebar Controller
  // =========================
  class SidebarController {
    constructor() {
      this.state = {
        profile: null,
        resumesMeta: {},
        activityLog: [],
        extensionStatus: { mcpConnected: false, ragConnected: false, notionConnected: false },
        resumeMode: { mode: 'auto', resumeId: null },
        ui: { activeTab: 'dashboard', activityPage: 0, ragCooldownUntil: 0 }
      };
      this.dom = {};
      this._init().catch(e => error('Sidebar init error', e));
    }

    // -------------------------
    // INIT
    // -------------------------
    async _init() {
      this._bindDomRefs();
      this._bindUiEvents();
      await this._loadInitialState();
      this._renderAll();
      this._bindBackgroundMessages();
      setInterval(() => this._refreshStatus(), 30_000);
      log('SidebarController initialized');
    }

    // tolerant DOM refs (works for many sidebar.html structures)
    _bindDomRefs() {
      // tabs and screens
      this.dom.tabs = $$('[data-ai-tab]') || [];
      this.dom.screens = $$('.ai-screen') || [];

      // status indicators
      this.dom.statusMcp = $('#ai-status-mcp') || $('#status-mcp');
      this.dom.statusRag = $('#ai-status-rag') || $('#status-rag');
      this.dom.statusNotion = $('#ai-status-notion') || $('#status-notion');

      // actions
      this.dom.btnAutoFill = $('#ai-btn-autofill') || $('#btn-autofill');
      this.dom.btnLogApply = $('#ai-btn-log-apply') || $('#btn-log-apply');
      this.dom.btnFetchRag = $('#ai-btn-refresh-rag') || $('#btn-refresh-rag');

      // profile inputs
      this.dom.p_fullName = $('#ai-in-fullname') || $('#in-fullname') || $('#profile-fullname');
      this.dom.p_email = $('#ai-in-email') || $('#in-email') || $('#profile-email');
      this.dom.p_phone = $('#ai-in-phone') || $('#in-phone') || $('#profile-phone');
      this.dom.p_linkedin = $('#ai-in-linkedin') || $('#in-linkedin') || $('#profile-linkedin');
      this.dom.p_location = $('#ai-in-location') || $('#in-location') || $('#profile-location');
      this.dom.p_coverLetter = $('#ai-in-coverletter') || $('#in-coverletter') || $('#profile-coverletter');
      this.dom.btnSaveProfile = $('#ai-btn-save-profile') || $('#btn-save-profile');

      // resume manager
      this.dom.resumeAutoBtn = $('#ai-resume-auto') || $('#resume-auto');
      this.dom.resumeManualList = $('#ai-resume-manual-list') || $('#resume-manual-list');
      this.dom.resumeModeToggle = $('#ai-resume-mode') || $('#resume-mode');

      // activity
      this.dom.activityList = $('#ai-activity-list') || $('#activity-list');
      this.dom.activityPrev = $('#ai-activity-prev') || $('#activity-prev');
      this.dom.activityNext = $('#ai-activity-next') || $('#activity-next');
      this.dom.activityPageInfo = $('#ai-activity-pageinfo') || $('#activity-pageinfo');

      // analysis & tools containers
      this.dom.analysisSummary = $('#ai-analysis-summary') || $('#analysis-summary');
      this.dom.analysisSkills = $('#ai-analysis-skills') || $('#analysis-skills');
      this.dom.toolsContainer = $('#ai-tools') || $('#tools');
      this.dom.btnExportCSV = $('#ai-btn-export-csv') || $('#btn-export-csv');
      this.dom.btnDownloadAllResumes = $('#ai-btn-download-resumes') || $('#btn-download-resumes');

      // loading overlay
      this.dom.loadingOverlay = $('#ai-loading-overlay') || null;
    }

    _bindUiEvents() {
      (this.dom.tabs || []).forEach(btn => {
        btn.addEventListener('click', () => {
          const t = btn.getAttribute('data-ai-tab');
          if (t) this._gotoTab(t);
        });
      });

      if (this.dom.btnSaveProfile) this.dom.btnSaveProfile.addEventListener('click', () => this._saveProfile());
      if (this.dom.btnAutoFill) this.dom.btnAutoFill.addEventListener('click', () => this._requestAutoFill());
      if (this.dom.btnLogApply) this.dom.btnLogApply.addEventListener('click', () => this._requestManualLogApply());
      if (this.dom.btnFetchRag) this.dom.btnFetchRag.addEventListener('click', () => this._fetchRagResume());

      if (this.dom.resumeManualList) {
        this.dom.resumeManualList.addEventListener('click', (e) => {
          const item = e.target.closest('.ai-resume-item');
          if (!item) return;
          const rid = item.dataset.resumeId;
          if (rid) this._setResumeModeManual(rid);
        });
      }

      if (this.dom.activityPrev) this.dom.activityPrev.addEventListener('click', () => this._changeActivityPage(-1));
      if (this.dom.activityNext) this.dom.activityNext.addEventListener('click', () => this._changeActivityPage(1));
      if (this.dom.btnExportCSV) this.dom.btnExportCSV.addEventListener('click', () => this._exportActivityCSV());
      if (this.dom.btnDownloadAllResumes) this.dom.btnDownloadAllResumes.addEventListener('click', () => this._downloadAllResumes());
    }

    // -------------------------
    // LOAD initial data
    // -------------------------
    async _loadInitialState() {
      // load profile from background (preferred) else storage.sync
      try {
        const pResp = await APIClient.getUserProfile();
        if (pResp && pResp.success && pResp.profile) {
          this.state.profile = pResp.profile;
        } else {
          this.state.profile = (await Storage.getSync(CFG.STORAGE_KEYS.PROFILE)) || {
            full_name: '', email: '', phone: '', linkedin: '', location: '', cover_letter: ''
          };
        }
      } catch (e) { warn('load profile error', e); this.state.profile = this.state.profile || {}; }

      // resumes meta
      try {
        const rResp = await APIClient.getResumesMeta();
        if (rResp && rResp.success && rResp.meta) this.state.resumesMeta = rResp.meta;
        else this.state.resumesMeta = (await Storage.getSync(CFG.STORAGE_KEYS.RESUMES_META)) || {};
      } catch (e) { warn('load resumes meta error', e); this.state.resumesMeta = this.state.resumesMeta || {}; }

      // activity log (local)
      try {
        const local = await Storage.getLocal(CFG.STORAGE_KEYS.ACTIVITY_LOG);
        this.state.activityLog = Array.isArray(local) ? local : [];
      } catch (e) { warn('load activity log error', e); this.state.activityLog = []; }

      // extension status
      await this._refreshStatus();
    }

    // -------------------------
    // RENDER
    // -------------------------
    _renderAll() {
      this._renderHeaderStatus();
      this._renderProfile();
      this._renderResumes();
      this._renderActivityPage();
      this._renderAnalysisPanel();
      this._renderTools();
    }

    _renderHeaderStatus() {
      const s = this.state.extensionStatus || {};
      this._applyIndicator(this.dom.statusMcp, !!s.mcpConnected);
      this._applyIndicator(this.dom.statusRag, !!s.ragConnected);
      this._applyIndicator(this.dom.statusNotion, !!s.notionConnected);
    }

    _applyIndicator(el, ok) {
      if (!el) return;
      el.classList.toggle('ok', !!ok);
      el.classList.toggle('bad', !ok);
      el.title = ok ? 'Connected' : 'Disconnected';
    }

    _renderProfile() {
      const p = this.state.profile || {};
      if (this.dom.p_fullName) this.dom.p_fullName.value = p.full_name || '';
      if (this.dom.p_email) this.dom.p_email.value = p.email || '';
      if (this.dom.p_phone) this.dom.p_phone.value = p.phone || '';
      if (this.dom.p_linkedin) this.dom.p_linkedin.value = p.linkedin || '';
      if (this.dom.p_location) this.dom.p_location.value = p.location || '';
      if (this.dom.p_coverLetter) this.dom.p_coverLetter.value = p.cover_letter || '';
    }

    _renderResumes() {
      const meta = this.state.resumesMeta || {};
      const keys = Object.keys(meta).slice(0, CFG.MAX_RESUMES_SHOW);
      const container = this.dom.resumeManualList;
      if (!container) return;
      container.innerHTML = '';
      if (!keys.length) {
        const empty = createResumeEmpty();
        container.appendChild(empty);
        return;
      }
      keys.forEach(k => {
        const m = meta[k] || {};
        const item = document.createElement('div');
        item.className = 'ai-resume-item';
        item.dataset.resumeId = k;
        item.innerHTML = `<div class="ai-resume-title">${escapeHtml(m.name || 'Resume')}</div>
                          <div class="ai-resume-meta">
                            <span class="ai-resume-score">${m.score ? m.score + '%' : ''}</span>
                            <span class="ai-resume-updated">${m.cached_at ? new Date(m.cached_at).toLocaleString() : ''}</span>
                          </div>`;
        if (this.state.resumeMode && this.state.resumeMode.resumeId === k && this.state.resumeMode.mode === 'manual') {
          item.classList.add('active');
        }
        container.appendChild(item);
      });

      function createResumeEmpty() {
        const e = document.createElement('div');
        e.className = 'ai-empty';
        e.textContent = 'No resumes saved yet. Generate via RAG or upload manually.';
        return e;
      }
    }

    _renderActivityPage() {
      const page = this.state.ui.activityPage || 0;
      const per = CFG.ACTIVITY_PAGE_SIZE;
      const start = page * per;
      const slice = (this.state.activityLog || []).slice(start, start + per);
      const container = this.dom.activityList;
      if (!container) return;
      container.innerHTML = '';
      if (!slice.length) {
        container.innerHTML = '<div class="ai-empty">No application activity yet.</div>';
      } else {
        slice.forEach(item => {
          const el = document.createElement('div');
          el.className = 'ai-activity-row';
          el.innerHTML = `
            <div class="ai-activity-left">
              <div class="title">${escapeHtml(item.job_title || 'Untitled')}</div>
              <div class="company">${escapeHtml(item.company || '')}</div>
            </div>
            <div class="ai-activity-right">
              <div class="time">${new Date(item.ts || item.timestamp || Date.now()).toLocaleString()}</div>
              <div class="actions">
                ${item.job_url ? `<button class="ai-open-btn" data-url="${escapeAttr(item.job_url)}">Open</button>` : ''}
              </div>
            </div>`;
          container.appendChild(el);
          const openBtn = el.querySelector('.ai-open-btn');
          if (openBtn) openBtn.addEventListener('click', () => this._openUrl(openBtn.dataset.url));
        });
      }
      this._updateActivityPager();
    }

    _updateActivityPager() {
      const total = Math.max(Math.ceil((this.state.activityLog || []).length / CFG.ACTIVITY_PAGE_SIZE), 1);
      const cur = this.state.ui.activityPage || 0;
      if (this.dom.activityPageInfo) this.dom.activityPageInfo.textContent = `Page ${cur+1} / ${total}`;
      this.dom.activityPrev && (this.dom.activityPrev.disabled = cur <= 0);
      this.dom.activityNext && (this.dom.activityNext.disabled = cur >= total - 1);
    }

    _renderAnalysisPanel() {
      const container = this.dom.analysisSummary;
      if (!container) return;
      container.innerHTML = '';
      const latest = (this.state.activityLog && this.state.activityLog[0]) || null;
      const jobPreview = latest ? `<div class="ai-jobpreview">
          <div class="ai-job-title">${escapeHtml(latest.job_title || 'Recent job')}</div>
          <div class="ai-job-company">${escapeHtml(latest.company || '')}</div>
          <div class="ai-job-meta">${latest.job_url ? `<a href="#" data-url="${escapeAttr(latest.job_url)}" class="ai-open-link">Open job</a>` : ''}</div>
        </div>` :
        `<div class="ai-empty">No recent job detected. Open a job page and click "Analyze".</div>`;
      container.innerHTML = jobPreview;
      const link = container.querySelector('.ai-open-link');
      if (link) {
        link.addEventListener('click', (e) => { e.preventDefault(); const u = link.dataset.url; this._openUrl(u); });
      }

      const meta = this.state.resumesMeta || {};
      const best = Object.values(meta).sort((a, b) => (b.score || 0) - (a.score || 0))[0];
      const skillsEl = this.dom.analysisSkills;
      if (skillsEl) {
        skillsEl.innerHTML = '';
        if (best) {
          skillsEl.innerHTML = `<div class="ai-recommend">
            <div class="label">Recommended Resume</div>
            <div class="res-item">
              <div class="name">${escapeHtml(best.name || 'Resume')}</div>
              <div class="score">${best.score ? best.score + '%' : '—'}</div>
              <div class="actions">
                <button id="ai-use-resume-${escapeAttr(best.resume_id || 'r')}" class="ai-use-resume">Select</button>
                <button id="ai-download-resume-${escapeAttr(best.resume_id || 'r')}" class="ai-download-resume">Download</button>
              </div>
            </div>
          </div>`;
          const useBtn = skillsEl.querySelector('.ai-use-resume');
          const dlBtn = skillsEl.querySelector('.ai-download-resume');
          if (useBtn) useBtn.addEventListener('click', () => {
            this._setResumeModeManual(best.resume_id);
            UI.toast('Manual resume selected');
          });
          if (dlBtn) dlBtn.addEventListener('click', () => this._downloadResume(best.resume_id));
        } else {
          skillsEl.innerHTML = `<div class="ai-empty">No resume recommendation yet. Use RAG to generate a resume recommendation.</div>`;
        }
      }
    }

    _renderTools() {
      const container = this.dom.toolsContainer;
      if (!container) return;
      // leave visual layout to HTML/CSS; controls already bound in _bindUiEvents
    }

    // -------------------------
    // ACTIONS
    // -------------------------
    _gotoTab(tab) {
      this.state.ui.activeTab = tab;
      (this.dom.screens || []).forEach(screen => {
        const s = screen.getAttribute('data-ai-screen') || screen.dataset.aiScreen;
        screen.classList.toggle('active', s === tab);
      });
      (this.dom.tabs || []).forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-ai-tab') === tab);
      });
    }

    async _saveProfile() {
      try {
        const profile = {
          full_name: (this.dom.p_fullName && this.dom.p_fullName.value.trim()) || '',
          email: (this.dom.p_email && this.dom.p_email.value.trim()) || '',
          phone: (this.dom.p_phone && this.dom.p_phone.value.trim()) || '',
          linkedin: (this.dom.p_linkedin && this.dom.p_linkedin.value.trim()) || '',
          location: (this.dom.p_location && this.dom.p_location.value.trim()) || '',
          cover_letter: (this.dom.p_coverLetter && this.dom.p_coverLetter.value.trim()) || ''
        };
        this.state.profile = profile;
        UI.toast('Saving profile...');
        const resp = await APIClient.updateProfile(profile);
        if (resp && resp.success) {
          UI.toast('Profile saved');
        } else {
          await Storage.setSync({ [CFG.STORAGE_KEYS.PROFILE]: profile });
          UI.toast('Profile saved locally');
        }
      } catch (e) {
        error('saveProfile failed', e);
        UI.toast('Failed to save profile');
      }
    }

    async _requestAutoFill() {
      try {
        UI.toast('Requesting autofill on active tab...');
        // ask background to inject and autofill first
        const injectResp = await APIClient.requestInjectAndAutofill(this.state.resumeMode);
        // also send direct content message to active tab as fallback
        const tabs = await new Promise((res) => chrome.tabs.query({ active: true, currentWindow: true }, res));
        const tab = tabs && tabs[0];
        if (tab && tab.id) {
          chrome.tabs.sendMessage(tab.id, { type: 'AUTO_FILL_FORM', resumeMode: this.state.resumeMode }, () => {});
        }
        if (!injectResp || !injectResp.success) {
          UI.toast('Auto-fill request sent (background unavailable)');
        }
      } catch (e) {
        warn('requestAutoFill error', e);
        UI.toast('Auto-fill request failed');
      }
    }

    async _requestManualLogApply() {
      try {
        UI.toast('Logging manual apply...');
        const tabs = await new Promise((res) => chrome.tabs.query({ active: true, currentWindow: true }, res));
        const tab = tabs && tabs[0];
        if (!tab || !tab.id) { UI.toast('No active tab'); return; }
        chrome.tabs.sendMessage(tab.id, { type: 'EXTRACT_JOB_DATA_AND_LOG' }, (ack) => {
          UI.toast('Apply log requested');
        });
      } catch (e) {
        warn('manualLogApply failed', e);
        UI.toast('Apply log failed');
      }
    }

    async _fetchRagResume() {
      try {
        const now = Date.now();
        if (this.state.ui.ragCooldownUntil && now < this.state.ui.ragCooldownUntil) {
          UI.toast('RAG request cooling down. Try again later.');
          return;
        }
        UI.toast('Requesting RAG resume recommendation...');
        this.state.ui.ragCooldownUntil = now + CFG.RAG_REFRESH_COOLDOWN_MS;
        const jobTitle = (this.state.profile && (this.state.profile.target_title || '')) || document.title || '';
        const resp = await APIClient.requestDynamicResume(jobTitle);
        if (resp && resp.success && resp.resume) {
          UI.toast('RAG resume generated and cached');
          await this._refreshResumesMeta();
          this._renderResumes();
        } else {
          UI.toast('RAG request failed');
        }
      } catch (e) {
        warn('fetchRagResume failed', e);
        UI.toast('RAG request failed');
      }
    }

    async _refreshResumesMeta() {
      try {
        const r = await APIClient.getResumesMeta();
        if (r && r.success && r.meta) {
          this.state.resumesMeta = r.meta;
        } else {
          this.state.resumesMeta = (await Storage.getSync(CFG.STORAGE_KEYS.RESUMES_META)) || this.state.resumesMeta;
        }
        this._renderResumes();
      } catch (e) { warn('refreshResumesMeta failed', e); }
    }

    async _setResumeModeManual(resumeId) {
      try {
        this.state.resumeMode = { mode: 'manual', resumeId };
        if (!this.state.profile) this.state.profile = {};
        this.state.profile.preferred_resume_mode = 'manual';
        this.state.profile.preferred_resume_id = resumeId;
        await APIClient.updateProfile(this.state.profile);
        UI.toast(`Manual resume selected: ${resumeId}`);
        this._renderResumes();
      } catch (e) {
        warn('setResumeModeManual failed', e);
      }
    }

    async _downloadResume(resumeId) {
      try {
        const resp = await APIClient.getResumeBlob(resumeId);
        if (!resp || !resp.success) { UI.toast('Resume not available'); return; }
        const base64 = resp.base64;
        const meta = resp.meta || {};
        const binary = atob(base64);
        const len = binary.length;
        const arr = new Uint8Array(len);
        for (let i = 0; i < len; i++) arr[i] = binary.charCodeAt(i);
        const blob = new Blob([arr.buffer], { type: meta.mime || 'application/pdf' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = meta.fileName || `resume_${resumeId}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 10000);
        UI.toast('Resume download started');
      } catch (e) {
        warn('downloadResume failed', e);
        UI.toast('Download failed');
      }
    }

    async _downloadAllResumes() {
      try {
        const meta = this.state.resumesMeta || {};
        const keys = Object.keys(meta);
        if (!keys.length) { UI.toast('No resumes to download'); return; }
        UI.toast('Preparing download of all resumes...');
        for (const k of keys) {
          await this._downloadResume(k);
          await new Promise(r => setTimeout(r, 600));
        }
      } catch (e) {
        warn('downloadAllResumes error', e);
        UI.toast('Download failed');
      }
    }

    // -------------------------
    // Activity helpers
    // -------------------------
    _changeActivityPage(delta) {
      const totalPages = Math.max(Math.ceil((this.state.activityLog || []).length / CFG.ACTIVITY_PAGE_SIZE), 1);
      let cur = this.state.ui.activityPage || 0;
      cur = Math.max(0, Math.min(totalPages - 1, cur + delta));
      this.state.ui.activityPage = cur;
      this._renderActivityPage();
    }

    async _openUrl(url) {
      if (!url) return;
      try { chrome.tabs.create({ url }); } catch (e) { warn('openUrl failed', e); }
    }

    async _exportActivityCSV() {
      try {
        const rows = this.state.activityLog || [];
        if (!rows.length) { UI.toast('No activity to export'); return; }
        const csv = rows.map(r => {
          return `"${(r.job_title || '').replace(/"/g, '""')}","${(r.company || '').replace(/"/g, '""')}","${(r.job_url || '').replace(/"/g, '""')}","${(r.ts || r.timestamp || '')}"`;
        }).join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ai_job_activity_${new Date().toISOString().slice(0,10)}.csv`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 5000);
        UI.toast('Activity CSV exported');
      } catch (e) {
        warn('exportActivityCSV failed', e);
        UI.toast('Export failed');
      }
    }

    // -------------------------
    // STATUS / BACKGROUND
    // -------------------------
    async _refreshStatus() {
      try {
        const resp = await APIClient.getExtensionStatus();
        if (resp && resp.success && resp.status) {
          this.state.extensionStatus = {
            mcpConnected: !!resp.status.mcpConnected,
            ragConnected: !!(resp.status.apiEndpoints && resp.status.apiEndpoints.rag),
            notionConnected: !!(resp.status.apiEndpoints && resp.status.apiEndpoints.notion)
          };
        } else {
          this.state.extensionStatus = { mcpConnected: false, ragConnected: false, notionConnected: false };
        }
        this._renderHeaderStatus();
      } catch (e) {
        warn('refreshStatus failed', e);
        this.state.extensionStatus = { mcpConnected: false, ragConnected: false, notionConnected: false };
        this._renderHeaderStatus();
      }
    }

    // -------------------------
    // BACKGROUND MESSAGE BIND
    // -------------------------
    _bindBackgroundMessages() {
      chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
        try {
          if (!msg || !msg.type) return;
          switch (msg.type) {
            case 'API_STATUS_UPDATE':
              this.state.extensionStatus = { ...this.state.extensionStatus, ...msg.status };
              this._renderHeaderStatus();
              break;
            case 'APPLICATION_COMPLETED':
              if (msg.payload) {
                const entry = {
                  job_title: msg.payload.job_title || 'Applied',
                  company: msg.payload.company || '',
                  job_url: msg.payload.job_url || '',
                  ts: msg.payload.timestamp || nowISO()
                };
                this.state.activityLog.unshift(entry);
                if (this.state.activityLog.length > CFG.MAX_ACTIVITY_ENTRIES) this.state.activityLog.length = CFG.MAX_ACTIVITY_ENTRIES;
                Storage.setLocal({ [CFG.STORAGE_KEYS.ACTIVITY_LOG]: this.state.activityLog }).catch(e => warn('persist activity failed', e));
                this._renderActivityPage();
                UI.toast('Application recorded');
              }
              break;
            case 'ANALYSIS_COMPLETED':
              if (msg.analysis) {
                const a = msg.analysis;
                UI.modal(`<div style="font-weight:700">${escapeHtml(a.job_title || 'Analysis')}</div>
                  <div style="margin-top:8px">Match: ${a.match_score || 'N/A'}%</div>
                  <div style="margin-top:10px">${escapeHtml(a.summary || '')}</div>`);
              }
              break;
            case 'JOB_DETECTED':
              if (msg.jobData) {
                const jd = msg.jobData;
                this.state.activityLog.unshift({ job_title: jd.title, company: jd.company, job_url: jd.url, ts: nowISO() });
                if (this.state.activityLog.length > CFG.MAX_ACTIVITY_ENTRIES) this.state.activityLog.length = CFG.MAX_ACTIVITY_ENTRIES;
                Storage.setLocal({ [CFG.STORAGE_KEYS.ACTIVITY_LOG]: this.state.activityLog }).catch(e => warn('persist activity failed', e));
                this._renderActivityPage();
                this._renderAnalysisPanel();
              }
              break;
            default:
              break;
          }
        } catch (e) { warn('sidebar onMessage error', e); }
      });
    }

  } // end SidebarController

  // instantiate after DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => { new SidebarController(); });
  } else {
    new SidebarController();
  }

})();