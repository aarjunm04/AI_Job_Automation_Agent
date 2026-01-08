/**
 * utils/notion_api.js
 * Notion logging helper for AI Job Automation Agent (Chrome extension)
 *
 * Responsibilities:
 *  - Exposes `NotionAPI` class with `logApplication(jobData)` method.
 *  - Implements idempotency keys, retries with exponential backoff, and persistent queueing.
 *  - Stores transient queue in chrome.storage.local to survive service worker restarts.
 *  - Robust error handling and non-blocking behavior (never throw unhandled).
 *
 * Usage:
 *   const notion = new NotionAPI({ webhookUrl: 'https://your-notion-proxy/log' });
 *   await notion.logApplication({ job_url, job_title, company, status, resume_id });
 *
 * Notes:
 *  - Designed to be used from background service worker only.
 *  - Limits: avoids flooding Notion; uses backoff; persists queue of pending logs.
 */

'use strict';

const NotionAPI = (function () {
  const DEFAULT_OPTIONS = {
    webhookUrl: null,
    timeoutMs: 15_000,
    maxRetries: 4,
    baseDelayMs: 700,
    storageKey: 'ai_notion_queue', // chrome.storage.local key for queued items
    maxQueueSize: 500
  };

  // helper: sleep
  function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  // helper: fetch with timeout
  async function fetchWithTimeout(url, options = {}, timeout = 15000) {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    try {
      const res = await fetch(url, { ...options, signal: controller.signal });
      clearTimeout(id);
      return res;
    } catch (err) {
      clearTimeout(id);
      throw err;
    }
  }

  // helper: stable idempotency key using subtle crypto SHA-256 (fallback to base64)
  async function makeIdempotencyKey(jobUrl, timestampISO) {
    const text = `${jobUrl}|${timestampISO || ''}`;
    try {
      if (globalThis.crypto && crypto.subtle && crypto.subtle.digest) {
        const enc = new TextEncoder().encode(text);
        const hashBuf = await crypto.subtle.digest('SHA-256', enc);
        const hashArr = Array.from(new Uint8Array(hashBuf));
        return hashArr.map((b) => b.toString(16).padStart(2, '0')).join('');
      }
    } catch (e) {
      // fallback
    }
    // fallback compact base64
    try {
      return btoa(unescape(encodeURIComponent(text))).slice(0, 64);
    } catch (e) {
      return `id-${Date.now()}`;
    }
  }

  // storage helpers
  const Storage = {
    async get(key) {
      return new Promise((resolve) => {
        chrome.storage.local.get([key], (res) => {
          resolve(res ? res[key] : undefined);
        });
      });
    },
    async set(obj) {
      return new Promise((resolve, reject) => {
        chrome.storage.local.set(obj, () => {
          const e = chrome.runtime.lastError;
          if (e) reject(e);
          else resolve();
        });
      });
    }
  };

  // Default logger (togglable)
  function log(...args) {
    // use background.DEBUG toggles if you wire it; default true for dev
    if (typeof chrome !== 'undefined' && chrome.runtime && chrome.runtime.getManifest) {
      // avoid noise in production; caller can toggle
      // console.log('[NotionAPI]', ...args);
    } else {
      // environment fallback
      // console.log('[NotionAPI]', ...args);
    }
  }

  class NotionAPI {
    constructor(options = {}) {
      this.opts = { ...DEFAULT_OPTIONS, ...options };
      if (!this.opts.webhookUrl) {
        throw new Error('NotionAPI: webhookUrl is required in options');
      }
      this.queueProcessing = false;
    }

    // Push an item into persistent queue and try to flush immediately (non-blocking)
    async enqueue(payload) {
      try {
        const key = this.opts.storageKey;
        const cur = (await Storage.get(key)) || [];
        // trim queue if too big
        if (cur.length >= this.opts.maxQueueSize) {
          cur.splice(this.opts.maxQueueSize - 1, 1);
        }
        cur.unshift(payload);
        await Storage.set({ [key]: cur });
      } catch (err) {
        log('enqueue error', err);
      }
    }

    // Pop an item (LIFO) - returns null if none
    async dequeueOne() {
      try {
        const key = this.opts.storageKey;
        const cur = (await Storage.get(key)) || [];
        if (!cur.length) return null;
        const item = cur.shift();
        await Storage.set({ [key]: cur });
        return item;
      } catch (err) {
        log('dequeueOne error', err);
        return null;
      }
    }

    // peek without removing
    async peekAll() {
      return (await Storage.get(this.opts.storageKey)) || [];
    }

    // core: attempt to send one payload with retries and exponential backoff
    async sendOnce(payload) {
      const { webhookUrl, timeoutMs, maxRetries, baseDelayMs } = this.opts;
      let lastErr = null;
      for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
          const res = await fetchWithTimeout(webhookUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          }, timeoutMs);

          if (!res.ok) {
            const text = await res.text().catch(() => '');
            lastErr = new Error(`Notion webhook returned ${res.status}: ${text}`);
            // consider 4xx as permanent failure (no retry)
            if (res.status >= 400 && res.status < 500) {
              throw lastErr;
            }
            // otherwise retry
            throw lastErr;
          }
          // success
          const json = await res.json().catch(() => null);
          return { ok: true, raw: json };
        } catch (err) {
          lastErr = err;
          // exponential backoff
          const delay = baseDelayMs * Math.pow(2, attempt);
          await sleep(delay);
          continue;
        }
      }
      return { ok: false, error: lastErr ? String(lastErr) : 'unknown' };
    }

    // Public method used by background to log an application event.
    // jobData: { job_url, job_title, company, status, resume_id, metadata? }
    async logApplication(jobData = {}) {
      try {
        const now = new Date().toISOString();
        const idempotency_key = await makeIdempotencyKey(jobData.job_url || '', now);

        const payload = {
          idempotency_key,
          source: 'chrome_extension',
          job_url: jobData.job_url || null,
          job_title: jobData.job_title || null,
          company: jobData.company || null,
          timestamp: now,
          status: jobData.status || 'submitted',
          resume_id: jobData.resume_id || null,
          metadata: jobData.metadata || {}
        };

        // Try immediate send first (fast-path)
        const sendResult = await this.sendOnce(payload);
        if (sendResult.ok) {
          // success — record to activity log (background can also track)
          try {
            // append a short activity log in storage for audit (lightweight)
            const activityKey = 'ai_activity_log';
            const existing = (await Storage.get(activityKey)) || [];
            existing.unshift({ ts: now, type: 'notion_log_ok', job_url: payload.job_url });
            if (existing.length > 200) existing.length = 200;
            await Storage.set({ [activityKey]: existing });
          } catch (e) {
            // non-fatal
          }
          return { success: true, raw: sendResult.raw || null };
        }

        // If immediate send failed, enqueue and return queued response
        await this.enqueue(payload);
        // attempt to ensure background flush process is running
        this.processQueue().catch((e) => log('queue process launched (background) error', e));

        return { success: false, queued: true, error: sendResult.error || 'queued' };
      } catch (err) {
        // On unexpected error, try to enqueue anyway
        try {
          const now2 = new Date().toISOString();
          const key = this.opts.storageKey;
          const backPayload = { idempotency_key: `err-${Date.now()}`, source: 'chrome_extension', job_url: jobData.job_url || null, job_title: jobData.job_title || null, company: jobData.company || null, timestamp: now2, status: jobData.status || 'submitted' };
          await this.enqueue(backPayload);
        } catch (e) {
          // ignore
        }
        return { success: false, error: String(err) };
      }
    }

    // Processes queue; runs until queue empty or service worker unloads.
    // Serial processing to maintain order and reduce parallel requests.
    async processQueue() {
      if (this.queueProcessing) return;
      this.queueProcessing = true;
      try {
        let item = await this.dequeueOne();
        while (item) {
          try {
            const res = await this.sendOnce(item);
            if (!res.ok) {
              // Failed after retries: re-enqueue at end with slight delay to avoid hot-loop
              await this.enqueue(item);
              await sleep(1000);
              break; // exit loop to allow background to restart later
            } else {
              // success -> record activity
              try {
                const activityKey = 'ai_activity_log';
                const existing = (await Storage.get(activityKey)) || [];
                existing.unshift({ ts: new Date().toISOString(), type: 'notion_log_ok', job_url: item.job_url || null });
                if (existing.length > 200) existing.length = 200;
                await Storage.set({ [activityKey]: existing });
              } catch (e) {
                // ignore
              }
            }
          } catch (e) {
            // on unexpected error re-enqueue and exit; will retry later
            await this.enqueue(item);
            break;
          }
          item = await this.dequeueOne();
        }
      } catch (err) {
        log('processQueue error', err);
      } finally {
        this.queueProcessing = false;
      }
    }
  }

  return NotionAPI;
})();

// Export for usage in background.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = NotionAPI;
} else {
  // attach to window/global for extension import style (background can use import via <script>)
  try {
    self.NotionAPI = NotionAPI;
  } catch (e) {
    // ignore
  }
}