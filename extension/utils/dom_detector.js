/* ══════════════════════════════════════════════════════════════════
   dom_detector.js — Job Agent Chrome Extension v1
   Shared DOM parsing, field classification, React input detection,
   value injection, and mutation observation utilities.
   ══════════════════════════════════════════════════════════════════ */

// ── Field Type Constants ──
const FIELD_TYPES = {
  TEXT: 'text',
  EMAIL: 'email',
  PHONE: 'phone',
  URL: 'url',
  NUMBER: 'number',
  SELECT: 'select',
  TEXTAREA: 'textarea',
  FILE: 'file',
  CHECKBOX: 'checkbox',
  RADIO: 'radio',
  UNKNOWN: 'unknown',
};

// ── ATS URL Patterns ──
const ATS_URL_PATTERNS = [
  { pattern: /greenhouse\.io/i, hint: 'greenhouse' },
  { pattern: /lever\.co/i, hint: 'lever' },
  { pattern: /myworkdayjobs\.com|wd[0-9]+\.myworkday/i, hint: 'workday' },
  { pattern: /linkedin\.com\/jobs/i, hint: 'linkedin' },
  { pattern: /indeed\.com/i, hint: 'indeed' },
  { pattern: /wellfound\.com/i, hint: 'wellfound' },
  { pattern: /arc\.dev/i, hint: 'arc' },
];

// ── ATS DOM Fingerprints ──
const ATS_DOM_FINGERPRINTS = [
  { selector: 'div#app_body', hint: 'greenhouse' },
  { selector: 'div.application-form', hint: 'lever' },
  { selector: 'div[data-automation-id]', hint: 'workday' },
  { selector: 'div.jobs-easy-apply-content', hint: 'linkedin' },
];

// ── Job Description Selectors (priority order) ──
const JD_SELECTORS = [
  '[class*="job-description"]',
  '[class*="jobDescription"]',
  '[class*="job-details"]',
  '[id*="job-description"]',
  '[class*="description-content"]',
  'article',
  'main',
];

// ── Phone Keywords ──
const PHONE_KEYWORDS = /phone|mobile|tel|contact/i;
// ── Email Keywords ──
const EMAIL_KEYWORDS = /email/i;
// ── URL Keywords ──
const URL_KEYWORDS = /website|portfolio|github|linkedin|url/i;
// ── Number Keywords ──
const NUMBER_KEYWORDS = /year|experience|salary|ctc|age/i;

// ── React Internal Keys Prefixes ──
const REACT_PREFIXES = [
  '__reactFiber',
  '__reactInternalInstance',
  '_reactProps',
  '__reactEventHandlers',
];

// ── Visibility Check ──
/**
 * Checks if an element is visible and interactable.
 * @param {HTMLElement} element
 * @returns {boolean}
 */
function isVisible(element) {
  try {
    if (element.type === 'hidden') return false;
    if (element.offsetParent === null && getComputedStyle(element).position !== 'fixed') return false;
    const style = getComputedStyle(element);
    if (style.display === 'none') return false;
    if (style.visibility === 'hidden') return false;
    return true;
  } catch (e) {
    return false;
  }
}

/**
 * Humanises a name string: replaces _ and - with spaces, title-cases words.
 * @param {string} str
 * @returns {string}
 */
function humanise(str) {
  if (!str) return '';
  return str
    .replace(/[_-]/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .trim();
}

/**
 * Returns a combined string of an element's id, name, placeholder, and label
 * for keyword matching purposes.
 * @param {HTMLElement} element
 * @param {string} labelText
 * @returns {string}
 */
function getMatchableText(element, labelText) {
  return [
    element.id || '',
    element.name || '',
    element.placeholder || '',
    labelText || '',
  ].join(' ');
}

// ══════════════════════════════════════════════════════════════════
// EXPORTED FUNCTIONS
// ══════════════════════════════════════════════════════════════════

/**
 * Resolves the human-readable label for a form field.
 * Resolution order:
 * 1. <label for="{element.id}"> text content
 * 2. element.closest("label") text content
 * 3. aria-label attribute
 * 4. aria-labelledby → getElementById → text content
 * 5. placeholder attribute
 * 6. name attribute (humanised: replace _ and - with space, title-case)
 * 7. "" (empty string fallback)
 * Max 80 chars, trimmed.
 * @param {HTMLElement} element
 * @param {Document|ShadowRoot} root
 * @returns {string}
 */
export function getLabelText(element, root) {
  try {
    // 1. <label for="id">
    if (element.id) {
      try {
        const label = root.querySelector('label[for="' + CSS.escape(element.id) + '"]');
        if (label && label.textContent.trim()) {
          return label.textContent.trim().slice(0, 80);
        }
      } catch (e) { /* selector may fail in shadow roots */ }
    }

    // 2. closest <label>
    try {
      const parentLabel = element.closest('label');
      if (parentLabel && parentLabel.textContent.trim()) {
        return parentLabel.textContent.trim().slice(0, 80);
      }
    } catch (e) { /* closest may fail in shadow roots */ }

    // 3. aria-label
    const ariaLabel = element.getAttribute('aria-label');
    if (ariaLabel && ariaLabel.trim()) {
      return ariaLabel.trim().slice(0, 80);
    }

    // 4. aria-labelledby
    const labelledBy = element.getAttribute('aria-labelledby');
    if (labelledBy) {
      try {
        const refEl = root.getElementById
          ? root.getElementById(labelledBy)
          : root.querySelector('#' + CSS.escape(labelledBy));
        if (refEl && refEl.textContent.trim()) {
          return refEl.textContent.trim().slice(0, 80);
        }
      } catch (e) { /* skip */ }
    }

    // 5. placeholder
    if (element.placeholder && element.placeholder.trim()) {
      return element.placeholder.trim().slice(0, 80);
    }

    // 6. name (humanised)
    if (element.name) {
      return humanise(element.name).slice(0, 80);
    }

    // 7. fallback
    return '';
  } catch (e) {
    return '';
  }
}

/**
 * Classifies an input element's semantic type based on type attribute,
 * name, id, placeholder, and label text.
 * @param {HTMLElement} element
 * @returns {string} FieldType
 */
export function classifyFieldType(element) {
  try {
    const tagName = element.tagName.toUpperCase();

    if (tagName === 'SELECT') return FIELD_TYPES.SELECT;
    if (tagName === 'TEXTAREA') return FIELD_TYPES.TEXTAREA;

    const inputType = (element.type || 'text').toLowerCase();

    // Direct type mappings
    if (inputType === 'email') return FIELD_TYPES.EMAIL;
    if (inputType === 'tel') return FIELD_TYPES.PHONE;
    if (inputType === 'url') return FIELD_TYPES.URL;
    if (inputType === 'number') return FIELD_TYPES.NUMBER;
    if (inputType === 'file') return FIELD_TYPES.FILE;
    if (inputType === 'checkbox') return FIELD_TYPES.CHECKBOX;
    if (inputType === 'radio') return FIELD_TYPES.RADIO;

    // Keyword-based classification for text/search inputs
    if (inputType === 'text' || inputType === 'search' || inputType === '') {
      const labelText = getLabelText(element, element.ownerDocument || document);
      const matchText = getMatchableText(element, labelText);

      if (PHONE_KEYWORDS.test(matchText)) return FIELD_TYPES.PHONE;
      if (EMAIL_KEYWORDS.test(matchText)) return FIELD_TYPES.EMAIL;
      if (URL_KEYWORDS.test(matchText)) return FIELD_TYPES.URL;
      if (NUMBER_KEYWORDS.test(matchText)) return FIELD_TYPES.NUMBER;

      return FIELD_TYPES.TEXT;
    }

    return FIELD_TYPES.TEXT;
  } catch (e) {
    return FIELD_TYPES.UNKNOWN;
  }
}

/**
 * Detects if an element is controlled by React.
 * Checks for: __reactFiber*, __reactInternalInstance*, _reactProps*,
 * __reactEventHandlers* keys on the element (prefix wildcard match).
 * @param {HTMLElement} element
 * @returns {boolean}
 */
export function isReactControlled(element) {
  try {
    const keys = Object.keys(element);
    for (let i = 0; i < keys.length; i++) {
      for (let j = 0; j < REACT_PREFIXES.length; j++) {
        if (keys[i].startsWith(REACT_PREFIXES[j])) {
          return true;
        }
      }
    }
    return false;
  } catch (e) {
    return false;
  }
}

/**
 * Detects the ATS platform from current page URL and DOM fingerprints.
 * @param {string} url - Current page URL
 * @param {Document} doc
 * @returns {string} ATS hint string
 */
export function detectAtsHint(url, doc) {
  try {
    // URL-based detection first
    for (let i = 0; i < ATS_URL_PATTERNS.length; i++) {
      if (ATS_URL_PATTERNS[i].pattern.test(url)) {
        return ATS_URL_PATTERNS[i].hint;
      }
    }

    // DOM fingerprint fallback
    for (let i = 0; i < ATS_DOM_FINGERPRINTS.length; i++) {
      try {
        if (doc.querySelector(ATS_DOM_FINGERPRINTS[i].selector)) {
          return ATS_DOM_FINGERPRINTS[i].hint;
        }
      } catch (e) { /* skip invalid selector */ }
    }

    return 'unknown';
  } catch (e) {
    return 'unknown';
  }
}

/**
 * Scans the document for all visible, interactable form fields.
 * Excludes hidden fields, disabled fields, and fields inside iframes
 * (iframes handled separately in content.js).
 * @param {Document|ShadowRoot} root - The root to scan (document or shadow root)
 * @returns {DetectedField[]}
 */
export function scanFormFields(root) {
  try {
    const elements = root.querySelectorAll('input, select, textarea');
    const fields = [];
    const atsHint = detectAtsHint(
      (typeof location !== 'undefined' ? location.href : ''),
      root.ownerDocument || root
    );

    for (let i = 0; i < elements.length; i++) {
      const el = elements[i];

      try {
        // Skip disabled fields
        if (el.disabled) continue;

        // Skip hidden fields
        if (!isVisible(el)) continue;

        // Skip elements inside iframes (handled separately)
        if (el.ownerDocument !== (root.ownerDocument || root)) continue;

        const isShadow = root instanceof ShadowRoot;
        const reactControlled = isReactControlled(el);
        const fieldType = classifyFieldType(el);
        const labelText = getLabelText(el, root);

        // Store live element reference in global array
        let elementRefIndex = -1;
        if (typeof window !== 'undefined' && Array.isArray(window.__jobAgentFields)) {
          elementRefIndex = window.__jobAgentFields.length;
          window.__jobAgentFields.push(el);
        }

        fields.push({
          index: fields.length,
          id: el.id || '',
          name: el.name || '',
          placeholder: el.placeholder || '',
          label_text: labelText,
          field_type: fieldType,
          tag_name: el.tagName.toUpperCase(),
          react_controlled: reactControlled,
          shadow_dom: isShadow,
          element_ref_index: elementRefIndex,
          ats_hint: atsHint,
        });
      } catch (elErr) {
        // Skip individual element errors
        continue;
      }
    }

    return fields;
  } catch (e) {
    return [];
  }
}

/**
 * Injects a value into a form field, handling both standard and
 * React-controlled inputs.
 * @param {HTMLElement} element
 * @param {string} value
 * @param {boolean} isReact
 * @returns {{success: boolean, error: string|null}}
 */
export function injectFieldValue(element, value, isReact) {
  try {
    const tagName = element.tagName.toUpperCase();

    // ── Checkbox ──
    if (element.type === 'checkbox') {
      const boolVal = value === true || value === 'true' || value === '1' || value === 1;
      element.checked = boolVal;
      element.dispatchEvent(new Event('change', { bubbles: true }));
      return { success: true, error: null };
    }

    // ── Radio ──
    if (element.type === 'radio') {
      const shouldCheck = value === true || value === 'true' || value === '1' ||
        value === 1 || element.value === String(value);
      element.checked = shouldCheck;
      element.dispatchEvent(new Event('change', { bubbles: true }));
      return { success: true, error: null };
    }

    // ── Select ──
    if (tagName === 'SELECT') {
      const options = Array.from(element.options);
      const valLower = String(value).toLowerCase();

      // Try value match first (case-insensitive)
      let matched = options.find(
        (opt) => opt.value.toLowerCase() === valLower
      );
      // Fallback: text match
      if (!matched) {
        matched = options.find(
          (opt) => opt.textContent.trim().toLowerCase() === valLower
        );
      }
      // Fallback: partial text match
      if (!matched) {
        matched = options.find(
          (opt) => opt.textContent.trim().toLowerCase().includes(valLower)
        );
      }

      if (matched) {
        element.value = matched.value;
        element.dispatchEvent(new Event('change', { bubbles: true }));
        return { success: true, error: null };
      }
      return { success: false, error: 'No matching option found for: ' + value };
    }

    // ── React-controlled input/textarea ──
    if (isReact) {
      try {
        const proto = tagName === 'TEXTAREA'
          ? HTMLTextAreaElement.prototype
          : HTMLInputElement.prototype;
        const nativeSetter = Object.getOwnPropertyDescriptor(proto, 'value').set;
        nativeSetter.call(element, String(value));
        element.dispatchEvent(new Event('input', { bubbles: true }));
        element.dispatchEvent(new Event('change', { bubbles: true }));
        return { success: true, error: null };
      } catch (reactErr) {
        // Fall through to standard injection
      }
    }

    // ── Standard input/textarea ──
    element.value = String(value);
    element.dispatchEvent(new Event('input', { bubbles: true }));
    element.dispatchEvent(new Event('change', { bubbles: true }));
    return { success: true, error: null };
  } catch (e) {
    return { success: false, error: e.message || 'Injection failed' };
  }
}

/**
 * Scans open shadow roots recursively and returns all fields found inside them.
 * @param {Document} doc
 * @returns {DetectedField[]}
 */
export function scanShadowDomFields(doc) {
  try {
    const fields = [];
    const allElements = doc.querySelectorAll('*');

    for (let i = 0; i < allElements.length; i++) {
      try {
        const shadowRoot = allElements[i].shadowRoot;
        if (shadowRoot) {
          const shadowFields = scanFormFields(shadowRoot);
          for (let j = 0; j < shadowFields.length; j++) {
            shadowFields[j].shadow_dom = true;
            fields.push(shadowFields[j]);
          }

          // Recurse into nested shadow roots
          const nested = scanShadowDomFields(shadowRoot);
          for (let k = 0; k < nested.length; k++) {
            fields.push(nested[k]);
          }
        }
      } catch (elErr) {
        // Skip inaccessible shadow roots (closed)
        continue;
      }
    }

    return fields;
  } catch (e) {
    return [];
  }
}

/**
 * Extracts the job description text from the page.
 * Tries candidate selectors in priority order, takes the longest match.
 * Cleans whitespace and returns max 3000 characters.
 * @param {Document} doc
 * @returns {string}
 */
export function extractJobDescription(doc) {
  try {
    let bestText = '';

    for (let i = 0; i < JD_SELECTORS.length; i++) {
      try {
        const elements = doc.querySelectorAll(JD_SELECTORS[i]);
        for (let j = 0; j < elements.length; j++) {
          const text = (elements[j].innerText || '').trim();
          if (text.length > bestText.length) {
            bestText = text;
          }
        }
      } catch (e) {
        // Skip invalid selectors
        continue;
      }
    }

    if (!bestText) return '';

    // Clean: collapse whitespace, remove null bytes
    bestText = bestText
      .replace(/\0/g, '')
      .replace(/[ \t]+/g, ' ')
      .replace(/\n{3,}/g, '\n\n')
      .trim();

    return bestText.slice(0, 3000);
  } catch (e) {
    return '';
  }
}

/**
 * Creates a debounced MutationObserver that calls callback(newFields)
 * whenever the DOM changes significantly (childList or subtree mutations
 * that add/remove INPUT, SELECT, TEXTAREA nodes).
 * @param {function(DetectedField[]): void} callback
 * @param {number} debounceMs
 * @param {Document} doc
 * @returns {{observer: MutationObserver, stop: function(): void}}
 */
export function createFieldScanner(callback, debounceMs, doc) {
  let timer = null;
  const FORM_TAGS = new Set(['INPUT', 'SELECT', 'TEXTAREA']);

  /**
   * Checks if a list of MutationRecords contains changes involving form elements.
   * @param {MutationRecord[]} mutations
   * @returns {boolean}
   */
  function hasFormFieldChanges(mutations) {
    for (let i = 0; i < mutations.length; i++) {
      const m = mutations[i];
      // Check added nodes
      for (let j = 0; j < m.addedNodes.length; j++) {
        const node = m.addedNodes[j];
        if (node.nodeType === Node.ELEMENT_NODE) {
          if (FORM_TAGS.has(node.tagName)) return true;
          if (node.querySelector && node.querySelector('input, select, textarea')) return true;
        }
      }
      // Check removed nodes
      for (let j = 0; j < m.removedNodes.length; j++) {
        const node = m.removedNodes[j];
        if (node.nodeType === Node.ELEMENT_NODE) {
          if (FORM_TAGS.has(node.tagName)) return true;
          if (node.querySelector && node.querySelector('input, select, textarea')) return true;
        }
      }
    }
    return false;
  }

  const observer = new MutationObserver((mutations) => {
    try {
      if (!hasFormFieldChanges(mutations)) return;

      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        try {
          const fields = scanFormFields(doc);
          callback(fields);
        } catch (e) {
          // Silently fail — never crash the host page
        }
      }, debounceMs);
    } catch (e) {
      // Silently fail
    }
  });

  observer.observe(doc.body || doc.documentElement, {
    childList: true,
    subtree: true,
  });

  function stop() {
    if (timer) clearTimeout(timer);
    observer.disconnect();
  }

  return { observer, stop };
}

// ══════════════════════════════════════════════════════════════════
// WINDOW GLOBAL SHIM — content script compatibility
// dom_detector.js is loaded as a content script (not ES module),
// so we expose all exports on window.JobAgentDetector.
// ══════════════════════════════════════════════════════════════════
if (typeof window !== 'undefined') {
  window.JobAgentDetector = {
    scanFormFields,
    scanShadowDomFields,
    detectAtsHint,
    classifyFieldType,
    getLabelText,
    isReactControlled,
    injectFieldValue,
    extractJobDescription,
    createFieldScanner,
  };
}
