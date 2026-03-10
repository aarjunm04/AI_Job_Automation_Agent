/* ══════════════════════════════════════════════════════════════════
   sidebar.js — AI Job Agent Chrome Extension v1
   Content script that listens for autofill messages from the popup
   and fills detected form fields with user profile data.
   Zero MCP references. Never throws uncaught exceptions.
   ══════════════════════════════════════════════════════════════════ */

'use strict';

// ── Autofill field-matching rules (priority order, first match wins) ──
var _FIELD_RULES = [
    // email
    { keys: ['email', 'e-mail', 'emailaddress'], profile: 'USER_EMAIL' },
    // phone
    { keys: ['phone', 'mobile', 'tel', 'telephone', 'cell', 'contact'], profile: 'USER_PHONE' },
    // first name (before full name so it wins on "first" tokens)
    { keys: ['firstname', 'first name', 'first-name', 'given name', 'givenname'], profile: 'first_name' },
    // last name
    { keys: ['lastname', 'last name', 'last-name', 'surname', 'family name', 'familyname'], profile: 'last_name' },
    // full name
    { keys: ['fullname', 'full name', 'full-name', 'your name', 'name', 'candidate name'], profile: 'USERNAME' },
    // LinkedIn
    { keys: ['linkedin', 'linkedin url', 'linkedin profile'], profile: 'USER_LINKEDIN_URL' },
    // portfolio / GitHub
    { keys: ['portfolio', 'github', 'website', 'personal url', 'personal site'], profile: 'USER_PORTFOLIO_URL' },
    // location
    { keys: ['location', 'city', 'country', 'residence', 'current location', 'where are you'], profile: 'USER_LOCATION' },
    // years of experience
    { keys: ['years of experience', 'years experience', 'experience years', 'how many years', 'total experience'], profile: 'USER_YEARS_EXPERIENCE' },
];

// ══════════════════════════════════════════════════════════════════
// FIELD DISCOVERY
// ══════════════════════════════════════════════════════════════════

/**
 * Returns all fillable input/textarea/select elements on the page
 * that are visible and not submit/button/hidden/file types.
 * @returns {Element[]}
 */
function _getFormFields() {
    var inputs = Array.from(document.querySelectorAll(
        'input:not([type="submit"]):not([type="button"]):not([type="hidden"])' +
        ':not([type="file"]):not([type="checkbox"]):not([type="radio"]):not([type="image"]),' +
        'textarea, select'
    ));
    return inputs.filter(function (el) {
        try {
            var style = window.getComputedStyle(el);
            return style.display !== 'none' && style.visibility !== 'hidden' && el.offsetParent !== null;
        } catch (_) {
            return true;
        }
    });
}

// ══════════════════════════════════════════════════════════════════
// FIELD MATCHING
// ══════════════════════════════════════════════════════════════════

/**
 * Derives a combined token string from a form field element:
 * name + id + placeholder + associated label text (if any).
 *
 * @param {Element} el - Form field element.
 * @returns {string} Lower-cased combined token string.
 */
function _fieldTokens(el) {
    var parts = [
        el.name || '',
        el.id || '',
        el.placeholder || '',
        el.getAttribute('aria-label') || '',
        el.getAttribute('data-label') || '',
    ];

    // Attempt to find an associated <label> via for="id" or wrapping element
    if (el.id) {
        var label = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
        if (label) parts.push(label.textContent || '');
    }
    if (!el.id || !document.querySelector('label[for="' + CSS.escape(el.id || '') + '"]')) {
        var parent = el.closest('label');
        if (parent) parts.push(parent.textContent || '');
    }

    return parts.join(' ').toLowerCase();
}

/**
 * Matches a field's combined token string against the rule table.
 * Returns the matching profile key name, or null if no match.
 *
 * @param {Element} el - Form field element.
 * @returns {string|null} Profile key (e.g. "USER_EMAIL") or null.
 */
function _matchField(el) {
    var tokens = _fieldTokens(el);
    for (var r = 0; r < _FIELD_RULES.length; r++) {
        var rule = _FIELD_RULES[r];
        for (var k = 0; k < rule.keys.length; k++) {
            if (tokens.indexOf(rule.keys[k]) !== -1) {
                return rule.profile;
            }
        }
    }
    return null;
}

// ══════════════════════════════════════════════════════════════════
// FIELD FILLING
// ══════════════════════════════════════════════════════════════════

/**
 * Fills a single form field with value, then dispatches 'input' and
 * 'change' events so React/Vue/Angular controlled inputs update.
 *
 * @param {Element} el - Target form element.
 * @param {string} value - Value to set.
 * @returns {boolean} True if fill succeeded without error.
 */
function _fillField(el, value) {
    try {
        var nativeInputValueSetter = Object.getOwnPropertyDescriptor(
            Object.getPrototypeOf(el), 'value'
        );
        if (nativeInputValueSetter && nativeInputValueSetter.set) {
            // React-compatible setter
            nativeInputValueSetter.set.call(el, value);
        } else {
            el.value = value;
        }
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
        return true;
    } catch (_) {
        return false;
    }
}

// ══════════════════════════════════════════════════════════════════
// AUTOFILL ORCHESTRATOR
// ══════════════════════════════════════════════════════════════════

/**
 * Builds a flat profile object from the /autofill response data,
 * adding synthesised first_name and last_name from USERNAME.
 *
 * @param {Object} data - Response from GET /autofill endpoint.
 * @returns {Object} Flat profile dict keyed by profile token names.
 */
function _buildProfile(data) {
    var fullName = (data.USERNAME || data.field_mappings && data.field_mappings.name) || '';
    var parts = fullName.trim().split(/\s+/);
    return {
        USERNAME: fullName,
        first_name: parts.length > 0 ? parts[0] : fullName,
        last_name: parts.length > 1 ? parts[parts.length - 1] : fullName,
        USER_EMAIL: data.USER_EMAIL || '',
        USER_PHONE: data.USER_PHONE || '',
        USER_LINKEDIN_URL: data.USER_LINKEDIN_URL || '',
        USER_PORTFOLIO_URL: data.USER_PORTFOLIO_URL || '',
        USER_LOCATION: data.USER_LOCATION || '',
        USER_YEARS_EXPERIENCE: String(data.USER_YEARS_EXPERIENCE || ''),
        default_resume: data.default_resume || '',
    };
}

/**
 * Iterates all visible form fields on the page, matches each to the
 * user profile by keyword heuristic, and fills matched fields.
 * Wraps each individual field fill in try/catch — never propagates.
 *
 * @param {Object} data - Autofill response payload from FastAPI.
 * @returns {number} Number of fields successfully filled.
 */
function _autofillPage(data) {
    var profile = _buildProfile(data);
    var fields = _getFormFields();
    var filledCount = 0;

    for (var i = 0; i < fields.length; i++) {
        try {
            var el = fields[i];
            var profileKey = _matchField(el);
            if (!profileKey) continue;
            var value = profile[profileKey];
            if (!value) continue;
            var ok = _fillField(el, value);
            if (ok) filledCount++;
        } catch (_) {
            // Per-field isolation — continue to next field
        }
    }

    console.log('[JobAgent] Autofill complete — filled', filledCount, 'of', fields.length, 'fields');
    return filledCount;
}

// ══════════════════════════════════════════════════════════════════
// MESSAGE LISTENER
// ══════════════════════════════════════════════════════════════════

/**
 * Listens for messages from the popup.
 * Handles action "autofill" by invoking _autofillPage().
 * Responds synchronously with filled field count.
 */
chrome.runtime.onMessage.addListener(function (message, _sender, sendResponse) {
    if (!message || message.action !== 'autofill') {
        sendResponse({ acknowledged: true });
        return false;
    }

    try {
        var count = _autofillPage(message.data || {});
        sendResponse({ filled: count });
    } catch (err) {
        // Catch-all — never let this crash the content script
        console.error('[JobAgent] Autofill error:', err);
        sendResponse({ filled: 0, error: err.message });
    }

    return false; // synchronous — no need to keep channel open
});
