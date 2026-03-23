document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------------
    // UI Elements
    // -------------------------------------------------------------------
    const uiResumePath = document.getElementById('resumePath');
    const uiSelectFile = document.getElementById('selectFile');
    const uiApiBaseUrl = document.getElementById('apiBaseUrl');
    const uiApiTestBtn = document.getElementById('apiTestBtn');
    const uiApiStatusText = document.getElementById('apiStatusText');
    
    // -------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------
    let currentResumePath = "";
    
    // -------------------------------------------------------------------
    // Initialization
    // -------------------------------------------------------------------
    chrome.storage.sync.get(['resume_pdf_path', 'api_base_url'], (data) => {
        if (data.resume_pdf_path) {
            currentResumePath = data.resume_pdf_path;
            uiResumePath.value = currentResumePath;
        }
        if (data.api_base_url) {
            uiApiBaseUrl.value = data.api_base_url;
        }
    });
    
    // -------------------------------------------------------------------
    // Event Listeners
    // -------------------------------------------------------------------
    
    // Pseudo file selection (since extensions cannot easily read raw filesystem paths securely without user gesture)
    uiSelectFile.addEventListener('click', () => {
        const path = prompt("Enter the absolute path to your resume PDF:\n(e.g., /Users/name/Documents/Resume.pdf)", currentResumePath);
        if (path !== null && path.trim() !== "") {
            currentResumePath = path.trim();
            uiResumePath.value = currentResumePath;
            chrome.storage.sync.set({ resume_pdf_path: currentResumePath }, () => {
                const originalText = uiSelectFile.textContent;
                uiSelectFile.textContent = "Saved ✓";
                setTimeout(() => uiSelectFile.textContent = originalText, 2000);
            });
        }
    });
    
    // Auto-save on manual edit
    uiResumePath.addEventListener('change', () => {
        currentResumePath = uiResumePath.value.trim();
        chrome.storage.sync.set({ resume_pdf_path: currentResumePath });
    });
    
    // API Base URL Auto-save
    uiApiBaseUrl.addEventListener('change', () => {
        let url = uiApiBaseUrl.value.trim();
        if (url && !url.endsWith('/')) {
            url += '/';
            uiApiBaseUrl.value = url;
        }
        chrome.storage.sync.set({ api_base_url: url });
    });
    
    // API Connection Test
    uiApiTestBtn.addEventListener('click', async () => {
        const baseUrl = uiApiBaseUrl.value.trim() || 'http://localhost:8000/';
        const url = baseUrl.endsWith('/') ? baseUrl + 'health' : baseUrl + '/health';
        
        uiApiStatusText.textContent = "Testing connection...";
        uiApiStatusText.className = "status-text status-pending";
        uiApiTestBtn.disabled = true;
        
        try {
            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json'
                },
                signal: AbortSignal.timeout(5000)
            });
            
            if (response.ok) {
                const data = await response.json();
                let dbStatus = "Unknown DB";
                if (data.db_connected !== undefined) {
                    dbStatus = data.db_connected ? "DB Connected" : "DB Disconnected";
                } else if (data.checks && data.checks.postgres) {
                    dbStatus = data.checks.postgres === 'ok' ? "DB Connected" : "DB Disconnected";
                } else if (data.db) {
                    dbStatus = data.db === 'connected' ? "DB Connected" : "DB Disconnected";
                }
                uiApiStatusText.textContent = `Success! API Version: ${data.version || 'Unknown'} | ${dbStatus}`;
                uiApiStatusText.className = "status-text status-success";
            } else {
                uiApiStatusText.textContent = `Server reachable, but returned error ${response.status}.`;
                uiApiStatusText.className = "status-text status-error";
            }
        } catch (err) {
            uiApiStatusText.textContent = `Connection failed: ${err.message}`;
            uiApiStatusText.className = "status-text status-error";
        } finally {
            uiApiTestBtn.disabled = false;
        }
    });
    
    // Add AbortSignal.timeout polyfill if needed
    if (!AbortSignal.timeout) {
        AbortSignal.timeout = function(ms) {
            const controller = new AbortController();
            setTimeout(() => controller.abort(new Error("TimeoutError")), ms);
            return controller.signal;
        };
    }
});
