const API_URL = "http://127.0.0.1:8000"; // Ensure this matches your FastAPI port

// --- 1. ANALYZE WEBSITE LOGIC ---
async function analyzeWebsite() {
    const urlInput = document.querySelector('.url-input-wrapper input');
    const summaryBox = document.querySelector('.summary-box');
    const papersContainer = document.querySelector('.papers-scroller');
    const btn = document.querySelector('.scrape-btn');
    
    const url = urlInput.value;
    if (!url) return alert("Please enter a URL");

    // UI: Show loading state
    btn.textContent = "Analyzing...";
    btn.disabled = true;
    summaryBox.innerHTML = "<p style='color:white; font-style:italic;'>Running AI analysis... this may take a moment.</p>";

    try {
        // Send data to FastAPI
        const response = await fetch(`${API_URL}/analyze`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: url })
        });

        const data = await response.json();

        // UI: Update Summary
        summaryBox.innerHTML = `<p style="color:white; line-height: 1.6;">${data.summary}</p>`;

        // UI: Update Papers (Clear old ones, add new ones)
        papersContainer.innerHTML = ""; 
        data.papers.forEach(paper => {
            const paperCard = `
                <div class="paper-card">
                    <div class="paper-icon">${paper.icon}</div>
                    <div class="paper-title">${paper.title}</div>
                </div>
            `;
            papersContainer.innerHTML += paperCard;
        });

        // Unlock Chat & Attachment Button
        document.querySelector('.chat-input').disabled = false;
        document.querySelector('.chat-send-btn').disabled = false;
        document.querySelector('.chat-attach-btn').disabled = false; // Unlock paperclip

    } catch (error) {
        console.error("Error:", error);
        summaryBox.innerHTML = "<p style='color:red;'>Error analyzing website. Is the backend running?</p>";
    } finally {
        btn.textContent = "Analyze Website";
        btn.disabled = false;
    }
}

// --- 2. FILE ATTACHMENT LOGIC ---
const fileInput = document.getElementById('chat-file-upload');
const previewArea = document.getElementById('file-preview-area');

// Listen for file selection
if(fileInput) {
    fileInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            // Show file name preview
            previewArea.style.display = 'block';
            previewArea.innerHTML = `
                <div class="file-selected-badge">
                    📄 ${file.name}
                    <span class="remove-file-btn" onclick="clearFile()" style="margin-left:8px; cursor:pointer; color:#ff6b6b;">✖</span>
                </div>
            `;
        }
    });
}

function clearFile() {
    fileInput.value = ''; // Clear the actual input
    previewArea.style.display = 'none'; // Hide preview text
    previewArea.innerHTML = '';
}

// --- 3. CHAT LOGIC ---
async function sendMessage() {
    const input = document.querySelector('.chat-input');
    const history = document.querySelector('.chat-history');
    const message = input.value;
    const file = fileInput ? fileInput.files[0] : null;

    // Only send if there is a message OR a file
    if (!message && !file) return;

    // 1. Add User Message to UI immediately
    let userHtml = `<div style="text-align:right; margin: 10px 0; color: #89c2ff;">`;
    if (message) userHtml += `<div><strong>You:</strong> ${message}</div>`;
    if (file) userHtml += `<div style="font-size: 0.85em; opacity: 0.8; margin-top:5px; background:rgba(137,194,255,0.1); padding:5px; display:inline-block; border-radius:4px;">📎 Attached: ${file.name}</div>`;
    userHtml += `</div>`;
    
    history.innerHTML += userHtml;
    history.scrollTop = history.scrollHeight; // Auto scroll down

    // Clear inputs immediately
    input.value = "";
    clearFile();

    // 2. Prepare data for Backend (Using FormData for files)
    const formData = new FormData();
    if (message) formData.append("message", message);
    if (file) formData.append("file", file);

    try {
        // 3. Send to FastAPI
        const response = await fetch(`${API_URL}/chat`, {
            method: "POST",
            body: formData // No JSON headers needed for FormData
        });
        
        const data = await response.json();

        // 4. Add AI Response to UI
        history.innerHTML += `<div style="text-align:left; margin: 10px 0; color: white;"><strong>AI:</strong> ${data.response}</div>`;
        history.scrollTop = history.scrollHeight; // Auto scroll down
        
    } catch (error) {
        console.error("Chat Error:", error);
        history.innerHTML += `<div style="text-align:left; margin: 10px 0; color: #ff6b6b;">Error connecting to server.</div>`;
    }
}

// --- 4. EVENT LISTENERS ---
// Wait for HTML to load before attaching clicks
document.addEventListener('DOMContentLoaded', () => {
    const scrapeBtn = document.querySelector('.scrape-btn');
    const sendBtn = document.querySelector('.chat-send-btn');
    const chatInput = document.querySelector('.chat-input');

    if(scrapeBtn) scrapeBtn.addEventListener('click', analyzeWebsite);
    if(sendBtn) sendBtn.addEventListener('click', sendMessage);
    
    // Allow pressing "Enter" to send chat
    if(chatInput) {
        chatInput.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }
});