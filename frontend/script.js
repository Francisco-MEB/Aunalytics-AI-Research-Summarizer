// ================================
// CONFIG
// ================================
const API_URL = "http://127.0.0.1:8000";
let conversationHistory = [];


// ================================
// 1. ANALYZE WEBSITE
// ================================
async function analyzeWebsite() {
    const urlInput = document.querySelector(".url-input-wrapper input");
    const summaryBox = document.querySelector(".summary-box");
    const papersContainer = document.querySelector(".papers-scroller");
    const btn = document.querySelector(".scrape-btn");

    const url = urlInput.value.trim();
    if (!url) return alert("Please enter a URL.");

    // UI state
    btn.textContent = "Analyzing...";
    btn.disabled = true;
    summaryBox.innerHTML = `<p style="color:white; font-style:italic;">Analyzing website…</p>`;

    try {
        const response = await fetch(`${API_URL}/scrape/analyze`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Backend error");

        // Show summary
        summaryBox.innerHTML = `<p style="color:white;">${data.summary}</p>`;

        // Research papers
        papersContainer.innerHTML = "";
        data.papers.forEach((paper) => {
            papersContainer.innerHTML += `
                <div class="paper-card" onclick="openPaper('${encodeURIComponent(paper.description)}', '${encodeURIComponent(paper.title)}')">
                    <div class="paper-icon">📄</div>
                    <div class="paper-title">${paper.title}</div>
                </div>
            `;
        });

        // Enable chat
        document.querySelector(".chat-input").disabled = false;
        document.querySelector(".chat-send-btn").disabled = false;
        document.querySelector(".chat-attach-btn").disabled = false;

    } catch (err) {
        summaryBox.innerHTML = `<p style="color:red;">Error analyzing website. Is the backend running?</p>`;
        console.error(err);
    } finally {
        btn.textContent = "Analyze Website";
        btn.disabled = false;
    }
}



// ================================
// 2. CLICKABLE PAPER → show paper text in chat (with typing animation)
// ================================
async function openPaper(encodedText, encodedTitle) {
    const text = decodeURIComponent(encodedText);
    const title = decodeURIComponent(encodedTitle);

    const historyBox = document.querySelector(".chat-history");
    historyBox.innerHTML = ""; // clear for clean display

    const container = document.createElement("div");
    container.style.textAlign = "left";
    container.style.margin = "10px 0";
    container.style.color = "white";
    container.style.lineHeight = "1.5";
    historyBox.appendChild(container);

    const fullText = `Paper Title: ${title}\n\n${text}`;

    // Type the text slowly
    await typeText(container, fullText, 8);

    historyBox.scrollTop = historyBox.scrollHeight;
}



// ================================
// ⭐ CHATGPT TYPING EFFECT
// ================================
async function typeText(element, text, speed = 10) {
    element.innerHTML = "";  
    for (let i = 0; i < text.length; i++) {
        element.innerHTML += text[i]
            .replace(/\n/g, "<br>");  
        await new Promise(resolve => setTimeout(resolve, speed));
    }
}



// ================================
// 3. FILE PREVIEW HANDLING
// ================================
const fileInput = document.getElementById("chat-file-upload");
const previewArea = document.getElementById("file-preview-area");

if (fileInput) {
    fileInput.addEventListener("change", function () {
        const file = this.files[0];
        if (file) {
            previewArea.style.display = "block";
            previewArea.innerHTML = `
                <div class="file-selected-badge">
                    📄 ${file.name}
                    <span class="remove-file-btn" onclick="clearFile()" 
                          style="margin-left:8px; cursor:pointer; color:#ff6b6b;">✖</span>
                </div>
            `;
        }
    });
}

function clearFile() {
    fileInput.value = "";
    previewArea.style.display = "none";
    previewArea.innerHTML = "";
}



// ================================
// 4. SEND CHAT MESSAGE (with typing animation)
// ================================
async function sendMessage() {
    const input = document.querySelector(".chat-input");
    const historyBox = document.querySelector(".chat-history");
    const message = input.value.trim();
    const file = fileInput.files[0] || null;

    if (!message && !file) return;

    // User message UI
    const userDiv = document.createElement("div");
    userDiv.style.textAlign = "right";
    userDiv.style.margin = "10px 0";
    userDiv.style.color = "#89c2ff";

    if (message) {
        userDiv.innerHTML = `<strong>You:</strong> ${message}`;
    }
    if (file) {
        userDiv.innerHTML += `<div style="font-size:.85em; opacity:.8; margin-top:5px;">📎 ${file.name}</div>`;
    }

    historyBox.appendChild(userDiv);
    historyBox.scrollTop = historyBox.scrollHeight;

    // Save conversation
    if (message) {
        conversationHistory.push({ role: "user", content: message });
    }

    input.value = "";
    clearFile();

    // Prepare payload
    const formData = new FormData();
    if (message) formData.append("message", message);
    if (file) formData.append("file", file);
    formData.append("history", JSON.stringify(conversationHistory));

    try {
        const response = await fetch(`${API_URL}/chat/`, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        // AI response container
        const aiDiv = document.createElement("div");
        aiDiv.style.textAlign = "left";
        aiDiv.style.margin = "10px 0";
        aiDiv.style.color = "white";
        historyBox.appendChild(aiDiv);

        // Type text slowly
        await typeText(aiDiv, data.response, 10);

        historyBox.scrollTop = historyBox.scrollHeight;

        conversationHistory.push({ role: "assistant", content: data.response });

    } catch (err) {
        const errDiv = document.createElement("div");
        errDiv.style.textAlign = "left";
        errDiv.style.margin = "10px 0";
        errDiv.style.color = "#ff6b6b";
        errDiv.innerHTML = "Error contacting server.";
        historyBox.appendChild(errDiv);
        console.error(err);
    }
}



// ================================
// 5. EVENT LISTENERS
// ================================
document.addEventListener("DOMContentLoaded", () => {
    const analyzeBtn = document.querySelector(".scrape-btn");
    const sendBtn = document.querySelector(".chat-send-btn");
    const chatInput = document.querySelector(".chat-input");

    if (analyzeBtn) analyzeBtn.addEventListener("click", analyzeWebsite);
    if (sendBtn) sendBtn.addEventListener("click", sendMessage);

    if (chatInput) {
        chatInput.addEventListener("keypress", (e) => {
            if (e.key === "Enter") sendMessage();
        });
    }
});
