const API_URL = "http://127.0.0.1:8000";
let conversationHistory = [];

// Generate a user_id for this session (in production, this would come from auth)
// FOR TESTING: Using fixed user_id to ensure consistency across sessions
const USE_FIXED_USER_ID = true;  // Set to false for production
const FIXED_USER_ID = 'test-user-123';

const USER_ID = USE_FIXED_USER_ID ? FIXED_USER_ID : (localStorage.getItem('user_id') || (() => {
    const id = crypto.randomUUID();
    localStorage.setItem('user_id', id);
    console.log('🆔 NEW USER ID GENERATED:', id);
    return id;
})());

console.log('🆔 Current User ID:', USER_ID);

async function analyzeWebsite() {
    const urlInput = document.querySelector(".url-input-wrapper input");
    const summaryBox = document.querySelector(".summary-box");
    const papersSection = document.querySelector(".papers-section");
    const papersContainer = document.querySelector(".papers-scroller");
    const btn = document.querySelector(".scrape-btn");

    const url = urlInput.value.trim();
    if (!url) return alert("Please enter a URL.");

    btn.textContent = "Analyzing...";
    btn.disabled = true;
    summaryBox.innerHTML = `<p style="color:white; font-style:italic;">Analyzing website…</p>`;
    papersSection.style.display = "none";

    try {
        const response = await fetch(`${API_URL}/scrape/analyze`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url, user_id: USER_ID })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Backend error");

        summaryBox.innerHTML = `<p style="color:white;">${data.summary}</p>`;
        
        // Show how many chunks were stored
        if (data.chunks_stored) {
            const total = data.chunks_stored.homepage + data.chunks_stored.papers + data.chunks_stored.summary;
            summaryBox.innerHTML += `<p style="color:#888; font-size:0.85em; margin-top:10px;">
                ✓ Stored ${total} chunks in your knowledge base (Homepage: ${data.chunks_stored.homepage}, 
                Papers: ${data.chunks_stored.papers}, Summary: ${data.chunks_stored.summary})
            </p>`;
        }

        papersContainer.innerHTML = "";
        data.papers.forEach((paper, index) => {
            papersContainer.innerHTML += `
                <div class="paper-card" style="animation-delay:${index * 0.08}s"
                     onclick="openPaper('${encodeURIComponent(paper.description)}', '${encodeURIComponent(paper.title)}')">
                    <div class="paper-icon">📄</div>
                    <div class="paper-title">${paper.title}</div>
                </div>
            `;
        });

        papersSection.style.display = "block";

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

async function openPaper(encodedText, encodedTitle) {
    const text = decodeURIComponent(encodedText);
    const title = decodeURIComponent(encodedTitle);

    const historyBox = document.querySelector(".chat-history");
    historyBox.innerHTML = "";

    const container = document.createElement("div");
    container.style.textAlign = "left";
    container.style.margin = "10px 0";
    container.style.color = "white";
    container.style.lineHeight = "1.5";
    historyBox.appendChild(container);

    const fullText = `Paper Title: ${title}\n\n${text}`;
    await typeText(container, fullText, 8);
    historyBox.scrollTop = historyBox.scrollHeight;
}

async function typeText(element, text, speed = 10) {
    element.innerHTML = "";
    for (let i = 0; i < text.length; i++) {
        element.innerHTML += text[i].replace(/\n/g, "<br>");
        await new Promise(resolve => setTimeout(resolve, speed));
    }
}

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

async function uploadFile(file) {
    const historyBox = document.querySelector(".chat-history");
    
    // Show uploading status
    const statusDiv = document.createElement("div");
    statusDiv.style.textAlign = "center";
    statusDiv.style.margin = "10px 0";
    statusDiv.style.color = "#89c2ff";
    statusDiv.style.fontStyle = "italic";
    statusDiv.innerHTML = `Uploading and processing ${file.name}...`;
    historyBox.appendChild(statusDiv);
    historyBox.scrollTop = historyBox.scrollHeight;
    
    try {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("user_id", USER_ID);
        
        const response = await fetch(`${API_URL}/ingest/`, {
            method: "POST",
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || "Upload failed");
        }
        
        // Update status with success
        statusDiv.style.color = "#4ade80";
        statusDiv.innerHTML = `✓ Successfully processed ${data.num_chunks} chunks from ${file.name}`;
        
        return data;
        
    } catch (err) {
        statusDiv.style.color = "#ff6b6b";
        statusDiv.innerHTML = `✗ Error uploading file: ${err.message}`;
        throw err;
    }
}

async function sendMessage() {
    const input = document.querySelector(".chat-input");
    const historyBox = document.querySelector(".chat-history");
    const message = input.value.trim();
    const file = fileInput.files[0] || null;

    if (!message && !file) return;

    // Display user message
    if (message) {
        const userDiv = document.createElement("div");
        userDiv.style.textAlign = "right";
        userDiv.style.margin = "10px 0";
        userDiv.style.color = "#89c2ff";
        userDiv.innerHTML = `<strong>You:</strong> ${message}`;
        historyBox.appendChild(userDiv);
        historyBox.scrollTop = historyBox.scrollHeight;
    }

    input.value = "";
    
    // Handle file upload first if present
    if (file) {
        try {
            await uploadFile(file);
            clearFile();
        } catch (err) {
            console.error("Upload error:", err);
            return; // Don't proceed with query if upload failed
        }
    }
    
    // If there's a message, query the knowledge base
    if (message) {
        const thinkingDiv = document.createElement("div");
        thinkingDiv.style.textAlign = "left";
        thinkingDiv.style.margin = "10px 0";
        thinkingDiv.style.color = "white";
        thinkingDiv.style.fontStyle = "italic";
        thinkingDiv.innerHTML = "Thinking...";
        historyBox.appendChild(thinkingDiv);
        historyBox.scrollTop = historyBox.scrollHeight;

        try {
            const formData = new FormData();
            formData.append("message", message);
            formData.append("user_id", USER_ID);
            
            // Check if it's a summary request to use more chunks
            const isSummary = message.toLowerCase().includes('summarize') || 
                            message.toLowerCase().includes('summary') ||
                            message.toLowerCase().includes('overview');
            if (isSummary) {
                formData.append("top_k", "15");
            }
            
            const response = await fetch(`${API_URL}/query/`, {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Query failed");
            }

            // Remove thinking message
            historyBox.removeChild(thinkingDiv);

            // Display AI response
            const aiDiv = document.createElement("div");
            aiDiv.style.textAlign = "left";
            aiDiv.style.margin = "10px 0";
            aiDiv.style.color = "white";
            historyBox.appendChild(aiDiv);

            await typeText(aiDiv, data.response, 10);
            
            // Show source count
            if (data.num_sources > 0) {
                const sourcesDiv = document.createElement("div");
                sourcesDiv.style.textAlign = "left";
                sourcesDiv.style.margin = "5px 0";
                sourcesDiv.style.color = "#888";
                sourcesDiv.style.fontSize = "0.85em";
                sourcesDiv.innerHTML = `<em>Based on ${data.num_sources} document chunks</em>`;
                historyBox.appendChild(sourcesDiv);
            }

            historyBox.scrollTop = historyBox.scrollHeight;

        } catch (err) {
            historyBox.removeChild(thinkingDiv);
            
            const errDiv = document.createElement("div");
            errDiv.style.textAlign = "left";
            errDiv.style.margin = "10px 0";
            errDiv.style.color = "#ff6b6b";
            errDiv.innerHTML = `Error: ${err.message}`;
            historyBox.appendChild(errDiv);
            console.error(err);
        }
    }
}

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
    
    // Show user ID in console for debugging
    console.log("Session User ID:", USER_ID);
});
