// Aunalytics RAG Research Summarizer - Frontend JavaScript
const API_BASE_URL = "http://localhost:8080";

// TEMPORARY: Use fixed test UUID until authentication is implemented
// TODO: Replace with proper auth system (sign in/sign up functionality)
const userId = "550e8400-e29b-41d4-a716-446655440000";

console.log('Session User ID:', userId);

// Wait for page to load
document.addEventListener('DOMContentLoaded', () => {
    const scrapeBtn = document.querySelector('.scrape-btn');
    const urlInput = document.querySelector('.url-input-wrapper input');
    const summaryBox = document.querySelector('.summary-box');
    const papersScroller = document.querySelector('.papers-scroller');
    const documentsList = document.getElementById('documents-list');
    const chatSendBtn = document.querySelector('.chat-send-btn');
    const chatInput = document.querySelector('.chat-input');
    const chatHistory = document.querySelector('.chat-history');
    const fileUpload = document.getElementById('chat-file-upload');
    const filePreview = document.getElementById('file-preview-area');
    const clearDataBtn = document.getElementById('clear-data-btn');

    // Load document list on page load
    if (documentsList) {
        loadDocumentList();
    }

    // Function to load and display document list
    async function loadDocumentList() {
        if (!documentsList) return;

        try {
            const response = await fetch(`${API_BASE_URL}/api/documents/${userId}/list`);
            if (!response.ok) throw new Error('Failed to load documents');

            const data = await response.json();

            if (data.documents.length === 0) {
                documentsList.innerHTML = '<p style="color: #666; text-align: center; padding: 20px;">No documents uploaded yet</p>';
                return;
            }

            documentsList.innerHTML = '';
            data.documents.forEach(doc => {
                const docCard = document.createElement('div');
                docCard.className = 'paper-card';
                docCard.style.display = 'flex';
                docCard.style.justifyContent = 'space-between';
                docCard.style.alignItems = 'center';
                
                docCard.innerHTML = `
                    <div style="flex: 1;">
                        <div class="paper-icon">📄</div>
                        <div class="paper-title">${doc.source}</div>
                        <div style="font-size: 0.75rem; color: #666; margin-top: 4px;">${doc.chunk_count} chunks</div>
                    </div>
                    <button class="delete-doc-btn" data-source="${doc.source}" 
                            style="background: #dc3545; color: white; border: none; padding: 6px 12px; 
                                   border-radius: 4px; cursor: pointer; font-size: 0.8rem;">
                        Delete
                    </button>
                `;

                // Add delete handler
                const deleteBtn = docCard.querySelector('.delete-doc-btn');
                deleteBtn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    const sourceName = deleteBtn.dataset.source;
                    
                    if (!confirm(`Delete "${sourceName}" and all its chunks?`)) return;

                    deleteBtn.disabled = true;
                    deleteBtn.textContent = 'Deleting...';

                    try {
                        const response = await fetch(
                            `${API_BASE_URL}/api/documents/${userId}/source/${encodeURIComponent(sourceName)}`,
                            { method: 'DELETE' }
                        );

                        if (response.ok) {
                            loadDocumentList(); // Reload list
                        } else {
                            throw new Error('Delete failed');
                        }
                    } catch (error) {
                        console.error('Delete error:', error);
                        alert('Failed to delete document');
                        deleteBtn.disabled = false;
                        deleteBtn.textContent = 'Delete';
                    }
                });

                documentsList.appendChild(docCard);
            });

        } catch (error) {
            console.error('Error loading documents:', error);
            documentsList.innerHTML = '<p style="color: #ff6b6b; text-align: center; padding: 20px;">Error loading documents</p>';
        }
    }

    // Clear all documents button
    if (clearDataBtn) {
        clearDataBtn.addEventListener('click', async () => {
            if (!confirm('This will delete all uploaded documents and scraped papers. Continue?')) {
                return;
            }

            clearDataBtn.disabled = true;
            clearDataBtn.textContent = 'Clearing...';

            try {
                const response = await fetch(`${API_BASE_URL}/api/documents/${userId}`, {
                    method: 'DELETE'
                });

                if (response.ok) {
                    // Reset UI
                    summaryBox.innerHTML = '<span class="placeholder-text">SUMMARY OUTPUT<br><small>(Content will appear here after analysis)</small></span>';
                    loadDocumentList(); // Reload document list
                    chatHistory.innerHTML = `
                        <div class="empty-state">
                            <div class="ai-icon">AI</div>
                            <p>Once the papers are analyzed, ask me anything about the research.</p>
                        </div>
                    `;
                    alert('All documents cleared successfully!');
                } else {
                    throw new Error('Failed to clear documents');
                }
            } catch (error) {
                console.error('Clear error:', error);
                alert('Error clearing documents: ' + error.message);
            } finally {
                clearDataBtn.disabled = false;
                clearDataBtn.textContent = 'Clear All Documents';
            }
        });
    }

    // Scrape website button
    if (scrapeBtn && urlInput) {
        scrapeBtn.addEventListener('click', async () => {
            const url = urlInput.value.trim();
            if (!url) {
                alert('Please enter a professor website URL');
                return;
            }

            scrapeBtn.disabled = true;
            scrapeBtn.textContent = 'Analyzing...';
            summaryBox.innerHTML = '<p style="color: #89c2ff;">Scraping website and extracting research papers...</p>';

            try {
                const response = await fetch(`${API_BASE_URL}/api/scrape`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        professor_url: url,
                        user_id: userId,
                        max_papers: 10
                    })
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || `HTTP ${response.status}`);
                }

                const data = await response.json();
                
                // Display summary
                summaryBox.innerHTML = `
                    <h3 style="color: #89c2ff; margin-bottom: 12px;">Analysis Complete</h3>
                    <p><strong>Source:</strong> ${data.source}</p>
                    <p><strong>Papers Found:</strong> ${data.papers_found}</p>
                    <p><strong>Chunks Embedded:</strong> ${data.chunks_created}</p>
                    <p style="margin-top: 12px;"><em>${data.message}</em></p>
                `;

                // Display papers
                if (data.papers && data.papers.length > 0) {
                    papersScroller.innerHTML = '';
                    data.papers.forEach(paper => {
                        const paperCard = document.createElement('div');
                        paperCard.className = 'paper-card';
                        paperCard.innerHTML = `
                            <div class="paper-icon">[PDF]</div>
                            <div class="paper-title">${paper.title} (${paper.year})</div>
                        `;
                        paperCard.title = `${paper.authors}\n\n${paper.abstract}`;
                        papersScroller.appendChild(paperCard);
                    });
                }

                // Enable chat
                if (chatHistory) {
                    chatHistory.innerHTML = `
                        <div class="ai-message">
                            <div class="ai-icon">AI</div>
                            <p>I've analyzed ${data.papers_found} research papers. Ask me anything about the professor's research!</p>
                        </div>
                    `;
                }

            } catch (error) {
                console.error('Scrape error:', error);
                
                // Better error message for common issues
                let errorMessage = error.message;
                if (errorMessage.includes('403') || errorMessage.includes('Forbidden') || errorMessage.includes('blocking')) {
                    errorMessage = `
                        <strong>Access Blocked</strong><br>
                        Research sites (ResearchGate, ASEE) are blocking automated access.<br><br>
                        <strong>Try instead:</strong><br>
                        - Use the "Upload Document" button to add papers directly<br>
                        - Provide a Google Scholar profile link<br>
                        - Manually download PDFs and upload them
                    `;
                }
                
                summaryBox.innerHTML = `<p style="color: #ff6b6b;">${errorMessage}</p>`;
            } finally {
                scrapeBtn.disabled = false;
                scrapeBtn.textContent = 'Analyze Website';
            }
        });
    }

    // Ask question
    const askQuestion = async () => {
        if (!chatInput || !chatHistory) return;
        
        const question = chatInput.value.trim();
        if (!question) return;

        // Add user message
        const userMsg = document.createElement('div');
        userMsg.className = 'user-message';
        userMsg.innerHTML = `<p>${question}</p>`;
        chatHistory.appendChild(userMsg);
        chatInput.value = '';
        chatHistory.scrollTop = chatHistory.scrollHeight;

        // Show loading
        const loadingMsg = document.createElement('div');
        loadingMsg.className = 'ai-message loading';
        loadingMsg.innerHTML = '<div class="ai-icon">...</div><p>Thinking...</p>';
        chatHistory.appendChild(loadingMsg);
        chatHistory.scrollTop = chatHistory.scrollHeight;

        try {
            const response = await fetch(`${API_BASE_URL}/api/question`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: question,
                    user_id: userId
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || `HTTP ${response.status}`);
            }

            const data = await response.json();

            // Remove loading
            loadingMsg.remove();

            // Add AI response
            const aiMsg = document.createElement('div');
            aiMsg.className = 'ai-message';
            aiMsg.innerHTML = `
                <div class="ai-icon">AI</div>
                <div>
                    <p>${data.answer}</p>
                    <small style="color: #666; margin-top: 8px; display: block;">
                        Based on ${data.num_sources} source(s)
                    </small>
                </div>
            `;
            chatHistory.appendChild(aiMsg);
            chatHistory.scrollTop = chatHistory.scrollHeight;

        } catch (error) {
            console.error('Question error:', error);
            loadingMsg.innerHTML = `
                <div class="ai-icon">!</div>
                <p style="color: #ff6b6b;">Error: ${error.message}</p>
            `;
        }
    };

    if (chatSendBtn && chatInput) {
        chatSendBtn.addEventListener('click', askQuestion);
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') askQuestion();
        });
    }

    // File upload
    if (fileUpload && filePreview) {
        fileUpload.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            filePreview.style.display = 'block';
            filePreview.textContent = `Uploading ${file.name}...`;

            const formData = new FormData();
            formData.append('file', file);
            formData.append('user_id', userId);

            try {
                const response = await fetch(`${API_BASE_URL}/api/upload`, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || `HTTP ${response.status}`);
                }

                const data = await response.json();
                filePreview.textContent = `Uploaded: ${file.name} (${data.chunks_created} chunks)`;
                
                // Reload document list
                if (documentsList) {
                    loadDocumentList();
                }
                
                setTimeout(() => {
                    filePreview.style.display = 'none';
                    fileUpload.value = '';
                }, 3000);

            } catch (error) {
                console.error('Upload error:', error);
                filePreview.textContent = `Upload failed: ${error.message}`;
            }
        });
    }
});
