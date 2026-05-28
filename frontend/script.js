document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatWindow = document.getElementById('chat-window');
    const sendBtn = document.getElementById('send-btn');
    const typingIndicator = document.getElementById('typing-indicator');
    const debugBtn = document.getElementById('debug-btn');
    const debugPanel = document.getElementById('debug-panel');
    const debugContent = document.getElementById('debug-content');

    // Generate a single session ID for the lifecycle of the page
    const sessionId = generateUUID();

    // Toggle debug panel
    debugBtn.addEventListener('click', () => {
        debugPanel.classList.toggle('active');
        // On smaller screens, slide over the chat
        if (window.innerWidth <= 1000) {
            if (debugPanel.classList.contains('active')) {
                debugPanel.style.display = 'block';
            } else {
                setTimeout(() => { debugPanel.style.display = 'none'; }, 300);
            }
        }
    });

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const message = chatInput.value.trim();
        if (!message) return;

        // 1. Add user message to UI
        addMessage(message, 'user');
        chatInput.value = '';
        
        // 2. Disable input and show typing
        chatInput.disabled = true;
        sendBtn.disabled = true;
        typingIndicator.classList.remove('hidden');
        chatWindow.scrollTop = chatWindow.scrollHeight;

        try {
            // 3. Send to API
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    session_id: sessionId
                })
            });

            if (!response.ok) {
                throw new Error(`API Error: ${response.status}`);
            }

            const data = await response.json();
            
            // 4. Add system response to UI
            addMessage(data.response, 'system');
            
            // 5. Update debug panel
            updateDebugPanel(data);

        } catch (error) {
            console.error('Chat error:', error);
            addMessage('⚠️ Sorry, I encountered an error connecting to the server.', 'system');
        } finally {
            // 6. Re-enable input
            chatInput.disabled = false;
            sendBtn.disabled = false;
            typingIndicator.classList.add('hidden');
            chatInput.focus();
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }
    });

    function addMessage(text, sender) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}-msg`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'msg-content';
        
        let formattedText = escapeHtml(text);
        if (sender === 'system') {
            formattedText = renderMarkdown(formattedText);
        } else {
            formattedText = `<p>${formattedText}</p>`;
        }
        
        contentDiv.innerHTML = formattedText;
        msgDiv.appendChild(contentDiv);
        
        // Insert before typing indicator
        chatWindow.insertBefore(msgDiv, typingIndicator);
    }

    function escapeHtml(value) {
        return value
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    function renderMarkdown(text) {
        const lines = text.split('\n');
        const blocks = [];
        let paragraph = [];
        let list = [];

        const flushParagraph = () => {
            if (paragraph.length) {
                blocks.push(`<p>${paragraph.join('<br>')}</p>`);
                paragraph = [];
            }
        };
        const flushList = () => {
            if (list.length) {
                blocks.push(`<ul>${list.map(item => `<li>${item}</li>`).join('')}</ul>`);
                list = [];
            }
        };

        lines.forEach((line) => {
            const listMatch = line.match(/^\s*[-*]\s+(.+)$/);
            if (listMatch) {
                flushParagraph();
                list.push(listMatch[1]);
            } else if (line.trim() === '') {
                flushParagraph();
                flushList();
            } else {
                flushList();
                paragraph.push(line);
            }
        });

        flushParagraph();
        flushList();

        return blocks.join('').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    }

    function updateDebugPanel(data) {
        const debugData = {
            intent: data.intent.intent,
            confidence: data.intent.confidence,
            sentiment: data.intent.sentiment,
            escalation: data.escalation,
            generation: data.generation,
            follow_up: data.pipeline_trace.follow_up,
            rewritten_query: data.pipeline_trace.augmented_query,
            previous_intent: data.retrieval.query_analysis.previous_intent,
            previous_category: data.retrieval.query_analysis.previous_category,
            reranking: data.retrieval.query_analysis.reranking,
            timings: data.pipeline_trace.timings,
            session_turn: data.session.turn_count,
            last_intent: data.session.last_intent,
            last_category: data.session.last_category,
            retrieval_count: data.retrieval.results.length
        };
        
        debugContent.textContent = JSON.stringify(debugData, null, 2);
    }

    function generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }
});
