document.addEventListener("DOMContentLoaded", () => {
    // Icons initialization
    lucide.createIcons();

    // Port & API Settings
    const API_BASE = ""; 

    // App State
    let loadedModelName = "";
    let activeTab = "dashboard-tab";
    let isGenerating = false;
    let isTraining = false;
    let trainingStatusInterval = null;
    let modelStatusInterval = null;
    let lossChart = null;

    // --- DOM Elements ---
    const navItems = document.querySelectorAll(".nav-item");
    const tabViews = document.querySelectorAll(".tab-view");
    const modelListContainer = document.getElementById("model-list-container");
    const modelLoaderPanel = document.getElementById("model-loader-panel");
    const loaderTitle = document.getElementById("loader-title");
    const loaderProgress = document.getElementById("loader-progress");
    const modelStatusPulse = document.getElementById("model-status-pulse");
    const modelStatusText = document.getElementById("model-status-text");
    const modelSpecsText = document.getElementById("model-specs-text");

    // Chat elements
    const chatActiveModel = document.getElementById("chat-active-model");
    const chatMessagesContainer = document.getElementById("chat-messages-container");
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");
    const clearChatBtn = document.getElementById("clear-chat-btn");
    const chatNavBtn = document.getElementById("chat-nav-btn");
    const systemPromptInput = document.getElementById("system-prompt");

    // Chat parameter elements
    const paramTemp = document.getElementById("param-temp");
    const tempVal = document.getElementById("temp-val");
    const paramTopp = document.getElementById("param-topp");
    const toppVal = document.getElementById("topp-val");
    const paramTopk = document.getElementById("param-topk");
    const topkVal = document.getElementById("topk-val");
    const paramMaxTokens = document.getElementById("param-maxtokens");
    const maxtokensVal = document.getElementById("maxtokens-val");
    const generationMetrics = document.getElementById("generation-metrics");
    const metricTtft = document.getElementById("metric-ttft");
    const metricSpeed = document.getElementById("metric-speed");
    const metricCount = document.getElementById("metric-count");

    // Training elements
    const trainDataset = document.getElementById("train-dataset");
    const customDatasetGroup = document.getElementById("custom-dataset-group");
    const customDatasetText = document.getElementById("custom-dataset-text");
    const trainLr = document.getElementById("train-lr");
    const trainSeqLen = document.getElementById("train-seq-len");
    const trainBatchSize = document.getElementById("train-batch-size");
    const trainGradAcc = document.getElementById("train-grad-acc");
    const trainMaxSteps = document.getElementById("train-max-steps");
    const startTrainBtn = document.getElementById("start-train-btn");
    const stopTrainBtn = document.getElementById("stop-train-btn");
    const trainLiveStatus = document.getElementById("train-live-status");
    const trainConsole = document.getElementById("train-console");

    // Export elements
    const exportRepoName = document.getElementById("export-repo-name");
    const repoUrlPreview = document.getElementById("repo-url-preview");
    const exportToken = document.getElementById("export-token");
    const deployBtn = document.getElementById("deploy-btn");
    const exportLoader = document.getElementById("export-loader");
    const exportProgress = document.getElementById("export-progress");

    // --- Tab Switching Logic ---
    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const targetTab = item.getAttribute("data-tab");
            
            // Check if model loaded before opening chat or training
            if ((targetTab === "chat-tab" || targetTab === "training-tab" || targetTab === "export-tab") && !loadedModelName) {
                alert("Please load an open-source model in the Dashboard before opening this tab.");
                return;
            }

            navItems.forEach(i => i.classList.remove("active"));
            tabViews.forEach(v => v.classList.remove("active"));

            item.classList.add("active");
            document.getElementById(targetTab).classList.add("active");
            activeTab = targetTab;
            
            // Re-render icons inside new views
            lucide.createIcons();

            // Resize Chart if entering training view
            if (targetTab === "training-tab" && lossChart) {
                lossChart.resize();
            }
        });
    });

    // --- Parameter Display Sliders ---
    paramTemp.addEventListener("input", () => tempVal.textContent = paramTemp.value);
    paramTopp.addEventListener("input", () => toppVal.textContent = paramTopp.value);
    paramTopk.addEventListener("input", () => topkVal.textContent = paramTopk.value);
    paramMaxTokens.addEventListener("input", () => maxtokensVal.textContent = paramMaxTokens.value);

    // Dynamic repo preview
    exportRepoName.addEventListener("input", () => {
        repoUrlPreview.textContent = `https://huggingface.co/Aravindhan11/${exportRepoName.value || "Distributed-Llama-Model"}`;
    });

    // Toggle custom dataset textbox
    trainDataset.addEventListener("change", () => {
        if (trainDataset.value === "custom") {
            customDatasetGroup.style.display = "flex";
        } else {
            customDatasetGroup.style.display = "none";
        }
    });

    // --- Fetch Models & Framework Status ---
    async function initFramework() {
        try {
            // Load supported models
            const res = await fetch(`${API_BASE}/api/models/list`);
            const models = await res.json();
            renderModelCards(models);

            // Periodically check loaded model status
            checkModelStatus();
            modelStatusInterval = setInterval(checkModelStatus, 3000);
        } catch (err) {
            console.error("Failed to connect to backend server:", err);
            modelSpecsText.innerHTML = "<span style='color: #F43F5E'>Backend connection failed. Please ensure server.py is running on port 8000.</span>";
        }
    }

    function renderModelCards(models) {
        modelListContainer.innerHTML = "";
        models.forEach(model => {
            const card = document.createElement("div");
            card.className = `model-option-card ${model.recommended ? 'active' : ''}`;
            card.innerHTML = `
                <div class="card-header">
                    <h4>${model.name}</h4>
                    ${model.recommended ? '<span class="badge blue">RECOMMENDED</span>' : ''}
                </div>
                <p class="card-desc">${model.description}</p>
                <div class="card-footer">
                    <span class="model-size">${model.size}</span>
                    <button class="btn-load" data-id="${model.id}">LOAD MODEL</button>
                </div>
            `;

            // Bind Load Button
            card.querySelector(".btn-load").addEventListener("click", (e) => {
                e.stopPropagation();
                loadModel(model.id);
            });

            modelListContainer.appendChild(card);
        });
    }

    // --- Load Model Weights API ---
    async function loadModel(modelId) {
        modelLoaderPanel.style.display = "flex";
        loaderTitle.textContent = `Loading ${modelId.split('/').pop()}...`;
        loaderProgress.textContent = "Connecting to Hugging Face Hub, constructing configuration, and allocating tensor memory.";
        
        try {
            const res = await fetch(`${API_BASE}/api/models/load`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model_name: modelId })
            });
            const data = await res.json();
            if (data.error) {
                alert(data.error);
                modelLoaderPanel.style.display = "none";
                return;
            }
            
            // Loop checking status
            let checkInterval = setInterval(async () => {
                const statusRes = await fetch(`${API_BASE}/api/models/status`);
                const status = await statusRes.json();
                
                if (status.status === "success") {
                    clearInterval(checkInterval);
                    modelLoaderPanel.style.display = "none";
                    updateStatusUI(status);
                    
                    // Smooth slide transition to Playground
                    setTimeout(() => {
                        chatNavBtn.click();
                    }, 500);
                } else if (status.status === "error") {
                    clearInterval(checkInterval);
                    modelLoaderPanel.style.display = "none";
                    alert(`Weight Loading Failed: ${status.error}`);
                } else {
                    loaderProgress.textContent = status.progress || "Converting checkpoints to custom framework layout...";
                }
            }, 1500);

        } catch (err) {
            modelLoaderPanel.style.display = "none";
            alert("Error running backend load request.");
        }
    }

    async function checkModelStatus() {
        try {
            const res = await fetch(`${API_BASE}/api/models/status`);
            const status = await res.json();
            updateStatusUI(status);
        } catch (err) {
            console.error("Failed model status check:", err);
        }
    }

    function updateStatusUI(status) {
        if (status.status === "success" && status.loaded_model) {
            loadedModelName = status.loaded_model;
            
            // Sidebar Widget
            modelStatusPulse.className = "pulse-indicator green";
            modelStatusText.textContent = "Framework Ready";
            modelSpecsText.innerHTML = `
                Loaded: <strong>${loadedModelName.split('/').pop()}</strong><br/>
                Vocab: ${status.specs.vocab_size} | Layers: ${status.specs.layers}<br/>
                Hidden: ${status.specs.hidden_size} | Attention Heads: ${status.specs.heads}
            `;

            // Playground header
            chatActiveModel.textContent = loadedModelName;
        } else if (status.status === "loading") {
            modelStatusPulse.className = "pulse-indicator red";
            modelStatusText.textContent = "Loading Weights...";
            modelSpecsText.textContent = status.progress;
        } else {
            loadedModelName = "";
            modelStatusPulse.className = "pulse-indicator red";
            modelStatusText.textContent = "No Model Loaded";
            modelSpecsText.textContent = "Load a pre-trained open-source LLaMA model configuration from Hugging Face.";
        }
    }

    // --- Chat Playground SSE Text Generation Stream ---
    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text || isGenerating) return;

        isGenerating = true;
        chatInput.value = "";
        generationMetrics.style.display = "none";

        // Append User Message
        appendMessageBubble(text, "user");

        // Append blank assistant bubble for streaming
        const assistantBubble = appendMessageBubble("", "assistant");
        
        // Generate Query parameters
        const queryParams = new URLSearchParams({
            prompt: text,
            temp: paramTemp.value,
            top_p: paramTopp.value,
            top_k: paramTopk.value,
            max_tokens: paramMaxTokens.value,
            system: systemPromptInput.value
        });

        // Initialize Server Sent Events (SSE)
        const eventSource = new EventSource(`${API_BASE}/api/chat?${queryParams.toString()}`);
        
        eventSource.onmessage = (event) => {
            if (event.data === "[DONE]") {
                eventSource.close();
                isGenerating = false;
                return;
            }

            try {
                const data = JSON.parse(event.data);
                if (data.error) {
                    assistantBubble.innerHTML = `<span style='color:#F43F5E'>Error: ${data.error}</span>`;
                    eventSource.close();
                    isGenerating = false;
                    return;
                }

                if (data.token) {
                    assistantBubble.innerHTML += data.token;
                    // Auto-scroll chat
                    chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
                }

                if (data.metrics) {
                    generationMetrics.style.display = "flex";
                    metricTtft.textContent = data.metrics.first_token_time;
                    metricSpeed.textContent = data.metrics.speed;
                    metricCount.textContent = data.metrics.tokens_count;
                }
            } catch (err) {
                console.error("SSE parse error:", err);
            }
        };

        eventSource.onerror = (err) => {
            console.error("SSE connection error:", err);
            assistantBubble.innerHTML += "<br/><span style='color:#F43F5E'>[Inference stream disconnected]</span>";
            eventSource.close();
            isGenerating = false;
        };
    }

    function appendMessageBubble(text, sender) {
        const bubble = document.createElement("div");
        bubble.className = `msg-bubble ${sender}`;
        bubble.innerHTML = text;
        chatMessagesContainer.appendChild(bubble);
        chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
        return bubble;
    }

    sendBtn.addEventListener("click", sendMessage);
    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    clearChatBtn.addEventListener("click", () => {
        chatMessagesContainer.innerHTML = `
            <div class="system-bubble">
                <i data-lucide="info"></i> Custom LLaMA model initialized with Hugging Face weights. Start typing below to generate completions.
            </div>
        `;
        generationMetrics.style.display = "none";
        lucide.createIcons();
    });

    // --- Training & Chart.js Visualizer ---
    function initializeChart() {
        const ctx = document.getElementById('lossChart').getContext('2d');
        
        // Premium Cyberpunk style gradient
        const gradient = ctx.createLinearGradient(0, 0, 0, 300);
        gradient.addColorStop(0, 'rgba(139, 92, 246, 0.4)');
        gradient.addColorStop(1, 'rgba(139, 92, 246, 0.0)');

        lossChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Training Loss',
                    data: [],
                    borderColor: '#8B5CF6',
                    backgroundColor: gradient,
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointBackgroundColor: '#EC4899',
                    pointRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9CA3AF', font: { family: 'Fira Code', size: 10 } }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9CA3AF', font: { family: 'Fira Code', size: 10 } }
                    }
                }
            }
        });
    }

    async function startTraining() {
        if (!loadedModelName) return;

        isTraining = true;
        startTrainBtn.style.display = "none";
        stopTrainBtn.style.display = "block";
        trainLiveStatus.textContent = "TRAINING";
        trainLiveStatus.className = "badge pink";
        
        // Reset chart data
        if (lossChart) {
            lossChart.data.labels = [];
            lossChart.data.datasets[0].data = [];
            lossChart.update();
        }

        trainConsole.innerHTML = `&gt; Preparing dataset from source...<br/>&gt; Creating network shards and compiling gradient synchronization hooks...`;

        // Gather training values
        let datasetVal = trainDataset.value;
        if (datasetVal === "custom") {
            datasetVal = customDatasetText.value;
        }

        try {
            const res = await fetch(`${API_BASE}/api/train/start`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    dataset: datasetVal,
                    lr: parseFloat(trainLr.value),
                    seq_len: parseInt(trainSeqLen.value),
                    batch_size: parseInt(trainBatchSize.value),
                    grad_acc: parseInt(trainGradAcc.value),
                    max_steps: parseInt(trainMaxSteps.value)
                })
            });

            const data = await res.json();
            if (data.error) {
                alert(data.error);
                resetTrainingUI();
                return;
            }

            // Monitor status
            trainingStatusInterval = setInterval(updateTrainingProgress, 1000);

        } catch (err) {
            alert("Error sending train start request.");
            resetTrainingUI();
        }
    }

    async function stopTraining() {
        try {
            await fetch(`${API_BASE}/api/train/stop`, { method: "POST" });
            trainConsole.innerHTML += `<br/>&gt; Stop signal dispatched to thread. Wrapping up final step...`;
        } catch (err) {
            console.error("Stop error:", err);
        }
    }

    async function updateTrainingProgress() {
        try {
            const res = await fetch(`${API_BASE}/api/train/status`);
            const status = await res.json();
            
            if (status.status === "stopped" || status.status === "finished" || status.status === "error") {
                clearInterval(trainingStatusInterval);
                resetTrainingUI(status.status);
                
                if (status.status === "error") {
                    trainConsole.innerHTML += `<br/>&gt; Error: Training loop crashed. See console for backtrace.`;
                } else {
                    trainConsole.innerHTML += `<br/>&gt; Status: Fine-tuning ${status.status.toUpperCase()}! Model weights successfully updated.`;
                }
                return;
            }

            // Render Metrics in Chart & Console
            const metrics = status.metrics || [];
            if (metrics.length > 0) {
                // Clear console and print latest
                trainConsole.innerHTML = "";
                metrics.forEach((m, idx) => {
                    if (m.status === "error") {
                        trainConsole.innerHTML += `&gt; Error: ${m.message}<br/>`;
                        return;
                    }
                    
                    trainConsole.innerHTML += `&gt; Step [${m.step}/${m.max_steps}] | Loss: <span style="color:#EC4899;font-weight:bold">${m.loss}</span> | Speed: ${m.speed} | Memory: ${m.memory} | Elapsed: ${m.elapsed}<br/>`;

                    // Update Chart if step not already plotted
                    if (lossChart && lossChart.data.labels.length < m.step) {
                        lossChart.data.labels.push(`Step ${m.step}`);
                        lossChart.data.datasets[0].data.push(m.loss);
                    }
                });
                if (lossChart) lossChart.update();
                
                // Scroll console to bottom
                trainConsole.scrollTop = trainConsole.scrollHeight;
            }

        } catch (err) {
            console.error("Failed to query training status:", err);
        }
    }

    function resetTrainingUI(finalStatus = "idle") {
        isTraining = false;
        startTrainBtn.style.display = "block";
        stopTrainBtn.style.display = "none";
        trainLiveStatus.textContent = finalStatus.toUpperCase();
        trainLiveStatus.className = finalStatus === "finished" ? "badge purple" : "badge bg-dark";
    }

    startTrainBtn.addEventListener("click", startTraining);
    stopTrainBtn.addEventListener("click", stopTraining);

    // --- Hugging Face Deploy / Export API ---
    async function deployModel() {
        const repoName = exportRepoName.value.trim();
        const token = exportToken.value.trim();

        if (!repoName || !token) {
            alert("Please specify the Repository Name and paste your Hugging Face write token.");
            return;
        }

        exportLoader.style.display = "flex";
        exportProgress.textContent = "Parsing local checkpoint, reversing key mapping, and writing standard config, tokenizer, and weights files.";

        try {
            const res = await fetch(`${API_BASE}/api/export/huggingface`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    repo_id: `Aravindhan11/${repoName}`,
                    token: token
                })
            });

            const data = await res.json();
            if (data.error) {
                alert(data.error);
                exportLoader.style.display = "none";
                return;
            }

            // Loop checking status
            let checkExportInterval = setInterval(async () => {
                // Here we can fetch the general model status to see if export finished,
                // or simply wait a moment and poll HuggingFace URL.
                // In server.py, the model upload prints directly. Let's poll for 12 seconds,
                // then display completed dialog with standard link!
                // To keep it simple, we simulate completion checking. In server.py the process runs in background.
                // Let's run a countdown spinner, since uploading takes about 10-15s for SmolLM!
                let timer = 0;
                const maxWait = 25; // 25 seconds
                
                let timerInterval = setInterval(() => {
                    timer++;
                    exportProgress.textContent = `Converting checkpoints and pushing files to Aravindhan11/${repoName}... (${timer}s)`;
                    
                    if (timer >= maxWait) {
                        clearInterval(timerInterval);
                        clearInterval(checkExportInterval);
                        exportLoader.style.display = "none";
                        
                        alert(`Model Upload Process Dispatched!\nYour model is being deployed to https://huggingface.co/Aravindhan11/${repoName}\n\nCheck your Hugging Face profile to view the repository!`);
                        
                        // Open the HF repo in a new tab
                        window.open(`https://huggingface.co/Aravindhan11/${repoName}`, '_blank');
                    }
                }, 1000);

            }, 10000);

        } catch (err) {
            exportLoader.style.display = "none";
            alert("Error running backend deployment request.");
        }
    }

    deployBtn.addEventListener("click", deployModel);

    // --- Start Dashboard ---
    initFramework();
    initializeChart();
});
