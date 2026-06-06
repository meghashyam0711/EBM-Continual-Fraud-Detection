




let currentFeatures = Array(29).fill(0.0);
let activeBatchIndex = null;
let activeRecordIndex = 0;
let currentProofData = null;
let currentThreshold = -5.9963;
let trainingPollInterval = null;
let auditLogs = [];
let mockAuditLogs = [];


const PRESETS = {
    normal: [
        -0.12, 0.45, -0.32, 0.1, -0.05, 0.23, -0.15, 0.08, -0.02, 0.12,
        -0.45, 0.21, -0.11, 0.05, 0.3, -0.18, 0.02, -0.05, 0.1, -0.12,
        0.05, -0.08, 0.14, -0.02, 0.22, -0.15, 0.04, 0.01, 0.15 
    ],
    fraud: [
        -1.35, 1.12, -2.45, 2.12, -1.89, 0.98, -1.45, 0.54, -1.12, -0.45,
        1.89, -2.12, 0.76, -1.54, -0.23, -0.87, -2.12, -0.45, 0.67, 0.23,
        -0.45, -0.12, 0.76, -0.21, -0.12, 0.45, 0.89, -0.12, 2.45 
    ],
    ood: [
        12.45, -15.12, 10.89, 8.45, -14.12, 11.23, -9.89, 13.45, -7.12, 9.89,
        -11.45, 10.12, -13.89, 12.12, 7.89, -8.45, 11.12, -10.45, 9.12, -11.12,
        6.45, -8.12, 12.12, -9.45, 14.12, -11.89, 8.45, 6.12, 15.00 
    ]
};


document.addEventListener("DOMContentLoaded", () => {
    const storedUrl = localStorage.getItem("backend_url") || "";
    const input = document.getElementById("input-backend-url");
    if (input) input.value = storedUrl;

    generateSliders();
    loadPreset('normal');
    fetchSystemStatus();
    loadAuditLogs();
    checkTrainingStatus();
    
    setInterval(fetchSystemStatus, 5000);
});

function getApiUrl(path) {
    const customUrl = localStorage.getItem("backend_url");
    if (customUrl) {
        return customUrl.replace(/\/$/, "") + path;
    }
    return path;
}

function saveBackendUrl(val) {
    if (val && val.trim()) {
        let cleaned = val.trim();
        if (!cleaned.toLowerCase().startsWith("http://") && !cleaned.toLowerCase().startsWith("https://")) {
            cleaned = "http://" + cleaned;
        }
        localStorage.setItem("backend_url", cleaned);
    } else {
        localStorage.removeItem("backend_url");
    }
    fetchSystemStatus();
    loadAuditLogs();
}


function generateSliders() {
    const grid = document.getElementById("sliders-grid");
    grid.innerHTML = "";
    
    for (let i = 0; i < 29; i++) {
        const isAmount = i === 28;
        const name = isAmount ? "Amount" : `V${i + 1}`;
        const min = isAmount ? 0 : -3;
        const max = isAmount ? 10 : 3;
        const step = 0.01;
        const initial = isAmount ? 1.0 : 0.0;
        
        const group = document.createElement("div");
        group.className = "slider-group";
        group.innerHTML = `
            <div class="slider-info">
                <span class="slider-label">${name}</span>
                <span class="slider-val" id="val-f-${i}">${initial.toFixed(2)}</span>
            </div>
            <input type="range" class="slider-input" id="input-f-${i}" 
                   min="${min}" max="${max}" step="${step}" value="${initial}"
                   oninput="updateFeatureValue(${i}, this.value)">
        `;
        grid.appendChild(group);
    }
}


function updateFeatureValue(index, val) {
    const numericVal = parseFloat(val);
    currentFeatures[index] = numericVal;
    document.getElementById(`val-f-${index}`).innerText = numericVal.toFixed(2);
}


function loadPreset(presetName) {
    
    document.querySelectorAll(".btn-preset").forEach(btn => btn.classList.remove("active"));
    const activeBtn = document.getElementById(`preset-${presetName}`);
    if (activeBtn) activeBtn.classList.add("active");
    
    const values = PRESETS[presetName];
    for (let i = 0; i < 29; i++) {
        currentFeatures[i] = values[i];
        const input = document.getElementById(`input-f-${i}`);
        if (input) {
            input.value = values[i];
            document.getElementById(`val-f-${i}`).innerText = values[i].toFixed(2);
        }
    }
}


function randomizeFeatures() {
    
    const isOOD = Math.random() > 0.6;
    const factor = isOOD ? 12.0 : 1.0;
    
    for (let i = 0; i < 29; i++) {
        const isAmount = i === 28;
        let randVal = 0;
        if (isAmount) {
            randVal = (Math.random() * 8.0) * factor;
        } else {
            randVal = ((Math.random() * 6.0) - 3.0) * factor;
        }
        updateFeatureValue(i, randVal);
        const input = document.getElementById(`input-f-${i}`);
        if (input) input.value = randVal;
    }
}


async function fetchSystemStatus() {
    try {
        const response = await fetch(getApiUrl("/ready"));
        const data = await response.json();
        
        document.getElementById("val-model-features").innerText = "29 Features";
        
        const apiStatus = document.getElementById("api-status");
        if (apiStatus) {
            apiStatus.querySelector(".status-dot").className = "status-dot green";
            apiStatus.querySelector(".pulse-text").innerText = "ONLINE";
        }
        
        const redisConnected = data.checks.redis_connected;
        const redisStatus = document.getElementById("redis-status");
        if (redisStatus) {
            if (redisConnected) {
                redisStatus.querySelector(".status-dot").className = "status-dot green";
                document.getElementById("val-redis-status").innerText = "ONLINE";
            } else {
                redisStatus.querySelector(".status-dot").className = "status-dot red";
                document.getElementById("val-redis-status").innerText = "OFFLINE";
            }
        }
        
    } catch (e) {
        console.warn("Could not retrieve system status: ", e);
        const apiStatus = document.getElementById("api-status");
        if (apiStatus) {
            apiStatus.querySelector(".status-dot").className = "status-dot orange";
            apiStatus.querySelector(".pulse-text").innerText = "SIMULATED";
        }
    }
}


async function runDetection() {
    const btn = document.getElementById("btn-detect");
    btn.disabled = true;
    btn.innerText = "Running Engine...";
    
    try {
        let result = null;
        let isMock = false;
        
        try {
            const response = await fetch(getApiUrl("/api/v1/predict"), {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    features: currentFeatures
                })
            });
            if (response.ok) {
                const data = await response.json();
                result = data.results[0];
            } else {
                isMock = true;
            }
        } catch (err) {
            isMock = true;
        }
        
        if (isMock) {
            const sumSquares = currentFeatures.reduce((sum, val) => sum + val * val, 0);
            let prediction = "LEGITIMATE";
            let energy_score = -6.1245 + (Math.random() - 0.5);
            let is_ood = false;
            let confidence = 0.99 - Math.random() * 0.02;
            
            const isOodPreset = currentFeatures[0] > 10;
            const isFraudPreset = currentFeatures[0] < -1 && currentFeatures[0] > -2;
            
            if (isOodPreset || sumSquares > 150) {
                prediction = "LEGITIMATE";
                energy_score = 4.5612 + Math.random() * 5;
                is_ood = true;
                confidence = 0.76 - Math.random() * 0.1;
            } else if (isFraudPreset || (sumSquares > 15 && sumSquares <= 150)) {
                prediction = "FRAUDULENT";
                energy_score = -4.1245 + Math.random();
                is_ood = false;
                confidence = 0.92 - Math.random() * 0.05;
            }
            
            result = {
                prediction: prediction,
                energy_score: energy_score,
                is_ood: is_ood,
                confidence: confidence,
                cached: false
            };
            
            const mockRoot = Array.from({length: 64}, () => Math.floor(Math.random()*16).toString(16)).join('');
            const newLog = {
                index: mockAuditLogs.length,
                timestamp: new Date().toISOString(),
                merkle_root: mockRoot,
                num_records: 1,
                tree_depth: 1
            };
            mockAuditLogs.unshift(newLog);
        }
        
        const resultsContainer = document.getElementById("inference-results");
        resultsContainer.classList.remove("hidden");
        resultsContainer.classList.remove("slide-in-up");
        void resultsContainer.offsetWidth; 
        resultsContainer.classList.add("slide-in-up");
        
        const predCard = document.getElementById("card-prediction");
        const predVal = document.getElementById("res-prediction");
        predVal.innerText = result.prediction;
        if (result.prediction === "FRAUDULENT") {
            predVal.className = "card-value fraud";
            predCard.style.borderColor = "var(--neon-red)";
        } else {
            predVal.className = "card-value legit";
            predCard.style.borderColor = "var(--neon-green)";
        }
        
        const energyVal = document.getElementById("res-energy");
        energyVal.innerText = result.energy_score.toFixed(4);
        if (result.is_ood) {
            energyVal.className = "card-value ood";
            document.getElementById("card-energy").style.borderColor = "var(--neon-amber)";
        } else {
            energyVal.className = "card-value";
            document.getElementById("card-energy").style.borderColor = "rgba(255, 255, 255, 0.05)";
        }
        
        document.getElementById("res-confidence").innerText = `${(result.confidence * 100).toFixed(1)}%`;
        
        const alertBanner = document.getElementById("ood-alert-banner");
        if (result.is_ood) {
            alertBanner.classList.add("active");
        } else {
            alertBanner.classList.remove("active");
        }
        
        loadAuditLogs();
        
    } catch (e) {
        alert("Inference request failed: " + e.message);
    } finally {
        btn.disabled = false;
        btn.innerText = "Run Anomaly Detection Engine";
    }
}


async function loadAuditLogs() {
    try {
        const response = await fetch(getApiUrl("/api/v1/audit-logs"));
        if (response.ok) {
            const data = await response.json();
            auditLogs = data;
            renderAuditLogs(auditLogs);
        } else {
            renderAuditLogs(mockAuditLogs);
        }
    } catch (e) {
        console.warn("Could not load audit logs, using mock logs: ", e);
        renderAuditLogs(mockAuditLogs);
    }
}

function renderAuditLogs(logs) {
    const listContainer = document.getElementById("audit-logs-list");
    listContainer.innerHTML = "";
    
    if (logs.length === 0) {
        listContainer.innerHTML = `<div class="empty-state">No transaction audits recorded yet. Run sandbox inferences to populate.</div>`;
        return;
    }
    
    logs.forEach(log => {
        const item = document.createElement("div");
        item.className = `audit-log-item ${activeBatchIndex === log.index ? 'active' : ''}`;
        item.onclick = () => selectAuditBatch(log.index);
        
        const localTime = new Date(log.timestamp).toLocaleTimeString();
        
        item.innerHTML = `
            <div class="audit-log-meta">
                <span>Batch #${log.index}</span>
                <span>${localTime} (${log.num_records} Tx)</span>
            </div>
            <div class="audit-log-root" title="${log.merkle_root}">Root: ${log.merkle_root.substring(0, 24)}...</div>
        `;
        listContainer.appendChild(item);
    });
}

async function selectAuditBatch(index) {
    activeBatchIndex = index;
    
    document.querySelectorAll(".audit-log-item").forEach((item, idx) => {
        if (idx === index) item.classList.add("active");
        else item.classList.remove("active");
    });
    
    
    activeRecordIndex = 0;
    fetchAuditProof();
}

async function fetchAuditProof() {
    const area = document.getElementById("proof-verification-area");
    area.innerHTML = `<div class="empty-state">Loading cryptographic audit proof...</div>`;
    
    try {
        let data = null;
        try {
            const response = await fetch(getApiUrl("/api/v1/audit-proof"), {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    batch_index: activeBatchIndex,
                    record_index: activeRecordIndex
                })
            });
            if (response.ok) {
                data = await response.json();
            }
        } catch (err) {
            console.warn("Could not fetch proof from server, using simulated proof: ", err);
        }
        
        if (!data) {
            const activeBatch = mockAuditLogs.find(log => log.index === activeBatchIndex);
            if (!activeBatch) throw new Error("Batch not found.");
            
            const mockRecord = [...currentFeatures];
            const mockLeafHash = Array.from({length: 64}, () => Math.floor(Math.random()*16).toString(16)).join('');
            const mockSiblingHash = Array.from({length: 64}, () => Math.floor(Math.random()*16).toString(16)).join('');
            
            data = {
                leaf_hash: mockLeafHash,
                proof: [[mockSiblingHash, "R"]],
                merkle_root: activeBatch.merkle_root,
                record: mockRecord
            };
        }
        
        currentProofData = data;
        
        const activeBatch = (auditLogs && auditLogs.length > 0 ? auditLogs : mockAuditLogs).find(log => log.index === activeBatchIndex);
        const numRecords = activeBatch ? activeBatch.num_records : 1;
        
        area.innerHTML = `
            <div class="proof-header">
                <h4>Audit Proof: Batch #${activeBatchIndex}</h4>
                <div class="validation-badge" id="proof-validation-badge">Verified</div>
            </div>
            <div class="proof-body">
                <div class="record-selector-group">
                    <label>Select Transaction Record in Batch:</label>
                    <select class="select-record" onchange="changeRecordIndex(this.value)">
                        ${Array.from({ length: numRecords }, (_, i) => `
                            <option value="${i}" ${activeRecordIndex === i ? 'selected' : ''}>Record #${i}</option>
                        `).join('')}
                    </select>
                </div>
                
                <div class="proof-details">
                    <div class="proof-hash-line">
                        <span>Leaf Hash (SHA-256):</span>
                        <span title="${data.leaf_hash}">${data.leaf_hash.substring(0, 18)}...</span>
                    </div>
                    <div class="proof-hash-line">
                        <span>Expected Root:</span>
                        <span title="${data.merkle_root}">${data.merkle_root.substring(0, 18)}...</span>
                    </div>
                </div>

                <h5>Audit Trail (Auth Path Steps)</h5>
                <div class="proof-visual-steps">
                    ${data.proof.map((step, i) => `
                        <div class="proof-step">
                            <span class="step-num">Step ${i+1}:</span>
                            <span class="sibling" title="${step[0]}">Sibling: ${step[0].substring(0, 14)}...</span>
                            <span class="side">${step[1] === 'R' ? 'Concat Right' : 'Concat Left'}</span>
                        </div>
                    `).join('')}
                </div>

                <div class="proof-actions">
                    <button class="btn btn-secondary" onclick="tamperLeafAndVerify()">Verify Tampered Hash</button>
                    <button class="btn btn-primary" onclick="verifyAuditProof()">Verify Cryptographic Root</button>
                </div>
            </div>
        `;
        
    } catch (e) {
        area.innerHTML = `<div class="empty-state text-red">Failed to load proof details: ${e.message}</div>`;
    }
}

function changeRecordIndex(val) {
    activeRecordIndex = parseInt(val);
    fetchAuditProof();
}

async function verifyAuditProof(tamperedHash = null) {
    if (!currentProofData) return;
    
    const badge = document.getElementById("proof-validation-badge");
    const hashToVerify = tamperedHash || currentProofData.leaf_hash;
    
    try {
        let is_valid = false;
        try {
            const response = await fetch(getApiUrl("/api/v1/verify-proof"), {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    leaf_hash: hashToVerify,
                    proof: currentProofData.proof,
                    merkle_root: currentProofData.merkle_root
                })
            });
            if (response.ok) {
                const data = await response.json();
                is_valid = data.is_valid;
            } else {
                is_valid = (hashToVerify === currentProofData.leaf_hash);
            }
        } catch (err) {
            is_valid = (hashToVerify === currentProofData.leaf_hash);
        }
        
        if (is_valid) {
            badge.innerText = "VERIFIED [OK]";
            badge.className = "validation-badge success";
        } else {
            badge.innerText = "TAMPERED [ALERT]";
            badge.className = "validation-badge fail";
        }
        
    } catch (e) {
        alert("Failed to verify audit proof: " + e.message);
    }
}

function tamperLeafAndVerify() {
    
    const tamperedHash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    verifyAuditProof(tamperedHash);
}


async function triggerTraining() {
    const btn = document.getElementById("btn-train");
    btn.disabled = true;
    
    try {
        const response = await fetch(getApiUrl("/api/v1/train"), { method: "POST" });
        const data = await response.json();
        
        document.getElementById("training-progress-area").classList.remove("hidden");
        pollTrainingStatus();
        
    } catch (e) {
        alert("Could not start training run: " + e.message);
        btn.disabled = false;
    }
}

function pollTrainingStatus() {
    if (trainingPollInterval) clearInterval(trainingPollInterval);
    
    trainingPollInterval = setInterval(async () => {
        try {
            const response = await fetch(getApiUrl("/api/v1/train/status"));
            const data = await response.json();
            
            updateTrainingUI(data);
            
            if (data.status === "completed" || data.status === "error") {
                clearInterval(trainingPollInterval);
                document.getElementById("btn-train").disabled = false;
                
                
                if (data.status === "completed") {
                    currentThreshold = data.calibrated_threshold;
                    document.getElementById("val-ood-threshold").innerText = currentThreshold.toFixed(4);
                    
                    addHistoryRow(data);
                }
            }
        } catch (e) {
            console.warn("Polling error: ", e);
        }
    }, 1500);
}

function checkTrainingStatus() {
    
    fetch(getApiUrl("/api/v1/train/status"))
        .then(res => res.json())
        .then(data => {
            if (data.status === "running") {
                document.getElementById("btn-train").disabled = true;
                document.getElementById("training-progress-area").classList.remove("hidden");
                pollTrainingStatus();
            }
        });
}

function updateTrainingUI(data) {
    const progressFill = document.getElementById("training-bar");
    const percentText = document.getElementById("training-percent");
    const statusText = document.getElementById("training-status-text");
    
    progressFill.style.width = `${data.progress}%`;
    percentText.innerText = `${data.progress}%`;
    
    if (data.status === "running") {
        statusText.innerText = `Epoch ${data.current_epoch}/${data.epochs} (Loss: ${data.loss.toFixed(4)}, ε: ${data.epsilon.toFixed(2)})`;
    } else if (data.status === "completed") {
        statusText.innerText = `Training Completed! Epsilon: ${data.epsilon.toFixed(2)}`;
        
        document.getElementById("budget-epsilon").innerText = data.epsilon.toFixed(2);
        const advantage = Math.exp(data.epsilon);
        document.getElementById("budget-advantage").innerText = `${advantage.toFixed(2)}x`;
    } else if (data.status === "error") {
        statusText.innerText = `Error: ${data.error}`;
    }
}

function addHistoryRow(data) {
    const tbody = document.getElementById("history-table-body");
    
    
    const emptyRow = tbody.querySelector(".empty-table-text");
    if (emptyRow) tbody.innerHTML = "";
    
    const row = document.createElement("tr");
    row.innerHTML = `
        <td class="font-mono">Run #${tbody.children.length + 1}</td>
        <td class="font-mono">${data.loss.toFixed(4)}</td>
        <td class="font-mono">${data.epsilon.toFixed(2)}</td>
        <td><span class="badge badge-purple">Spectral Filtered</span></td>
        <td><span class="badge badge-cyan" style="background:rgba(57,255,20,0.1);color:var(--neon-green)">Success</span></td>
    `;
    tbody.appendChild(row);
}
