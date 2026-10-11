const API_BASE = '';

// Tab switching
function switchTab(tabId) {
  document.querySelectorAll('.nav-item').forEach(el => {
    el.classList.toggle('active', el.dataset.tab === tabId);
  });

  document.querySelectorAll('.view-panel').forEach(el => {
    el.style.display = 'none';
  });

  const activePanel = document.getElementById(`view-${tabId}`);
  if (activePanel) activePanel.style.display = 'block';

  const titles = {
    dashboard: 'Dashboard',
    upload: 'Upload Scans',
    results: 'Processing Results',
    review: 'Review & Correction',
    records: 'Academic Records',
    search: 'Search Records',
    evaluation: 'Evaluation Benchmark'
  };
  document.getElementById('page-title').textContent = titles[tabId] || 'Dashboard';

  // Load relevant data on tab change
  if (tabId === 'dashboard') loadDashboard();
  if (tabId === 'results') loadResultsDocuments();
  if (tabId === 'review') loadPendingReviews();
  if (tabId === 'records') loadAcademicRecords();
  if (tabId === 'evaluation') loadEvaluation();
}

function showAlert(message, type = 'success') {
  const box = document.getElementById('alert-box');
  box.innerHTML = `<div class="alert alert-${type}">${message}</div>`;
  setTimeout(() => { box.innerHTML = ''; }, 4500);
}

// ---------------- DASHBOARD ----------------

async function loadDashboard() {
  try {
    const res = await fetch('/api/dashboard/stats');
    const json = await res.json();
    if (json.status === 'success') {
      const data = json.data;
      document.getElementById('stat-docs').textContent = data.total_documents;
      document.getElementById('stat-students').textContent = data.digitized_students;
      document.getElementById('stat-results').textContent = data.digitized_results;
      document.getElementById('stat-reviews').textContent = data.fields_requiring_review;

      const tbody = document.getElementById('dashboard-recent-table');
      if (data.recent_documents.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:20px; color:var(--text-muted)">No documents uploaded yet. Click "Seed Demo Records" to load samples.</td></tr>`;
        return;
      }

      tbody.innerHTML = data.recent_documents.map(doc => `
        <tr>
          <td>${doc.id}</td>
          <td><strong>${doc.filename}</strong></td>
          <td>${doc.document_type === 'result_broadsheet' ? 'Result Broadsheet' : 'Transcript Sheet'}</td>
          <td><span class="badge ${getStatusBadgeClass(doc.status)}">${formatStatus(doc.status)}</span></td>
          <td>${doc.created_at ? doc.created_at.substring(0, 19) : '--'}</td>
          <td>
            ${doc.status === 'uploaded' ? 
              `<button class="btn btn-primary btn-sm" onclick="processDoc(${doc.id})">Process</button>` :
              `<button class="btn btn-outline btn-sm" onclick="viewDocResults(${doc.id})">View Details</button>`
            }
          </td>
        </tr>
      `).join('');
    }
  } catch (err) {
    console.error("Dashboard error:", err);
  }
}

function getStatusBadgeClass(status) {
  if (status === 'completed') return 'badge-success';
  if (status === 'review_required') return 'badge-warning';
  if (status === 'processing') return 'badge-secondary';
  if (status === 'failed') return 'badge-danger';
  return 'badge-secondary';
}

function formatStatus(status) {
  if (status === 'review_required') return 'Review Required';
  if (status === 'completed') return 'Completed';
  if (status === 'uploaded') return 'Uploaded';
  if (status === 'processing') return 'Processing';
  return status;
}

// Seed Demo Documents
document.getElementById('btn-seed').addEventListener('click', async () => {
  try {
    showAlert('Seeding demonstration records from ATBU legacy dataset...', 'warning');
    const res = await fetch('/api/seed-demo', { method: 'POST' });
    const json = await res.json();
    if (json.status === 'success') {
      showAlert(json.message, 'success');
      loadDashboard();
    } else {
      showAlert(json.message, 'danger');
    }
  } catch (e) {
    showAlert('Failed to seed demonstration records', 'danger');
  }
});

// ---------------- UPLOAD ----------------

document.getElementById('upload-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById('upload-file');
  const docType = document.getElementById('upload-doc-type').value;

  if (!fileInput.files[0]) return;

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  formData.append('document_type', docType);

  const btn = document.getElementById('btn-upload-submit');
  btn.disabled = true;
  btn.textContent = 'Uploading...';

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });
    const json = await res.json();
    btn.disabled = false;
    btn.textContent = 'Upload Record';

    if (json.status === 'success') {
      showAlert('Document uploaded successfully. Ready for OCR processing.', 'success');
      fileInput.value = '';
      switchTab('results');
      setTimeout(() => {
        const select = document.getElementById('results-doc-select');
        select.value = json.data.id;
        loadDocumentDetails(json.data.id);
      }, 500);
    } else {
      showAlert(json.message || 'Upload failed', 'danger');
    }
  } catch (err) {
    btn.disabled = false;
    btn.textContent = 'Upload Record';
    showAlert('Network error during upload', 'danger');
  }
});

// ---------------- PROCESSING & RESULTS ----------------

async function loadResultsDocuments() {
  try {
    const res = await fetch('/api/documents');
    const json = await res.json();
    if (json.status === 'success') {
      const select = document.getElementById('results-doc-select');
      const docs = json.data;
      if (docs.length === 0) {
        select.innerHTML = `<option value="">No documents available</option>`;
        return;
      }
      select.innerHTML = docs.map(d => `
        <option value="${d.id}">${d.filename} (${formatStatus(d.status)})</option>
      `).join('');

      loadDocumentDetails(docs[0].id);
    }
  } catch (err) {
    console.error("Results load error:", err);
  }
}

document.getElementById('results-doc-select').addEventListener('change', (e) => {
  if (e.target.value) {
    loadDocumentDetails(e.target.value);
  }
});

async function loadDocumentDetails(docId) {
  try {
    const res = await fetch(`/api/documents/${docId}`);
    const json = await res.json();
    if (json.status === 'success') {
      const doc = json.data.document;
      const fields = json.data.fields;

      // Images
      const imgOrig = document.getElementById('img-original');
      const imgPre = document.getElementById('img-preprocessed');
      const imgLay = document.getElementById('img-layout');

      // Check if dataset or uploads
      let origUrl = `/uploads/${doc.original_path}`;
      if (doc.original_path.endsWith('.png') && !doc.original_path.includes('_17')) {
        origUrl = `/dataset/pages/${doc.original_path}`;
      }
      imgOrig.src = origUrl;
      imgPre.src = doc.preprocessed_path ? `/uploads/${doc.preprocessed_path}` : origUrl;
      imgLay.src = doc.layout_path ? `/uploads/${doc.layout_path}` : origUrl;

      // Fields table
      const tbody = document.getElementById('results-fields-table');
      if (fields.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:20px; color:var(--text-muted)">Document has not been processed yet. Click "Process Document" to extract records.</td></tr>`;
        return;
      }

      tbody.innerHTML = fields.map(f => `
        <tr>
          <td><strong>${f.field_name}</strong></td>
          <td>${f.field_value}</td>
          <td>${Math.round(f.confidence * 100)}%</td>
          <td>
            <span class="badge ${f.needs_review ? 'badge-warning' : 'badge-success'}">
              ${f.needs_review ? 'Review Needed' : 'Validated'}
            </span>
          </td>
          <td style="color:var(--text-muted); font-size:0.8rem;">${f.validation_notes || '--'}</td>
          <td>
            <button class="btn btn-outline btn-sm" onclick="openEditModal(${f.id}, '${f.field_name}', '${f.field_value}')">Edit</button>
          </td>
        </tr>
      `).join('');
    }
  } catch (err) {
    console.error("Doc details error:", err);
  }
}

async function processDoc(docId) {
  try {
    showAlert(`Running image preprocessing, layout analysis, and TrOCR extraction on document #${docId}...`, 'warning');
    const res = await fetch(`/api/documents/${docId}/process`, { method: 'POST' });
    const json = await res.json();
    if (json.status === 'success') {
      showAlert(`Processing complete in ${json.data.processing_time_sec}s. Extracted ${json.data.fields_count} fields.`, 'success');
      viewDocResults(docId);
    } else {
      showAlert(json.message || 'Processing failed', 'danger');
    }
  } catch (err) {
    showAlert('Error during OCR processing', 'danger');
  }
}

document.getElementById('btn-reprocess').addEventListener('click', () => {
  const select = document.getElementById('results-doc-select');
  if (select.value) {
    processDoc(select.value);
  }
});

function viewDocResults(docId) {
  switchTab('results');
  setTimeout(() => {
    const select = document.getElementById('results-doc-select');
    select.value = docId;
    loadDocumentDetails(docId);
  }, 200);
}

// ---------------- REVIEW & CORRECTION ----------------

async function loadPendingReviews() {
  try {
    const res = await fetch('/api/documents');
    const json = await res.json();
    if (json.status !== 'success') return;

    let pendingList = [];
    for (const doc of json.data) {
      const dRes = await fetch(`/api/documents/${doc.id}`);
      const dJson = await dRes.json();
      if (dJson.status === 'success') {
        const pFields = dJson.data.fields.filter(f => f.needs_review === 1);
        pFields.forEach(pf => pf.doc_id = doc.id);
        pendingList = pendingList.concat(pFields);
      }
    }

    const tbody = document.getElementById('review-pending-table');
    if (pendingList.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:24px; color:var(--text-muted)">All academic fields are verified. No review required.</td></tr>`;
      return;
    }

    tbody.innerHTML = pendingList.map(f => `
      <tr>
        <td>#${f.doc_id}</td>
        <td><strong>${f.field_name}</strong></td>
        <td>${f.field_value}</td>
        <td>${Math.round(f.confidence * 100)}%</td>
        <td><span class="badge badge-warning">${f.validation_notes || 'Low confidence'}</span></td>
        <td>
          <button class="btn btn-primary btn-sm" onclick="openEditModal(${f.id}, '${f.field_name}', '${f.field_value}')">Verify / Correct</button>
        </td>
      </tr>
    `).join('');
  } catch (err) {
    console.error("Pending review error:", err);
  }
}

function openEditModal(fieldId, fieldName, fieldValue) {
  document.getElementById('edit-field-id').value = fieldId;
  document.getElementById('edit-field-name').value = fieldName;
  document.getElementById('edit-field-val').value = fieldValue;
  document.getElementById('edit-modal').style.display = 'flex';
}

function closeEditModal() {
  document.getElementById('edit-modal').style.display = 'none';
}

document.getElementById('edit-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fieldId = document.getElementById('edit-field-id').value;
  const newVal = document.getElementById('edit-field-val').value;
  const reason = document.getElementById('edit-reason').value;

  try {
    const res = await fetch(`/api/fields/${fieldId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: jsonBody = JSON.stringify({ field_value: newVal, reason: reason })
    });
    const json = await res.json();
    if (json.status === 'success') {
      showAlert('Field corrected and recorded in verification audit trail.', 'success');
      closeEditModal();
      loadPendingReviews();
      const currentDoc = document.getElementById('results-doc-select').value;
      if (currentDoc) loadDocumentDetails(currentDoc);
      loadDashboard();
    } else {
      showAlert(json.message || 'Correction failed', 'danger');
    }
  } catch (err) {
    showAlert('Error saving correction', 'danger');
  }
});

// ---------------- ACADEMIC RECORDS ----------------

async function loadAcademicRecords() {
  try {
    const res = await fetch('/api/records');
    const json = await res.json();
    if (json.status === 'success') {
      const tbody = document.getElementById('records-table');
      const records = json.data;
      if (records.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding:24px; color:var(--text-muted)">No records digitized yet. Process a document to extract records.</td></tr>`;
        return;
      }

      tbody.innerHTML = records.map(r => `
        <tr>
          <td><strong>${r.reg_no}</strong></td>
          <td>${r.student_name}</td>
          <td>${r.course_code}</td>
          <td>${r.credit_unit}</td>
          <td>${r.score}</td>
          <td><span class="badge ${r.grade === 'A' ? 'badge-success' : 'badge-secondary'}">${r.grade}</span></td>
          <td>${r.session_name || '--'}</td>
          <td>${r.semester}</td>
        </tr>
      `).join('');
    }
  } catch (err) {
    console.error("Records error:", err);
  }
}

// ---------------- SEARCH ----------------

async function executeSearch() {
  const query = document.getElementById('search-input').value.trim();
  if (!query) return;

  try {
    const res = await fetch(`/api/records/search?q=${encodeURIComponent(query)}`);
    const json = await res.json();
    if (json.status === 'success') {
      const tbody = document.getElementById('search-results-table');
      const results = json.data;
      if (results.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding:24px; color:var(--text-muted)">No matching records found for "${query}".</td></tr>`;
        return;
      }

      tbody.innerHTML = results.map(r => `
        <tr>
          <td><strong>${r.reg_no}</strong></td>
          <td>${r.student_name}</td>
          <td>${r.department || 'Computer Science'}</td>
          <td>${r.course_code}</td>
          <td>${r.score}</td>
          <td><span class="badge ${r.grade === 'A' ? 'badge-success' : 'badge-secondary'}">${r.grade}</span></td>
          <td>${r.session_name || '--'}</td>
          <td>${r.semester}</td>
        </tr>
      `).join('');
    }
  } catch (err) {
    console.error("Search error:", err);
  }
}

document.getElementById('btn-search-exec').addEventListener('click', executeSearch);
document.getElementById('search-input').addEventListener('keypress', (e) => {
  if (e.key === 'Enter') executeSearch();
});

// ---------------- EVALUATION ----------------

async function loadEvaluation() {
  try {
    const res = await fetch('/api/evaluation');
    const json = await res.json();
    if (json.status === 'success') {
      const rep = json.data;
      if (rep.proposed_model) {
        document.getElementById('eval-trocr-cer').textContent = `${(rep.proposed_model.cer * 100).toFixed(2)}%`;
        document.getElementById('eval-base-cer').textContent = `${(rep.baseline_model.cer * 100).toFixed(2)}%`;
        document.getElementById('eval-cer-gain').textContent = `-${rep.baseline_improvement.cer_reduction_pct}%`;
        document.getElementById('eval-field-acc').textContent = `${rep.proposed_model.field_accuracy_pct}%`;

        const tbody = document.getElementById('eval-comparison-table');
        tbody.innerHTML = `
          <tr>
            <td><strong>Character Error Rate (CER)</strong></td>
            <td>${(rep.proposed_model.cer * 100).toFixed(2)}%</td>
            <td>${(rep.baseline_model.cer * 100).toFixed(2)}%</td>
            <td style="color:var(--success)"><strong>-${rep.baseline_improvement.cer_reduction_pct}% Error Reduction</strong></td>
          </tr>
          <tr>
            <td><strong>Word Error Rate (WER)</strong></td>
            <td>${(rep.proposed_model.wer * 100).toFixed(2)}%</td>
            <td>${(rep.baseline_model.wer * 100).toFixed(2)}%</td>
            <td style="color:var(--success)"><strong>-${rep.baseline_improvement.wer_reduction_pct}% Error Reduction</strong></td>
          </tr>
          <tr>
            <td><strong>Field Extraction Accuracy</strong></td>
            <td>${rep.proposed_model.field_accuracy_pct}%</td>
            <td>${rep.baseline_model.field_accuracy_pct}%</td>
            <td style="color:var(--primary)"><strong>+${rep.baseline_improvement.field_accuracy_gain_pct}% Gain</strong></td>
          </tr>
          <tr>
            <td><strong>Average Processing Time / Page</strong></td>
            <td>${rep.proposed_model.avg_processing_time_sec}s</td>
            <td>${rep.baseline_model.avg_processing_time_sec}s</td>
            <td style="color:var(--primary)"><strong>${rep.baseline_improvement.speedup_factor}x Speedup</strong></td>
          </tr>
        `;
      }
    }
  } catch (err) {
    console.error("Evaluation error:", err);
  }
}

document.getElementById('btn-run-eval').addEventListener('click', async () => {
  showAlert('Running evaluation on all 20 held-out test pages...', 'warning');
  try {
    const res = await fetch('/api/evaluation/run', { method: 'POST' });
    const json = await res.json();
    if (json.status === 'success') {
      showAlert('Benchmark evaluation successfully executed across held-out test set.', 'success');
      loadEvaluation();
    } else {
      showAlert(json.message || 'Evaluation run failed', 'danger');
    }
  } catch (err) {
    showAlert('Error running evaluation', 'danger');
  }
});

// Navigation listener
document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', () => {
    switchTab(item.dataset.tab);
  });
});

// Initial boot
document.addEventListener('DOMContentLoaded', () => {
  loadDashboard();
});
