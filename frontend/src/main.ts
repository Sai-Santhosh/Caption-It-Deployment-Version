/**
 * Caption-It Frontend
 * Production-grade image captioning UI - TypeScript
 */
import "./style.css";

// Use empty for same-origin (proxy); set VITE_API_URL when API is on different host
const API_BASE = import.meta.env.VITE_API_URL ?? "";

interface CaptionResponse {
  caption: string;
  model_id: string;
  inference_time_ms?: number;
}

async function captionFromFile(file: File): Promise<CaptionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE || ""}/api/v1/caption`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

async function captionFromUrl(url: string): Promise<CaptionResponse> {
  const res = await fetch(`${API_BASE || ""}/api/v1/caption/url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

function createApp(): void {
  const app = document.querySelector<HTMLDivElement>("#app")!;
  app.innerHTML = `
    <div class="container">
      <header>
        <h1>Caption-It</h1>
        <p class="subtitle">AI-powered image captioning. Upload or paste a URL.</p>
      </header>

      <div class="tabs">
        <button type="button" class="tab active" data-tab="upload">Upload Image</button>
        <button type="button" class="tab" data-tab="url">From URL</button>
      </div>

      <div class="panel" data-panel="upload">
        <label class="dropzone" id="dropzone">
          <input type="file" accept="image/jpeg,image/png,image/webp" id="fileInput" hidden>
          <span class="dropzone-text">Drag & drop or click to select</span>
          <span class="dropzone-hint">JPEG, PNG, WebP — Max 10MB</span>
        </label>
        <div class="preview-area" id="previewArea" hidden></div>
        <button type="button" class="btn primary" id="captionFile" disabled>Generate Caption</button>
      </div>

      <div class="panel hidden" data-panel="url">
        <div class="url-input-group">
          <input type="url" id="urlInput" placeholder="https://example.com/image.jpg">
          <button type="button" class="btn primary" id="captionUrl">Generate Caption</button>
        </div>
        <div class="preview-area" id="urlPreview" hidden></div>
      </div>

      <div class="result" id="result" hidden>
        <h3>Caption</h3>
        <p class="caption-text" id="captionText"></p>
        <p class="meta" id="resultMeta"></p>
      </div>

      <div class="error" id="error" hidden>
        <p id="errorText"></p>
      </div>

      <footer>
        <a href="${API_BASE || ""}/api/docs" target="_blank" rel="noopener">API Docs</a>
      </footer>
    </div>
  `;

  const dropzone = app.querySelector<HTMLLabelElement>("#dropzone")!;
  const fileInput = app.querySelector<HTMLInputElement>("#fileInput")!;
  const previewArea = app.querySelector<HTMLDivElement>("#previewArea")!;
  const captionFileBtn = app.querySelector<HTMLButtonElement>("#captionFile")!;
  const urlInput = app.querySelector<HTMLInputElement>("#urlInput")!;
  const captionUrlBtn = app.querySelector<HTMLButtonElement>("#captionUrl")!;
  const urlPreview = app.querySelector<HTMLDivElement>("#urlPreview")!;
  const resultDiv = app.querySelector<HTMLDivElement>("#result")!;
  const captionText = app.querySelector<HTMLParagraphElement>("#captionText")!;
  const resultMeta = app.querySelector<HTMLParagraphElement>("#resultMeta")!;
  const errorDiv = app.querySelector<HTMLDivElement>("#error")!;
  const errorText = app.querySelector<HTMLParagraphElement>("#errorText")!;

  let selectedFile: File | null = null;

  function hideError(): void {
    errorDiv.hidden = true;
  }
  function showError(msg: string): void {
    errorText.textContent = msg;
    errorDiv.hidden = false;
    resultDiv.hidden = true;
  }
  function showResult(data: CaptionResponse): void {
    captionText.textContent = data.caption;
    let meta = `Model: ${data.model_id}`;
    if (data.inference_time_ms != null) {
      meta += ` • ${data.inference_time_ms.toFixed(0)}ms`;
    }
    resultMeta.textContent = meta;
    resultDiv.hidden = false;
    errorDiv.hidden = true;
  }

  function setLoading(loading: boolean): void {
    captionFileBtn.disabled = loading || !selectedFile;
    captionUrlBtn.disabled = loading;
    captionFileBtn.textContent = loading ? "Generating…" : "Generate Caption";
    captionUrlBtn.textContent = loading ? "Generating…" : "Generate Caption";
  }

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const file = e.dataTransfer?.files?.[0];
    if (file && file.type.startsWith("image/")) {
      selectFile(file);
    }
  });
  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) selectFile(file);
  });

  function selectFile(file: File): void {
    selectedFile = file;
    captionFileBtn.disabled = false;
    previewArea.hidden = false;
    previewArea.innerHTML = `<img src="${URL.createObjectURL(file)}" alt="Preview" class="preview-img">`;
  }

  captionFileBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    hideError();
    setLoading(true);
    try {
      const data = await captionFromFile(selectedFile);
      showResult(data);
    } catch (e) {
      showError(e instanceof Error ? e.message : "Caption failed");
    } finally {
      setLoading(false);
    }
  });

  captionUrlBtn.addEventListener("click", async () => {
    const url = urlInput.value.trim();
    if (!url) {
      showError("Please enter an image URL");
      return;
    }
    hideError();
    setLoading(true);
    try {
      const data = await captionFromUrl(url);
      showResult(data);
      urlPreview.hidden = false;
      urlPreview.innerHTML = `<img src="${url}" alt="Preview" class="preview-img" onerror="this.style.display='none'">`;
    } catch (e) {
      showError(e instanceof Error ? e.message : "Caption failed");
    } finally {
      setLoading(false);
    }
  });

  app.querySelectorAll<HTMLButtonElement>(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      app.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      app.querySelectorAll(".panel").forEach((p) => p.classList.add("hidden"));
      tab.classList.add("active");
      app.querySelector(`[data-panel="${tab.dataset.tab}"]`)?.classList.remove("hidden");
      hideError();
    });
  });
}

createApp();
