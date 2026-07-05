const editorToken = new URLSearchParams(window.location.search).get("token") || "";

export async function requestJSON(path, options = {}) {
  const response = await fetch(path, withEditorToken(options));
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

export async function postJSON(path, payload) {
  return requestJSON(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function requestBlob(path, options = {}) {
  const response = await fetch(path, withEditorToken(options));
  if (!response.ok) {
    let message = response.statusText;
    try {
      const payload = await response.json();
      message = payload.error || message;
    } catch {
      // Binary endpoints can fail before JSON is available.
    }
    throw new Error(message);
  }
  return response;
}

function withEditorToken(options) {
  const headers = new Headers(options.headers || {});
  if (editorToken) headers.set("X-SuperGLM-Editor-Token", editorToken);
  return { ...options, headers };
}
