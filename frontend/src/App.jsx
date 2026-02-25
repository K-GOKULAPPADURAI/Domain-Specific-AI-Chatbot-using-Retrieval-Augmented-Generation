import React, { useState, useCallback, useEffect, useRef } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import {
  ArrowUp,
  Bot,
  FileText,
  Loader2,
  Plus,
  ShieldCheck,
  Sparkles,
  Trash2,
  UploadCloud,
  User,
  X,
} from 'lucide-react';

const normalizeBase = (base) => (base || '').trim().replace(/\/+$/, '');

const uniqueBases = (bases) => {
  const seen = new Set();
  return bases.filter((base) => {
    const normalized = normalizeBase(base);
    if (!normalized || seen.has(normalized)) return false;
    seen.add(normalized);
    return true;
  });
};

const getApiCandidates = () => {
  const configured = normalizeBase(import.meta.env.VITE_API_BASE_URL);
  if (configured) return [configured];

  if (import.meta.env.DEV) {
    return ['http://127.0.0.1:8000', 'http://localhost:8000'];
  }

  const origin =
    typeof window !== 'undefined' && window.location?.origin?.startsWith('http')
      ? normalizeBase(window.location.origin)
      : '';

  return uniqueBases([origin, 'http://127.0.0.1:8000', 'http://localhost:8000']);
};

const API_CANDIDATES = getApiCandidates();

const RETRYABLE_HTTP_STATUS = new Set([404, 405]);

const formatBackendTargets = () => API_CANDIDATES.join(', ');

const extractErrorDetail = (error) => {
  const data = error?.response?.data;
  if (typeof data?.detail === 'string' && data.detail.trim()) return data.detail;
  if (typeof data?.message === 'string' && data.message.trim()) return data.message;
  if (typeof data === 'string' && data.trim()) return data;
  if (typeof error?.message === 'string' && error.message.trim()) return error.message;
  return '';
};

const buildUserError = (error, fallback) => {
  const detail = extractErrorDetail(error);
  if (!error?.response) {
    const networkMessage = detail && detail !== 'Network Error' ? detail : fallback;
    return `${networkMessage} Tried backend targets: ${formatBackendTargets()}.`;
  }
  if (detail) return detail;
  return `${fallback} Tried backend targets: ${formatBackendTargets()}.`;
};

const joinUrl = (base, path) => `${normalizeBase(base)}${path}`;

async function axiosWithFallback(requestFactory, candidates = API_CANDIDATES) {
  const bases = uniqueBases(candidates);
  let lastError = null;

  for (const base of bases) {
    try {
      const response = await requestFactory(base);
      return { response, base };
    } catch (error) {
      const status = error?.response?.status;
      if (status && RETRYABLE_HTTP_STATUS.has(status)) {
        lastError = error;
        continue;
      }
      if (!error?.response) {
        lastError = error;
        continue;
      }
      throw error;
    }
  }

  throw lastError || new Error('Backend is unreachable.');
}

async function fetchWithFallback(path, init, candidates = API_CANDIDATES) {
  const bases = uniqueBases(candidates);
  let lastNetworkError = null;
  let lastHttpResponse = null;
  let lastBase = bases[0] || '';

  for (const base of bases) {
    lastBase = base;
    try {
      const response = await fetch(joinUrl(base, path), init);
      if (RETRYABLE_HTTP_STATUS.has(response.status)) {
        lastHttpResponse = response;
        continue;
      }
      return { response, base };
    } catch (error) {
      lastNetworkError = error;
    }
  }

  if (lastHttpResponse) {
    return { response: lastHttpResponse, base: lastBase };
  }
  throw lastNetworkError || new Error('Backend is unreachable.');
}

const STARTER_TEXT =
  'Upload your domain documents, then ask questions. Responses are grounded in your uploaded files.';

const QUICK_PROMPTS = [
  'Summarize the key points from all uploaded documents.',
  'What are the most important deadlines or dates mentioned?',
  'List all action items I should focus on next.',
  'Give me a concise FAQ based on these documents.',
];

function formatFileSize(size) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function App() {
  const [messages, setMessages] = useState([{ role: 'ai', text: STARTER_TEXT }]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [composerFocused, setComposerFocused] = useState(false);

  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  useEffect(() => {
    const area = textareaRef.current;
    if (!area) return;
    area.style.height = '0px';
    area.style.height = `${Math.min(area.scrollHeight, 180)}px`;
  }, [input]);

  // Fetch existing documents on mount
  useEffect(() => {
    axiosWithFallback((base) => axios.get(joinUrl(base, '/documents')))
      .then(({ response }) => {
        setUploadedFiles(response.data.files || []);
      })
      .catch(() => {});
  }, []);

  const appendAssistantMessage = (text) => {
    setMessages((prev) => [...prev, { role: 'ai', text }]);
  };

  const onDrop = useCallback(async (acceptedFiles) => {
    if (!acceptedFiles.length) return;

    setUploading(true);
    const formData = new FormData();
    acceptedFiles.forEach((file) => formData.append('files', file));

    try {
      const { response: uploadRes, base } = await axiosWithFallback((candidate) =>
        axios.post(joinUrl(candidate, '/upload'), formData)
      );
      // Refresh file list from server
      const { response: docsRes } = await axiosWithFallback(
        (candidate) => axios.get(joinUrl(candidate, '/documents')),
        [base, ...API_CANDIDATES]
      );
      setUploadedFiles(docsRes.data.files || []);
      let msg = uploadRes?.data?.message || `Indexed ${acceptedFiles.length} file(s).`;
      if (uploadRes?.data?.warnings?.length) {
        msg += '\n\nWarnings:\n' + uploadRes.data.warnings.join('\n');
      }
      appendAssistantMessage(msg);
    } catch (err) {
      appendAssistantMessage(buildUserError(err, 'Upload failed. Make sure the backend is running and the files are readable.'));
    } finally {
      setUploading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
    },
    multiple: true,
    disabled: uploading,
    noClick: true,
    noKeyboard: true,
  });

  const handleSend = async (prefilledQuestion) => {
    const question = (prefilledQuestion ?? input).trim();
    if (!question || loading) return;

    setMessages((prev) => [...prev, { role: 'user', text: question }]);
    setInput('');
    setLoading(true);

    // Streaming response
    try {
      const formData = new FormData();
      formData.append('question', question);

      const { response } = await fetchWithFallback('/chat/stream', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let serverMessage = '';
        try {
          const payload = await response.json();
          serverMessage = payload?.detail || payload?.message || '';
        } catch {
          // ignore parsing error
        }
        throw new Error(serverMessage || `Stream request failed (${response.status}).`);
      }

      const reader = response.body?.getReader?.();
      if (!reader) {
        throw new Error('Stream is unavailable from the backend response.');
      }

      const decoder = new TextDecoder();
      let fullAnswer = '';
      let messageAdded = false;
      let streamBuffer = '';
      let doneSignalReceived = false;
      let statusPlaceholderActive = false;

      const upsertAssistantMessage = () => {
        if (!messageAdded) {
          setMessages((prev) => [...prev, { role: 'ai', text: fullAnswer }]);
          messageAdded = true;
          return;
        }
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: 'ai', text: fullAnswer };
          return updated;
        });
      };

      const processSseEvent = (eventPayload) => {
        const lines = eventPayload.split('\n');
        for (const rawLine of lines) {
          const line = rawLine.trimEnd();
          if (!line.startsWith('data:')) continue;

          const data = line.slice(5).trimStart();
          if (!data) continue;
          if (data === '[DONE]') {
            doneSignalReceived = true;
            return;
          }

          try {
            const parsed = JSON.parse(data);
            if (parsed.status && !messageAdded) {
              setMessages((prev) => [...prev, { role: 'ai', text: 'Thinking...' }]);
              messageAdded = true;
              statusPlaceholderActive = true;
              continue;
            }
            if (parsed.error) {
              fullAnswer = `Error: ${parsed.error}`;
              doneSignalReceived = true;
              upsertAssistantMessage();
              return;
            }
            if (parsed.token) {
              fullAnswer += parsed.token;
              statusPlaceholderActive = false;
              upsertAssistantMessage();
            }
          } catch {
            // Ignore malformed SSE payloads and continue with next event.
          }
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        streamBuffer += decoder.decode(value, { stream: true });
        const events = streamBuffer.split('\n\n');
        streamBuffer = events.pop() ?? '';

        for (const eventPayload of events) {
          processSseEvent(eventPayload);
          if (doneSignalReceived) break;
        }

        if (doneSignalReceived) {
          await reader.cancel();
          break;
        }
      }

      streamBuffer += decoder.decode();
      if (streamBuffer.trim() && !doneSignalReceived) {
        processSseEvent(streamBuffer);
      }

      if (!messageAdded) {
        appendAssistantMessage(fullAnswer || 'No answer returned from the server.');
      } else if (statusPlaceholderActive && !fullAnswer) {
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: 'ai', text: 'No answer returned from the server.' };
          return updated;
        });
      }
    } catch {
      // Fallback to non-streaming
      try {
        const formData = new FormData();
        formData.append('question', question);
        const { response } = await axiosWithFallback((base) =>
          axios.post(joinUrl(base, '/chat'), formData)
        );
        appendAssistantMessage(response?.data?.answer || 'No answer returned.');
      } catch (err) {
        appendAssistantMessage(buildUserError(err, 'Could not reach the AI server.'));
      }
    } finally {
      setLoading(false);
    }
  };

  const handleComposerKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  const handleNewChat = async () => {
    setMessages([{ role: 'ai', text: STARTER_TEXT }]);
    setInput('');
    try {
      await axiosWithFallback((base) => axios.post(joinUrl(base, '/chat/clear')));
    } catch {
      // ignore
    }
  };

  const handleDeleteFile = async (filename) => {
    try {
      await axiosWithFallback((base) =>
        axios.delete(joinUrl(base, `/documents/${encodeURIComponent(filename)}`))
      );
      setUploadedFiles((prev) => prev.filter((f) => f.name !== filename));
      appendAssistantMessage(`Deleted ${filename}.`);
    } catch (err) {
      appendAssistantMessage(buildUserError(err, `Failed to delete ${filename}.`));
    }
  };

  const handleClearDocuments = async () => {
    try {
      await axiosWithFallback((base) => axios.delete(joinUrl(base, '/documents')));
      setUploadedFiles([]);
      appendAssistantMessage('All documents and index cleared. Upload new files to start fresh.');
    } catch (err) {
      appendAssistantMessage(buildUserError(err, 'Failed to clear documents.'));
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">
          <div className="brand-mark">
            <Sparkles size={16} />
          </div>
          <div className="brand-copy">
            <h1>Domain RAG Assistant</h1>
            <p>Fast, grounded answers from your private knowledge</p>
          </div>
        </div>

        <div className="header-actions">
          <button className="ghost-btn" onClick={handleNewChat}>
            <Plus size={14} />
            New chat
          </button>
          <button className="ghost-btn danger" onClick={handleClearDocuments} disabled={!uploadedFiles.length}>
            <Trash2 size={14} />
            Clear all docs
          </button>
        </div>
      </header>

      <main className="workspace">
        <aside className="docs-pane">
          <div className="pane-head">
            <h2>Knowledge Base</h2>
            <span className="status-pill">{uploading ? 'Indexing...' : `${uploadedFiles.length} files`}</span>
          </div>

          <div {...getRootProps()} className={`upload-zone ${isDragActive ? 'dragging' : ''} ${uploading ? 'busy' : ''}`}>
            <input {...getInputProps()} />
            <div className="upload-zone-copy">
              <div className="upload-title-row">
                <UploadCloud size={15} />
                <span>{uploading ? 'Uploading documents' : 'Upload PDF/TXT files'}</span>
              </div>
              <p>{isDragActive ? 'Drop files to start indexing' : 'Drag and drop files here, or choose manually'}</p>
            </div>
            <button className="upload-btn" onClick={open} type="button" disabled={uploading}>
              {uploading ? <Loader2 size={14} className="spin" /> : 'Choose files'}
            </button>
          </div>

          <div className="docs-list" aria-label="Uploaded documents">
            {uploadedFiles.length ? (
              uploadedFiles.map((file) => (
                <article key={file.name} className="doc-row">
                  <div className="doc-icon">
                    <FileText size={14} />
                  </div>
                  <div className="doc-meta">
                    <p className="doc-name">{file.name}</p>
                    <p className="doc-size">{formatFileSize(file.size)}</p>
                  </div>
                  <button
                    className="doc-delete-btn"
                    onClick={() => handleDeleteFile(file.name)}
                    title={`Delete ${file.name}`}
                    type="button"
                  >
                    <X size={12} />
                  </button>
                </article>
              ))
            ) : (
              <p className="empty-docs">No documents uploaded yet.</p>
            )}
          </div>

          <div className="pane-foot">
            <ShieldCheck size={14} />
            <span>Answers are constrained by retrieved document context.</span>
          </div>
        </aside>

        <section className="chat-pane">
          <div className="messages-panel">
            {messages.map((message, index) => (
              <article key={`${message.role}-${index}`} className={`message-row ${message.role}`}>
                <div className={`message-avatar ${message.role}`}>
                  {message.role === 'ai' ? <Bot size={14} /> : <User size={14} />}
                </div>
                <div className={`message-bubble ${message.role}`}>{message.text}</div>
              </article>
            ))}

            {loading && !messages[messages.length - 1]?.text?.length && (
              <article className="message-row ai">
                <div className="message-avatar ai">
                  <Bot size={14} />
                </div>
                <div className="message-bubble ai thinking-card">
                  <div className="typing-dots" aria-hidden="true">
                    <span />
                    <span />
                    <span />
                  </div>
                  <div className="thinking-lines" aria-hidden="true">
                    <span />
                    <span />
                  </div>
                  <span className="thinking-label">Searching and generating answer...</span>
                </div>
              </article>
            )}

            <div ref={messagesEndRef} />
          </div>

          {!loading && messages.length <= 2 && (
            <div className="prompt-chips" aria-label="Prompt suggestions">
              {QUICK_PROMPTS.map((prompt) => (
                <button key={prompt} type="button" className="prompt-chip" onClick={() => handleSend(prompt)}>
                  {prompt}
                </button>
              ))}
            </div>
          )}

          <div className={`composer ${composerFocused ? 'focus' : ''}`}>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleComposerKeyDown}
              onFocus={() => setComposerFocused(true)}
              onBlur={() => setComposerFocused(false)}
              placeholder="Ask something about your uploaded documents..."
              rows={1}
            />
            <button
              className="send-btn"
              onClick={() => handleSend()}
              disabled={loading || !input.trim()}
              type="button"
              aria-label="Send message"
            >
              {loading ? <Loader2 size={15} className="spin" /> : <ArrowUp size={15} />}
            </button>
          </div>

          <p className="composer-hint">Enter to send, Shift+Enter for a new line.</p>
        </section>
      </main>
    </div>
  );
}

export default App;
