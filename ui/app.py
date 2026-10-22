"""Streamlit chat UI. Run: streamlit run app.py

The backend is the source of truth: history and files are reloaded from the API
on every rerun, and the current thread id lives in the URL (?thread=...) so a
page refresh or a shared link reopens the same conversation.
"""

import os
from decimal import Decimal
from typing import Any

import httpx
import streamlit as st

from api_client import ApiError, ChatApi, Delta, Done

API_URL = os.environ.get("API_URL", "http://localhost:8000")
FILE_TYPES = ["pdf", "docx", "txt", "md", "csv"]

st.set_page_config(page_title="Chat", page_icon=":speech_balloon:", layout="centered")

state = st.session_state
state.setdefault("uploader_key", 0)  # bumped to clear the file uploader after each batch
state.setdefault("conflicts", [])  # uploads waiting for a "replace or keep both?" answer
state.setdefault("notices", [])  # one-shot messages to show after a rerun


@st.cache_resource
def get_api() -> ChatApi:
    return ChatApi(API_URL)


def usage_caption(usage: dict[str, Any] | None) -> str:
    if not usage:
        return ""
    parts = [
        f"{usage['prompt_tokens']:,} in",
        f"{usage['cached_tokens']:,} cached",
        f"{usage['completion_tokens']:,} out",
    ]
    if usage.get("cost") is not None:
        parts.append(f"${Decimal(str(usage['cost'])):.6f}")
    return " · ".join(parts)


def file_label(file: dict[str, Any]) -> str:
    version = f" (v{file['version']})" if file["version"] > 1 else ""
    return f":material/attach_file: {file['filename']}{version}"


def open_thread(thread_id: str | None) -> None:
    if thread_id:
        st.query_params["thread"] = thread_id
    else:
        st.query_params.pop("thread", None)


def ensure_thread(model: str) -> str:
    thread_id = st.query_params.get("thread")
    if not thread_id:
        thread_id = api.create_thread(model)["id"]
        open_thread(thread_id)
    return thread_id


def upload(thread_id: str, name: str, data: bytes, on_conflict: str | None = None) -> None:
    """Upload one file; queue a conflict question or a notice for the next rerun."""
    try:
        file = api.upload_file(thread_id, name, data, on_conflict)
    except ApiError as exc:
        if exc.code == "file_name_conflict":
            state.conflicts.append({"name": name, "data": data, "message": exc.message})
        else:
            state.notices.append(("error", f"{name}: {exc.message}"))
        return
    if file["version"] > 1:
        state.notices.append(("success", f"{file['filename']} updated (v{file['version']})"))
    else:
        state.notices.append(("success", f"{file['filename']} attached"))


api = get_api()

try:
    models = api.models()
except httpx.HTTPError:
    st.error(
        f"Can't reach the chat API at `{API_URL}`. Start it with `docker compose up` "
        "(or set the API_URL environment variable)."
    )
    st.stop()

thread_id = st.query_params.get("thread")
thread: dict[str, Any] | None = None
history: list[dict[str, Any]] = []
files: list[dict[str, Any]] = []
if thread_id:
    try:
        thread = api.get_thread(thread_id)
        history = api.messages(thread_id)
        files = api.files(thread_id)
    except ApiError as exc:
        if exc.status != 404:
            raise
        st.warning("That conversation no longer exists. Starting a new one.")
        open_thread(None)
        thread_id = None

pending = [f for f in files if f["message_id"] is None and f["superseded_by_id"] is None]

with st.sidebar:
    st.header("Chat")
    model = st.selectbox(
        "Model for new chats",
        models["allowed"],
        index=models["allowed"].index(models["default"]),
    )
    if st.button("New chat", width="stretch", type="primary"):
        open_thread(None)
        state.conflicts.clear()
        st.rerun()

    st.divider()
    st.subheader("Files")
    uploaded = st.file_uploader(
        "Attach files (sent with your next message)",
        type=FILE_TYPES,
        accept_multiple_files=True,
        key=f"uploader-{state.uploader_key}",
    )
    if uploaded:
        target = ensure_thread(model)
        with st.spinner("Uploading..."):
            for item in uploaded:
                upload(target, item.name, item.getvalue())
        state.uploader_key += 1  # clears the widget so files aren't uploaded twice
        st.rerun()

    # Same name, different content: the user decides what the upload means.
    for i, conflict in enumerate(list(state.conflicts)):
        st.warning(conflict["message"])
        replace, keep, cancel = st.columns(3)
        choice = None
        if replace.button("Replace", key=f"replace-{i}", width="stretch"):
            choice = "replace"
        if keep.button("Keep both", key=f"keep-{i}", width="stretch"):
            choice = "keep_both"
        if cancel.button("Cancel", key=f"cancel-{i}", width="stretch"):
            state.conflicts.remove(conflict)
            st.rerun()
        if choice:
            state.conflicts.remove(conflict)
            upload(ensure_thread(model), conflict["name"], conflict["data"], choice)
            st.rerun()

    for kind, text in state.notices:
        (st.success if kind == "success" else st.error)(text)
    state.notices.clear()

    if pending:
        st.caption("Waiting to be sent with your next message:")
        for file in pending:
            st.markdown(file_label(file))

    if thread:
        st.divider()
        st.caption(f"**Model:** `{thread['model']}`")
        usages = [m["usage"] for m in history if m.get("usage")]
        total_cost = sum(Decimal(str(u["cost"])) for u in usages if u.get("cost") is not None)
        prompt_total = sum(u["prompt_tokens"] for u in usages)
        cached_total = sum(u["cached_tokens"] for u in usages)
        st.metric("Context size (tokens)", f"{thread['est_context_tokens']:,}")
        st.metric("Thread cost", f"${total_cost:.6f}")
        if prompt_total:
            st.metric("Cache hit rate", f"{cached_total / prompt_total:.0%}")

# Conversation so far.
if not history:
    st.markdown(
        "#### Ask anything, or attach a file from the sidebar and ask about it.\n"
        "Follow-up questions remember the whole conversation."
    )
for message in history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for attachment in message.get("attachments", []):
            st.caption(file_label(attachment))
        if message["role"] == "assistant" and message.get("usage"):
            st.caption(usage_caption(message["usage"]))

# New turn.
if prompt := st.chat_input("Message"):
    thread_id = ensure_thread(model)

    with st.chat_message("user"):
        st.markdown(prompt)
        for file in pending:
            st.caption(file_label(file))

    with st.chat_message("assistant"):
        done: dict[str, Any] = {}

        def reply_text() -> Any:
            for event in api.stream_reply(thread_id, prompt):
                if isinstance(event, Delta):
                    yield event.text
                elif isinstance(event, Done):
                    done.update(event.data)

        try:
            st.write_stream(reply_text())
        except ApiError as exc:
            st.error(exc.message)
        except httpx.HTTPError:
            st.error("Lost the connection to the chat API. Please try again.")
        else:
            # Reload from the API so history, files and totals include this turn.
            st.rerun()
