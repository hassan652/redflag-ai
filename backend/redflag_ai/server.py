"""
FastAPI server for RedFlag AI web UI.

Provides a WebSocket endpoint for real-time workflow streaming
and serves the single-page HTML interface.
"""

import json
import asyncio
import os
import zipfile
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, UploadFile, File, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .workflow import (
    workflow,
    InputEvent,
    ToolCallEvent,
    GoDeeperEvent,
    AskHumanEvent,
    HumanAnswerEvent,
    ExplorationEndEvent,
    IngestEvent,
    RawAnswerEvent,
    get_agent,
    reset_agent,
    set_provider,
)
from .llm import generate_text
from .fs import parse_file
from .playground import (
    session_manager,
    MAX_UPLOAD_SIZE_MB,
    MAX_FILES_PER_SESSION,
)

app = FastAPI(title="RedFlag AI", description="AI-powered litigation file intelligence")

# CORS for Vercel frontend -> VPS backend
ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:8000,http://[::]:3000,http://[::1]:3000,https://redflag-ai.com,https://www.redflag-ai.com"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TaskRequest(BaseModel):
    """Request model for task submission."""
    task: str
    folder: str = "."


# =============================================================================
# Playground API (for public beta)
# =============================================================================


@app.get("/api/playground/status")
async def playground_status():
    """Check playground capacity — frontend shows queue/availability."""
    return await session_manager.get_status()


@app.post("/api/playground/session")
async def create_playground_session():
    """
    Create a new playground session.
    Returns session_id (used as auth token for subsequent requests).
    """
    session = await session_manager.create_session()
    if session is None:
        return JSONResponse(
            {"error": "Playground is full. Please try again in a few minutes."},
            status_code=503,
        )
    return {
        "session_id": session.session_id,
        "upload_dir": str(session.upload_dir),
        "max_files": MAX_FILES_PER_SESSION,
        "max_file_size_mb": MAX_UPLOAD_SIZE_MB,
    }


@app.post("/api/playground/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    x_session_id: str = Header(alias="X-Session-ID"),
):
    """
    Upload documents to a playground session.
    Accepts PDFs, DOCX, and text files. Also accepts .zip archives.
    """
    session = await session_manager.get_session(x_session_id)
    if session is None:
        return JSONResponse(
            {"error": "Invalid or expired session. Please start a new session."},
            status_code=401,
        )

    uploaded = []
    errors = []

    for file in files:
        # Check file count limit
        if session.files_uploaded >= MAX_FILES_PER_SESSION:
            errors.append(f"{file.filename}: file limit reached ({MAX_FILES_PER_SESSION})")
            break

        # Read file content
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)

        # Check size limit
        if size_mb > MAX_UPLOAD_SIZE_MB:
            errors.append(f"{file.filename}: exceeds {MAX_UPLOAD_SIZE_MB}MB limit")
            continue

        # Sanitize filename (prevent path traversal)
        safe_name = Path(file.filename).name
        if not safe_name or safe_name.startswith("."):
            errors.append(f"{file.filename}: invalid filename")
            continue

        # Handle zip files — extract contents
        if safe_name.lower().endswith(".zip"):
            try:
                zip_path = session.upload_dir / safe_name
                zip_path.write_bytes(content)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    # Security: check for path traversal in zip
                    for info in zf.infolist():
                        if info.filename.startswith("/") or ".." in info.filename:
                            continue
                        if info.is_dir():
                            continue
                        extracted_name = Path(info.filename).name
                        zf.extract(info, session.upload_dir)
                        session.files_uploaded += 1
                        uploaded.append(extracted_name)
                zip_path.unlink()  # Remove the zip after extraction
            except zipfile.BadZipFile:
                errors.append(f"{safe_name}: invalid zip file")
        else:
            # Regular file
            dest = session.upload_dir / safe_name
            dest.write_bytes(content)
            session.files_uploaded += 1
            uploaded.append(safe_name)

    return {
        "uploaded": uploaded,
        "errors": errors,
        "total_files": session.files_uploaded,
    }


@app.delete("/api/playground/session")
async def end_playground_session(x_session_id: str = Header(alias="X-Session-ID")):
    """End a session and clean up uploaded files."""
    await session_manager.end_session(x_session_id)
    return {"status": "session ended"}


@app.post("/api/playground/session/end")
async def end_playground_session_beacon(request: Request):
    """End a session via sendBeacon on page unload (POST, no custom headers)."""
    try:
        body = await request.json()
        session_id = body.get("session_id")
    except Exception:
        return JSONResponse({"error": "invalid body"}, status_code=400)
    if session_id:
        await session_manager.end_session(session_id)
    return {"status": "session ended"}


# =============================================================================
# Original UI + API
# =============================================================================


@app.get("/")
async def get_root():
    """API root — UI is served separately."""
    return JSONResponse({"service": "RedFlag AI API", "status": "running"})


@app.get("/api/folders")
async def list_folders(path: str = "."):
    """
    List folders in the given path.
    Returns list of folder names and current path info.
    """
    try:
        base_path = Path(path).resolve()
        if not base_path.exists():
            return JSONResponse({"error": "Path not found"}, status_code=404)
        if not base_path.is_dir():
            return JSONResponse({"error": "Not a directory"}, status_code=400)
        
        # Get folders (non-hidden)
        folders = sorted([
            f.name for f in base_path.iterdir()
            if f.is_dir() and not f.name.startswith('.')
        ])
        
        # Get parent path (if not at root)
        parent = str(base_path.parent) if base_path != base_path.parent else None
        
        return {
            "current": str(base_path),
            "parent": parent,
            "folders": folders,
            "files_count": len([f for f in base_path.iterdir() if f.is_file()]),
        }
    except PermissionError:
        return JSONResponse({"error": "Permission denied"}, status_code=403)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.websocket("/ws/explore")
async def websocket_explore(websocket: WebSocket):
    """
    WebSocket endpoint for real-time exploration streaming.
    
    Protocol:
    1. Client sends: {"task": "user question", "session_id": "..."}
    2. Server streams events: {"type": "...", "data": {...}}
    3. Final event: {"type": "complete", "data": {...}}
    """
    await websocket.accept()
    
    try:
        # Receive the task
        data = await websocket.receive_json()
        task = data.get("task", "")
        folder = data.get("folder")
        provider = data.get("provider", "openrouter")
        model = data.get("model") or None
        api_key = data.get("api_key") or None
        session_id = data.get("session_id")
        chat_only = data.get("chat_only", False)
        compact_context = bool(data.get("compact_context", False))
        
        if not task:
            await websocket.send_json({
                "type": "error",
                "data": {"message": "No task provided"}
            })
            return

        if chat_only:
            await websocket.send_json({
                "type": "start",
                "data": {"task": task, "provider": provider, "mode": "chat"}
            })

            # Build prompt — include uploaded document text + prior conversation history
            prompt = task
            docs_scanned = 0
            session_obj = None
            if session_id:
                session_obj = await session_manager.get_session(session_id)
                if session_obj and session_obj.files_uploaded > 0:
                    doc_parts: list[str] = []
                    for f in sorted(session_obj.upload_dir.iterdir()):
                        if f.is_file():
                            content = parse_file(str(f))
                            if content and not content.startswith("Error") and not content.startswith("Unsupported"):
                                doc_parts.append(f"### {f.name}\n{content}")
                                docs_scanned += 1
                    if doc_parts:
                        docs_block = "\n\n---\n\n".join(doc_parts)
                        prompt = (
                            f"The user has uploaded {docs_scanned} document(s). "
                            f"Use them to answer the question.\n\n"
                            f"{docs_block}\n\n"
                            f"---\n\nUser question: {task}"
                        )

            # Prepend prior conversation turns so the LLM can follow up
            history_prefix = ""
            if session_obj and session_obj.chat_history:
                turns = []
                for turn in session_obj.chat_history[-6:]:  # keep last 3 Q&A pairs
                    turns.append(f"User: {turn['user']}\nAssistant: {turn['assistant']}")
                history_prefix = "Previous conversation:\n\n" + "\n\n".join(turns) + "\n\n---\n\n"
                prompt = history_prefix + prompt

            answer, usage = await generate_text(
                prompt=prompt,
                provider=provider,
                model=model,
                api_key=api_key,
            )

            # Store this exchange in the session history
            if session_obj is not None:
                session_obj.chat_history.append({"user": task, "assistant": answer})
                # Cap history at 20 turns to avoid unbounded growth
                if len(session_obj.chat_history) > 20:
                    session_obj.chat_history = session_obj.chat_history[-20:]

            input_cost, output_cost, total_cost = usage._calculate_cost()

            await websocket.send_json({
                "type": "complete",
                "data": {
                    "final_result": answer,
                    "error": None,
                    "stats": {
                        "steps": 1,
                        "api_calls": usage.api_calls,
                        "documents_scanned": docs_scanned,
                        "documents_parsed": docs_scanned,
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                        "tool_result_chars": usage.tool_result_chars,
                        "estimated_cost": round(total_cost, 6),
                        "mode": "chat",
                    }
                }
            })
            return
        
        # If session_id is provided, use the session's upload directory
        if session_id:
            session = await session_manager.get_session(session_id)
            if session is None:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Invalid or expired session."}
                })
                return
            if session.files_uploaded == 0:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "No files uploaded yet. Please upload documents first."}
                })
                return
            folder = str(session.upload_dir)
            session.is_processing = True

        if not folder:
            await websocket.send_json({
                "type": "error",
                "data": {"message": "No documents available. Please upload files first."}
            })
            return
        
        if not task:
            await websocket.send_json({
                "type": "error",
                "data": {"message": "No task provided"}
            })
            return
        
        # Validate folder
        folder_path = Path(folder).resolve()
        if not folder_path.exists() or not folder_path.is_dir():
            await websocket.send_json({
                "type": "error",
                "data": {"message": f"Invalid folder: {folder}"}
            })
            return
        
        # Change to target folder
        original_cwd = os.getcwd()
        os.chdir(folder_path)
        
        # Set provider and reset agent for fresh state
        set_provider(provider, model, api_key)
        reset_agent()
        
        # Send start event
        await websocket.send_json({
            "type": "start",
            "data": {"task": task, "folder": str(folder_path), "provider": provider}
        })
        
        # Run the workflow
        step_number = 0
        handler = workflow.run(start_event=InputEvent(task=task, compact_context=compact_context))
        cancelled = False
        
        async def listen_for_cancel():
            """Background listener for cancel/human_response messages from client."""
            nonlocal cancelled
            try:
                while True:
                    msg = await websocket.receive_json()
                    if msg.get("type") == "cancel":
                        cancelled = True
                        return
                    elif msg.get("type") == "human_response":
                        # Forward human responses to the workflow
                        handler.ctx.send_event(
                            HumanAnswerEvent(response=msg.get("response", ""))
                        )
            except (WebSocketDisconnect, Exception):
                cancelled = True
        
        # Start listening for cancel in the background
        cancel_listener = asyncio.create_task(listen_for_cancel())
        
        try:
            async for event in handler.stream_events():
                if cancelled:
                    break
                
                if isinstance(event, IngestEvent):
                    step_number += 1
                    await websocket.send_json({
                        "type": "ingest",
                        "data": {
                            "step": step_number,
                            "message": event.message,
                            "documents_found": event.documents_found,
                            "hierarchy_summary": event.hierarchy_summary,
                        }
                    })
                
                elif isinstance(event, ToolCallEvent):
                    step_number += 1
                    await websocket.send_json({
                        "type": "tool_call",
                        "data": {
                            "step": step_number,
                            "tool_name": event.tool_name,
                            "tool_input": event.tool_input,
                            "reason": event.reason,
                        }
                    })
                    
                elif isinstance(event, GoDeeperEvent):
                    step_number += 1
                    await websocket.send_json({
                        "type": "go_deeper",
                        "data": {
                            "step": step_number,
                            "directory": event.directory,
                            "reason": event.reason,
                        }
                    })
                    
                elif isinstance(event, AskHumanEvent):
                    step_number += 1
                    await websocket.send_json({
                        "type": "ask_human",
                        "data": {
                            "step": step_number,
                            "question": event.question,
                            "reason": event.reason,
                        }
                    })
                    # Human responses are handled by the cancel_listener
                
                elif isinstance(event, RawAnswerEvent):
                    await websocket.send_json({
                        "type": "verifying",
                        "data": {
                            "message": "Verifying citations against source documents...",
                        }
                    })
        finally:
            cancel_listener.cancel()
            try:
                await cancel_listener
            except asyncio.CancelledError:
                pass
        
        if cancelled:
            await websocket.send_json({
                "type": "cancelled",
                "data": {"message": "Query cancelled by user."}
            })
            return
        
        # Get final result
        result = await handler
        
        # Get token usage
        agent = get_agent()
        usage = agent.token_usage
        input_cost, output_cost, total_cost = usage._calculate_cost()
        
        await websocket.send_json({
            "type": "complete",
            "data": {
                "final_result": result.final_result,
                "error": result.error,
                "stats": {
                    "steps": step_number,
                    "api_calls": usage.api_calls,
                    "documents_scanned": usage.documents_scanned,
                    "documents_parsed": usage.documents_parsed,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "tool_result_chars": usage.tool_result_chars,
                    "estimated_cost": round(total_cost, 6),
                }
            }
        })
        
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "data": {"message": str(e)}
        })
    finally:
        # Restore original working directory
        if 'original_cwd' in locals():
            os.chdir(original_cwd)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import sys
    _host = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    _port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    run_server(host=_host, port=_port)

