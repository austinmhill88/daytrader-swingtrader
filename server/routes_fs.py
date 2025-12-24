"""
File system routes - Safe file access within configured sandbox.
"""
import io
import difflib
from pathlib import Path
from typing import List

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse


def make_router(
    root_dir: str, 
    allow_write: bool, 
    allowed_globs: List[str], 
    denied_globs: List[str]
):
    """
    Create filesystem sandbox router.
    
    Args:
        root_dir: Root directory for file operations
        allow_write: Whether to allow write operations
        allowed_globs: List of allowed glob patterns
        denied_globs: List of denied glob patterns (takes precedence)
        
    Returns:
        FastAPI router with filesystem endpoints
    """
    router = APIRouter()
    ROOT = Path(root_dir).resolve()

    def _ensure_in_root(rel_path: str) -> Path:
        """
        Ensure path is within root directory.
        
        Args:
            rel_path: Relative path within root
            
        Returns:
            Resolved absolute path
            
        Raises:
            RuntimeError: If path escapes sandbox
        """
        p = (ROOT / rel_path).resolve()
        if ROOT not in p.parents and p != ROOT:
            raise RuntimeError("Path escapes sandbox root.")
        return p

    def _match_globs(path: Path, patterns: List[str]) -> bool:
        """
        Check if path matches any glob pattern.
        
        Args:
            path: Path to check
            patterns: List of glob patterns
            
        Returns:
            True if path matches any pattern
        """
        if not patterns:
            return True
        
        rel = path.relative_to(ROOT).as_posix()
        for pat in patterns:
            # Accept both "glob" and "/glob"
            if Path(rel).match(pat) or Path("/" + rel).match(pat):
                return True
        return False

    def _is_allowed(path: Path) -> bool:
        """
        Check if path is allowed by policy.
        
        Args:
            path: Path to check
            
        Returns:
            True if path is allowed
        """
        # Denied patterns take precedence
        if denied_globs and _match_globs(path, denied_globs):
            return False
        
        # Check allowed patterns
        if allowed_globs:
            return _match_globs(path, allowed_globs)
        
        return True

    @router.get("/fs/list")
    async def list_dir(path: str = Query("", description="Relative path within root")):
        """
        List files and directories.
        
        Args:
            path: Relative path within root (empty string for root)
            
        Returns:
            List of entries with name, path, type, size, modified
        """
        try:
            base = _ensure_in_root(path)
            
            if not base.exists():
                return JSONResponse(status_code=404, content={"error": "Path not found"})
            
            if base.is_file():
                return JSONResponse(
                    status_code=400, 
                    content={"error": "Path is a file; use /fs/read"}
                )
            
            entries = []
            for entry in sorted(base.iterdir()):
                if not _is_allowed(entry):
                    continue
                
                st = entry.stat()
                entries.append({
                    "name": entry.name,
                    "path": str(entry.relative_to(ROOT)),
                    "type": "dir" if entry.is_dir() else "file",
                    "size": st.st_size,
                    "modified": int(st.st_mtime),
                })
            
            return {"path": str(base.relative_to(ROOT)), "entries": entries}
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

    @router.get("/fs/read")
    async def read_file(
        path: str = Query(..., description="Relative path to file"), 
        max_bytes: int = Query(512_000, description="Maximum bytes to read")
    ):
        """
        Read file contents.
        
        Args:
            path: Relative path to file
            max_bytes: Maximum bytes to read
            
        Returns:
            File content, size, and truncation info
        """
        try:
            fp = _ensure_in_root(path)
            
            if not fp.exists() or not fp.is_file():
                return JSONResponse(status_code=404, content={"error": "File not found"})
            
            if not _is_allowed(fp):
                return JSONResponse(
                    status_code=403, 
                    content={"error": "Access denied by policy"}
                )
            
            data = fp.read_bytes()
            truncated = False
            
            if len(data) > max_bytes:
                data = data[:max_bytes]
                truncated = True
            
            return {
                "path": str(fp.relative_to(ROOT)),
                "size": fp.stat().st_size,
                "truncated": truncated,
                "content": data.decode("utf-8", errors="replace"),
            }
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

    @router.get("/fs/tail")
    async def tail_file(
        path: str = Query(..., description="Relative path to file"), 
        lines: int = Query(200, description="Number of lines to return")
    ):
        """
        Get last N lines of file (like tail command).
        
        Args:
            path: Relative path to file
            lines: Number of lines to return
            
        Returns:
            Last N lines of file
        """
        try:
            fp = _ensure_in_root(path)
            
            if not fp.exists() or not fp.is_file():
                return JSONResponse(status_code=404, content={"error": "File not found"})
            
            if not _is_allowed(fp):
                return JSONResponse(
                    status_code=403, 
                    content={"error": "Access denied by policy"}
                )
            
            data = fp.read_text(encoding="utf-8", errors="replace")
            tail = "\n".join(data.splitlines()[-lines:])
            
            return {
                "path": str(fp.relative_to(ROOT)), 
                "lines": lines, 
                "content": tail
            }
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

    @router.post("/fs/diff")
    async def diff(body: dict):
        """
        Generate unified diff without writing.
        
        Request body:
            - path: Relative path to file
            - new_content: New content to compare
            
        Returns:
            Unified diff of changes
        """
        try:
            path = body.get("path")
            new_content = body.get("new_content", "")
            
            fp = _ensure_in_root(path)
            
            if not fp.exists() or not fp.is_file():
                return JSONResponse(status_code=404, content={"error": "File not found"})
            
            if not _is_allowed(fp):
                return JSONResponse(
                    status_code=403, 
                    content={"error": "Access denied by policy"}
                )
            
            old = fp.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            new = io.StringIO(new_content).read().splitlines(keepends=True)
            
            udiff = difflib.unified_diff(
                old, new, 
                fromfile=str(fp.relative_to(ROOT)), 
                tofile=str(fp.relative_to(ROOT)), 
                lineterm=""
            )
            patch = "".join(udiff)
            
            return {"path": str(fp.relative_to(ROOT)), "diff": patch}
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

    @router.post("/fs/write")
    async def write_file(body: dict):
        """
        Write file (only if allow_write=true).
        
        Request body:
            - path: Relative path to file
            - new_content: Content to write
            
        Returns:
            Written file info (path, size, modified timestamp)
        """
        try:
            if not allow_write:
                return JSONResponse(
                    status_code=403, 
                    content={"error": "Writes disabled (allow_write=false)"}
                )
            
            path = body.get("path")
            new_content = body.get("new_content", "")
            
            fp = _ensure_in_root(path)
            
            if not _is_allowed(fp):
                return JSONResponse(
                    status_code=403, 
                    content={"error": "Access denied by policy"}
                )
            
            if fp.exists() and fp.is_dir():
                return JSONResponse(
                    status_code=400, 
                    content={"error": "Path is a directory"}
                )
            
            # Create parent directories if needed
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(new_content, encoding="utf-8")
            
            st = fp.stat()
            return {
                "path": str(fp.relative_to(ROOT)), 
                "size": st.st_size, 
                "modified": int(st.st_mtime)
            }
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

    return router
