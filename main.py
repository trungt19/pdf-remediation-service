#!/usr/bin/env python3
"""
PDF Remediation Microservice
FastAPI service for automated PDF accessibility remediation.

Endpoints:
    POST /remediate - Upload and remediate a PDF
    POST /analyze - Analyze PDF without modifying
    GET /health - Health check

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import io
import base64
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pikepdf
import httpx

# Ollama configuration - can be overridden by environment variable
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

app = FastAPI(
    title="PDF Remediation Service",
    description="Automated PDF accessibility remediation API",
    version="1.0.0"
)

# CORS for WordPress plugin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check for optional dependencies
OCRMYPDF_AVAILABLE = False
OLLAMA_AVAILABLE = False

try:
    result = subprocess.run(["ocrmypdf", "--version"], capture_output=True)
    OCRMYPDF_AVAILABLE = result.returncode == 0
except FileNotFoundError:
    pass

try:
    import httpx
    # Check if Ollama is running
    response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
    OLLAMA_AVAILABLE = response.status_code == 200
except:
    pass


class PDFRemediator:
    """Core PDF remediation logic."""

    @staticmethod
    def analyze_pdf(file_path: str) -> Dict[str, Any]:
        """Analyze a PDF for accessibility issues."""
        result = {
            "has_title": False,
            "has_language": False,
            "has_tags": False,
            "is_encrypted": False,
            "has_forms": False,
            "page_count": 0,
            "pdf_version": "",
            "issues": [],
            "score": 100
        }

        try:
            with pikepdf.open(file_path) as pdf:
                result["page_count"] = len(pdf.pages)
                result["pdf_version"] = str(pdf.pdf_version)

                # Check for title
                if pdf.docinfo and "/Title" in pdf.docinfo:
                    title = str(pdf.docinfo.get("/Title", ""))
                    result["has_title"] = bool(title.strip())

                # Check XMP metadata for title
                with pdf.open_metadata() as meta:
                    xmp_title = meta.get("{http://purl.org/dc/elements/1.1/}title", "")
                    if xmp_title:
                        result["has_title"] = True

                # Check for language
                root = pdf.Root
                if "/Lang" in root:
                    result["has_language"] = bool(str(root.get("/Lang", "")))

                # Check for tags (MarkInfo)
                if "/MarkInfo" in root:
                    mark_info = root.get("/MarkInfo", {})
                    result["has_tags"] = mark_info.get("/Marked", False)

                # Check for encryption
                result["is_encrypted"] = pdf.is_encrypted

                # Check for forms
                if "/AcroForm" in root:
                    result["has_forms"] = True

                # Calculate score and issues
                if not result["has_tags"]:
                    result["issues"].append({
                        "type": "critical",
                        "rule": "pdf-tagged",
                        "message": "PDF is not tagged (no document structure)"
                    })
                    result["score"] -= 30

                if not result["has_language"]:
                    result["issues"].append({
                        "type": "serious",
                        "rule": "pdf-lang",
                        "message": "PDF does not specify document language"
                    })
                    result["score"] -= 10

                if not result["has_title"]:
                    result["issues"].append({
                        "type": "moderate",
                        "rule": "pdf-title",
                        "message": "PDF does not have a document title set"
                    })
                    result["score"] -= 5

                if result["is_encrypted"]:
                    result["issues"].append({
                        "type": "warning",
                        "rule": "pdf-encrypted",
                        "message": "PDF is encrypted which may affect assistive technology"
                    })
                    result["score"] -= 5

                result["score"] = max(0, result["score"])

        except Exception as e:
            result["issues"].append({
                "type": "error",
                "rule": "pdf-readable",
                "message": f"Could not read PDF: {str(e)}"
            })
            result["score"] = 0

        return result

    @staticmethod
    def fix_metadata(
        input_path: str,
        output_path: str,
        title: Optional[str] = None,
        language: str = "en-US"
    ) -> Dict[str, Any]:
        """Fix PDF metadata (title, language)."""
        fixes = []

        try:
            with pikepdf.open(input_path) as pdf:
                # Derive title from filename if not provided
                if not title:
                    title = Path(input_path).stem
                    title = title.replace("_", " ").replace("-", " ")
                    title = " ".join(word.capitalize() for word in title.split())

                # Set title in docinfo
                if pdf.docinfo is None:
                    pdf.docinfo = pikepdf.Dictionary()

                if "/Title" not in pdf.docinfo or not str(pdf.docinfo.get("/Title", "")):
                    pdf.docinfo["/Title"] = title
                    fixes.append(f"Set title: {title}")

                # Set title in XMP metadata
                with pdf.open_metadata() as meta:
                    if not meta.get("{http://purl.org/dc/elements/1.1/}title"):
                        meta["{http://purl.org/dc/elements/1.1/}title"] = title

                # Set language
                root = pdf.Root
                if "/Lang" not in root or not str(root.get("/Lang", "")):
                    root["/Lang"] = language
                    fixes.append(f"Set language: {language}")

                pdf.save(output_path)

            return {"success": True, "fixes": fixes}

        except Exception as e:
            return {"success": False, "error": str(e), "fixes": fixes}

    @staticmethod
    def apply_ocr(input_path: str, output_path: str) -> Dict[str, Any]:
        """Apply OCR to a scanned PDF."""
        if not OCRMYPDF_AVAILABLE:
            return {"success": False, "error": "ocrmypdf not installed"}

        try:
            cmd = [
                "ocrmypdf",
                "--skip-text",
                "--optimize", "1",
                "--deskew",
                "-l", "eng",
                input_path,
                output_path
            ]

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if proc.returncode == 0:
                return {"success": True, "message": "OCR applied successfully"}
            elif proc.returncode == 6:
                return {"success": True, "message": "PDF already has text layer"}
            else:
                return {"success": False, "error": proc.stderr}

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "OCR timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "capabilities": {
            "metadata_fix": True,
            "ocr": OCRMYPDF_AVAILABLE,
            "ai_alt_text": OLLAMA_AVAILABLE
        }
    }


@app.post("/analyze")
async def analyze_pdf(file: UploadFile = File(...)):
    """
    Analyze a PDF for accessibility issues without modifying it.

    Returns accessibility score and list of issues.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        analysis = PDFRemediator.analyze_pdf(tmp_path)
        analysis["filename"] = file.filename
        analysis["file_size"] = len(content)
        return JSONResponse(content=analysis)
    finally:
        os.unlink(tmp_path)


@app.post("/remediate")
async def remediate_pdf(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    language: str = Form("en-US"),
    apply_ocr: bool = Form(False),
    return_file: bool = Form(True)
):
    """
    Remediate a PDF for accessibility.

    Fixes:
    - Document title (from filename or provided)
    - Document language
    - OCR for scanned documents (if apply_ocr=True)

    Args:
        file: PDF file to remediate
        title: Document title (optional, derived from filename if not provided)
        language: Language code (default: en-US)
        apply_ocr: Whether to apply OCR (default: False)
        return_file: Return remediated file (True) or just results (False)

    Returns:
        Remediated PDF file or JSON with results
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_in:
        content = await file.read()
        tmp_in.write(content)
        input_path = tmp_in.name

    # Create output path
    output_path = input_path + ".remediated.pdf"

    try:
        result = {
            "filename": file.filename,
            "original_size": len(content),
            "fixes_applied": [],
            "errors": [],
            "before_score": 0,
            "after_score": 0
        }

        # Analyze before
        before = PDFRemediator.analyze_pdf(input_path)
        result["before_score"] = before["score"]
        result["before_analysis"] = before

        # Step 1: Fix metadata
        meta_result = PDFRemediator.fix_metadata(
            input_path,
            output_path,
            title=title,
            language=language
        )

        if meta_result["success"]:
            result["fixes_applied"].extend(meta_result["fixes"])
            current_path = output_path
        else:
            result["errors"].append(meta_result.get("error", "Metadata fix failed"))
            current_path = input_path

        # Step 2: Apply OCR if requested
        if apply_ocr and OCRMYPDF_AVAILABLE:
            ocr_output = current_path + ".ocr.pdf"
            ocr_result = PDFRemediator.apply_ocr(current_path, ocr_output)

            if ocr_result["success"]:
                result["fixes_applied"].append(ocr_result.get("message", "OCR applied"))
                if os.path.exists(ocr_output):
                    current_path = ocr_output
            else:
                result["errors"].append(ocr_result.get("error", "OCR failed"))
        elif apply_ocr and not OCRMYPDF_AVAILABLE:
            result["errors"].append("OCR requested but ocrmypdf not available")

        # Analyze after
        after = PDFRemediator.analyze_pdf(current_path)
        result["after_score"] = after["score"]
        result["after_analysis"] = after
        result["score_improvement"] = after["score"] - before["score"]

        if return_file and os.path.exists(current_path):
            # Return the remediated file
            with open(current_path, "rb") as f:
                pdf_content = f.read()

            result["remediated_size"] = len(pdf_content)

            # Clean up temp files
            for path in [input_path, output_path, output_path + ".ocr.pdf"]:
                if os.path.exists(path):
                    os.unlink(path)

            # Return file with metadata in headers
            headers = {
                "X-Before-Score": str(result["before_score"]),
                "X-After-Score": str(result["after_score"]),
                "X-Fixes-Applied": ",".join(result["fixes_applied"]),
                "Content-Disposition": f'attachment; filename="remediated-{file.filename}"'
            }

            return StreamingResponse(
                io.BytesIO(pdf_content),
                media_type="application/pdf",
                headers=headers
            )
        else:
            # Clean up and return JSON result
            for path in [input_path, output_path, output_path + ".ocr.pdf"]:
                if os.path.exists(path):
                    os.unlink(path)

            return JSONResponse(content=result)

    except Exception as e:
        # Clean up on error
        for path in [input_path, output_path, output_path + ".ocr.pdf"]:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-analyze")
async def batch_analyze(files: List[UploadFile] = File(...)):
    """
    Analyze multiple PDFs at once.

    Returns analysis results for all files.
    """
    results = []

    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            results.append({
                "filename": file.filename,
                "error": "Not a PDF file"
            })
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            analysis = PDFRemediator.analyze_pdf(tmp_path)
            analysis["filename"] = file.filename
            analysis["file_size"] = len(content)
            results.append(analysis)
        finally:
            os.unlink(tmp_path)

    return JSONResponse(content={
        "total": len(results),
        "results": results,
        "summary": {
            "average_score": sum(r.get("score", 0) for r in results) / len(results) if results else 0,
            "with_issues": sum(1 for r in results if r.get("issues")),
            "tagged": sum(1 for r in results if r.get("has_tags")),
            "with_language": sum(1 for r in results if r.get("has_language")),
            "with_title": sum(1 for r in results if r.get("has_title"))
        }
    })


@app.post("/generate-alt-text")
async def generate_alt_text(
    file: UploadFile = File(...),
    ollama_url: Optional[str] = Form(None),
    model: str = Form("llava")
):
    """
    Generate alt-text for images in a PDF using Ollama LLaVA.

    Args:
        file: PDF file or image file
        ollama_url: Ollama server URL (optional, uses env var if not provided)
        model: Ollama model to use (default: llava)

    Returns:
        JSON with generated alt-text for each image
    """
    ollama_endpoint = ollama_url or OLLAMA_URL

    # Check if Ollama is available
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{ollama_endpoint}/api/tags")
            if response.status_code != 200:
                raise HTTPException(status_code=503, detail="Ollama not available")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Cannot connect to Ollama: {str(e)}")

    content = await file.read()
    filename = file.filename.lower()

    results = {
        "filename": file.filename,
        "images": [],
        "model": model,
        "ollama_url": ollama_endpoint
    }

    # Handle image files directly
    if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
        alt_text = await _generate_alt_text_for_image(content, ollama_endpoint, model)
        results["images"].append({
            "index": 0,
            "alt_text": alt_text,
            "source": "direct_image"
        })
        return JSONResponse(content=results)

    # Handle PDF files - extract images
    if not filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF or image")

    # Save PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Extract images from PDF using pikepdf
        images = await _extract_images_from_pdf(tmp_path)

        for i, img_data in enumerate(images):
            alt_text = await _generate_alt_text_for_image(img_data["bytes"], ollama_endpoint, model)
            results["images"].append({
                "index": i,
                "page": img_data.get("page", 0),
                "width": img_data.get("width", 0),
                "height": img_data.get("height", 0),
                "alt_text": alt_text
            })

        results["total_images"] = len(results["images"])
        return JSONResponse(content=results)

    finally:
        os.unlink(tmp_path)


async def _extract_images_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract images from a PDF file."""
    images = []

    try:
        with pikepdf.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                if "/Resources" not in page:
                    continue

                resources = page["/Resources"]
                if "/XObject" not in resources:
                    continue

                xobjects = resources["/XObject"]

                for name, obj in xobjects.items():
                    if obj.get("/Subtype") == "/Image":
                        try:
                            # Get image data
                            width = int(obj.get("/Width", 0))
                            height = int(obj.get("/Height", 0))

                            # Try to extract raw image data
                            raw_data = obj.read_raw_bytes()

                            # For DCTDecode (JPEG), we can use directly
                            filter_type = obj.get("/Filter")
                            if filter_type == "/DCTDecode":
                                images.append({
                                    "bytes": raw_data,
                                    "page": page_num + 1,
                                    "width": width,
                                    "height": height,
                                    "format": "jpeg"
                                })
                            # For other formats, try to convert
                            elif raw_data:
                                images.append({
                                    "bytes": raw_data,
                                    "page": page_num + 1,
                                    "width": width,
                                    "height": height,
                                    "format": "raw"
                                })
                        except Exception as e:
                            # Skip problematic images
                            continue
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract images: {str(e)}")

    return images


async def _generate_alt_text_for_image(image_bytes: bytes, ollama_url: str, model: str) -> str:
    """Generate alt-text for an image using Ollama LLaVA."""

    # Convert to base64
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    prompt = """Describe this image for a blind person. Be concise but descriptive.
Focus on:
- Main subject/content
- Important text visible
- Key colors and layout
- Any important details for understanding

Keep the description under 125 characters if possible. Do not start with "This image shows" or similar."""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False
                }
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return f"Error generating alt-text: {response.status_code}"

    except Exception as e:
        return f"Error: {str(e)}"


@app.post("/set-ollama-url")
async def set_ollama_url(url: str = Form(...)):
    """Update the Ollama URL at runtime."""
    global OLLAMA_URL
    OLLAMA_URL = url

    # Test connection
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return {
                    "success": True,
                    "url": url,
                    "models": [m.get("name") for m in models]
                }
    except Exception as e:
        return {
            "success": False,
            "url": url,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
