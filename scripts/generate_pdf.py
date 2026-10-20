import os
import sys
import subprocess
import re
from pathlib import Path

# Paths
EDGE_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS_STYLE = """
@page {
    size: letter;
    margin: 11mm 15mm 11mm 15mm;
    @bottom-right {
        content: counter(page);
    }
}

*, *:before, *:after {
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 9.3pt;
    line-height: 1.42;
    color: #24292f;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
}

h1 {
    font-size: 19pt;
    font-weight: 700;
    color: #1f2328;
    border-bottom: 2px solid #0969da;
    padding-bottom: 5px;
    margin-top: 0;
    margin-bottom: 10px;
}

h2 {
    font-size: 13pt;
    font-weight: 600;
    color: #0969da;
    border-bottom: 1px solid #d8dee4;
    padding-bottom: 3px;
    margin-top: 11px;
    margin-bottom: 5px;
    page-break-after: avoid;
}

h3 {
    font-size: 10.5pt;
    font-weight: 600;
    color: #1f2328;
    margin-top: 9px;
    margin-bottom: 4px;
    page-break-after: avoid;
}

h4 {
    font-size: 9.5pt;
    font-weight: 600;
    color: #24292f;
    margin-top: 7px;
    margin-bottom: 3px;
    page-break-after: avoid;
}

p {
    margin-top: 0;
    margin-bottom: 5px;
}

ul, ol {
    margin-top: 0;
    margin-bottom: 5px;
    padding-left: 20px;
}

li {
    margin-bottom: 2px;
}

hr {
    height: 1px;
    background-color: #d8dee4;
    border: none;
    margin: 10px 0;
}

blockquote {
    margin: 5px 0 8px 0;
    padding: 5px 12px;
    color: #57606a;
    background-color: #f6f8fa;
    border-left: 3.5px solid #0969da;
    font-size: 9pt;
}

blockquote p {
    margin-bottom: 3px;
}
blockquote p:last-child {
    margin-bottom: 0;
}

.callout {
    margin: 8px 0 10px 0;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 9pt;
    page-break-inside: avoid;
}
.callout-note {
    background-color: #ddf4ff;
    border-left: 4px solid #0969da;
    color: #0969da;
}
.callout-warning {
    background-color: #fff8c5;
    border-left: 4px solid #bf8700;
    color: #9a6700;
}
.callout-important {
    background-color: #fbefff;
    border-left: 4px solid #8250df;
    color: #6639ba;
}
.callout-content {
    color: #24292f;
    margin-top: 4px;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin-top: 8px;
    margin-bottom: 10px;
    font-size: 8.5pt;
    page-break-inside: avoid;
}

table th, table td {
    border: 1px solid #d0d7de;
    padding: 4px 7px;
    text-align: left;
}

table th {
    background-color: #f6f8fa;
    font-weight: 600;
    color: #1f2328;
}

table tr:nth-child(even) {
    background-color: #fcfcfc;
}

code {
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
    font-size: 8.5pt;
    background-color: #eff1f3;
    padding: 1.5px 4px;
    border-radius: 3px;
    color: #cf222e;
}

pre {
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
    font-size: 7.8pt;
    line-height: 1.35;
    background-color: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 4px;
    padding: 8px 10px;
    margin-top: 6px;
    margin-bottom: 10px;
    overflow-x: auto;
    page-break-inside: avoid;
}

pre code {
    background-color: transparent;
    padding: 0;
    color: #24292f;
    font-size: inherit;
}

a {
    color: #0969da;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}

.diagram-box {
    page-break-inside: avoid;
}
"""

def convert_md_to_html(md_path: Path, html_path: Path):
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    
    # Process callouts: > [!NOTE], > [!WARNING], > [!IMPORTANT]
    # Replace blockquotes of that style
    def replace_callout(match):
        ctype = match.group(1).lower()
        content = match.group(2).strip()
        label = match.group(1).upper()
        return f'<div class="callout callout-{ctype}"><strong>{label}:</strong><div class="callout-content">{content}</div></div>'

    # Run marked to get HTML body
    temp_md = md_path.with_suffix(".tmp.md")
    with open(temp_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    cmd = ["npx", "--yes", "marked", "-i", str(temp_md)]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", shell=True)
    if temp_md.exists():
        temp_md.unlink()
        
    raw_html = res.stdout
    
    # Post-process callouts in HTML
    def replace_callout(match):
        ctype = match.group(1).lower()
        inner = match.group(2).strip()
        label = match.group(1).upper()
        if inner.startswith("</p>"):
            inner = inner[4:].strip()
        elif inner.startswith("<br>"):
            inner = inner[4:].strip()
        if inner.endswith("</p>"):
            inner = inner[:-4].strip()
        return f'<div class="callout callout-{ctype}"><strong>{label}:</strong><div class="callout-content">{inner}</div></div>'

    callout_pattern = re.compile(
        r'<blockquote>\s*<p>\s*\[!(NOTE|WARNING|IMPORTANT|TIP|CAUTION)\]\s*(?:<br\s*/?>)?\s*(.*?)</blockquote>',
        re.DOTALL | re.IGNORECASE
    )
    raw_html = callout_pattern.sub(replace_callout, raw_html)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{md_path.stem}</title>
<style>
{CSS_STYLE}
</style>
</head>
<body>
{raw_html}
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Generated HTML: {html_path}")

def convert_html_to_pdf(html_path: Path, pdf_path: Path):
    cmd = [
        EDGE_EXE,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={str(pdf_path.resolve())}",
        str(html_path.resolve())
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Edge headless failed: {res.stderr}")
    print(f"Generated PDF: {pdf_path}")
    
    # Also copy to Downloads if available
    downloads_dir = Path(r"C:\Users\Omkar\Downloads")
    if downloads_dir.exists():
        target_dl = downloads_dir / pdf_path.name
        import shutil
        shutil.copy2(pdf_path, target_dl)
        print(f"Copied PDF to Downloads: {target_dl}")

def inspect_pdf(pdf_path: Path, render_dir: Path = None):
    import pymupdf
    doc = pymupdf.open(str(pdf_path))
    page_count = len(doc)
    print(f"PDF Page Count: {page_count} pages ({pdf_path.name})")
    
    if render_dir:
        render_dir.mkdir(parents=True, exist_ok=True)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=150)
            png_file = render_dir / f"{pdf_path.stem}_page_{i+1}.png"
            pix.save(str(png_file))
            print(f"Rendered: {png_file}")
    doc.close()
    return page_count

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_pdf.py <markdown_file> [render_png]")
        sys.exit(1)
    
    md_file = Path(sys.argv[1])
    html_file = md_file.with_suffix(".html")
    pdf_file = md_file.with_suffix(".pdf")
    
    render_png = len(sys.argv) > 2 and sys.argv[2] == "render"
    
    convert_md_to_html(md_file, html_file)
    convert_html_to_pdf(html_file, pdf_file)
    
    render_dir = Path("rendered_pages") if render_png else None
    count = inspect_pdf(pdf_file, render_dir)
