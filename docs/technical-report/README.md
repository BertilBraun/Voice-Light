# Technical report source

The report is built from `voice-light-technical-report.tex`. Its figures use the repository assets
under `docs/assets`, including the light-theme interaction diagnostics captured from the public
browser client.

From the repository root, build the PDF with Tectonic:

```powershell
New-Item -ItemType Directory -Force output\pdf | Out-Null
tectonic --outdir output\pdf docs\technical-report\voice-light-technical-report.tex
```

The resulting file is `output/pdf/voice-light-technical-report.pdf`. The published release PDF is
built from the commit tagged for that release and visually inspected page by page before upload.
