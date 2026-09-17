# Technical report source

The report is built from `voice-light-technical-report.tex`. Its figures use the repository assets
under `docs/assets`, including an observable speech-and-playback trace captured from the public
browser client. Raw adapter heads remain available in the browser diagnostics but are not presented
as calibrated behavioral probabilities in the report.

From the repository root, build the PDF with Tectonic:

```powershell
New-Item -ItemType Directory -Force output\pdf | Out-Null
tectonic --outdir output\pdf docs\technical-report\voice-light-technical-report.tex
```

The resulting file is `output/pdf/voice-light-technical-report.pdf`. The published release PDF is
built from the commit tagged for that release and visually inspected page by page before upload.
