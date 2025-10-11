# Troubleshooting Guide

Common issues and solutions for Book Chunker.

## Installation Issues

### `ModuleNotFoundError: No module named 'fitz'`

**Problem:** PyMuPDF not installed correctly.

**Solution:**
```bash
pip install PyMuPDF
# or
pip install -r requirements.txt
```

### `ImportError: cannot import name 'QdrantClient'`

**Problem:** Qdrant client not installed.

**Solution:**
```bash
pip install qdrant-client
```

## Extraction Issues

### No Bookmarks Found

**Symptoms:**
```
Warning: No outline found in PDF
Empty JSON output
```

**Cause:** PDF has no embedded bookmarks/TOC.

**Solutions:**

1. **Check if bookmarks exist:**
```bash
python book_extractor.py book.pdf --toc
```

2. **Use font-based extraction:**
```bash
python experimental/font_aware_processor.py book.pdf
```

3. **Manually create structure:**
Edit the JSON after basic extraction.

### Weird Paragraph Breaks

**Symptoms:**
- Paragraphs split mid-sentence
- Multiple paragraphs merged incorrectly

**Causes:**
- Unusual PDF formatting
- Complex layouts (multi-column, sidebars)
- Inconsistent spacing

**Solutions:**

1. **Adjust merging sensitivity:**
Edit `utils/paragraph_extraction.py`, line ~108:
```python
# Make less aggressive (fewer breaks)
vertical_gap > 2.0 * (bbox[3] - bbox[1])  # Was 1.5

# Make more aggressive (more breaks)
vertical_gap > 1.0 * (bbox[3] - bbox[1])  # Was 1.5
```

2. **Check PDF quality:**
Try exporting PDF from original source with better settings.

### Image Extraction Fails

**Symptoms:**
```
Warning: Failed to extract image
Page X: Successfully extracted 0 out of 3 images
```

**Causes:**
- Images are vector graphics
- Embedded in unusual format
- Corrupted PDF

**Solutions:**

1. **Check log details:**
```bash
python book_extractor.py book.pdf --extract-images --verbose
cat book_extraction.log | grep -A5 "Image"
```

2. **Try different PDF:**
Re-save or export PDF using different tool.

3. **Manual extraction:**
Use external tools like `pdfimages`:
```bash
pdfimages -all book.pdf output_dir/
```

### Extraction Hangs/Very Slow

**Symptoms:**
- Script runs for hours
- High CPU usage
- No progress output

**Causes:**
- Very large PDF
- Many images
- Complex layouts

**Solutions:**

1. **Process in chunks:**
```bash
python book_extractor.py book.pdf --pages 1-100
python book_extractor.py book.pdf --pages 101-200
# Merge JSONs manually
```

2. **Skip images:**
```bash
python book_extractor.py book.pdf
# Without --extract-images flag
```

3. **Monitor progress:**
```bash
tail -f book_extraction.log
```

## Vectorization Issues

### OpenAI API Error

**Symptoms:**
```
Error: OpenAI API request failed
AuthenticationError
RateLimitError
```

**Solutions:**

1. **Check API key:**
```bash
# Verify .env file
cat .env | grep OPENAI

# Test key directly
python -c "import openai; openai.api_key='sk-...'; print(openai.models.list())"
```

2. **Rate limiting:**
Add delays between requests in `vectorize.py`:
```python
import time
time.sleep(1)  # After each vectorize_text() call
```

3. **Check credits:**
Visit https://platform.openai.com/usage

### Qdrant Connection Failed

**Symptoms:**
```
Error: Qdrant connection test failed
Connection refused
```

**Solutions:**

1. **Check if Qdrant is running:**
```bash
docker ps | grep qdrant
# Or
curl http://localhost:6333
```

2. **Start Qdrant:**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

3. **Check environment:**
```bash
cat .env | grep QDRANT
```

4. **Test connection:**
```python
from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333")
print(client.get_collections())
```

### Text Too Long Error

**Symptoms:**
```
Warning: Text too long (45000 chars). Truncating to 30000
```

**Cause:** Section text exceeds OpenAI token limit.

**Solutions:**

1. **Use chunking:**
```bash
python vectorize.py book.json --collection books --chunk-size 3
```

2. **Adjust limit:**
Edit `vectorize.py`, line ~84:
```python
max_chars = 20000  # Reduce from 30000
```

### Memory Issues

**Symptoms:**
```
MemoryError
Killed
Process terminated
```

**Solutions:**

1. **Process fewer books at once:**
```bash
# Instead of
python vectorize.py *.json --collection books

# Do
for file in *.json; do
  python vectorize.py "$file" --collection books
done
```

2. **Reduce chunk size:**
```bash
python vectorize.py book.json --collection books --chunk-size 2
```

3. **Monitor memory:**
```bash
# Linux/Mac
top -p $(pgrep -f vectorize.py)
```

## JSON Issues

### Invalid JSON

**Symptoms:**
```
JSONDecodeError: Expecting ',' delimiter
```

**Causes:**
- Corrupted file
- Incomplete extraction
- Manual editing error

**Solutions:**

1. **Validate JSON:**
```bash
python -m json.tool book_structure.json > /dev/null
# Or
jq . book_structure.json > /dev/null
```

2. **Find error location:**
```bash
python -c "import json; json.load(open('book_structure.json'))"
```

3. **Re-extract:**
```bash
rm book_structure.json
python book_extractor.py book.pdf
```

### Large JSON Files

**Symptoms:**
- Multi-GB JSON files
- Slow loading
- Memory issues

**Causes:**
- Many embedded images
- Large book

**Solutions:**

1. **Extract without images:**
```bash
python book_extractor.py book.pdf
# Without --extract-images
```

2. **Store images separately:**
Modify extraction to save image files instead of embedding.

3. **Compress:**
```bash
gzip book_structure.json
# Decompress when needed
gunzip book_structure.json.gz
```

## Environment Issues

### .env File Not Loaded

**Symptoms:**
```
KeyError: 'OPENAI_API_KEY'
Environment variable not found
```

**Solutions:**

1. **Check file exists:**
```bash
ls -la .env
```

2. **Check file format:**
```bash
cat .env
# Should be:
# KEY=value
# Not: export KEY=value
```

3. **Manual load:**
```bash
export OPENAI_API_KEY=sk-...
python vectorize.py book.json --collection books
```

### Permission Denied

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**

1. **Check file permissions:**
```bash
ls -l book.pdf
chmod 644 book.pdf
```

2. **Check directory permissions:**
```bash
ls -ld .
chmod 755 .
```

3. **Run with proper user:**
```bash
# Don't use sudo unless necessary
python book_extractor.py book.pdf
```

## Performance Issues

### Slow Extraction

**Benchmarks:**
- Simple PDF: ~2-5 pages/second
- With images: ~0.5-2 pages/second
- Complex layouts: ~0.2-1 pages/second

**If slower:**

1. **Check CPU:**
```bash
top
```

2. **Check disk I/O:**
```bash
iostat -x 1
```

3. **Profile code:**
```bash
python -m cProfile -o profile.stats book_extractor.py book.pdf
```

### Slow Vectorization

**Benchmarks:**
- ~1-5 sections/second
- Limited by OpenAI API rate

**Solutions:**

1. **Use batch processing:**
Modify code to batch embed multiple texts per API call.

2. **Parallel processing:**
```python
# Use concurrent.futures for multiple books
from concurrent.futures import ThreadPoolExecutor
```

3. **Monitor API latency:**
Add timing logs to track bottlenecks.

## Debugging Tips

### Enable Verbose Logging

```bash
python book_extractor.py book.pdf --verbose
python vectorize.py book.json --collection books --verbose
```

### Check Log Files

```bash
tail -f book_extraction.log
tail -f vectorization.log
grep ERROR *.log
```

### Inspect Intermediate Results

```bash
# Pretty print JSON
cat book_structure.json | jq . | less

# Check specific section
cat book_structure.json | jq '.outline[0]'

# Count paragraphs
cat book_structure.json | jq '[.outline[].paragraphs | length] | add'
```

### Test Individual Components

```python
# Test paragraph extraction
from utils.paragraph_extraction import extract_paragraphs_from_pages
import fitz
doc = fitz.open("book.pdf")
paras = extract_paragraphs_from_pages(None, 1, 10, doc)
print(len(paras))

# Test vectorization
from vectorize import vectorize_text
embedding, size = vectorize_text("Test text")
print(f"Vector size: {len(embedding)}")
```

## Getting Help

### Before Asking

1. Check this troubleshooting guide
2. Search existing issues
3. Review logs for error details
4. Try minimal reproduction case

### When Asking

Include:
1. Complete error message
2. Relevant log excerpts
3. PDF characteristics (pages, size, source)
4. Command used
5. Environment (OS, Python version)
6. What you've already tried

### Useful Commands

```bash
# System info
python --version
pip list | grep -E 'PyPDF|PyMuPDF|qdrant|openai'

# File info
file book.pdf
pdfinfo book.pdf

# Docker info
docker --version
docker ps
docker logs <container_id>
```

## Known Limitations

1. **Scanned PDFs**: No OCR support (yet)
2. **Tables**: Not parsed to structured data
3. **Equations**: LaTeX not preserved
4. **Fonts**: Complex font analysis optional
5. **Multi-column**: May have ordering issues
6. **Languages**: Best with English, LTR text

## Workarounds

### No Bookmarks
→ Use `experimental/font_aware_processor.py`

### Complex Tables
→ Extract manually or use dedicated table extraction tool

### Scanned PDFs
→ Pre-process with OCR tool (Tesseract, Adobe Acrobat)

### Very Large Books
→ Split PDF, process parts, merge JSONs

### Rate Limits
→ Use different OpenAI tier or local embeddings

## Still Having Issues?

1. Create GitHub issue with details
2. Join discussions for community help
3. Check documentation: `docs/`
4. Review examples: `examples/`

