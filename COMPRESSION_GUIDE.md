# CZI Converter Compression Guide

## Quick Reference

| Use Case | Command | File Size | Quality | Color Accuracy |
|----------|---------|-----------|---------|----------------|
| **Minimal storage** | `--quality-preference smallest` | **Smallest** | Acceptable | ~9 pixel diff |
| **Web/display** | `--quality-preference balanced --compression jp2k` | **Medium** | Very Good | ~3 pixel diff |
| **Research/medical** | `--quality-preference best` | **Largest** | Perfect | 0 pixel diff |

## Detailed Options

### 1. Smallest File Size (Minimal Storage)
```bash
python bdsa_czi_converter.py \
  --input "sample.czi" \
  --output "output.tiff" \
  --quality-preference smallest
```
- **Compression**: JPEG2000 lossy
- **File Size**: ~30% of deflate size
- **Quality**: Acceptable for most applications
- **Color Accuracy**: Max difference ~9 pixels
- **Best for**: Storage-constrained environments, web previews, quick analysis

### 2. Balanced Quality (Recommended Default)
```bash
python bdsa_czi_converter.py \
  --input "sample.czi" \
  --output "output.tiff" \
  --quality-preference balanced \
  --compression jp2k
```
- **Compression**: JPEG2000 lossless
- **File Size**: ~65% of deflate size
- **Quality**: Very Good
- **Color Accuracy**: Max difference ~3 pixels
- **Best for**: General use, good balance of quality and file size

### 3. Perfect Quality (Research/Medical)
```bash
python bdsa_czi_converter.py \
  --input "sample.czi" \
  --output "output.tiff" \
  --quality-preference best
```
- **Compression**: Deflate
- **File Size**: Largest (baseline)
- **Quality**: Perfect
- **Color Accuracy**: 0 pixel difference
- **Best for**: Medical imaging, research requiring exact color fidelity

## File Size Comparison

Based on test results with `/wsi_archive/DUGGER_LAB/Batch1/1-960-MFG_4G8.czi`:

| Option | File Size | Relative Size | Quality | Use Case |
|--------|-----------|---------------|---------|----------|
| JPEG2000 lossy | 246M | **30%** | Acceptable | Minimal storage |
| JPEG2000 lossless | 542M | **65%** | Very Good | Balanced |
| Deflate | 837M | **100%** | Perfect | Research/medical |

## Processing Time Comparison

| Compression | Compression Speed | Decompression Speed | Memory Usage |
|-------------|-------------------|---------------------|--------------|
| JPEG2000 lossy | Fast | Slow | Low |
| JPEG2000 lossless | Medium | Slow | Low |
| Deflate | Slow | Fast | Medium |

## Recommendations by Use Case

### Medical/Research Applications
```bash
# For diagnostic imaging or research requiring exact color fidelity
python bdsa_czi_converter.py \
  --input "medical_sample.czi" \
  --output "output.tiff" \
  --quality-preference best
```

### Web/Display Applications
```bash
# For web viewers or display systems
python bdsa_czi_converter.py \
  --input "web_sample.czi" \
  --output "output.tiff" \
  --quality-preference balanced \
  --compression jp2k
```

### Storage-Constrained Environments
```bash
# For systems with limited storage
python bdsa_czi_converter.py \
  --input "sample.czi" \
  --output "output.tiff" \
  --quality-preference smallest
```

### Batch Processing
```bash
# For processing large numbers of files
python bdsa_czi_converter.py \
  --input "batch_sample.czi" \
  --output "output.tiff" \
  --quality-preference balanced \
  --compression jp2k \
  --workers 8
```

## Advanced Options

### Manual Compression Control
```bash
# Force specific compression with warnings
python bdsa_czi_converter.py \
  --input "sample.czi" \
  --output "output.tiff" \
  --compression jp2k \
  --lossless false  # Force lossy JPEG2000
```

### Legacy Color Fidelity Option
```bash
# Use the original color-fidelity parameter
python bdsa_czi_converter.py \
  --input "sample.czi" \
  --output "output.tiff" \
  --compression jp2k \
  --color-fidelity perfect  # Will switch to deflate
```

## Troubleshooting

### File Too Large
- Use `--quality-preference smallest` for minimal file size
- Consider JPEG2000 lossy compression
- Check if you need the full resolution

### Color Issues
- Use `--quality-preference best` for perfect color fidelity
- Switch to deflate compression
- Verify the original CZI file colors

### Processing Too Slow
- Increase `--workers` for parallel processing
- Use JPEG2000 for faster compression
- Consider processing smaller regions with `--test-mode`

## Migration from Previous Versions

### Old Command (with color issues)
```bash
python bdsa_czi_converter.py \
  --input "sample.czi" \
  --output "output.tiff" \
  --compression jp2k
```

### New Command (with quality control)
```bash
python bdsa_czi_converter.py \
  --input "sample.czi" \
  --output "output.tiff" \
  --quality-preference balanced \
  --compression jp2k
```

## Performance Tips

1. **For large files**: Use `--quality-preference smallest` to reduce processing time
2. **For multiple files**: Use `--workers 8` or more for parallel processing
3. **For testing**: Use `--test-mode` to process a small region first
4. **For storage**: Use `--tmpdir` to specify a fast temporary directory

## Summary

The enhanced converter now provides three quality levels:

- **Smallest**: 30% file size, acceptable quality
- **Balanced**: 65% file size, very good quality  
- **Best**: 100% file size, perfect quality

Choose based on your specific requirements for file size vs quality trade-offs. 