import argparse

import time
import numpy as np
from aicspylibczi import CziFile
import pyvips
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
import tempfile
import zarr
import shutil
import json
from datetime import datetime

def verify_lossless_conversion(orig_path, jp2k_path):
    import numpy as np

    print(f"Verifying round-trip between:\n  Original: {orig_path}\n  JP2K:     {jp2k_path}")

    # Use pyvips to read both images
    orig_img = pyvips.Image.new_from_file(orig_path)
    recon_img = pyvips.Image.new_from_file(jp2k_path)

    # Convert to numpy arrays
    orig = np.ndarray(
        buffer=orig_img.write_to_memory(),
        dtype=np.uint8,
        shape=[orig_img.height, orig_img.width, orig_img.bands],
    )
    recon = np.ndarray(
        buffer=recon_img.write_to_memory(),
        dtype=np.uint8,
        shape=[recon_img.height, recon_img.width, recon_img.bands],
    )

    if orig.shape != recon.shape:
        print(f"Shape mismatch: {orig.shape} vs {recon.shape}")
        return False

    diff = np.abs(orig.astype(int) - recon.astype(int))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff:.4f}")

    if max_diff == 0:
        print("✅ Images are pixel-perfect identical (lossless)")
        return True
    else:
        print("⚠️ Images differ (not truly lossless)")
        print(f"   Max pixel difference: {max_diff}")
        print(f"   Mean pixel difference: {mean_diff:.4f}")
        return False



# Configure libvips for parallel processing
pyvips.cache_set_max(0)  # Disable operation cache
# pyvips.concurrency_set(os.cpu_count())  # Use all available CPU cores
pyvips.leak_set(True)  # Enable memory leak detection

def get_native_tile_size(czi):
    tiles = czi.get_all_mosaic_tile_bounding_boxes()
    if hasattr(tiles, 'keys') and len(tiles) > 0:
        first_key = list(tiles.keys())[0]
        first_tile = tiles[first_key]
        return (first_tile.w, first_tile.h)
    else:
        return None

def numpy_to_vips(np_array):
    if np_array.ndim == 4 and np_array.shape[0] == 1 and np_array.shape[3] == 3:
        np_array = np_array[0]
    elif np_array.ndim == 3 and np_array.shape[2] == 3:
        pass
    else:
        raise ValueError(f"Expected RGB image (3 channels), got shape {np_array.shape}")
    
    # Convert BGR to RGB (CZI files are typically BGR24)
    # Swap the first and last channels (BGR -> RGB)
    np_array = np_array[..., ::-1]
    
    height, width, bands = np_array.shape
    if not np_array.flags['C_CONTIGUOUS']:
        np_array = np.ascontiguousarray(np_array)
    return pyvips.Image.new_from_memory(
        np_array.tobytes(),
        width,
        height,
        bands,
        pyvips.BandFormat.UCHAR if np_array.dtype == np.uint8 else pyvips.BandFormat.SHORT
    )

def convert_tiff_to_png(tiff_path, png_path, max_size=2000):
    image = pyvips.Image.new_from_file(tiff_path)
    if image.width > max_size or image.height > max_size:
        scale = max_size / max(image.width, image.height)
        image = image.resize(scale)
    image.pngsave(png_path, compression=9)
    print(f"Created preview PNG at: {png_path}")

def check_tmpdir_space(tmpdir, n_tiles, tile_size, bands, dtype=np.uint8):
    # Estimate size per tile (bytes)
    dtype_size = np.dtype(dtype).itemsize
    tile_bytes = tile_size[0] * tile_size[1] * bands * dtype_size
    total_bytes = tile_bytes * n_tiles
    # Require at least 2x this space for safety
    min_bytes = total_bytes * 2
    stat = shutil.disk_usage(tmpdir)
    print(f"Temp dir: {tmpdir}, Free: {stat.free/1e9:.2f} GB, Needed: {min_bytes/1e9:.2f} GB")
    if stat.free < min_bytes:
        raise RuntimeError(f"Not enough space in temp dir {tmpdir}: {stat.free/1e9:.2f} GB free, need at least {min_bytes/1e9:.2f} GB.")

def process_tile(args):
    czi_path, region, tile_idx, total_tiles, verbose, temp_dir = args
    worker_id = os.getpid()
    if verbose:
        print(f"Worker {worker_id} processing tile {tile_idx+1}/{total_tiles}")
    czi = CziFile(czi_path)
    read_start = time.time()
    tile = czi.read_mosaic(region=region, C=0)
    read_end = time.time()
    h, w = region[3], region[2]
    tile = tile[:h, :w]
    vips_tile = numpy_to_vips(tile)
    temp_filename = os.path.join(temp_dir, f"tile_{worker_id}_{tile_idx}.tiff")
    vips_tile.tiffsave(temp_filename, compression="none")
    if verbose:
        print(f"Worker {worker_id} saved tile {tile_idx+1}/{total_tiles} to {temp_filename}")
    return (region, temp_filename, read_end - read_start)

def process_tile_batch(args):
    czi_path, regions, batch_idx, verbose, temp_dir = args
    worker_id = os.getpid()
    if verbose:
        print(f"Worker {worker_id} starting batch {batch_idx} with {len(regions)} tiles")
    czi = CziFile(czi_path)
    results = []
    for i, region in enumerate(regions):
        if verbose:
            print(f"Worker {worker_id} processing tile {i+1}/{len(regions)} in batch {batch_idx}")
        read_start = time.time()
        tile = czi.read_mosaic(region=region, C=0)
        read_end = time.time()
        h, w = region[3], region[2]
        tile = tile[:h, :w]
        vips_tile = numpy_to_vips(tile)
        # Use the file-specific temp directory
        temp_filename = os.path.join(temp_dir, f"tile_{worker_id}_{batch_idx}_{i}.tiff")
        vips_tile.tiffsave(temp_filename, compression="none")
        if verbose:
            print(f"Worker {worker_id} saved tile {i+1}/{len(regions)} in batch {batch_idx} to {temp_filename}")
        results.append((region, temp_filename, read_end - read_start))
    if verbose:
        print(f"Worker {worker_id} completed batch {batch_idx}")
    return results

def write_zarr_pyramid(np_img, zarr_output, chunk_size):
    try:
        from skimage.transform import resize
    except ImportError:
        raise ImportError("scikit-image is required for Zarr pyramid generation. Please install with 'pip install scikit-image'.")
    level = 0
    shape = np_img.shape
    current = np_img
    group = zarr.open_group(zarr_output, mode='w')
    print(f"Writing Zarr pyramid to: {zarr_output}")
    while True:
        print(f"  Writing level {level} with shape {current.shape}")
        z = group.create_dataset(str(level), data=current, chunks=(chunk_size[1], chunk_size[0], current.shape[2]), dtype=current.dtype, overwrite=True)
        # Downsample by 2 for next level
        new_h = max(1, current.shape[0] // 2)
        new_w = max(1, current.shape[1] // 2)
        if new_h < 256 or new_w < 256:
            break
        # skimage resize expects (h, w, c), anti_aliasing for quality
        current = resize(current, (new_h, new_w, current.shape[2]), order=1, preserve_range=True, anti_aliasing=True).astype(np_img.dtype)
        level += 1
    print(f"Successfully wrote Zarr pyramid at: {zarr_output}")

def create_tiled_tiff_vips_parallel(input_czi_path, output_path, tile_size=(1024, 1024), test_mode=False, workers=2, verbose=False, compression="deflate", output_tile_size=(256, 256), zarr_output=None, tmpdir=None, args=None):
    import pyvips
    start_time = time.time()
    print(f"Main process ID: {os.getpid()}")
    
    # Create a unique subdirectory for this input file
    input_basename = os.path.splitext(os.path.basename(input_czi_path))[0]
    if tmpdir:
        file_tmpdir = os.path.join(tmpdir, f"czi_{input_basename}")
        print(f"Creating temporary directory for {input_basename} in {tmpdir}")
        os.makedirs(file_tmpdir, exist_ok=True)
        print(f"Created temporary directory: {file_tmpdir}")
    else:
        print(f"No tmpdir provided, creating temporary directory for {input_basename}")
        file_tmpdir = tempfile.mkdtemp(prefix=f"czi_{input_basename}_")
        print(f"Created temporary directory: {file_tmpdir}")
    
    # Initialize timing dictionary
    timing_info = {
        "start_time": datetime.now().isoformat(),
        "parameters": {
            "input_file": input_czi_path,
            "output_file": output_path,
            "tile_size": tile_size,
            "test_mode": test_mode,
            "workers": workers,
            "compression": compression,
            "output_tile_size": output_tile_size,
            "zarr_output": zarr_output,
            "tmpdir": file_tmpdir
        },
        "timing": {
            "total_tile_read_save": 0,
            "tiff_write": 0,
            "image_reassembly": 0,
            "cleanup": 0,
            "total_time": 0
        },
        "file_info": {}
    }

    czi = CziFile(input_czi_path)
    bbox = czi.get_mosaic_bounding_box()
    x_offset, y_offset = bbox.x, bbox.y
    width, height = bbox.w, bbox.h
    print(f"Mosaic size: {width} x {height}, offset: ({x_offset}, {y_offset})")

    # Add file info to timing dictionary
    timing_info["file_info"] = {
        "mosaic_size": {"width": width, "height": height},
        "offset": {"x": x_offset, "y": y_offset}
    }

    if test_mode:
        print("Test mode: Processing a 10Kx10K region from the center")
        center_x = x_offset + width // 2
        center_y = y_offset + height // 2
        test_size = 10000  # 10Kx10K region
        test_x = center_x - test_size // 2
        test_y = center_y - test_size // 2
        x_offset = test_x
        y_offset = test_y
        width = test_size
        height = test_size
        print(f"Test region: starting at ({x_offset}, {y_offset}), size {width}x{height}")

    n_tiles_x = (width + tile_size[0] - 1) // tile_size[0]
    n_tiles_y = (height + tile_size[1] - 1) // tile_size[1]
    total_tiles = n_tiles_x * n_tiles_y
    print(f"Processing {n_tiles_x} x {n_tiles_y} tiles ({total_tiles} total) with {workers} workers...")

    # Add tile info to timing dictionary
    timing_info["file_info"]["tiles"] = {
        "n_tiles_x": n_tiles_x,
        "n_tiles_y": n_tiles_y,
        "total_tiles": total_tiles
    }

    regions = []
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            x0 = x_offset + tx * tile_size[0]
            y0 = y_offset + ty * tile_size[1]
            w = min(tile_size[0], width - tx * tile_size[0])
            h = min(tile_size[1], height - ty * tile_size[1])
            region = (x0, y0, w, h)
            regions.append(region)

    # Use custom temp dir if provided
    if tmpdir:
        temp_dir = tmpdir
    else:
        temp_dir = file_tmpdir
    print(f"Using temp dir: {temp_dir}")
    check_tmpdir_space(temp_dir, n_tiles_x * n_tiles_y, tile_size, 3)  # assume RGB, uint8

    # Create smaller batches for more frequent updates
    batch_size = max(1, (len(regions) + (workers * 4) - 1) // (workers * 4))  # More batches for more frequent updates
    batches = [regions[i:i + batch_size] for i in range(0, len(regions), batch_size)]
    print(f"Created {len(batches)} batches of size ~{batch_size} for more frequent updates")

    # Use file_tmpdir instead of tmpdir for batch processing
    batch_args = [(input_czi_path, batch, idx, verbose, file_tmpdir) for idx, batch in enumerate(batches)]

    tile_read_start = time.time()
    vips_tiles_rows = [[None for _ in range(n_tiles_x)] for _ in range(n_tiles_y)]
    temp_files = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        if verbose:
            print("Executor created, submitting jobs...")
        futures = {executor.submit(process_tile_batch, arg): idx for idx, arg in enumerate(batch_args)}
        if verbose:
            print(f"Submitted {len(futures)} jobs to executor")
        if tqdm:
            pbar = tqdm(total=total_tiles, desc='Tiles')
        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                batch_results = future.result()
                if tqdm:
                    pbar.update(len(batch_results))
                for region, temp_filename, read_time in batch_results:
                    tx = (region[0] - x_offset) // tile_size[0]
                    ty = (region[1] - y_offset) // tile_size[1]
                    vips_tiles_rows[ty][tx] = temp_filename
                    temp_files.append(temp_filename)
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
        if tqdm:
            pbar.close()
    tile_read_end = time.time()
    timing_info["timing"]["total_tile_read_save"] = tile_read_end - tile_read_start
    print(f"Total tile read+save time: {tile_read_end - tile_read_start:.2f} sec")

    # Reassemble the image from temp files using parallel processing
    reassembly_start = time.time()
    vips_tiles = []
    for row in vips_tiles_rows:
        row_images = [pyvips.Image.new_from_file(tile, access='sequential') for tile in row if tile is not None]
        if row_images:
            vips_row = pyvips.Image.arrayjoin(row_images, across=len(row_images))
            vips_tiles.append(vips_row)
    if not vips_tiles:
        raise RuntimeError("No valid tiles were processed")
    vips_image = pyvips.Image.arrayjoin(vips_tiles, across=1)
    reassembly_end = time.time()
    timing_info["timing"]["image_reassembly"] = reassembly_end - reassembly_start
    print(f"Image reassembly time: {reassembly_end - reassembly_start:.2f} sec")

    # If zarr_output is set, write Zarr pyramid and skip TIFF
    if zarr_output:
        print(f"Writing Zarr pyramid to: {zarr_output}")
        # Convert vips_image to numpy array (RGB)
        np_img = np.ndarray(
            buffer=vips_image.write_to_memory(),
            dtype=np.uint8,
            shape=[vips_image.height, vips_image.width, vips_image.bands],
        )
        write_zarr_pyramid(np_img, zarr_output, output_tile_size)
    else:
        tiff_write_start = time.time()
        print(f"Writing TIFF with compression: {compression}, output tile size: {output_tile_size}, pyramid: True")
        
        # Add specific settings for JPEG2000 compression (only valid pyvips args)
        if compression == 'jp2k':
            vips_image.tiffsave(
                output_path,
                tile=True,
                tile_width=output_tile_size[0],
                tile_height=output_tile_size[1],
                compression='jp2k',
                lossless=args.lossless if args else True,  # Use lossless parameter or default to True
                bigtiff=True,
                pyramid=True,
                subifd=True,
            )
        else:
            vips_image.tiffsave(
                output_path,
                tile=True,
                tile_width=output_tile_size[0],
                tile_height=output_tile_size[1],
                compression=compression,
                bigtiff=True,
                pyramid=True,
                subifd=True,
            )
        tiff_write_end = time.time()
        timing_info["timing"]["tiff_write"] = tiff_write_end - tiff_write_start
        print(f"Successfully created tiled TIFF at: {output_path}")
        print(f"TIFF write time: {tiff_write_end - tiff_write_start:.2f} sec")
        # Run lossless check on first tile if using jp2k
        if compression == 'jp2k' and temp_files:
            first_tile = temp_files[0]
            temp_jp2k = first_tile + ".jp2k.tiff"
            pyvips.Image.new_from_file(first_tile).tiffsave(
                temp_jp2k,
                compression='jp2k',
                lossless=True,
                Q=100,
                tile=True,
                tile_width=output_tile_size[0],
                tile_height=output_tile_size[1],
                bigtiff=True,
            )
            verify_lossless_conversion(first_tile, temp_jp2k)

    # Clean up temp files and directory
    cleanup_start = time.time()
    print(f"Cleaning up temporary files in {file_tmpdir}")
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            if verbose:
                print(f"Removed temporary file: {temp_file}")
        except Exception as e:
            if verbose:
                print(f"Could not remove temp file {temp_file}: {e}")
    
    # Only remove the temp directory if we created it (not if user provided tmpdir)
    if not tmpdir:
        print(f"Removing temporary directory: {file_tmpdir}")
        try:
            shutil.rmtree(file_tmpdir)
            print(f"Successfully removed temporary directory: {file_tmpdir}")
        except Exception as e:
            if verbose:
                print(f"Could not remove temp directory {file_tmpdir}: {e}")
    else:
        print(f"Keeping temporary directory {file_tmpdir} as it was user-provided")
    cleanup_end = time.time()
    timing_info["timing"]["cleanup"] = cleanup_end - cleanup_start
    print(f"Cleanup time: {cleanup_end - cleanup_start:.2f} sec")

    if test_mode and not zarr_output:
        # Only create PNG if size is reasonable (e.g., <10000 px on a side)
        if width <= 10000 and height <= 10000:
            png_path = output_path.replace('.tiff', '.png')
            convert_tiff_to_png(output_path, png_path)
        else:
            print("Skipping PNG preview: image too large")
    end_time = time.time()
    timing_info["timing"]["total_time"] = end_time - start_time
    print(f"Total script time: {end_time - start_time:.2f} sec")

    # Write timing info to JSON file
    timing_file = output_path + ".timing.json"
    with open(timing_file, 'w') as f:
        json.dump(timing_info, f, indent=2)
    print(f"Timing information written to: {timing_file}")

def get_output_filename(base_output, compression, tile_size, output_tile_size, workers, default_tile_size=False, test_mode=False):
    """Generate output filename with parameter information."""
    # Get base name and extension
    base, ext = os.path.splitext(base_output)
    
    # Build parameter suffix
    params = []
    params.append(compression)
    
    # Add tile size information
    if default_tile_size:
        params.append("native_tile")
    else:
        params.append(f"tile{tile_size[0]}x{tile_size[1]}")
    
    # Add output tile size
    params.append(f"out_tile{output_tile_size[0]}x{output_tile_size[1]}")
    
    # Add number of workers
    params.append(f"w{workers}")
    
    # Add test mode if enabled
    if test_mode:
        params.append("test")
    
    # Join parameters with underscores
    param_suffix = "_".join(params)
    
    return f"{base}_{param_suffix}{ext}"

def main():
    parser = argparse.ArgumentParser(description="Parallel CZI to tiled TIFF/Zarr converter.")
    parser.add_argument('--input', type=str, default="input_czi_files/1-254-MFG_AT8.czi", help="Input CZI file")
    parser.add_argument('--output', type=str, default="output_tiffs/1-254-MFG_AT8_tiled_vips_parallel.tiff", help="Output TIFF file")
    parser.add_argument('--tile-size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), default=None, help="Tile size (width height) for reading from CZI (native tile size recommended)")
    parser.add_argument('--default-tile-size', action='store_true', help="Use native tile size from CZI file for reading")
    parser.add_argument('--output-tile-size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), default=[256, 256], help="Tile size (width height) for writing output TIFF/Zarr (default: 256 256)")
    parser.add_argument('--test-mode', action='store_true', help="Test mode: process only a small region from the center")
    parser.add_argument('--workers', type=int, default=2, help="Number of parallel workers")
    parser.add_argument('--verbose', action='store_true', help="Enable detailed logging")
    parser.add_argument('--compression', type=str, default='deflate', choices=['deflate', 'lzw', 'none', 'jp2k'], help="TIFF compression type")
    parser.add_argument('--lossless', action='store_true', help="Use lossless compression (for JPEG2000)")
    parser.add_argument('--color-fidelity', choices=['perfect', 'acceptable'], default='acceptable', help='Color fidelity requirement (perfect=deflate, acceptable=jp2k)')
    parser.add_argument('--quality-preference', choices=['smallest', 'balanced', 'best'], default='balanced', help='Quality preference: smallest=minimal file size, balanced=good balance, best=perfect quality')
    parser.add_argument('--zarr-output', type=str, default=None, help="Output Zarr store path (if set, skip TIFF)")
    parser.add_argument('--tmpdir', type=str, default=None, help="Directory for temporary tile files (default: system temp dir)")
    args = parser.parse_args()

    # Handle quality preference and compression selection
    if args.quality_preference == 'smallest':
        print("Quality preference: smallest file size")
        args.compression = 'jp2k'
        args.lossless = False
        print("Using JPEG2000 lossy compression for minimal file size")
        print("WARNING: This may introduce color variations (max intensity diff ~9/255)")
    elif args.quality_preference == 'balanced':
        print("Quality preference: balanced (good quality, reasonable file size)")
        if args.compression == 'jp2k':
            args.lossless = True
            print("Using JPEG2000 lossless compression for good balance")
            print("WARNING: Minor color variations possible (max intensity diff ~3/255)")
        elif args.compression == 'deflate':
            print("Using deflate compression for good balance")
    elif args.quality_preference == 'best':
        print("Quality preference: best quality (perfect color fidelity)")
        args.compression = 'deflate'
        print("Using deflate compression for perfect color preservation")
    
    # Handle legacy color-fidelity parameter
    if args.color_fidelity == 'perfect' and args.compression == 'jp2k':
        print("WARNING: JPEG2000 compression may introduce minor color variations.")
        print("Switching to deflate compression for perfect color fidelity")
        args.compression = 'deflate'
    elif args.compression == 'jp2k' and args.quality_preference == 'balanced':
        print("WARNING: JPEG2000 compression may introduce minor color variations.")
        print("For applications requiring perfect color fidelity, consider using 'deflate' compression or --quality-preference best")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    czi = CziFile(args.input)
    if args.default_tile_size:
        native_tile_size = get_native_tile_size(czi)
        if native_tile_size:
            print(f"Using native tile size for reading: {native_tile_size}")
            tile_size = native_tile_size
        else:
            print("Could not determine native tile size, using default 1024x1024.")
            tile_size = (1024, 1024)
    elif args.tile_size:
        tile_size = tuple(args.tile_size)
        print(f"Using user-specified tile size for reading: {tile_size}")
    else:
        tile_size = (1024, 1024)
        print(f"Using default tile size for reading: {tile_size}")

    # Output tile size for writing
    output_tile_size = tuple(args.output_tile_size)
    print(f"Output TIFF/Zarr tile size: {output_tile_size}")

    # Generate output filename with parameter information
    output_with_params = get_output_filename(
        args.output, 
        args.compression, 
        tile_size, 
        output_tile_size,
        args.workers,
        args.default_tile_size,
        args.test_mode
    )
    print(f"Output file will be: {output_with_params}")

    create_tiled_tiff_vips_parallel(
        args.input, output_with_params, tile_size=tile_size, test_mode=args.test_mode, 
        workers=args.workers, verbose=args.verbose, compression=args.compression, 
        output_tile_size=output_tile_size, zarr_output=args.zarr_output, tmpdir=args.tmpdir, args=args
    )

if __name__ == "__main__":
    main() 
