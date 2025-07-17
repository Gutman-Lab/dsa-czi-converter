#!/usr/bin/env python3
"""
CZI Metadata Explorer - Extract metadata from CZI files using aicspylibczi
"""

import json
import argparse
from aicspylibczi import CziFile
import xml.etree.ElementTree as ET

def get_pixel_resolution(czi):
    """Extract pixel resolution (physical size per pixel) from CZI metadata"""
    try:
        meta = czi.meta
        # If meta is an Element, use it directly
        if isinstance(meta, ET.Element):
            root = meta
        else:
            root = ET.fromstring(meta)
        
        # Find Scaling/Distance for X and Y
        x_res, y_res = None, None
        for dist in root.findall(".//Scaling/Items/Distance"):
            dist_id = dist.get("Id")
            value_elem = dist.find("Value")
            if value_elem is not None and value_elem.text:
                if dist_id == "X":
                    x_res = float(value_elem.text)
                elif dist_id == "Y":
                    y_res = float(value_elem.text)
        
        return x_res, y_res
    except Exception as e:
        print(f"Error extracting pixel resolution: {e}")
        return None, None

def get_zeiss_metadata(czi):
    """Extract specific Zeiss metadata keys from CZI XML"""
    try:
        meta = czi.meta
        # If meta is an Element, use it directly
        if isinstance(meta, ET.Element):
            root = meta
        else:
            root = ET.fromstring(meta)
        
        zeiss_data = {}
        
        # Extract microscope information
        microscopes = root.findall(".//Microscopes/Microscope")
        for i, microscope in enumerate(microscopes, 1):
            prefix = f"zeiss.Information.Instrument.Microscopes.Microscope:{i}"
            zeiss_data[f"{prefix}.Id"] = microscope.get("Id", "")
            zeiss_data[f"{prefix}.Name"] = microscope.get("Name", "")
            zeiss_data[f"{prefix}.Type"] = microscope.get("Type", "")
        
        # Extract objective information
        objectives = root.findall(".//Objectives/Objective")
        for i, objective in enumerate(objectives, 1):
            prefix = f"zeiss.Information.Instrument.Objectives.Objective:{i}"
            zeiss_data[f"{prefix}.Id"] = objective.get("Id", "")
            zeiss_data[f"{prefix}.Name"] = objective.get("Name", "")
            zeiss_data[f"{prefix}.Immersion"] = objective.get("Immersion", "")
            zeiss_data[f"{prefix}.ImmersionRefractiveIndex"] = objective.get("ImmersionRefractiveIndex", "")
            zeiss_data[f"{prefix}.LensNA"] = objective.get("LensNA", "")
            zeiss_data[f"{prefix}.NominalMagnification"] = objective.get("NominalMagnification", "")
            
            # Get manufacturer model
            manufacturer = objective.find("Manufacturer")
            if manufacturer is not None:
                zeiss_data[f"{prefix}.Manufacturer.Model"] = manufacturer.get("Model", "")
        
        # Extract scaling information
        auto_scaling = root.find(".//AutoScaling")
        if auto_scaling is not None:
            scaling_prefix = "zeiss.Scaling.AutoScaling"
            for child in auto_scaling:
                zeiss_data[f"{scaling_prefix}.{child.tag}"] = child.text or ""
        
        # Extract scaling items (X, Y resolution)
        scaling_items = root.findall(".//Scaling/Items/Distance")
        for item in scaling_items:
            item_id = item.get("Id", "")
            if item_id in ["X", "Y"]:
                prefix = f"zeiss.Scaling.Items.{item_id}"
                zeiss_data[f"{prefix}.Id"] = item_id
                value_elem = item.find("Value")
                if value_elem is not None and value_elem.text:
                    zeiss_data[f"{prefix}.Value"] = value_elem.text
                unit_elem = item.find("DefaultUnitFormat")
                if unit_elem is not None and unit_elem.text:
                    zeiss_data[f"{prefix}.DefaultUnitFormat"] = unit_elem.text
        
        return zeiss_data
        
    except Exception as e:
        print(f"Error extracting Zeiss metadata: {e}")
        return {}

def get_basic_metadata(czi_path):
    """Get basic metadata from CZI file"""
    czi = CziFile(czi_path)
    
    # Basic file info
    print(f"=== Basic File Info ===")
    print(f"File: {czi_path}")
    
    # Get pixel resolution
    x_res, y_res = get_pixel_resolution(czi)
    if x_res and y_res:
        print(f"Pixel resolution: {x_res:.2e} x {y_res:.2e} microns/pixel")
    else:
        print("Pixel resolution: Not found in metadata")
    
    # Get Zeiss metadata
    zeiss_data = get_zeiss_metadata(czi)
    if zeiss_data:
        print(f"\n=== Zeiss Metadata ===")
        for key, value in zeiss_data.items():
            if value:  # Only print non-empty values
                print(f"{key}\t{value}")
    
    # Get mosaic bounding box
    bbox = czi.get_mosaic_bounding_box()
    print(f"\nMosaic size: {bbox.w} x {bbox.h}")
    print(f"Offset: ({bbox.x}, {bbox.y})")
    
    # Get dimensions
    dims = czi.get_dims_shape()
    print(f"Dimensions: {dims}")
    
    # Get all mosaic tile bounding boxes
    tiles = czi.get_all_mosaic_tile_bounding_boxes()
    print(f"Number of tiles: {len(tiles)}")
    
    if tiles:
        first_key = list(tiles.keys())[0]
        first_tile = tiles[first_key]
        print(f"First tile size: {first_tile.w} x {first_tile.h}")
        print(f"First tile position: ({first_tile.x}, {first_tile.y})")
    
    # Get pixel type
    pixel_type = czi.pixel_type
    print(f"Pixel type: {pixel_type}")
    
    # Get size info
    size_info = czi.size
    print(f"Size info: {size_info}")
    
    # Check if it's a mosaic
    is_mosaic = czi.is_mosaic()
    print(f"Is mosaic: {is_mosaic}")
    
    return czi

def get_detailed_metadata(czi):
    """Get detailed metadata including XML metadata"""
    print(f"\n=== XML Metadata Summary ===")
    
    # Get XML metadata using the 'meta' attribute
    try:
        xml_metadata = czi.meta
        if isinstance(xml_metadata, ET.Element):
            print(f"XML metadata: {xml_metadata.tag} element with {len(xml_metadata)} children")
        else:
            print(f"XML metadata: {len(str(xml_metadata))} characters")
        
        return xml_metadata
        
    except Exception as e:
        print(f"Error getting XML metadata: {e}")
        return None

def get_subblock_metadata(czi):
    """Get subblock metadata"""
    print(f"\n=== Tile Information ===")
    
    try:
        # Get metadata for first few subblocks
        tiles = czi.get_all_mosaic_tile_bounding_boxes()
        if tiles:
            print(f"Total tiles: {len(tiles)}")
            
            # Show first few tiles with readable info
            print(f"First 3 tiles:")
            for i, (key, tile) in enumerate(list(tiles.items())[:3]):
                print(f"  {i+1}: Size {tile.w}x{tile.h} at ({tile.x}, {tile.y})")
                
    except Exception as e:
        print(f"Error getting tile metadata: {e}")

def get_scene_info(czi):
    """Get scene information"""
    print(f"\n=== Scene Information ===")
    
    try:
        # Get all scene bounding boxes
        scenes = czi.get_all_mosaic_scene_bounding_boxes()
        print(f"Number of scenes: {len(scenes)}")
        
        if scenes:
            first_scene_key = list(scenes.keys())[0]
            first_scene = scenes[first_scene_key]
            print(f"First scene size: {first_scene.w} x {first_scene.h}")
            print(f"First scene position: ({first_scene.x}, {first_scene.y})")
            
    except Exception as e:
        print(f"Error getting scene info: {e}")

def save_metadata_to_json(czi_path, output_path):
    """Save all metadata to JSON file"""
    czi = CziFile(czi_path)
    
    metadata = {
        'file_path': czi_path,
        'mosaic_info': {
            'width': czi.get_mosaic_bounding_box().w,
            'height': czi.get_mosaic_bounding_box().h,
            'x_offset': czi.get_mosaic_bounding_box().x,
            'y_offset': czi.get_mosaic_bounding_box().y
        },
        'dimensions': czi.get_dims_shape(),
        'pixel_type': str(czi.pixel_type),
        'size_info': czi.size,
        'is_mosaic': czi.is_mosaic(),
        'tiles': {
            'count': len(czi.get_all_mosaic_tile_bounding_boxes()),
            'first_tile': None
        },
        'scenes': {
            'count': len(czi.get_all_mosaic_scene_bounding_boxes())
        }
    }
    
    # Add first tile info
    tiles = czi.get_all_mosaic_tile_bounding_boxes()
    if tiles:
        first_key = list(tiles.keys())[0]
        first_tile = tiles[first_key]
        metadata['tiles']['first_tile'] = {
            'width': first_tile.w,
            'height': first_tile.h,
            'x': first_tile.x,
            'y': first_tile.y,
            'key': first_key
        }
    
    # Add XML metadata
    try:
        metadata['xml_metadata'] = czi.meta
    except Exception as e:
        metadata['xml_metadata_error'] = str(e)
    
    # Add subblock metadata for first tile
    try:
        if tiles:
            first_key = list(tiles.keys())[0]
            metadata['first_subblock_metadata'] = czi.read_subblock_metadata(first_key)
    except Exception as e:
        metadata['subblock_metadata_error'] = str(e)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract metadata from CZI files")
    parser.add_argument('czi_file', help='Path to CZI file')
    parser.add_argument('--json', help='Save metadata to JSON file')
    parser.add_argument('--xml-only', action='store_true', help='Show only XML metadata')
    
    args = parser.parse_args()
    
    if args.xml_only:
        # Just show XML metadata
        czi = CziFile(args.czi_file)
        try:
            meta = czi.meta
            if isinstance(meta, ET.Element):
                xml_str = ET.tostring(meta, encoding='unicode')
                print(xml_str)
            else:
                print(meta)
        except Exception as e:
            print(f"Error: {e}")
        return
    
    # Get basic metadata
    czi = get_basic_metadata(args.czi_file)
    
    # Get scene info
    get_scene_info(czi)
    
    # Get subblock metadata
    get_subblock_metadata(czi)
    
    # Get detailed metadata
    xml_metadata = get_detailed_metadata(czi)
    
    # Save to JSON if requested
    if args.json:
        save_metadata_to_json(args.czi_file, args.json)

    # Dump the full XML metadata only if requested
    if args.xml_only:
        print("\n=== Full CZI XML Metadata (meta tag) ===")
        try:
            czi = CziFile(args.czi_file)
            meta = czi.meta
            import xml.etree.ElementTree as ET
            if isinstance(meta, ET.Element):
                xml_str = ET.tostring(meta, encoding='unicode')
                print(xml_str)
            else:
                print(meta)
        except Exception as e:
            print(f"Error dumping czi.meta: {e}")
    else:
        print("\n=== Summary ===")
        print("Use --xml-only to see the full XML metadata")
        print("Use --json <filename> to save all metadata to JSON file")

if __name__ == "__main__":
    main() 