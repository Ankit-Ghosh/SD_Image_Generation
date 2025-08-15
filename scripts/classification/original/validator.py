import os
import shutil
from PIL import Image
import pathlib
import argparse
import sys

def validate_image(image_path):
    """Check if file is a valid image in supported format (JPEG, PNG, GIF, BMP)"""
    try:
        with Image.open(image_path) as img:
            # Try to verify the image
            img.verify()
            
            # Check if it's one of the supported formats
            if img.format not in ['JPEG', 'PNG', 'GIF', 'BMP']:
                return False, f"Unsupported format: {img.format}"
            
            return True, None
    except Exception as e:
        return False, str(e)

def fix_dataset(input_dir, output_dir):
    """Validate all images and create a clean dataset"""
    input_path = pathlib.Path(input_dir)
    output_path = pathlib.Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' doesn't exist")
        return
    
    # Create output directory if it doesn't exist
    if not output_path.exists():
        output_path.mkdir(parents=True)
    
    print(f"Scanning dataset at: {input_dir}")
    print(f"Clean dataset will be created at: {output_dir}")
    
    # Stats
    total_files = 0
    valid_files = 0
    invalid_files = 0
    
    # Track problem files
    problem_files = []
    
    # Process each class directory
    for class_dir in input_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")
        
        # Create output class directory
        output_class_dir = output_path / class_name
        if not output_class_dir.exists():
            output_class_dir.mkdir(parents=True)
        
        # Process each image in the class
        class_files = 0
        class_valid = 0
        class_invalid = 0
        
        for img_path in class_dir.glob('*'):
            if not img_path.is_file():
                continue
                
            total_files += 1
            class_files += 1
            
            # Validate the image
            is_valid, error = validate_image(img_path)
            
            if is_valid:
                # Copy valid image to output directory
                shutil.copy2(img_path, output_class_dir / img_path.name)
                valid_files += 1
                class_valid += 1
            else:
                # Record invalid image
                invalid_files += 1
                class_invalid += 1
                problem_files.append((str(img_path), error))
                print(f"  Problem with {img_path.name}: {error}")
        
        print(f"  {class_name}: {class_valid}/{class_files} valid images ({class_invalid} invalid)")
    
    # Print summary
    print("\n" + "="*50)
    print(f"DATASET VALIDATION SUMMARY:")
    print(f"  Total files scanned: {total_files}")
    print(f"  Valid images: {valid_files}")
    print(f"  Invalid images: {invalid_files}")
    print("="*50)
    
    # Save problem files report
    if problem_files:
        report_path = output_path / "invalid_images_report.txt"
        with open(report_path, 'w') as f:
            f.write("INVALID IMAGES REPORT\n")
            f.write("="*50 + "\n\n")
            for path, error in problem_files:
                f.write(f"{path}\n  Error: {error}\n\n")
        print(f"\nDetailed report of invalid images saved to: {report_path}")
    
    print(f"\nClean dataset ready at: {output_dir}")
    return valid_files, invalid_files

def check_extension(file_path):
    """Check if file has a valid image extension"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    return file_path.suffix.lower() in valid_extensions

def print_extension_summary(input_dir):
    """Print summary of file extensions in dataset"""
    input_path = pathlib.Path(input_dir)
    extensions = {}
    
    print("\nFile extension analysis:")
    for class_dir in input_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        for file_path in class_dir.glob('*'):
            if not file_path.is_file():
                continue
                
            ext = file_path.suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1
    
    # Print sorted by count
    print("  Extension counts:")
    for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True):
        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            print(f"  {ext}: {count} (supported)")
        else:
            print(f"  {ext}: {count} (UNSUPPORTED)")
    
    return extensions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and fix image dataset for TensorFlow")
    parser.add_argument("input_dir", help="Input directory containing image classes")
    parser.add_argument("--output_dir", help="Output directory for clean dataset (default: input_dir + '_clean')")
    
    args = parser.parse_args()
    
    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = args.input_dir + "_clean"
    
    # Analyze file extensions first
    print_extension_summary(args.input_dir)
    
    # Confirm before proceeding
    print(f"\nThis will create a clean copy of your dataset at: {args.output_dir}")
    response = input("Continue? (y/n): ")
    
    if response.lower() != 'y':
        print("Operation cancelled.")
        sys.exit(0)
    
    # Process the dataset
    fix_dataset(args.input_dir, args.output_dir)