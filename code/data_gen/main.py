from pathlib import Path

from datagen import SimpleStarGenerator

# Example usage
if __name__ == "__main__":
    gen = SimpleStarGenerator(size=128, seed=42)

    # Generate and save to a directory
    output_dir = Path("./star_training_data")
    output_dir.mkdir(exist_ok=True)

    for i in range(10):
        filepath = output_dir / f"star_field_{i:03d}.fits"
        gen.generate_and_save(num_stars=100, output_path=filepath)
        print(f"Saved: {filepath}")

    # Later, load one file back
    loaded_img, loaded_labels = SimpleStarGenerator.load_fits(
        output_dir / "star_field_000.fits"
    )
    print(f"\nLoaded image shape: {loaded_img.shape}")
    print(f"Loaded {len(loaded_labels)} stars")
    print("First star:", loaded_labels[0])
