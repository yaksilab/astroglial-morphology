import os
import argparse
import shutil

def find_pairs(folder):
    png_files = [f for f in os.listdir(folder) if f.endswith('.png')]
    pairs = []
    for png in png_files:
        base = png[:-4]
        npy = f"{base}_seg.npy"
        if npy in os.listdir(folder):
            pairs.append((png, npy))
    return pairs

def duplicate_pairs(folder, pairs, n):
    for png, npy in pairs:
        base = png[:-4]
        for i in range(1, n+1):
            new_png = f"{base}_{i}.png"
            new_npy = f"{base}_{i}_seg.npy"
            shutil.copy2(os.path.join(folder, png), os.path.join(folder, new_png))
            shutil.copy2(os.path.join(folder, npy), os.path.join(folder, new_npy))

def main():
    parser = argparse.ArgumentParser(description="Duplicate image/npy pairs n times.")
    parser.add_argument("folder", help="Folder containing .png and *_seg.npy files")
    parser.add_argument("n", type=int, help="Number of copies to make")
    args = parser.parse_args()

    pairs = find_pairs(args.folder)
    duplicate_pairs(args.folder, pairs, args.n)

if __name__ == "__main__":
    main()
