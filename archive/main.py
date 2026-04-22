from pathlib import Path
import csv
import re
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# USER SETTINGS
# =========================================================
folder_path = r"C:\Users\jsayl\OneDrive\Documents\GitHub\Landmark\Landmark Locations 20260422"  # <- change if needed
search_subfolders = True
connect_snapshots = True
save_summary_csv = True


# =========================================================
# HELPERS
# =========================================================
def natural_sort_key(text):
    """
    Sort strings in human order:
    file2 < file10
    """
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', text)]


def find_csv_files(folder, recursive=True):
    folder = Path(folder)
    if recursive:
        files = list(folder.rglob("*.csv"))
    else:
        files = list(folder.glob("*.csv"))
    return sorted(files, key=lambda p: natural_sort_key(p.name))


def is_number(value):
    try:
        float(value)
        return True
    except:
        return False


def extract_first_row_landmarks(csv_file):
    """
    Reads one CSV file and extracts the first numeric row of:
    landmark_11, landmark_12, landmark_13

    Expected file pattern:
    - one row containing:
      Landmark:landmark_11, Landmark:landmark_12, Landmark:landmark_13
    - later, first numeric row begins with frame number
    """
    with open(csv_file, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 5:
        raise ValueError("file too short to parse")

    # Find landmark header row
    landmark_row_idx = None
    for i, row in enumerate(rows):
        joined = " ".join(cell.strip() for cell in row)
        if (
            "Landmark:landmark_11" in joined
            and "Landmark:landmark_12" in joined
            and "Landmark:landmark_13" in joined
        ):
            landmark_row_idx = i
            break

    if landmark_row_idx is None:
        raise ValueError("could not find landmark header row")

    landmark_row = rows[landmark_row_idx]

    # Find start columns for each landmark
    landmark_cols = {}
    for idx, cell in enumerate(landmark_row):
        cell_clean = cell.strip()
        if cell_clean == "Landmark:landmark_11":
            landmark_cols["landmark_11"] = idx
        elif cell_clean == "Landmark:landmark_12":
            landmark_cols["landmark_12"] = idx
        elif cell_clean == "Landmark:landmark_13":
            landmark_cols["landmark_13"] = idx

    needed = {"landmark_11", "landmark_12", "landmark_13"}
    if set(landmark_cols.keys()) != needed:
        raise ValueError("missing one or more required landmarks")

    # Find first numeric data row after header
    data_row = None
    for row in rows[landmark_row_idx + 1:]:
        cleaned = [c.strip() for c in row]

        # Must at least contain enough columns
        if len(cleaned) < max(landmark_cols.values()) + 3:
            continue

        # First column should be frame number
        if cleaned[0] != "" and is_number(cleaned[0]):
            data_row = cleaned
            break

    if data_row is None:
        raise ValueError("no numeric data row found")

    extracted = {}
    for lm, start_col in landmark_cols.items():
        try:
            x = float(data_row[start_col])
            y = float(data_row[start_col + 1])
            z = float(data_row[start_col + 2])
        except Exception as e:
            raise ValueError(f"failed reading {lm} values: {e}")

        extracted[lm] = (x, y, z)

    return extracted


def plot_projection(lm11, lm12, lm13, ordered_names, all_data,
                    connect_snapshots=True, plane="xy"):
    """
    plane options:
    - 'xy'
    - 'xz'
    - 'yz'
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    if plane == "xy":
        i, j = 0, 1
        xlabel, ylabel = "X (mm)", "Y (mm)"
        title = "XY Projection"
    elif plane == "xz":
        i, j = 0, 2
        xlabel, ylabel = "X (mm)", "Z (mm)"
        title = "XZ Projection"
    elif plane == "yz":
        i, j = 1, 2
        xlabel, ylabel = "Y (mm)", "Z (mm)"
        title = "YZ Projection"
    else:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")

    # Scatter points
    ax.scatter(lm11[:, i], lm11[:, j], s=40, label="landmark_11")
    ax.scatter(lm12[:, i], lm12[:, j], s=40, label="landmark_12")
    ax.scatter(lm13[:, i], lm13[:, j], s=40, label="landmark_13")

    # Connect same landmark across files
    if connect_snapshots:
        ax.plot(lm11[:, i], lm11[:, j])
        ax.plot(lm12[:, i], lm12[:, j])
        ax.plot(lm13[:, i], lm13[:, j])

    # Draw triangle for each snapshot
    for name in ordered_names:
        triad = np.array([
            all_data[name]["landmark_11"],
            all_data[name]["landmark_12"],
            all_data[name]["landmark_13"],
            all_data[name]["landmark_11"]
        ])
        ax.plot(triad[:, i], triad[:, j], alpha=0.3)

    # Label first and last snapshot
    ax.text(lm11[0, i], lm11[0, j], ordered_names[0])
    ax.text(lm11[-1, i], lm11[-1, j], ordered_names[-1])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# MAIN
# =========================================================
csv_files = find_csv_files(folder_path, recursive=search_subfolders)

if not csv_files:
    raise FileNotFoundError("No CSV files found in the selected folder.")

all_data = {}
failed_files = []

for file in csv_files:
    # Skip summary/output CSV files so they do not get parsed as inputs
    if file.name.lower() in {"landmark_summary.csv", "_output_landmark_summary.csv"}:
        continue

    base_name = file.stem

    try:
        coords = extract_first_row_landmarks(file)
        all_data[base_name] = coords
        print(f"Loaded: {file.name}")
    except Exception as e:
        failed_files.append((file.name, str(e)))
        print(f"Skipped: {file.name} --> {e}")

print("\nDone.")
print(f"Successful files: {len(all_data)}")
print(f"Failed files: {len(failed_files)}")

if failed_files:
    print("\nFiles that failed:")
    for fname, err in failed_files:
        print(f"  {fname}: {err}")


# =========================================================
# SAVE SUMMARY CSV
# =========================================================
if save_summary_csv and all_data:
    summary_path = Path(folder_path) / "landmark_summary.csv"

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file",
            "lm11_x", "lm11_y", "lm11_z",
            "lm12_x", "lm12_y", "lm12_z",
            "lm13_x", "lm13_y", "lm13_z"
        ])

        for name in sorted(all_data.keys(), key=natural_sort_key):
            d = all_data[name]
            writer.writerow([
                name,
                *d["landmark_11"],
                *d["landmark_12"],
                *d["landmark_13"]
            ])

    print(f"\nSummary saved to: {summary_path}")


# =========================================================
# BUILD ARRAYS FOR PLOTTING
# =========================================================
if all_data:
    ordered_names = sorted(all_data.keys(), key=natural_sort_key)

    lm11 = np.array([all_data[name]["landmark_11"] for name in ordered_names])
    lm12 = np.array([all_data[name]["landmark_12"] for name in ordered_names])
    lm13 = np.array([all_data[name]["landmark_13"] for name in ordered_names])

    # Plot XY
    plot_projection(
        lm11, lm12, lm13, ordered_names, all_data,
        connect_snapshots=connect_snapshots,
        plane="xy"
    )

    # Plot XZ
    plot_projection(
        lm11, lm12, lm13, ordered_names, all_data,
        connect_snapshots=connect_snapshots,
        plane="xz"
    )

    # Plot YZ
    plot_projection(
        lm11, lm12, lm13, ordered_names, all_data,
        connect_snapshots=connect_snapshots,
        plane="yz"
    )

else:
    print("No valid data available for plotting.")