from pathlib import Path
import csv
import re
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# USER SETTINGS
# =========================================================
search_subfolders = True
connect_snapshots = True
save_summary_csv = True

folder_path = r"C:\Users\jsayl\OneDrive\Documents\GitHub\Landmark\ymaze"  # <- change if needed

# =========================================================
# HELPERS
# =========================================================
def natural_sort_key(text):
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


def find_header_and_data_rows(rows):
    """
    Finds:
    - the row containing marker/item names
    - the first numeric data row after that

    Assumes structure like:
    row N: marker names
    row N+1: Frame Sub Frame X Y Z X Y Z ...
    row N+2: units
    row N+3+: numeric rows
    """
    header_row_idx = None
    data_row_idx = None

    for i, row in enumerate(rows):
        cleaned = [c.strip() for c in row]

        # header row usually contains object names and not just X/Y/Z
        nonempty = [c for c in cleaned if c != ""]
        if len(nonempty) >= 1:
            # look ahead to see if next row is Frame/Sub Frame/X/Y/Z
            if i + 1 < len(rows):
                next_row = [c.strip() for c in rows[i + 1]]
                joined_next = " ".join(next_row).lower()
                if "frame" in joined_next and "sub frame" in joined_next and "x" in joined_next and "y" in joined_next and "z" in joined_next:
                    header_row_idx = i
                    break

    if header_row_idx is None:
        raise ValueError("could not find marker/item header row")

    for j in range(header_row_idx + 1, len(rows)):
        cleaned = [c.strip() for c in rows[j]]
        if len(cleaned) > 0 and cleaned[0] != "" and is_number(cleaned[0]):
            data_row_idx = j
            break

    if data_row_idx is None:
        raise ValueError("could not find first numeric data row")

    return header_row_idx, data_row_idx


def get_available_items(csv_file):
    """
    Reads one CSV and returns available marker/item names in the header row.
    """
    with open(csv_file, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)

    header_row_idx, _ = find_header_and_data_rows(rows)
    header_row = [c.strip() for c in rows[header_row_idx]]

    items = []
    for cell in header_row:
        if cell != "":
            items.append(cell)

    return items


def parse_selected_items(csv_file, selected_items):
    """
    Extract first numeric row coordinates for the selected items.

    Returns:
    {
        "maze:take_off": (x, y, z),
        "maze:enter_right": (x, y, z),
        ...
    }
    """
    with open(csv_file, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)

    header_row_idx, data_row_idx = find_header_and_data_rows(rows)
    header_row = [c.strip() for c in rows[header_row_idx]]
    data_row = [c.strip() for c in rows[data_row_idx]]

    # Find each item's starting column
    item_cols = {}
    for idx, cell in enumerate(header_row):
        if cell in selected_items:
            item_cols[cell] = idx

    missing = [item for item in selected_items if item not in item_cols]
    if missing:
        raise ValueError(f"missing selected items: {missing}")

    extracted = {}
    for item, start_col in item_cols.items():
        if len(data_row) < start_col + 3:
            raise ValueError(f"not enough columns for {item}")

        try:
            x = float(data_row[start_col])
            y = float(data_row[start_col + 1])
            z = float(data_row[start_col + 2])
        except Exception as e:
            raise ValueError(f"failed reading XYZ for {item}: {e}")

        extracted[item] = (x, y, z)

    return extracted


def sanitize_name(name):
    """
    Safe filename/column-friendly name
    """
    return re.sub(r'[^A-Za-z0-9_\-]+', '_', name)


def prompt_user_for_items(available_items):
    print("\nAvailable markers/items found:\n")
    for i, item in enumerate(available_items, start=1):
        print(f"{i:>2}. {item}")

    print("\nChoose items to plot:")
    print(" - Enter numbers separated by commas, e.g. 1,3,5")
    print(" - Or type 'all' to use all items")

    while True:
        choice = input("\nYour selection: ").strip().lower()

        if choice == "all":
            return available_items

        try:
            indices = [int(x.strip()) for x in choice.split(",") if x.strip() != ""]
            selected = []
            for idx in indices:
                if idx < 1 or idx > len(available_items):
                    raise ValueError
                selected.append(available_items[idx - 1])

            if len(selected) == 0:
                print("No valid selections entered.")
                continue

            return selected

        except:
            print("Invalid input. Please enter something like 1,2,4 or 'all'.")


def plot_projection(all_data, ordered_names, selected_items, plane="xy", connect_snapshots=True):
    fig, ax = plt.subplots(figsize=(9, 9))

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

    # plot each selected item
    for item in selected_items:
        coords = np.array([all_data[name][item] for name in ordered_names])
        ax.scatter(coords[:, i], coords[:, j], s=40, label=item)

        if connect_snapshots:
            ax.plot(coords[:, i], coords[:, j])

    # connect all selected items within each snapshot
    for name in ordered_names:
        shape = np.array([all_data[name][item] for item in selected_items])
        if len(shape) > 1:
            ax.plot(shape[:, i], shape[:, j], alpha=0.25)

    # label first snapshot for the first selected item
    first_item = selected_items[0]
    first_coords = np.array([all_data[name][first_item] for name in ordered_names])
    ax.text(first_coords[0, i], first_coords[0, j], ordered_names[0])
    ax.text(first_coords[-1, i], first_coords[-1, j], ordered_names[-1])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# =========================================================
# MAIN
# =========================================================
csv_files = find_csv_files(folder_path, recursive=search_subfolders)

if not csv_files:
    raise FileNotFoundError("No CSV files found in the selected folder.")

# Skip summary/output files
input_csv_files = [
    f for f in csv_files
    if f.name.lower() not in {"landmark_summary.csv", "_output_landmark_summary.csv"}
]

if not input_csv_files:
    raise FileNotFoundError("No valid input CSV files found.")

# Use the first valid file to discover available items
reference_file = input_csv_files[0]
available_items = get_available_items(reference_file)

if not available_items:
    raise ValueError("No markers/items found in the reference CSV.")

selected_items = prompt_user_for_items(available_items)

print("\nSelected items:")
for item in selected_items:
    print(f" - {item}")

all_data = {}
failed_files = []

for file in input_csv_files:
    base_name = file.stem
    try:
        coords = parse_selected_items(file, selected_items)
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

# Save summary
if save_summary_csv and all_data:
    summary_path = Path(folder_path) / "selected_item_summary.csv"

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = ["file"]
        for item in selected_items:
            safe = sanitize_name(item)
            header.extend([f"{safe}_x", f"{safe}_y", f"{safe}_z"])
        writer.writerow(header)

        for name in sorted(all_data.keys(), key=natural_sort_key):
            row = [name]
            for item in selected_items:
                row.extend(all_data[name][item])
            writer.writerow(row)

    print(f"\nSummary saved to: {summary_path}")

# Plot
if all_data:
    ordered_names = sorted(all_data.keys(), key=natural_sort_key)

    plot_projection(
        all_data,
        ordered_names,
        selected_items,
        plane="xy",
        connect_snapshots=connect_snapshots
    )

    plot_projection(
        all_data,
        ordered_names,
        selected_items,
        plane="xz",
        connect_snapshots=connect_snapshots
    )

    plot_projection(
        all_data,
        ordered_names,
        selected_items,
        plane="yz",
        connect_snapshots=connect_snapshots
    )
else:
    print("No valid data available for plotting.")